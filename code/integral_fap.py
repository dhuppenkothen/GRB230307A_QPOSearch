import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

import numpy as np
import scipy.io
import pickle

import jax
# Important to enable 64-bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
import tensorflow_probability.substrates.jax as tfp

from tinygp import GaussianProcess, kernels
from jaxns.utils import resample

import stingray
from stingray import Lightcurve, Powerspectrum
from stingray.modeling.gpmodeling import get_kernel, get_mean
from stingray.modeling.gpmodeling import get_prior, get_log_likelihood, get_gp_params
from stingray.modeling.gpmodeling import GPResult

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# abbreviations for tensorflow distributions + bijectors
tfpd = tfp.distributions
tfpb = tfp.bijectors

datadir = "/Users/daniela/work/data/grb230307A/"


def load_integral_data(fpath):
    """
    Load the Integral data from 
    the IDL .sav file. The file has two columns: 
    `barytime` and `counts`

    Parameters
    ----------
    fpath : str
        The path to the file with the data

    Returns
    -------
    lc : stingray.Lightcurve object
        The barycentred light curve
    """
    data = scipy.io.readsav(fpath)
    barytime = np.array(data["barytime"], dtype=float)
    counts = np.array(data["counts"], dtype=float)
    mean_bkg = np.mean(counts[-100:])

    lc = Lightcurve(barytime, counts-mean_bkg)
    minind = lc.time.searchsorted(0.0)
    maxind = lc.time.searchsorted(60.0)
    lc = lc.truncate(start=minind, stop=maxind, method="index")

    return lc

def resample_posterior(res, rkey):
    """
    Resample Nested Sampling posterior sample 
    based on the weights to provide unbiased
    posterior samples.

    Parameters
    ----------
    res : jaxns.Results object
        The object with the posterior results of 
        the Nested Sampling run

    rkey : jax.RandomState key
        A random key for reproducibility

    Returns
    -------
    samples_resampled : dict
        A dictionary with the resampled samples
    """
    log_p = res.log_dp_mean #log-prob

    # array for resampled samples
    samples_resampled = {}

    # go through samples, resample with weights to get 
    # a weighted posterior sample
    for name in res.samples.keys():
        samples = res.samples[name]

        weights = jnp.where(jnp.isfinite(samples), jnp.exp(log_p), 0.0)
        log_weights = jnp.where(jnp.isfinite(samples), log_p, -jnp.inf)
        sr = resample(
            rkey, samples, log_weights, S=max(10, int(res.ESS)), replace=True
    )
        samples_resampled[name] = sr

    return samples_resampled


def simulate_grb(pars_log, time):
    """
    Simulate a GRB light curve given a set of parameters.
    
    Parameters
    ----------
    pars_log : dict
        A dictionary with the parameters for the 
        data generation process
        
    time : numpy.ndarray
        An array of time stamps
        
    Returns
    -------
    lcsim : stingray.Lightcurve
        A simulated light curve
    """
    dt = time[1]-time[0]
    
    kernel_type = "RN"
    mean_type = "skew_gaussian"
    params_list = get_gp_params(kernel_type= kernel_type, mean_type = mean_type)

    pars = {}
    for params in params_list:
        if params[0:4] == "log_":
            pars[params[4:]] = jnp.exp(pars_log[params])
        else:
            pars[params] = pars_log[params]

    mean = get_mean(mean_type=mean_type, mean_params=pars)
    kernel = get_kernel(kernel_type=kernel_type, kernel_params=pars)

    gp = GaussianProcess(kernel, time, mean_value=mean(time))
    sample = gp.sample(random.PRNGKey(np.random.randint(0, 1000)))
    
    sample = sample.at[sample <= 0].set(1e-10)
    sample = np.random.poisson(sample)
    lcsim = Lightcurve(time, sample, dt=dt, skip_checks=True)
    
    return lcsim
    


def sample_drw(lc):
    """
    Set up a Damped Random Walk + mean model and sample from 
    the posterior for a given Lightcurve.
    
    Parameters
    ----------
    lc : stingray.Lightcurve object
        The light curve to model with a DRW
        
    Returns
    -------
    gpresult_sim : stingray.modeling.GPResult object
        The object with the sampling results
    """
    times = lc.time
    counts = lc.counts
    
    kernel_type = "RN"
    mean_type = "skew_gaussian"
    params_list = get_gp_params(kernel_type= kernel_type, mean_type = mean_type)

    print("parameters list", params_list)

    total_time = times[-1] - times[0]
    f = 1/(times[1]- times[0])
    span = jnp.max(counts) - jnp.min(counts)

    # The prior dictionary, with suitable tfpd prior distributions
    prior_dict = {
        "t0": tfpd.Uniform(low = 0.0, high = 20.0),
        "log_A": tfpd.Uniform(5, 15),
        "log_sig1": tfpd.Uniform(-1, 3.5),
        "log_sig2": tfpd.Uniform(1, 4.0),
        "log_arn": tfpd.Uniform(2, 20),
        "log_crn": tfpd.Uniform(-10, 10)
    }

    params_list2 = ["log_arn", "log_crn", "log_A", "t0", "log_sig1", "log_sig2"]

    prior_model = get_prior(params_list2, prior_dict)

    log_likelihood_model = get_log_likelihood(params_list2, kernel_type= kernel_type, mean_type = mean_type, 
                                              times = times, counts = counts)
    
    gpresult_sim = GPResult(lc = lc)
    gpresult_sim.sample(prior_model = prior_model, likelihood_model = log_likelihood_model,
                   max_samples=1e5)
    
    return gpresult_sim

def sample_drw_qpo(lc):
    """
    Set up a Damped Random Walk + QPO + mean model and sample from 
    the posterior for a given Lightcurve.
    
    Parameters
    ----------
    lc : stingray.Lightcurve object
        The light curve to model with a DRW
        
    Returns
    -------
    gpresult_sim : stingray.modeling.GPResult object
        The object with the sampling results
    """
    times = lc.time
    counts = lc.counts
    
    kernel_type = "QPO_plus_RN"
    mean_type = "skew_gaussian"
    params_list = get_gp_params(kernel_type= kernel_type, mean_type = mean_type)
    
    total_time = times[-1] - times[0]
    f = 1/(times[1]- times[0])
    span = jnp.max(counts) - jnp.min(counts)

    # The prior dictionary, with suitable tfpd prior distributions
    prior_dict = {
        "t0": tfpd.Uniform(low = 0.0, high = 20.0),
        "log_A": tfpd.Uniform(5, 15),
        "log_sig1": tfpd.Uniform(-1, 3.5),
        "log_sig2": tfpd.Uniform(1, 4.0),
        "log_freq": tfpd.Uniform(np.log(1.0), np.log(3.0)),
        "log_aqpo": tfpd.Uniform(2, 20),
        "log_cqpo": tfpd.Uniform(-10, 10),
        "log_arn": tfpd.Uniform(2, 20),
        "log_crn": tfpd.Uniform(-10, 10)
    }

    prior_model = get_prior(params_list, prior_dict)

    log_likelihood_model = get_log_likelihood(params_list, kernel_type= kernel_type, mean_type = mean_type, 
                                              times = times, counts = counts)

    gpresult_sim = GPResult(lc = lc)
    gpresult_sim.sample(prior_model = prior_model, likelihood_model = log_likelihood_model,
                   max_samples=1e5)
    
    return gpresult_sim

def main():

    # load Integral data
    fpath = datadir + "acs_lc_bary.sav"
    lc = load_integral_data(fpath)

    with open(datadir+"intregal_drw_res.pkl", "rb") as f:
       res = pickle.load(f)

    rkey = random.PRNGKey(12345) 

    samples_resampled = resample_posterior(res, rkey)

    nsim = 100
    nsamples = samples_resampled[list(samples_resampled.keys())[0]].shape[0]

    idx_all = np.random.choice(np.arange(0, nsamples, 1, dtype=int), size=nsim, replace=False)

    kernel_type = "RN"
    mean_type = "skew_gaussian"

    kernel_params = stingray.modeling.gpmodeling._get_kernel_params(kernel_type)
    mean_params = stingray.modeling.gpmodeling._get_mean_params(mean_type)

    all_params = np.hstack([mean_params, kernel_params])

    #mean_samples = dict((k, samples_resampled[k]) for k in mean_params)
    #kernel_samples = dict((k, samples_resampled[k] for k in kernel_params))

    lcsim_all = []

    drw_logz_file = f"{datadir}integral_sim_drw_logz.txt"
    qpo_logz_file = f"{datadir}integral_sim_qpo_logz.txt"

   # with open(drw_logz_file, "w") as f:
   #     f.write("# logz \t d_logz \n")

   # with open(qpo_logz_file, "w") as f:
   #     f.write("# logz \t d_logz \n")

    for i, idx in enumerate(idx_all):
        i+= 10
        print(f"I am on simulation {i}")
        pars_log = dict((k, samples_resampled[k][idx]) for k in all_params)

        lcsim = simulate_grb(pars_log, lc.time)
        
        np.savetxt(f"{datadir}integral_lcsim{i}.dat", np.array([lcsim.time, lcsim.counts]).T)
        lcsim_all.append(lcsim)
        
        print("Sampling the DRW ...")
        # sample the DRW
        gpresult_drw = sample_drw(lcsim)
        
        with open(f"{datadir}intregal_drw_res_sim{i}.pkl", "wb") as f:
            pickle.dump(gpresult_drw.results, f)
            
        with open(drw_logz_file, "a") as f:
            logz_drw = f"{gpresult_drw.results.log_Z_mean} \t {gpresult_drw.results.log_Z_uncert} \n"
            f.write(logz_drw)
            
        print("Sampling the DRW + QPO model ...")
        # sample the DRW + QPO
        gpresult_qpo = sample_drw_qpo(lcsim)
        
        with open(f"{datadir}intregal_drw_qpo_res_sim{i}.pkl", "wb") as f:
            pickle.dump(gpresult_qpo.results, f)
            
        with open(qpo_logz_file, "a") as f:
            logz_qpo = f"{gpresult_qpo.results.log_Z_mean} \t {gpresult_qpo.results.log_Z_uncert} \n"
            f.write(logz_qpo)

        print("... and done! \n")

    return
    
if __name__ == "__main__":
    main()