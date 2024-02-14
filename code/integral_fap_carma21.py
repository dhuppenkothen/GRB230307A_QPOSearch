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
from tinygp.kernels.quasisep import CARMA

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

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.nested_sampling import NestedSampler

import arviz as az
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


def skewgaussian(t, params):
    #parameter_names = ("logA", "t0", "logsig1", "logsig2", "logconst")
    logA = params[0]
    t0 = params[1]
    logsig1 = params[2]
    logsig2 = params[3]
    logconst = params[4]

    y = jnp.exp(logA) * jnp.where(
            t > t0,
            jnp.exp(-((t - t0) ** 2) / (2 * (jnp.exp(logsig2)**2))),
            jnp.exp(-((t - t0) ** 2) / (2 * (jnp.exp(logsig1)**2))),
        )
    return y + jnp.exp(logconst)

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
    
    p = 2
    q = 1
    mean_var_names = ["log_amp", "t0", "log_sig1",
                    'log_sig2', "log_const"]

    mean_params = [pars_log["log_amp"],
                   pars_log["t0"],
                   pars_log["log_sig1"],
                   pars_log["log_sig2"],
                   pars_log["log_const"]]

    alpha = [jnp.exp(pars_log["log_alpha1"]), jnp.exp(pars_log["log_alpha2"])]
    beta = [jnp.exp(pars_log["log_beta"])]

    mean_val = skewgaussian(time, mean_params)
    kernel = CARMA.init(alpha=alpha, beta=beta)

    gp = GaussianProcess(kernel, time, diag=0.01)

    key = random.PRNGKey(np.random.randint(0, 1000))
    sample = gp.sample(key) + mean_val
    
    sample = sample.at[sample <= 0].set(1e-10)
    sample = np.random.poisson(sample)
    lcsim = Lightcurve(time, sample, dt=dt, skip_checks=True)
    
    return lcsim
    
def carma21_integral(t, y=None):

    mean_bkg = np.mean(y[-100:])
    # mean parameters
    log_amp = numpyro.sample('log_amp', dist.Uniform(9, 12.0))
    t0 = numpyro.sample("t0", dist.Uniform(0.0, 10))
    log_sig1 = numpyro.sample("log_sig1", dist.Uniform(-5, 1.5))
    log_sig2 = numpyro.sample("log_sig2", dist.Uniform(1, 4))
    log_const = numpyro.sample("log_const", dist.Normal(mean_bkg, mean_bkg/5))

    params = [log_amp, t0, log_sig1,log_sig2, log_const]
    
    mean = skewgaussian(t, params)
    
    # kernel parameters
    log_alpha1 = numpyro.sample("log_alpha1", dist.Uniform(-10, 20))
    log_alpha2 = numpyro.sample("log_alpha2", dist.Uniform(-10, 20))
    
    log_beta = numpyro.sample("log_beta", dist.Uniform(-10, 20))
        
    kernel = CARMA.init(alpha=[jnp.exp(log_alpha1), jnp.exp(log_alpha2)], 
                        beta=[jnp.exp(log_beta)])
    
    gp = GaussianProcess(kernel, t, diag=y, mean_value=mean)
        
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        numpyro.deterministic("pred", gp.condition(y, t).gp.loc)

def sample_carma21(lc):
    """
    Set up a CARMA(2,1) + mean model and sample from 
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
    
    ns = NestedSampler(carma21_integral)
    ns.run(random.PRNGKey(200), lc.time, y=lc.counts)   
    
    return ns

def carma21_integral_with_qpo(t, y=None):

    mean_bkg = np.mean(y[-100:])

     # mean parameters
    log_amp = numpyro.sample('log_amp', dist.Uniform(9, 12.0))
    t0 = numpyro.sample("t0", dist.Uniform(0.0, 10))
    log_sig1 = numpyro.sample("log_sig1", dist.Uniform(-5, 1.5))
    log_sig2 = numpyro.sample("log_sig2", dist.Uniform(1, 4))
    log_const = numpyro.sample("log_const", dist.Normal(mean_bkg, mean_bkg/5))

    params = [log_amp, t0, log_sig1,log_sig2, log_const]
    
    mean = skewgaussian(t, params)
    
    # kernel parameters
    log_alpha1 = numpyro.sample("log_alpha1", dist.Uniform(-10, 20))
    log_alpha2 = numpyro.sample("log_alpha2", dist.Uniform(-10, 20))
    
    log_beta = numpyro.sample("log_beta", dist.Uniform(-10, 20))

    kernel_carma = CARMA.init(alpha=[jnp.exp(log_alpha1), jnp.exp(log_alpha2)], 
                        beta=[jnp.exp(log_beta)])
        
    # QPO kernel parameters
    log_aqpo = numpyro.sample("log_aqpo", dist.Uniform(1, 15))
    log_cqpo = numpyro.sample("log_cqpo", dist.Uniform(-20, 2))
    log_freq = numpyro.sample("log_freq", dist.Uniform(np.log(0.1), np.log(5)))
    
    # QPO kernel
    kernel_qpo = kernels.quasisep.Celerite(
            a=jnp.exp(log_aqpo),
            b=0.0,
            c=jnp.exp(log_cqpo),
            d=2 * jnp.pi * jnp.exp(log_freq),
        )
    
    # add kernels together
    kernel = kernel_carma + kernel_qpo
    
    # gaussian process
    gp = GaussianProcess(kernel, t, diag=y, mean_value=mean)
    
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        numpyro.deterministic("pred", gp.condition(y, t).gp.loc)

def sample_carma21_qpo(lc):
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
    ns_qpo = NestedSampler(carma21_integral_with_qpo)
    ns_qpo.run(random.PRNGKey(2), lc.time, y=lc.counts)
    return ns_qpo

def main():

    # load Integral data
    fpath = datadir + "acs_lc_bary.sav"
    lc = load_integral_data(fpath)

    with open(datadir + "integral_carma21_ns.pkl", "rb") as f:
        ns = pickle.load(f)
    
    samples, weights = ns.get_weighted_samples()

    all_params = list(samples.keys())

    nsim = 100
    nsamples = samples[list(samples.keys())[0]].shape[0]

    idx_all = np.random.choice(np.arange(0, nsamples, 1, dtype=int), size=nsim, replace=False)

    lcsim_all = []

    carma21_logz_file = f"{datadir}integral_sim_carma21_logz.txt"
    qpo_logz_file = f"{datadir}integral_sim_carma21_qpo_logz.txt"

    for i, idx in enumerate(idx_all):
        print(f"I am on simulation {i}")
        pars_log = dict((k, samples[k][idx]) for k in all_params)

        lcsim = simulate_grb(pars_log, lc.time)
        
        np.savetxt(f"{datadir}integral_lcsim{i}.dat", np.array([lcsim.time, lcsim.counts]).T)
        lcsim_all.append(lcsim)
        
        print("Sampling the CARMA(2,1) process ...")
        # sample the DRW
        ns_rn = sample_carma21(lcsim)
        
        with open(f"{datadir}integral_carm21_res_sim{i}.pkl", "wb") as f:
            pickle.dump(ns_rn._results, f)
            
        with open(carma21_logz_file, "a") as f:
            logz_drw = f"{ns_rn._results.log_Z_mean} \t {ns_rn._results.log_Z_uncert} \n"
            f.write(logz_drw)
            
        print("Sampling the CARMA(2,1) + QPO model ...")
        # sample the DRW + QPO
        ns_qpo = sample_carma21_qpo(lcsim)
        
        with open(f"{datadir}intregal_drw_qpo_res_sim{i}.pkl", "wb") as f:
            pickle.dump(ns_qpo._results, f)
            
        with open(qpo_logz_file, "a") as f:
            logz_qpo = f"{ns_qpo._results.log_Z_mean} \t {ns_qpo._results.log_Z_uncert} \n"
            f.write(logz_qpo)

        print("... and done! \n")

    return
    
if __name__ == "__main__":
    main()
