import numpy as np
import scipy.io

# Need both Jax and Tensorflow Probability 
import jax
# Important to enable 64-bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
import tensorflow_probability.substrates.jax as tfp

# TinyGP for Gaussian Processes, JaxNS and numpyro for sampling
from tinygp import GaussianProcess, kernels
from tinygp.kernels.quasisep import CARMA

from jaxns.utils import resample
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoNormal

#stingray imports
import stingray
from stingray import Lightcurve, Powerspectrum
from stingray.modeling.gpmodeling import get_kernel, get_mean
from stingray.modeling.gpmodeling import get_prior, get_log_likelihood, get_gp_params
from stingray.modeling.gpmodeling import _skew_gaussian

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# abbreviations for tensorflow distributions + bijectors
tfpd = tfp.distributions
tfpb = tfp.bijectors

def power_spectrum(freq, sigma, ar_coef, ma_coefs=[1.0]):
    """
    FROM CARMApack (Kelly et al, 2014)
    
    Return the power spectrum for a CARMA(p,q) process calculated at the input frequencies.

    :param freq: The frequencies at which to calculate the PSD.
    :param sigma: The standard deviation driving white noise.
    :param ar_coef: The CARMA model autoregressive coefficients.
    :param ma_coefs: Coefficients of the moving average polynomial

    :rtype : The power spectrum at the input frequencies, a numpy array.
    """
    try:
        len(ma_coefs) <= len(ar_coef)
    except ValueError:
        "Size of ma_coefs must be less or equal to size of ar_roots."

    ma_poly = np.polyval(ma_coefs[::-1], 2.0 * np.pi * 1j * freq)  # Evaluate the polynomial in the PSD numerator
    ar_poly = np.polyval(ar_coef, 2.0 * np.pi * 1j * freq)  # Evaluate the polynomial in the PSD denominator
    pspec = sigma ** 2 * np.abs(ma_poly) ** 2 / np.abs(ar_poly) ** 2
    return pspec

def skewgaussian(t, params):
    #parameter_names = ("logA", "t0", "logsig1", "logsig2")
    logA = params[0]
    t0 = params[1]
    logsig1 = params[2]
    logsig2 = params[3]

    y = jnp.exp(logA) * jnp.where(
            t > t0,
            jnp.exp(-((t - t0) ** 2) / (2 * (jnp.exp(logsig2)**2))),
            jnp.exp(-((t - t0) ** 2) / (2 * (jnp.exp(logsig1)**2))),
        )
    return y

def generate_data():
    # Make a time array
    dt = 0.05
    tseg = 60.0
    
    time = np.linspace(0, 60, int(60/0.05))
    
    alpha = [3.5, 4.5, 2.0]
    beta = [1.5]
    sigma = 1.5
    
    skew_amp = 400
    skew_t0 = 6.0
    skew_sig1 = 2.0
    skew_sig2 = 10.0
    const = 10.0
    
    
    mean_params_true = [np.log(skew_amp),
                   skew_t0,
                   np.log(skew_sig1),
                   np.log(skew_sig2),]
    
    
    mean_val = skewgaussian(time, mean_params_true)
    
    kernel_params_true = np.hstack([alpha, beta, sigma])
    
    kernel = CARMA.init(alpha=alpha, beta=beta, sigma=sigma)
    gp = GaussianProcess(kernel, time, diag=0.01)
    
    key = random.PRNGKey(1200)
    sample = jnp.exp(gp.sample(key)) * mean_val + const
    sample = np.random.poisson(sample)
    
    gp.condition(sample)
    
    lcsample = Lightcurve(time, sample)
    pssample = Powerspectrum(lcsample, norm="frac")

    return lcsample, pssample

def model_with_mean(x, y=None):
 
    # mean parameters
    log_amp = numpyro.sample('log_amp', dist.Uniform(2, 7))
    t0 = numpyro.sample("t0", dist.Uniform(2, 10))
    log_sig1 = numpyro.sample("log_sig1", dist.Uniform(-1, 1.5))
    log_sig2 = numpyro.sample("log_sig2", dist.Uniform(1, 3))

    #params = [log_amp, t0, log_sig1,log_sig2]
    
    #mean = skewgaussian(x, params)
    
    # kernel parameters
    log_alpha1 = numpyro.sample("log_alpha1", dist.Uniform(-1, 2.5))
    log_alpha2 = numpyro.sample("log_alpha2", dist.Uniform(-1, 2.5))
    log_alpha3 = numpyro.sample("log_alpha3", dist.Uniform(-1, 2.5))
    
    log_beta = numpyro.sample("log_beta", dist.Uniform(-1, 2.5))
    log_sigma = numpyro.sample("log_sigma", dist.Uniform(0, 5.3))
    
    
    kernel = CARMA.init(alpha=[jnp.exp(log_alpha1), jnp.exp(log_alpha2), jnp.exp(log_alpha3)], 
                        beta=[jnp.exp(log_beta)], sigma=jnp.exp(log_sigma))
    
    gp = GaussianProcess(kernel, x, diag=0.01)  
        
    # This parameter has shape (num_data,) and it encodes our beliefs about
    # the process rate in each bin
    log_rate = numpyro.sample("log_rate", gp.numpyro_dist())

    # Finally, our observation model is Poisson
    numpyro.sample("obs", dist.Poisson(jnp.exp(log_rate)), obs=y)

def main():
    lc, ps = generate_data()

    npoints = lc.n
    optim = numpyro.optim.Adam(0.01)
    guide = AutoNormal(model_with_mean)
    svi = numpyro.infer.SVI(model_with_mean, guide, optim, numpyro.infer.Trace_ELBO(10))
    results = svi.run(jax.random.PRNGKey(100), 3000, lc.time[:npoints], y=lc.counts[:npoints], progress_bar=False)

    return

if __name__ == "__main__":
    main()    
