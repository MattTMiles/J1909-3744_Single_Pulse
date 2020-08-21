#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise with unknown variance.

"""
from __future__ import division
import bilby
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import hist, diag
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.special import gamma, factorial

# A few simple setup steps
label = 'linear_regression_unknown_noise'
outdir = 'outdir_normalised_function_result_real_50bins_truenoise'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

fdfs = pd.read_pickle("./Freq_small_df.pkl")
Edata = fdfs["snr"]

#Gauss components
def gauss1(x,f,mu,sigma):
    C1 = 2/((2**(1+(1/2)))*sigma*gamma(1/2))
    return f*C1*np.exp(-0.5*(np.abs((x-mu)/sigma)**2))

def gauss2(x,f,mu,sigma,alpha):
    C2 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return (1-f)*C2*np.exp(-0.5*(np.abs((x-mu)/sigma)**alpha))

def bimodal(x,f,mu1,sigma1,mu2,sigma2,alpha2):
    return gauss1(x,f,mu1,sigma1)+gauss2(x,f,mu2,sigma2,alpha2)

#Model that's in use
def model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]
    for unit in x:

        C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
        C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))
        sigma_noise = 1.1
        result = integrate.quad(lambda xdash: (1/(np.sqrt(2*np.pi*(sigma_noise**2))))*np.exp(-0.5*((xdash**2)/(sigma_noise**2)))*((f*(C1*np.exp(-0.5*np.abs(((unit-xdash)-mu1)/sigma1)**alpha1)))+((1-f)*C2*np.exp(-0.5*np.abs(((unit-xdash)-mu2)/sigma2)**alpha2))),x.min(),x.max())[0]

        resultbin.append(result)

    a = np.asarray(resultbin)
    return a

def deconv_model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin2 =[]
    for unit in x:

        C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
        C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))
        sigma_noise = 1.1
        result2 = integrate.quad(lambda xdash: ((f*(C1*np.exp(-0.5*np.abs(((unit-xdash)-mu1)/sigma1)**alpha1)))+((1-f)*C2*np.exp(-0.5*np.abs(((unit-xdash)-mu2)/sigma2)**alpha2))),x.min(),x.max())[0]

        resultbin2.append(result2)

    a2 = np.asarray(resultbin2)
    return a2



#An analytic gaussian function for comparison where needed
def analytic(x,mu1,sigma1):
    return (1/(np.sqrt(2*np.pi*(1+(sigma1**2)))))*np.exp(-0.5*(((x-mu1)**2)/(1+(sigma1**2))))

# Create random gaussian noise
rmu1, rsigma1 = 0.5, 1.1
s1 = np.random.normal(rmu1, rsigma1, 1000)

rmu2, rsigma2, = 6, 1
s2 = np.random.normal(rmu2, rsigma2, 1000)

noise_mu, noise_sigma = 0, 1
snoise = np.random.normal(noise_mu, noise_sigma, 1000)

#T_s = s1
T_s = np.concatenate((snoise+s1,snoise+s2))

#Introduce the requirements for the SNR probability density functions
E_y,E_x,E_=hist(Edata,50,alpha=.3,label='On-Pulse', density=True)
E_x = (E_x[1:]+E_x[:-1])/2

#Call the likelihood function that is required
likelihood = bilby.core.likelihood.GaussianLikelihood(E_x, E_y, model)

priors = dict()
priors['f'] = bilby.core.prior.Uniform(1e-5, 1-(1e-5), 'f')
#priors['f'] = 1
priors['mu1'] = bilby.core.prior.Uniform(0, 5, 'mu1')
priors['sigma1'] = bilby.core.prior.Uniform(0, 5, 'sigma1')
#priors['A1'] = bilby.core.prior.Uniform(0, 10000, 'A1')
#priors['alpha1'] = bilby.core.prior.Uniform(2, 6, 'alpha1')
priors['alpha1'] = 2
priors['mu2'] = bilby.core.prior.Uniform(5, 10, 'mu2')
priors['sigma2'] = bilby.core.prior.Uniform(0.5, 5, 'sigma2')
#priors['A2'] = bilby.core.prior.Uniform(0, 10000, 'A2')
priors['alpha2'] = bilby.core.prior.Uniform(1.5, 6, 'alpha2')
#priors['alpha2'] = 2
priors['sigma'] = bilby.core.prior.Uniform(1e-5, 500, 'sigma')

# And run sampler\
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=250,
    sample='unif', injection_parameters=None, outdir=outdir,
    label=label)
result.plot_corner()
