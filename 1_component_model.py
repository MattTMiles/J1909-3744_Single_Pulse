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
outdir = 'outdir_single_component_model_2'
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

#Model that's in use (single component)
def model(x,mu1,sigma1,alpha1):

    resultbin =[]
    for unit in x:

        C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
        sigma_noise = 1.1
        result = integrate.quad(lambda xdash: \
            (1/(np.sqrt(2*np.pi*(sigma_noise**2))))*np.exp(-0.5*((xdash**2)/(sigma_noise**2)))*\
            (C1*np.exp(-0.5*np.abs(((unit-xdash)-mu1)/sigma1)**alpha1))\
                    ,x.min(),x.max())[0]

        resultbin.append(result)

    a = np.asarray(resultbin)
    return a

#An analytic gaussian function for comparison where needed
def analytic(x,mu1,sigma1):
    return (1/(np.sqrt(2*np.pi*(1+(sigma1**2)))))*np.exp(-0.5*(((x-mu1)**2)/(1+(sigma1**2))))

#Introduce the requirements for the SNR probability density functions
E_y,E_x,E_=hist(Edata,50,alpha=.3,label='On-Pulse', density=True)
E_x = (E_x[1:]+E_x[:-1])/2

#Call the likelihood function that is required
likelihood = bilby.core.likelihood.GaussianLikelihood(E_x, E_y, model)

priors = dict()

priors['mu1'] = bilby.core.prior.Uniform(0, 20, 'mu1')
priors['sigma1'] = bilby.core.prior.Uniform(0, 20, 'sigma1')
priors['alpha1'] = bilby.core.prior.Uniform(1, 100, 'alpha1')
priors['sigma'] = bilby.core.prior.Uniform(1e-5, 500, 'sigma')

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=250,
    sample='unif', injection_parameters=None, outdir=outdir,
    label=label)
result.plot_corner()
