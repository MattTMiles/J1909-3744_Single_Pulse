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
outdir = 'outdir_Positive_Pulses_nlive_2000'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

fdfs = pd.read_pickle("./pol_df.pkl")
Edata = fdfs["snr"]

#Model that's in use
def model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]

    C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
    C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))

    #Normalisation integral
    Cbig = integrate.quad(lambda x: \
        (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))\
            +((1-f)*C2*np.exp((-0.5*np.abs((x-mu2)/sigma2)**alpha2)))\
                ,0,x.max())[0]
    

    #Numerical integration
    for unit in x:

        sigma_noise = 1.1
        result = integrate.quad(lambda xdash: \
            (1/(np.sqrt(2*np.pi*(sigma_noise**2))))*np.exp(-0.5*(((unit-xdash)**2)/(sigma_noise**2)))*\
                ((f*C1*np.exp(-0.5*np.abs((xdash-mu1)/sigma1)**alpha1))\
                    +((1-f)*C2*np.exp(-0.5*np.abs((xdash-mu2)/sigma2)**alpha2)))\
                        ,0,x.max())[0]

        resultC = result/Cbig
        resultbin.append(resultC)

    a = np.asarray(resultbin)
    return a


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
priors['sigma2'] = bilby.core.prior.Uniform(0.5, 10, 'sigma2')
#priors['A2'] = bilby.core.prior.Uniform(0, 10000, 'A2')
priors['alpha2'] = bilby.core.prior.Uniform(1.5, 6, 'alpha2')
#priors['alpha2'] = 2
priors['sigma'] = bilby.core.prior.Uniform(1e-5, 500, 'sigma')

# And run sampler\
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=2000,
    sample='unif', injection_parameters=None, outdir=outdir, npool=16,
    label=label)
result.plot_corner()
