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

# A few simple setup steps
label = 'linear_regression_unknown_noise'
outdir = 'outdir_alpha26_1_simulated'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

fdfs = pd.read_pickle("./Freq_small_df.pkl")
Edata = fdfs["snr"]
'''
# First, we define our "signal model", in this case a simple linear function
def gauss3(x,f,mu,sigma,A):
    return f*(A/((2*np.pi*(sigma**2))**0.5))*np.exp(-0.5*(np.abs(((x-mu)/sigma)**2)))

def gauss4(x,f,mu,sigma,A):
    return (1-f)*(A/((2*np.pi*(sigma**2))**0.5))*np.exp(-0.5*(np.abs(((x-mu)/sigma)**2)))

def model(x,f,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss3(x,f,mu1,sigma1,A1)+gauss4(x,f,mu2,sigma2,A2)
'''
#Lets create another version that uses numerical integration

def gauss1(x,f,mu,sigma,A,alpha):
    return f*A*np.exp(-0.5*(np.abs((x-mu)/sigma)**alpha))

def gauss2(x,f,mu,sigma,A,alpha):
    return (1-f)*A*np.exp(-0.5*(np.abs((x-mu)/sigma)**alpha))

def bimodal(x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2):
    return gauss1(x,f,mu1,sigma1,A1,alpha1)+gauss2(x,f,mu2,sigma2,A2,alpha2)

def model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]
    for unit in x:
        result = integrate.quad(lambda xdash: (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(xdash**2))*((f*((1/np.sqrt(2*np.pi*sigma1**alpha1))*np.exp(-0.5*np.abs(((unit-xdash)-mu1)/sigma1)**alpha1)))+((1-f)*(1/np.sqrt(2*np.pi*sigma2**alpha2))*np.exp(-0.5*np.abs(((unit-xdash)-mu2)/sigma2)**alpha2))),x.min(),x.max())[0]
        
        resultbin.append(result)

    a = np.asarray(resultbin)
    return a
'''
def model(x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2):
    resultbin =[]
    xmin = x.min()
    xmax = x.max()
    for unit in x:
        result = integrate.quad(lambda xdash: (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(xdash**2))*bimodal(unit-xdash,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2),xmin,xmax)[0]
        resultbin.append(result)
    
    a = np.asarray(resultbin)
    return a
'''

def analytic(x,mu1,sigma1):
    return (1/(np.sqrt(2*np.pi*(1+(sigma1**2)))))*np.exp(-0.5*(((x-mu1)**2)/(1+(sigma1**2))))

#def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
#    return gauss1(x,mu1,sigma1,A1)+gauss2(x,mu2,sigma2,A2)


# Now we define the injection parameters which we make simulated data with
#injection_parameters = dict(m=0.5, c=0.2)


# Create random gaussian noise
rmu1, rsigma1 = 0.5, 1.1
s1 = np.random.normal(rmu1, rsigma1, 1000)

rmu2, rsigma2, = 6, 1
s2 = np.random.normal(rmu2, rsigma2, 1000)

noise_mu, noise_sigma = 0, 1
snoise = np.random.normal(noise_mu, noise_sigma, 1000)

#T_s = s1
T_s = np.concatenate((snoise+s1,snoise+s2))

'''
# For this example, we'll inject standard Gaussian noise
sigma = 1

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, 'o', label='data')
ax.plot(time, model(time, **injection_parameters), '--r', label='signal')
ax.set_xlabel('time')
ax.set_ylabel('y')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))
'''
#sigma = 1
#injection_parameters.update(dict(sigma=1))

# Now lets instantiate the built-in GaussianLikelihood, giving it
# the time, data and signal model. Note that, because we do not give it the
# parameter, sigma is unknown and marginalised over during the sampling

E_y,E_x,E_=hist(T_s,50,alpha=.3,label='On-Pulse', density=True)
E_x = (E_x[1:]+E_x[:-1])/2

likelihood = bilby.core.likelihood.GaussianLikelihood(E_x, E_y, model)

priors = dict()
priors['f'] = bilby.core.prior.Uniform(1e-5, 1-(1e-5), 'f')
#priors['f'] = 1
priors['mu1'] = bilby.core.prior.Uniform(0, 5, 'mu1')
priors['sigma1'] = bilby.core.prior.Uniform(0, 5, 'sigma1')
#priors['A1'] = bilby.core.prior.Uniform(0, 10000, 'A1')
priors['alpha1'] = bilby.core.prior.Uniform(2, 6, 'alpha1')
#priors['alpha1'] = 2
priors['mu2'] = bilby.core.prior.Uniform(5, 10, 'mu2')
priors['sigma2'] = bilby.core.prior.Uniform(0.5, 5, 'sigma2')
#priors['A2'] = bilby.core.prior.Uniform(0, 10000, 'A2')
priors['alpha2'] = bilby.core.prior.Uniform(2, 6, 'alpha2')
#priors['alpha2'] = 2
priors['sigma'] = bilby.core.prior.Uniform(1e-5, 500, 'sigma')

# And run sampler\
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=250,
    sample='unif', injection_parameters=None, outdir=outdir,
    label=label)
result.plot_corner()
