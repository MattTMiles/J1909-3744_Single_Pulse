from __future__ import division
import numpy as np
from scipy import signal
from scipy.signal import convolve, deconvolve
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pylab import hist, diag
import scipy.integrate as integrate
from scipy.special import gamma, factorial

fdfs = pd.read_pickle("./Freq_small_df.pkl")
Edata = fdfs["snr"]

def gaussian(x,mu,sigma):
    return np.exp(-0.5*(np.abs((x-mu)/sigma)**2))

#Convolved, normalised gaussian function
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

#These are the paramaters found from bilby. 
#They represent: f, mu1, sigma1, alpha1, mu2, sigma2, alpha2
#Alpha1 has been preset as alpha1=2 prior to bilby.
Bilby_params = (0.16,0.58,0.55,2,5.97,4.54,3.57)

#Based on the real data lets bound our distribution
E_y,E_x,E_=hist(Edata,50,alpha=.3,label='On-Pulse', density=True)
E_x = (E_x[1:]+E_x[:-1])/2

#This is the convolved model fitted to the data based on the parameters found by Bilby
convolved_model = model(E_x, *Bilby_params)

#Define here a signal such that convolved_model == convolve(signal_True,signal_Noise)
#We want to isolate signal_True

#This is the noise in the signal as defined by the off-pulse window SNR distribution
#It's important to make the signal shorter than the total
x_noise = E_x[:25]
signal_Noise = gaussian(x_noise,0,1.1)

#This is the true signal without noise built from a gaussian noise distribution and the model
signal_True = deconvolve(convolved_model,signal_Noise)[0]

#The deconvolution comes out as len(E_x)-len(signal_Noise)+1
n = len(E_x)-len(signal_Noise)+1
#So we can expand it on both sides by the factor below to make it work
s = (len(E_x)-n)/2

signal_True_res = np.zeros(len(E_x))
signal_True_res[s:len(E_x)-s-1] = signal_True
signal_True = signal_True_res
#This should mean that signal_True now contains the doconvoluted version
#Expanded to the original shape
