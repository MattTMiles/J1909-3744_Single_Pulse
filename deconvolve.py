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

#This is the deconvolved version of the above.
def dc_model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):
    
    C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
    C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))

    pulse_PDF = (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))+((1-f)*C2*np.exp(-0.5*np.abs((x-mu2)/sigma2)**alpha2))
    return pulse_PDF

def gausscomp1(x,f,mu,sigma,alpha):
    C1 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return f*C1*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

def gausscomp2(x,f,mu,sigma,alpha):
    C2 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return (1-f)*C2*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

#These are the paramaters found from bilby. 
#They represent: f, mu1, sigma1, alpha1, mu2, sigma2, alpha2
#Alpha1 has been preset as alpha1=2 prior to bilby.
Bilby_params = (0.16,0.58,0.55,2,5.97,4.54,3.57)

#Based on the real data lets bound our distribution
E_y,E_x,E_=hist(Edata, 50, alpha=.3, label='On-Pulse', density=True)
E_x = (E_x[1:]+E_x[:-1])/2
min_x = E_x.min()
max_x = E_x.max()
xarray = np.linspace(min_x,max_x,200)
#plt.figure()
#This is the convolved model fitted to the data based on the parameters found by Bilby
convolved_model = model(xarray, *Bilby_params)

#This is the deconvolved model fitted to the data based on the parameters found by Bilby
deconvolved_model = dc_model(xarray, *Bilby_params)

#This is a model of just the noise, 0 mean, 1.1 sigma

noise_model = 1/(1.1*np.sqrt(2*np.pi))*np.exp(-0.5*np.abs(xarray/1.1)**2)

#As a sanity check let's convolve the noise model with t`he deconvolved model and make sure we get the same thing
#(This does work but only where we plot with xarray2*2 for the reconvolved array, uncerain why)
reconvolve = np.convolve(deconvolved_model, noise_model, "full")
xarray2 = np.linspace(min_x,max_x,len(reconvolve))
dx = xarray[1]-xarray[0]
reconvolve = (reconvolve/(max(reconvolve)+sys.float_info[3]))

convolved_modelnorm = model(xarray2, *Bilby_params)
#convolved_modelnorm = convolved_modelnorm/(max(convolved_modelnorm)+sys.float_info[3])
#plt.plot(xarray2*2,reconvolve,label="reconvolved using scipy")
gauss1 = gausscomp1(xarray,0.16,0.58,0.55,2)
gauss2 = gausscomp2(xarray,0.16,5.97,4.54,3.57)

plt.plot(xarray2,convolved_modelnorm, label= "Model")
plt.plot(xarray,deconvolved_model, label= "Deconvolved Model")
plt.plot(xarray,gauss1, label = "Weak mode")
plt.plot(xarray, gauss2, label = "Strong Mode")

plt.xlabel("SNR")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
'''


area_weak = integrate.trapz(gauss1)
area_strong = integrate.trapz(gauss2)

reldist_weak = (area_weak/(area_weak+area_strong))*100
reldist_strong = (area_strong/(area_strong+area_weak))*100

print(f'Weak pulses are {reldist_weak:.2f}% of the total distribution, and strong pulses are {reldist_strong:.2f}%.')

plt.plot(xarray,convolved_model,label="Convolved Distribution")
plt.plot(xarray,deconvolved_model,label="Deconvolved Distribution")
plt.plot(xarray,gauss1,label="Weak Mode Distribution")
plt.plot(xarray,gauss2,label="Strong Mode Distribution")

#plt.plot(xarray,noise_model, label="Noise Model")
#plt.ylim(10e-5,1)
#plt.xlim(min_x,max_x)

plt.legend()
plt.show()
'''