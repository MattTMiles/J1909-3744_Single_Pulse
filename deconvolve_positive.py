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
import os

os.chdir('/fred/oz002/users/mmiles/SinglePulse/')

fdfs = pd.read_pickle("./pol_df.pkl")
Edata = fdfs["snr"]

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

        #resultC = result/Cbig
        resultbin.append(result)

    a = np.asarray(resultbin)
    a = a/Cbig
    return a

def dc_model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]
    Call = []

    C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
    C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))

    #Normalisation integral
    Cbig = integrate.quad(lambda x: \
        (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))\
            +((1-f)*C2*np.exp((-0.5*np.abs((x-mu2)/sigma2)**alpha2)))\
                ,0,x.max())[0]


    result = (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))\
            +((1-f)*C2*np.exp(-0.5*np.abs((x-mu2)/sigma2)**alpha2))
        #print(result)
        #resultbin.append(result)

    
    resultbin = np.asarray(resultbin)
    result = result/Cbig
    
    a = result
    return a, Cbig

def gausscomp1(x,f,mu,sigma,alpha):
    C1 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return f*C1*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

def gausscomp2(x,f,mu,sigma,alpha):
    C2 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return (1-f)*C2*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

Bilby_params = (0.16,0.13,0.28,2,7.28,7.29,4.11)

E_y,E_x,E_=hist(Edata, 50, alpha=.3, label='On-Pulse', density=True)
E_x = (E_x[1:]+E_x[:-1])/2
min_x = E_x.min()
max_x = E_x.max()
xarray = np.linspace(min_x,max_x,200)

xarray_0 = np.linspace(0,max_x,200)

convolved_model = model(xarray, *Bilby_params)

deconvolved_model, Cbig = dc_model(xarray, *Bilby_params)

heavi_x = np.heaviside(xarray,1)
heavi_x[25]=0.565
#heavi_dc, heavi_Cbig = dc_model(heavi_x, *Bilby_params)

noise_model = (1/(1.1*np.sqrt(2*np.pi)))*np.exp(-0.5*(np.abs(xarray/1.1)**2))


anothernoise = (1/(1.1*np.sqrt(2*np.pi))*np.exp(-0.5*np.abs((xarray)/1.1)**2))/Cbig

reconvolve = np.convolve(deconvolved_model, noise_model, "full")

heavi_dc = heavi_x*deconvolved_model

heavi_reconvolve = np.convolve(heavi_dc, noise_model, "full")
#heavi_reconvolve2 = np.convolve(heavi_dc, noise_model, "full") 

#heavi_reconvolve2 = (heavi_reconvolve2/(max(heavi_reconvolve2)))

heavi_reconvolve = (heavi_reconvolve/(max(heavi_reconvolve)))

xarray2 = np.linspace(min_x,max_x,len(reconvolve))
dx = xarray[1]-xarray[0]
reconvolve = (reconvolve/(max(reconvolve)+sys.float_info[3]))

convolved_modelnorm = model(xarray2, *Bilby_params)
convolved_modelnorm = convolved_modelnorm/(max(convolved_modelnorm))

gauss1 = gausscomp1(xarray,Bilby_params[0],Bilby_params[1],Bilby_params[2],Bilby_params[3])/Cbig
gauss1 = heavi_x*gauss1
gauss2 = gausscomp2(xarray,Bilby_params[0],Bilby_params[4],Bilby_params[5],Bilby_params[6])/Cbig
gauss2 = heavi_x*gauss2

#deconvolved_modelnorm = dc_model(xarray2, *Bilby_params)
plt.plot(xarray,heavi_dc,label='heaviside deconvolved')
plt.plot(xarray,convolved_model, label= "Model")
#plt.plot(xarray2, convolved_modelnorm, label= "Normalised model")

#plt.plot(xarray2*2,reconvolve, label= "reconvolved")
#plt.plot(xarray2*2, heavi_reconvolve,label='heaviside reconvolve')
#plt.plot(xarray2*2, heavi_reconvolve2,label='heaviside reconvolve 2')
plt.plot(xarray,gauss1, label = "Heaviside weak mode")
plt.plot(xarray, gauss2, label = "Heaviside strong Mode")
#plt.plot(xarray2*2,reconvolve2, label= "reconvolve2")
#plt.plot(xarray_0,deconvolved_model, label= "Deconvolved Model")
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel("SNR")
plt.ylabel("Probability Density")
plt.legend(prop={'size':6})
plt.show()
