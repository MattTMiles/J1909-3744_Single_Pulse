#Numerically integrates over a region to create a probability distribution

from __future__ import division
import bilby
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import hist, diag
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.optimize import curve_fit

def gauss1(x,f,mu,sigma,A,alpha):
    return f*A*np.exp(-0.5*(np.abs((x-mu)/sigma)**alpha))

def gauss2(x,f,mu,sigma,A,alpha):
    return (1-f)*A*np.exp(-0.5*(np.abs((x-mu)/sigma)**alpha))

def bimodal(x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2):
    return gauss1(x,f,mu1,sigma1,A1,alpha1)+gauss2(x,f,mu2,sigma2,A2,alpha2)

def model(x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2):
    resultbin =[]
    
    for unit in x:
        result = integrate.quad(lambda xdash: (1/(np.sqrt(2*(np.pi**2))))*np.exp(-0.5*(xdash**2))*bimodal(unit-xdash,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2),-np.inf,np.inf)[0]
        
        resultbin.append(result)
    
    return resultbin

def model2(x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2):
    resultbin2 =[]
    xdash = np.linspace(x.min(),x.max(),100)    
    for unit in x:
        result2 = integrate.simps((1/(np.sqrt(2*(np.pi**2))))*np.exp(-0.5*(xdash**2))*bimodal(unit-xdash,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2),xdash)
        #result2 = integrate.simps((1/(np.sqrt(2*(np.pi**2))))*np.exp(-0.5*(xdash**2))*gauss1(unit-xdash,f,mu1,sigma1,A1,alpha1),xdash)
       
        resultbin2.append(result2)
    
    return resultbin2

def gauss3(x,f,mu,sigma,A):
    return f*(A/((2*np.pi*(sigma**2))**0.5))*np.exp(-0.5*(np.abs(((x-mu)/sigma)**2)))

def gauss4(x,f,mu,sigma,A):
    return (1-f)*(A/((2*np.pi*(sigma**2))**0.5))*np.exp(-0.5*(np.abs(((x-mu)/sigma)**2)))

def model2(x,f,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss3(x,f,mu1,sigma1,A1)+gauss4(x,f,mu2,sigma2,A2)


fdfs = pd.read_pickle("./Freq_small_df.pkl")
Edata = fdfs["snr"]

E_y,E_x,E_=hist(Edata,100,alpha=.3,label='On-Pulse')
E_x = (E_x[1:]+E_x[:-1])/2
f = 0.5

mu1, sigma1, A1, mu2, sigma2, A2 = (0.96,1.1,4000,7.3,2.7,4000)
expected = (0.5,0.96,1.1,4000,2,7.3,2.7,4000,2)
#alpha1 = 2
#alpha2 = 2
raise a
E_params,E_cov=curve_fit(model,E_x,E_y,expected)
E_sigma=np.sqrt(diag(E_cov))
E_output = model(E_x,*E_params)

print("Curve-fit parameters are: f={:.4f}, mu1={:.4f}, sigma1={:.4f}, A1={:.4f}, alpha1={:.4f}, mu2={:.4f}, sigma2={:.4f}, A2={:.4f}, alpha2={:.4f}".format(E_params[0], E_params[1], E_params[2], E_params[3], E_params[4], E_params[5], E_params[6], E_params[7], E_params[8]))
print("With errors: f=+/-{:.4f}, mu1=+/-{:.4f}, sigma1=+/-{:.4f}, A1=+/-{:.4f}, mu2=+/-{:.4f}, sigma2=+/-{:.4f}, A2=+/-{:.4f}".format(np.sqrt(E_cov[0,0]),np.sqrt(E_cov[1,1]),np.sqrt(E_cov[2,2]),np.sqrt(E_cov[3,3]),np.sqrt(E_cov[4,4]),np.sqrt(E_cov[5,5]),np.sqrt(E_cov[6,6]),np.sqrt(E_cov[7,7]),np.sqrt(E_cov[8,8])))
#for x in E_x:
#Eanswer = model(E_x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2)
#output2 = model2(E_x,f,mu1,sigma1,A1,mu2,sigma2,A2)
#resultbin.append(result)
plt.plot(E_x,E_output,label="Total Num Int")
#plt.plot(E_x,gauss1(E_x,E_params[0],E_params[1],E_params[2],E_params[3],E_params[4]),label="Weak Gauss")
#plt.plot(E_x,gauss2(E_x,E_params[0],E_params[5],E_params[6],E_params[7],E_params[8]),label="Strong Gauss")
plt.legend()
plt.show()
#result = model(E_x,f,mu1,sigma1,A1,alpha1,mu2,sigma2,A2,alpha2)
