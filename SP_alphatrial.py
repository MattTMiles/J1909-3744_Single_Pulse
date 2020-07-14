import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pylab
import pandas as pd
import scipy
from scipy.stats import norm
import scipy.signal
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from pylab import hist, diag

set1 = np.linspace(2,5,100)
set2 = np.linspace(2,5,100)
results = []
'''
def gauss1(x,mu,sigma,A):
    #return A*np.exp(-(x-mu)**2/2/sigma**2)
    return A*np.exp(-0.5*(np.abs((x-mu)/(sigma)))**alpha1)

def gauss2(x,mu,sigma,A):
    return A*np.exp(-0.5*(np.abs((x-mu)/(sigma)))**alpha2)
    
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss1(x,mu1,sigma1,A1,alpha1)+gauss2(x,mu2,sigma2,A2,alpha2)
'''
fdfs = pd.read_pickle("./Freq_small_df.pkl")
Edata = fdfs["Fluence"]/fdfs["Fluence"].mean()

fig, axs = plt.subplots(2)
Normalised_offpulse = fdfs['Baseband Noise']/fdfs["Fluence"].mean()
#Normalised_offpulse.hist(bins=100, label = "Off-Pulse region")

E_y,E_x,E_=hist(Edata,100,alpha=.3,label='On-Pulse')
E_x = (E_x[1:]+E_x[:-1])/2

E_expected = (0.43,0.3,1110,1.23,0.5,1100)

results = []
labels = ["alpha1","alpha2","Residual_fit"]
for alpha1 in set1:
    def gauss1(x,mu,sigma,A):
        #return A*np.exp(-(x-mu)**2/2/sigma**2)
        return A*np.exp(-0.5*(np.abs((x-mu)/(sigma)))**alpha1)

    for alpha2 in set2:

        def gauss2(x,mu,sigma,A):
            return A*np.exp(-0.5*(np.abs((x-mu)/(sigma)))**alpha2)

        def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
            return gauss1(x,mu1,sigma1,A1)+gauss2(x,mu2,sigma2,A2)

        try:
            E_params,E_cov=curve_fit(bimodal,E_x,E_y,E_expected)
            E_output = bimodal(E_x,*E_params)
            E_residual = E_y-E_output
            fit = sum(np.abs(E_residual))
            row = [alpha1,alpha2,fit]
            results.append(row)
        except RuntimeError:
            continue



df_results = pd.DataFrame(results, columns=labels)
