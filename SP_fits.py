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

#Import the good frequency channels (4 -> 13)
fdfs = pd.read_pickle("./Freq_small_df.pkl")
rawdata = np.load("fdata_smaller_scrunched.npy")
'''
#Fit for the mean and std of the off pulse
mu_offpulse, std_offpulse = norm.fit(fdfs['snr_off'])

#Create the probability density function and plot the snr hist
fdfs['snr_off'].hist(bins=100, label = "Off-Pulse region")

x = np.linspace(fdfs['snr_off'].min(), fdfs['snr_off'].max(), 100)
p_offpulse = norm.pdf(x, mu_offpulse, std_offpulse)

plt.plot(x, p_offpulse, 'k', linewidth=2,label="Off-Pulse dist")
#title = "Fit Results for off-pulse window: mu = %.2f, std = %.2f" % (mu_offpulse, std_offpulse)
#plt.title(title)
#plt.figure()
'''
'''
#Create a histogram that has the off-pulse region subtracted
subtracted_snr = fdfs['snr'] - fdfs['snr_off']

subtracted_snr.hist(density=True, bins=100)
x_subtracted = np.linspace(subtracted_snr.min(), subtracted_snr.max(), 100)

#Create a fit for this
mu_subtracted, std_subtracted = norm.fit(subtracted_snr)
p_subtracted = norm.pdf(x_subtracted, mu_subtracted, std_subtracted)

plt.plot(x_subtracted, p_subtracted, 'k', linewidth=2)
title2 = "Fit Results for subtracted snr: mu = %.2f, std = %.2f" % (mu_subtracted, std_subtracted)
plt.title(title2)
plt.figure()
plt.show()
'''

f = 0.793243
data = fdfs['snr']

def gauss1(x,mu,sigma,A):
    return f*(A/((2*np.pi*(1+(sigma**2)))**0.5))*np.exp(-0.5*(np.abs(((x-mu)**3.000000)/(1+(sigma**3.000000)))))

def gauss2(x,mu,sigma,A):
    return (1-f)*(A/((2*np.pi*(1+(sigma**2)))**0.5))*np.exp(-0.5*(np.abs(((x-mu)**2.000000)/(1+(sigma**2.000000)))))

def gauss3(x,mu,sigma,A):
    return (A/((2*np.pi*(sigma**2))**0.5))*np.exp(-0.5*(np.abs(((x-mu)/sigma)**2)))

def gauss4(x,mu,sigma,A):
    return (A/((2*np.pi*(sigma**2))**0.5))*np.exp(-0.5*(np.abs(((x-mu)/sigma)**2)))

def gauss_norm(x,mu,sigma,A):
    return A*np.exp(-0.5*(np.abs((x-mu)/(sigma)))**2)
    
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss1(x,mu1,sigma1,A1)+gauss2(x,mu2,sigma2,A2)

def bimodal2(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss3(x,mu1,sigma1,A1)+gauss4(x,mu2,sigma2,A2)

'''
y,x,_=hist(data,100,alpha=.3,label='data')
x = (x[1:]+x[:-1])/2

#x_total = np.linspace(fdfs['snr'].min(), fdfs['snr'].max(),100)
#mu_total, std_total = norm.fit(fdfs['snr'])
#p_total = norm.pdf(x_total, mu_total, std_total)

expected = (-0.01,1.11,140,7.22,3,128)
params,cov=curve_fit(bimodal,x,y,expected)
sigma=np.sqrt(diag(cov))
output = bimodal(x,*params)
plt.plot(x,bimodal(x,*params),label = "Total")
#plt.plot(x_total, p_total, 'k', linewidth=2)
plt.title("total snr data")

#Plot the gaussians that scipy has worked out
mu_curve1 = params[0]
sigma_curve1 = params[1]
amp_curve1 = params[2]

mu_curve2 = params[3]
sigma_curve2 = params[4]
amp_curve2 = params[5]

plt.plot(x,gauss(x,mu_curve1, sigma_curve1, amp_curve1),label="Weak")
plt.plot(x,gauss(x,mu_curve2, sigma_curve2, amp_curve2),label ="Strong")
plt.legend()
plt.show()
'''
'''
deconv, err = scipy.signal.deconvolve(output, p_offpulse)
deconv = deconv-err

n = len(output)-len(p_offpulse)+1
s = (len(output)-n)/2

deconv_res = np.zeros(len(output))
deconv_res[int(s):int(len(output)-s)] = deconv
deconv = deconv_res

plt.plot(deconv)
plt.show()
'''

'''
#Version for the density distribution rather than the actual number
fig, axs = plt.subplots(2)
fdfs['snr_off'].hist(bins=100, label = "Off-Pulse region",density=True)

d_y,d_x,d_=hist(data,1000,alpha=.3,label='On-Pulse',density=True)
d_x = (d_x[1:]+d_x[:-1])/2

d_expected = (0.7,1.11,0.11,7.3,3,0.09)
d_params,d_cov=curve_fit(bimodal,d_x,d_y,d_expected)
d_sigma=np.sqrt(diag(d_cov))
d_output = bimodal(d_x,*d_params)

axs[1].plot(d_x,bimodal(d_x,*d_params),label = "Total")
#plt.plot(x_total, p_total, 'k', linewidth=2)
axs[1].plot(d_x,gauss(d_x,d_params[0], d_params[1], d_params[2]),label="Weak")
axs[1].plot(d_x,gauss(d_x,d_params[3], d_params[4], d_params[5]),label ="Strong")
plt.legend()

residual = d_y-d_output
axs[0].plot(d_x,residual)

axs[1].set(xlabel="Signal-to-noise ratio", ylabel="Percentage of toal pulses")
axs[0].set(ylabel="Residual")
plt.show()
'''


#Version for the the energy normalised from the mean fluence
'''
Edata = fdfs["Fluence"]/fdfs["Fluence"].mean()

fig, axs = plt.subplots(2)
Normalised_offpulse = fdfs['Baseband Noise']/fdfs["Fluence"].mean()
#Normalised_offpulse.hist(bins=100, label = "Off-Pulse region",density=True)

E_y,E_x,E_=hist(Edata,100,alpha=.3,label='On-Pulse',density=True)
E_x = (E_x[1:]+E_x[:-1])/2

E_expected = (0.43,0.3,0.6,1.23,0.6,0.61)
E_expected2 = (0.43,0.3,0.6,1.23)
E_params,E_cov=curve_fit(bimodal,E_x,E_y,E_expected)
E_sigma=np.sqrt(diag(E_cov))
E_output = bimodal(E_x,*E_params)

#test = bimodal2(E_x,E_params[0], E_params[1], E_params[2],E_params[3])
#axs[1].plot(E_x,test)

axs[1].plot(E_x,bimodal(E_x,*E_params),label = "Total")

axs[1].plot(E_x,gauss1(E_x,E_params[0], E_params[1], E_params[2]),label="Weak")
axs[1].plot(E_x,gauss2(E_x,E_params[3], E_params[4], E_params[5]),label ="Strong")
Strong_gauss = gauss2(E_x,E_params[3], E_params[4], E_params[5])
#Strong_p = p_fit(E_x, E_params[3], E_params[4], E_params[5])
Weak_gauss = gauss1(E_x,E_params[0], E_params[1], E_params[2])

plt.legend()

E_residual = E_y-E_output

axs[0].plot(E_x,E_residual)
#axs[0].errorbar(E_x,E_residual,yerr=error)

axs[1].set(xlabel="Normalised Energy", ylabel="Percentage of toal pulses")
axs[0].set(ylabel="Residual")
plt.show()
'''

#Version that isn't using density
Edata = fdfs["snr"]
#Enorm = (Edata - fdfs["Baseline Noise"].mean())/fdfs["off_p_sigma"]

fig, axs = plt.subplots(2)
#Normalised_offpulse = (fdfs['Baseline Noise'] - fdfs["Baseline Noise"].mean())/fdfs["off_p_sigma"]
Normalised_offpulse = fdfs["snr_off"]
Off_y,Off_x,Off_ = hist(Normalised_offpulse,100,label = "off pulse")
Off_x = (Off_x[1:]+Off_x[:-1])/2

#Normalised_offpulse.hist(bins=100, label = "Off-Pulse region")
mu_offpulse, std_offpulse = norm.fit(fdfs['snr_off'])
xp = np.linspace(fdfs['snr_off'].min(), Edata.max(), 1000)
#p_offpulse = norm.pdf(xp, mu_offpulse, std_offpulse)
Off_expected = (mu_offpulse,std_offpulse,1720)
Off_params,Off_cov = curve_fit(gauss_norm,Off_x,Off_y,Off_expected)
p_offpulse = gauss_norm(xp,*Off_params)
axs[1].plot(xp,p_offpulse)

Off_mu_error = np.sqrt(Off_cov[0,0])
Off_std_error = np.sqrt(Off_cov[1,1])

'''
print("Off pulse parameters are: mu={:.4f}+/-{:.4f}, sigma={:.4f}+/-{:.4f}".format(Off_params[0],np.sqrt(Off_cov[0,0]), Off_params[1], np.sqrt(Off_cov[1,1]))
'''

E_y,E_x,E_=hist(Edata,100,alpha=.3,label='On-Pulse')
E_x = (E_x[1:]+E_x[:-1])/2

E_expected = (0.96,1.1,1157,7.3,2.7,1140)
bilby_params = (1.83,1.20,1297.78,6.36,1.57,1297.75)
E_params,E_cov=curve_fit(bimodal2,E_x,E_y,bilby_params)
E_sigma=np.sqrt(diag(E_cov))
E_output = bimodal2(E_x,*E_params)

bilby_output = bimodal2(E_x, *bilby_params)

#plt.plot(E_x,bimodal2(E_x,*E_params),label = "Total")
#plt.plot(E_x,gauss4(E_x,E_params[0], E_params[1], E_params[2]),label="Weak")
#plt.plot(E_x,gauss3(E_x,E_params[3], E_params[4], E_params[5]),label ="Strong")


axs[1].plot(E_x,bimodal2(E_x,*E_params),label = "Total")

axs[1].plot(E_x,gauss4(E_x,E_params[0], E_params[1], E_params[2]),label="Weak")
axs[1].plot(E_x,gauss3(E_x,E_params[3], E_params[4], E_params[5]),label ="Strong")
Strong_gauss = gauss3(E_x,E_params[3], E_params[4], E_params[5])
Weak_gauss = gauss4(E_x,E_params[0], E_params[1], E_params[2])

plt.legend()

E_residual = E_y-E_output
#print(sum(np.abs(E_residual)))
rms = np.sqrt((sum(E_output**2))/len(E_output))

axs[0].plot(E_x,E_residual)
error = E_output/np.sqrt(np.abs(E_y))
axs[0].errorbar(E_x,E_residual,yerr=error)

axs[1].set(xlabel="Normalised Energy", ylabel="Percentage of toal pulses")
axs[0].set(ylabel="Residual")
plt.figure()
#Everything should be set up now, let's try and deconvolve
E_lengthmatch = bimodal2(xp,*E_params)
#plt.plot(xp,E_lengthmatch,label="Model fit") 
#plt.plot(xp,p_offpulse,label="Noise PDF")
Subtracted = E_lengthmatch-p_offpulse
Subtracted = [(i>0)*i for i in Subtracted]
#plt.plot(xp,Subtracted,label="Model with noise subtracted")

#E_output2 = bimodal2(E_x,*E_params)
#Strong_gauss2 = gauss3(E_x,E_params[3], E_params[4], E_params[5])
#Weak_gauss2 = gauss4(E_x,E_params[0], E_params[1], E_params[2])

#plt.plot(E_x,E_output2,label="Deconvolved version?")
#plt.legend()
#plt.figure()
plt.show()
