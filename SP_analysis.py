#This is a library of different plots and code that work on the data output by SP_timephase.py

#Uncomment as appropriate to make them work

import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pylab
import pandas as pd
import scipy
from scipy import fft
import scipy.signal as signal
from astropy.timeseries import LombScargle

os.chdir("/home/mmiles/soft/SP/")
from ACF import auto_correlation_function
from CCF import cross_correlation_function
#This is the main directory
os.chdir("/fred/oz002/users/mmiles/SinglePulse")

df = pd.read_pickle("./Main_df.pkl")
fdf = pd.read_pickle("./Freq_df.pkl")
fdfs = pd.read_pickle("./Freq_small_df.pkl")
#frequency_data = np.load("frequency_data.npy")
useabledata = np.load("useabledata.npy")
good_fdata_scrunched = np.load("good_fdata_scrunched.npy")
fdata_smaller = np.load("fdata_smaller.npy")
fdata_smaller_scrunched = np.load("fdata_smaller_scrunched.npy")


#polarisation_data = np.load("polarisation_data.npy")

'''
#line plot of Fluence vs pulse
df.reset_index().plot(kind='line',x ='index',y=['Baseband Noise','Fluence'])
#plt.ylabel('non-pulse-signal')
plt.xlabel('Pulse Index')
plt.ylabel('Flux ($W/m^2$)')
plt.show()
'''
'''
#Line plot each of the 32 frequency channels
for freq in np.arange(0,32,1):
    trialdata = []
    for sp_freq in frequency_data[:,freq,:]:
        flu_freq=sum(sp_freq[1350:1550])
        trialdata.append(flu_freq)
    plt.plot(trialdata,label=freq)
'''
'''
#Autocorrelation plot
acf ,lags = auto_correlation_function(df['Fluence'])
plt.plot(acf)
plt.xlabel('Pulse Index')
plt.ylabel('ACF')
plt.show()
'''

'''
#Histogram of snr and Fluence
df.hist(column='snr',bins=100, log=True)
df.hist(column='Fluence',bins=100, log=True)
plt.show()
'''

'''
#Fourier Transform
yf = fft(df['Fluence'])
#yf = yf[1000:]
FFT2 = np.abs(yf)**2
N=len(yf)
#Create the random noise version as well
mu, sigma = df['Fluence'].mean(), df['Fluence'].std(axis =0)
s = np.random.normal(mu, sigma, len(df['Fluence']))
yf_s = fft(s)
FFT2_s = np.abs(yf_s)**2
plt.plot(FFT2[1:N//2],label="Data")
plt.plot(FFT2_s[1:N//2],label="Simulated")
plt.legend()
plt.show()
'''

'''
#Scipy Lomb-Scargle Periodogram
x = np.linspace(1,len(df['Fluence']),len(df['Fluence']))
#f = np.linspace(1,len(df['Fluence'])//2,1000)
periods = np.linspace(0.01,2.5,len(df['Fluence']))
f = 2*np.pi/periods
pgram = signal.lombscargle(x,df['Fluence']-df['Fluence'].mean(),f, normalize=True)
mu, sigma = df['Fluence'].mean(), df['Fluence'].std(axis =0)
s = np.random.normal(mu, sigma, len(df['Fluence']))
pgram_noise = signal.lombscargle(x,s-s.mean(),f, normalize=True)
plt.plot(periods,pgram,label="Data")
plt.plot(periods,pgram_noise,label="Simulated")
plt.ylabel('L-S Power (Normalised)')
plt.xlabel('Periods (Pulses)')
plt.legend()
plt.show()
'''

'''
#Astropy Lomb-Scargle Periodogram
x = np.linspace(1,len(df['Fluence']),len(df['Fluence']))
s = np.random.normal(mu, sigma, len(df['Fluence']))
frequency, power = LombScargle(x, df['Fluence']).autopower()
plt.plot(frequency, power, label="Data")
sim_frequency, sim_power = LombScargle(x, s).autopower()
plt.plot(sim_frequency, sim_power, label="Simulated")
plt.legend()
plt.show()
'''

'''
plt.imshow(useabledata.reshape(-1,2048), cmap='afmhot', aspect='auto',interpolation='none', origin='lower')
plt.xlabel('Pulse Phase')
plt.ylabel('Pulse Index')
#plt.savefig('/fred/oz002/users/mmiles/SinglePulse/First100_dedisperse.jpeg')
plt.show()
'''