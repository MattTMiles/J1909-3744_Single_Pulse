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
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

fig,ax = plt.subplots()
a = np.arange(0,len(fdata_smaller_scrunched[1]))
b = np.linspace(0,1,len(a))
ax.imshow(fdata_smaller_scrunched[:1000].reshape(-1,2048), cmap='afmhot', aspect='auto',interpolation='none', origin='lower')
#plt.xticks(a, np.linspace(0,1,len(a)))
#ax.xaxis.set_major_locator(plt.MaxNLocator(4))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#my_xticks=['0','0.25','0.5','0.75','1']
#plt.xticks(a,my_xticks)
plt.xlabel('Pulse Phase')
plt.ylabel('Pulse Number')
'''
axins = zoomed_inset_axes(ax, 2, loc='center left')
axins.imshow(fdata_smaller_scrunched[:1000].reshape(-1,2048), cmap='afmhot', interpolation="nearest", origin="lower")
x1,x2,y1,y2 = 1250,1700,400,600
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
'''

#plt.savefig('/fred/oz002/users/mmiles/SinglePulse/First100_dedisperse.jpeg')
#plt.tight_layout()
#mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="1")

plt.draw()
plt.show()
