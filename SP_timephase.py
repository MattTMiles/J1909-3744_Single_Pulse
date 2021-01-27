import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
#mport pylab
import pandas as pd
#import scipy
#from scipy import fft
#from ACF import auto_correlation_function
#import scipy.signal as signal
#from astropy.timeseries import LombScargle

'''
#Change to the testing directory for the single pulses
os.chdir("/fred/oz002/users/mmiles/SinglePulse/test_sample")
#Director reference for the source data
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

#Create the first numpy array to play around with
arch = []
#parch = []
#Comment out the 'for' line and chdir line depending if you want to use the active data or the source data
os.chdir("/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data")
for rawdata in sorted(os.listdir(source)):
#for rawdata in sorted(os.listdir(os.getcwd())):
    if rawdata.startswith('pulse'):
        try:
            archive = psrchive.Archive_load(rawdata)
            arch.append(archive)
            #parch.append(archive)
        except RuntimeError:
            pass

#Now we have the raw archives in arch, create a version with the data
#Create another version with full frequency resolution, and another with full polarisation resolution
data = []
#frequency_data = []
#polarisation_data = []

for parchives in parch:
    parchives.remove_baseline()
    parchives.dedisperse()
    parchives.fscrunch()
    data_pol = parchives.get_data()
    polarisation_data.append(data_pol)

for archives in arch:
    #archives.pscrunch()
    archives.remove_baseline()
    archives.dedisperse()
    #data_freq = archives.get_data()
    #frequency_data.append(data_freq)
    archives.fscrunch()
    #archives.bscrunch_to_nbin(128)
    data_SP = archives.get_data()
    data.append(data_SP)

#Make a useable array and bin the useless dimensions
#frequency_data = np.array(frequency_data)
#frequency_data = frequency_data[:,0,0,:,:]

useabledata = []

for entries in data:
    entries = entries[:,0,:,:]
    useabledata.append(entries)

useabledata = np.array(useabledata)
useabledata = useabledata[:,0,:]

#Saves the data as .npy files in the main directory
os.chdir("/fred/oz002/users/mmiles/SinglePulse")
np.save("poldata",useabledata)
#np.save("frequency_data",frequency_data)
#np.save("polarisation_data",polarisation_data)
'''
os.chdir("/fred/oz002/users/mmiles/SinglePulse")

useabledata = np.load('useabledata.npy')

labels = ['Fluence','bins','sigma','snr','Baseband Noise','IP_Fluence','test_nopulse']
dataframe = []
#nopulse = []
for sp in useabledata:
    S = sum(sp[1350:1450])
    N = 1450-1350
    IP_Fluence = sum(sp[510:710])
    lower = sp[:1350]
    upper = sp[1450:]
    using = np.concatenate((lower,upper),axis=None)
    #nopulse.append(using)
    ave = sum(using)/len(using)
    using = np.subtract(using,ave)
    using = np.square(using)
    total = sum(using)
    sigma = (total/len(using))**0.5
    snr = (S-(N*ave))/(sigma*np.sqrt(N))
    non_pulse_S = sum(sp[1050:1250])
    non_pulse_s2 = sum(sp[100:300])
    window_test = non_pulse_S-non_pulse_s2
    d = [S, N, sigma, snr, non_pulse_S, IP_Fluence, window_test]
    dataframe.append(d)

df = pd.DataFrame(dataframe,columns=labels)
#Save the dataframe to the main directory
df.to_pickle("/fred/oz002/users/mmiles/SinglePulse/offwindow_df.pkl")

limit = 1.49511
fdfs = pd.read_pickle("./Freq_small_df.pkl")
strong = fdfs[fdfs['Fluence']>=limit]
weak = fdfs[fdfs['Fluence']<limit]

active = np.load("fdata_smaller_scrunched.npy")

strongoff = active[strong.index]
weakoff = active[weak.index]

#strongoff_array = strongoff['Fluence']
#weakoff_array = weakoff['Fluence']
strongoff_profile = np.sum(strongoff,axis=0)/len(strongoff)
weakoff_profile = np.sum(weakoff,axis=0)/len(weakoff)

plt.plot(strongoff_profile,label='strongoff')
plt.plot(weakoff_profile,label='weakoff')
plt.legend()
plt.show()

#ave_snr = sum(df['snr'])/len(df['snr'])
#nopulse = np.array(nopulse)

#Save the dataframe to the main directory
#np.save("nopulse",nopulse)
