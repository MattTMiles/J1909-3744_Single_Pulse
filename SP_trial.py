#This loads required data using the psrchive python commands
#A script can be added to the end of this to do whatever is needed
#For trial manipulation/playing around with the data I would recommend using IPython. To run this in an IPython kernel use: run SP_trial.py


import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy

#Change to the testing directory for the single pulses
os.chdir("/fred/oz002/users/mmiles/SinglePulse/test_sample")
#Directory reference for the source data
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

#Create the first empty numpy array to play around with
arch = []

#This goes into the data files and loads them from psrchive into a python readable version
os.chdir("/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data")
for rawdata in sorted(os.listdir(source)):
    if rawdata.startswith('pulse'):
        #This try/except loop is just there in case the data is already loaded, I noticed it can sometimes get stuck when that happens
        try:
            archive = psrchive.Archive_load(rawdata)
            arch.append(archive)
        except RuntimeError:
            pass

#Now we have the raw archives in arch, create a version with the actual data

#Create an empty array to house the full data 
data = []

#This for loop p-scrunches, removes the baseline, dedisperses, and fscrunches
#I've commented it out but there is a bin scrunch in there that shows you how you would scrunch to a certain number of bins
#i.e. bsrunch_to_nbin(128) scrunches the data into 128 bins. This works with the other scrunching commands as well if phrased like that
for archives in arch:
    archives.pscrunch()
    archives.remove_baseline()
    archives.dedisperse()
    archives.fscrunch()
    #archives.bscrunch_to_nbin(128)
    
    #This get_data() command grabs the raw data from the file as a numpy array
    #It comes out with 4 components that are: (Nsub, Npol, Nchan, Nbin)
    data_SP = archives.get_data()
    data.append(data_SP)

#This is just me getting rid of one of the dimensions I didn't need (polarisation) and putting it into a different array
useabledata = []
for entries in data:
    entries = entries[:,0,:,:]
    useabledata.append(entries)
useabledata = np.array(useabledata)
useabledata = useabledata[:,0,:]

#Saves the data as .npy files in the main directory, this is an easy way to locally store the files so they don't need to be reloaded each time
os.chdir("/fred/oz002/users/mmiles/SinglePulse")
np.save("useabledata",useabledata)
#np.save("frequency_data",frequency_data)
#np.save("polarisation_data",polarisation_data)

#Another way to store the data is in a pandas dataframe, I find this a little easier to navigate so my method so far has been to keep the raw data in a numpy array and then the manipulations in a dataframe
labels = ['Fluence','bins','sigma','snr','Baseband Noise','IP_Fluence','test_nopulse']
dataframe = []
nopulse = []
for sp in useabledata:
    S = sum(sp[1350:1550])
    N = 1550-1350
    IP_Fluence = sum(sp[510:710])
    lower = sp[:1350]
    upper = sp[1550:]
    using = np.concatenate((lower,upper),axis=None)
    nopulse.append(using)
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
df.to_pickle("/fred/oz002/users/mmiles/SinglePulse/Main_df.pkl")

#This is just creating an off pulse array to compare to
nopulse = np.array(nopulse)

#Save the array to the main directory
np.save("nopulse",nopulse)
