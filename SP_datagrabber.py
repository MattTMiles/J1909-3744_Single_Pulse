#This is used to grab the data for the SP's individually rather than as a large script
#Large script had issues for some reason that isn't completely clear

import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pylab
import pandas as pd
import scipy
from scipy import fft
from ACF import auto_correlation_function
import scipy.signal as signal
from astropy.timeseries import LombScargle

#Change to the testing directory for the single pulses
os.chdir("/fred/oz002/users/mmiles/SinglePulse/")
#Director reference for the source data
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

parch = []

os.chdir(source)
for rawdata in sorted(os.listdir(source))[:100]:
#for rawdata in sorted(os.listdir(os.getcwd())):
    if rawdata.startswith('pulse'):
        try:
            archive = psrchive.Archive_load(rawdata)
            parch.append(archive)
        except RuntimeError:
            pass

pol_data = []
for archives in parch:
    #archives.pscrunch()
    archives.remove_baseline()
    archives.dedisperse()
    data_pol = archives.get_data()
    #data_pol = data_freq[0,0,:,:]
    pol_data.append(data_pol)

pol_data = np.array(pol_data)
#frequency_data = frequency_data[:,0,0,:,:]
os.chdir("/fred/oz002/users/mmiles/SinglePulse/")
np.save("pol_data",pol_data)
