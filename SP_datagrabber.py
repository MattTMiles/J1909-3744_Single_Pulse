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

farch = []

os.chdir(source)
for rawdata in sorted(os.listdir(source))[:53000]:
#for rawdata in sorted(os.listdir(os.getcwd())):
    if rawdata.startswith('pulse'):
        try:
            archive = psrchive.Archive_load(rawdata)
            farch.append(archive)
        except RuntimeError:
            pass

frequency_data = []
for archives in farch:
    archives.pscrunch()
    archives.remove_baseline()
    archives.dedisperse()
    data_freq = archives.get_data()
    data_freq = data_freq[0,0,:,:]
    frequency_data.append(data_freq)

frequency_data = np.array(frequency_data)
#frequency_data = frequency_data[:,0,0,:,:]
os.chdir("/fred/oz002/users/mmiles/SinglePulse/")
np.save("frequency_data",frequency_data)
