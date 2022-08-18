import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pylab
import pandas as pd
import scipy
import subprocess as sproc 

home_dir = '/fred/oz002/users/mmiles/SinglePulse'
data_dir = '/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data'

snr_pdmp = []
for data in sorted(os.listdir(data_dir))[:53000]:
    if data.startswith('pulse'):
        p = sproc.check_output("psrstat -c snr=pdmp -c snr "+data+" | awk '{print($NF)}'",shell = True)
        p = p.decode()
        p = p.split('snr=')
        p = p[1]
        p = p.strip('\n')
        snr = float(p)
        snr_pdmp.append(snr)

np.save('SP_pdmp_data',snr_pdmp)

        