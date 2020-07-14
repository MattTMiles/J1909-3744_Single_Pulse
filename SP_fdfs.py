import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pylab
import pandas as pd
import scipy

os.chdir("/fred/oz002/users/mmiles/SinglePulse")

active = np.load("fdata_smaller_scrunched.npy")
labels = ['Fluence','bins','sigma','snr','Baseline Noise','IP_Fluence','test_nopulse','snr_off']
dataframe = []
fdfs_nopulse = []

for sp in active:
    S = sum(sp[1350:1550])
    N = 1550-1350
    IP_Fluence = sum(sp[510:710])
    lower = sp[:1350]
    upper = sp[1550:]
    using = np.concatenate((lower,upper),axis=None)
    fdfs_nopulse.append(using)
    ave = sum(using)/len(using)
    using = np.subtract(using,ave)
    using = np.square(using)
    total = sum(using)
    sigma = (total/len(using))**0.5
    snr = (S-(N*ave))/(sigma*np.sqrt(N))
    non_pulse_S = sum(sp[1050:1250])
    non_pulse_s2 = sum(sp[100:300])
    window_test = non_pulse_S-non_pulse_s2
    op1 = sp[:1050]
    op2 = sp[1250:1350]
    op3 = sp[1550:]
    using2 = np.concatenate((op1,op2,op3),axis=None)
    ave2 = (sum(using2)/len(using2))
    using2 = np.subtract(using2,ave2)
    using2 = np.square(using2)
    total2 = sum(using2)
    sigma2 = (total2/len(using2))**0.5
    snr_offpulse = (non_pulse_S-(N*ave2))/(sigma2*np.sqrt(N))
    d = [S, N, sigma, snr, non_pulse_S, IP_Fluence, window_test,snr_offpulse]
    dataframe.append(d)

fdfs = pd.DataFrame(dataframe,columns=labels)
