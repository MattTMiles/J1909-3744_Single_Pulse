#Computes the cross correlation function

import numpy as np
from scipy.signal import correlate

def cross_correlation_function(signal1,signal2):
    
    signal1_hat = np.mean(signal1)
    signal2_hat = np.mean(signal2)
    spec_to_corr1 = (signal1 - signal1_hat)
    spec_to_corr2 = (signal2 - signal2_hat)

    ccf = np.correlate(spec_to_corr1, spec_to_corr2,"full")
    #ccf = ccf/(len(signal1)*signal1.std()*signal2.std())
    ccf = ccf/max(ccf)
    #lags = np.linspace(0, (len(acf)/2)+1,len(acf)/2)
    #lags = np.arange(-len(signal1)+1,len(signal1))

    ccf = ccf[ccf.size//2:]

    return ccf