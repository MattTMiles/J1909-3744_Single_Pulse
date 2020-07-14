#Computes the autocorrelation function

import numpy as np
from scipy.signal import correlate


def auto_correlation_function(spectrum):
    
    spectrum_hat = np.mean(spectrum)
    spec_to_corr = (spectrum - spectrum_hat)/np.sqrt(len(spectrum))
    
    acf = correlate(spec_to_corr, spec_to_corr)
    acf = acf/max(acf)
    #lags = np.linspace(0, (len(acf)/2)+1,len(acf)/2)
    lags = np.linspace((-len(acf)//2), (len(acf)//2)-1, len(acf))+1

    self_noise = np.argwhere(acf == max(acf))
    acf = np.delete(acf, self_noise)
    lags = np.delete(lags, self_noise)
    acf = acf[acf.size//2:]

    return acf, lags