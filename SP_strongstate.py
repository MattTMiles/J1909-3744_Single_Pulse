import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pylab import hist
from collections import Counter
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson

fdfs = pd.read_pickle("./Freq_small_df.pkl")
active = np.load("fdata_smaller_scrunched.npy")

bright = np.zeros(len(fdfs))
limit = 1.32123

bright[fdfs['Fluence']>limit] = 1

#Create dataframes for the strong and weak pulses
strong = fdfs[fdfs['Fluence']>=limit]
weak = fdfs[fdfs['Fluence']<limit]

Strongdf = pd.DataFrame(strong)
Weakdf = pd.DataFrame(weak)
Strongdf.to_pickle("/fred/oz002/users/mmiles/SinglePulse/Strong_state.pkl")
Weakdf.to_pickle("/fred/oz002/users/mmiles/SinglePulse/Weak_state.pkl")

#Create the arrays that contain the strong and weak pulses
strong_array = active[fdfs['Fluence']>=limit]
weak_array = active[fdfs['Fluence']<limit]

#Create the profiles for the strong and weak modes
strong_profile = np.sum(strong_array,axis=0)/len(strong_array)
weak_profile = np.sum(weak_array,axis=0)/len(weak_array)

#Get the strong arrival times
arrtim_strong = psrchive.ArrivalTime()
arrtim_strong.set_shift_estimator('FDM')
arrtim_strong.set_format('Tempo2')

arrtim_strong.set_standard(strong_profile)
arrtim_strong.set_observations(strong_array)
strong_toas = arrtim_strong.get_toas()

#Get the weak arrival times
arrtim_weak = psrchive.ArrivalTime()
arrtim_weak.set_shift_estimator('FDM')
arrtim_weak.set_format('Tempo2')

arrtim_weak.set_standard(strong_profile)
arrtim_weak.set_observations(strong_array)
weak_toas = arrtim_weak.get_toas()

condition = bright==1
condition2 = bright==0
countbright = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:], [True])))[0])[::2]

countweak = np.diff(np.where(np.concatenate(([condition2[0]],condition2[:-1] != condition2[1:], [True])))[0])[::2]
'''
def p_fit(k, lamb):
    return poisson.pmf(k, lamb)

bins = np.arange(1,len(Counter(countbright))+1)-0.5
entries, bin_edges, patches = plt.hist(countbright,bins=bins,label="Consecutive bright pulses")

bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
x = np.arange(1,len(Counter(countbright))+1)

P_params, P_cov = curve_fit(p_fit, bin_middles, entries)

#plt.plot(x, p_fit(x, *P_params),marker='o',linestyle='')
plt.legend()
plt.figure()
'''
'''
entries2, bin_edges2, patches2 = plt.hist(countweak,bins=bins,label="Consecutive weak pulses",density=True)

bin_middles2 = 0.5*(bin_edges2[1:] + bin_edges2[:-1])
x2 = np.arange(1,len(Counter(countweak))+1)
P_params2, P_cov2 = curve_fit(p_fit, bin_middles2, entries2)

plt.plot(x2, p_fit(x2, *P_params2),marker='o',linestyle='')

plt.legend()
plt.figure()
plt.show()
'''
'''
####Version for the off-pulse window for comparison

bright2 = np.zeros(len(fdfs))
window = active[:,100:300]
window = np.sum(window,axis=1)

limit_off = 3*window

bright2[fdfs['Baseband Noise']>limit] = 1
condition = bright2==1
condition2 = bright2==0
countbright_off = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:], [True])))[0])[::2]

countweak_off = np.diff(np.where(np.concatenate(([condition2[0]],condition2[:-1] != condition2[1:], [True])))[0])[::2]
plt.hist(countbright_off,bins=len(Counter(countbright_off)),label="Consecutive bright pulses - off",density=True)
plt.legend()
plt.figure()

plt.hist(countweak_off,bins=len(Counter(countweak_off)),label="Consecutive weak pulses - off",density=True)
plt.legend()
plt.figure()

plt.show()
'''