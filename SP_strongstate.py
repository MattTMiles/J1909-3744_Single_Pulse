import os
import psrchive
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
#from pylab import hist
#from collections import Counter
#from scipy.optimize import curve_fit
#from scipy.special import factorial
#from scipy.stats import poisson
#from sklearn import preprocessing
from scipy.ndimage.interpolation import shift
import subprocess as sproc 

#Source of the SP data
Main = "/fred/oz002/users/mmiles/SinglePulse/"
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

fdfs = pd.read_pickle("./Freq_small_df.pkl")
active = np.load("fdata_smaller_scrunched.npy")

bright = np.zeros(len(fdfs))
#limit = 1.32123
limit = 1.49511
'''
bright[fdfs['Fluence']>limit] = 1

#Create dataframes for the strong and weak pulses
strong = fdfs[fdfs['Fluence']>=limit]
weak = fdfs[fdfs['Fluence']<limit]

#The below code snippet is to copy the data into the strong and weak mode folders. I'm leaving this in in case the limit changes 
# and this needs to be done again. If that happens, you should probably clear the Strong_data and Weak_data folders 
bulk = []
archivestrong = []
archiveweak = []

os.chdir(source)

for rawdata in sorted(os.listdir(source))[:53001]:
    bulk.append(rawdata)

for rawdata in sorted(bulk):
    os.system('cp '+rawdata+' ../bulk_data2')

strongbulk = [bulk[x] for x in strong.index]
weakbulk = [bulk[x] for x in weak.index]

for rawdata_s in sorted(strongbulk):
    #os.system('cp '+rawdata_s+' ../Strong_data')
    p = sproc.Popen('cp '+rawdata_s+' ../Strong_data2',shell=True)
    p.wait()

for rawdata_w in sorted(weakbulk):
    p = sproc.Popen('cp '+rawdata_w+' ../Weak_data2',shell = True)
    p.wait()

os.chdir(Main)


Strongdf = pd.DataFrame(strong)
Weakdf = pd.DataFrame(weak)
Strongdf.to_pickle("/fred/oz002/users/mmiles/SinglePulse/Strong_state.pkl")
Weakdf.to_pickle("/fred/oz002/users/mmiles/SinglePulse/Weak_state.pkl")
'''
#Create the arrays that contain the strong and weak pulses
strong_array = active[fdfs['Fluence']>=limit]
weak_array = active[fdfs['Fluence']<limit]

#Create the profiles for the strong and weak modes
Total_profile = np.sum(active,axis=0)/len(active)
strong_profile = np.sum(strong_array,axis=0)/len(strong_array)

weak_profile = np.sum(weak_array,axis=0)/len(weak_array)
#First and second half of the weak pulse
#weak_profile1st = np.sum(weak_array[:7000],axis=0)/len(weak_array[:7000])
#weak_profile2nd = np.sum(weak_array[7000:],axis=0)/len(weak_array[7000:])

def norm(data):
    return data-min(data)/(max(data)-min(data))
'''
jitter_weak = []
jitter_strong = []

for pulse in strong_array:
    max_strong = np.argmax(pulse)
    jitter_strong.append(max_strong)

for pulse in weak_array:
    max_weak = np.argmax(pulse)
    jitter_weak.append(max_weak)


#These are plots of the jitter for the weak and strong pulses
fig, ax = plt.subplots()
ax.set_title('Weak Pulse Max Index')
ax.hist(jitter_weak, bins=len(weak_array[0]), color='tab:blue')
fig.tight_layout()

fig,ax = plt.subplots()
ax.set_title('Strong Pulse Max Index')
ax.hist(jitter_strong, bins=len(strong_array[0]), color='tab:blue')
fig.tight_layout()

plt.show()
'''
'''
fig, ax = plt.subplots()
ax.set_title('Weak Profile')
ax.plot(weak_profile)
#plt.axvline(1350)
#plt.axvline(1550)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Strong Profile')
ax.plot(strong_profile)
#plt.axvline(1350)
#plt.axvline(1550)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Total Profile')
ax.plot(Total_profile)
#plt.axvline(1350)
#plt.axvline(1550)
fig.tight_layout()
'''

weak_baseline = sum(weak_profile[:400])/len(weak_profile[:400])
strong_baseline = sum(strong_profile[:400])/len(strong_profile[:400])
'''
fig, ax = plt.subplots()
ax.set_title('Profile Comparison')
ax.plot(strong_profile,label = 'Strong Profile')
ax.plot(weak_profile, label = 'Weak Profile')
#plt.axvline(1450)
#plt.axvline(1550)
#plt.axvline(510)
#plt.axvline(710)
fig.legend()
fig.tight_layout()
'''
scaled_strong = strong_profile - strong_baseline
scaled_weak = weak_profile - weak_baseline
residual = ((scaled_strong/(max(scaled_strong)))) - (scaled_weak/(max(scaled_weak)))
xarray = np.arange(0,len(strong_profile),1)
shiftedstrong = shift((scaled_strong/(max(scaled_strong))),-6)
shiftedresidual = shiftedstrong - (scaled_weak/(max(scaled_weak)))
'''
fig, ax = plt.subplots()
ax.set_title('Scaled Profile Comparison')
ax.plot(scaled_strong/(max(scaled_strong)),label = 'Strong Profile')
ax.plot(scaled_weak/(max(scaled_weak)), label = 'Weak Profile')
#plt.axvline(1450)
#plt.axvline(1550)
fig.legend()
fig.tight_layout()

#Shifted version
fig, ax = plt.subplots()
ax.set_title('Scaled Shifted Profile Comparison')
ax.plot(xarray,shift((scaled_strong/(max(scaled_strong))),-6),label = 'Strong Profile')
ax.plot(xarray,(scaled_weak/(max(scaled_weak))), label = 'Weak Profile')
#plt.axvline(1450)
#plt.axvline(1550)
fig.legend()
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Residual')
ax.plot(residual, label='Strong - Weak')
fig.legend()
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Shifted Residual')
ax.plot(shiftedresidual, label='Strong - Weak shifted to max value')
fig.legend()
fig.tight_layout()

fig, axs = plt.subplots(2, gridspec_kw={'height_ratios':[1,2]})
axs[0].plot(residual, label='Strong - Weak residual',c='tab:green')
axs[0].set_xlim(1400,1600)
axs[1].plot(scaled_strong/(max(scaled_strong)),label = 'Strong Profile')
axs[1].plot(scaled_weak/(max(scaled_weak)), label = 'Weak Profile')
axs[1].set_xlim(1400,1600)
axs[0].set_title('Scaled comparison')
fig.legend()
fig.tight_layout()

fig, axs = plt.subplots(2, gridspec_kw={'height_ratios':[1,2]})
axs[0].plot(shiftedresidual, label='Strong - Weak residual',c='tab:green')
axs[1].plot(xarray,shift((scaled_strong/(max(scaled_strong))),-6),label = 'Strong Profile')
axs[1].plot(xarray,(scaled_weak/(max(scaled_weak))), label = 'Weak Profile')
axs[0].set_xlim(1400,1600)
axs[1].set_xlim(1400,1600)
axs[0].set_title('Scaled shifted comparison')
fig.legend()
fig.tight_layout()

plt.show()
'''

##TOA computation stuff below here
'''
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


fig, ax = plt.subplots()
ax.set_title('Weak pulse waterfall plot')
ax.imshow(weak_array.reshape(-1,2048)[:500], cmap='afmhot', aspect='auto',interpolation='none', origin='lower')
ax.set_xlabel('Pulse Phase')
ax.set_ylabel('Pulse Index')
#plt.savefig('/fred/oz002/users/mmiles/SinglePulse/First100_dedisperse.jpeg')
fig.tight_layout()


fig, ax = plt.subplots()
ax.set_title('Strong pulse waterfall plot')
ax.imshow(strong_array.reshape(-1,2048)[:500], cmap='afmhot', aspect='auto',interpolation='none', origin='lower')
ax.set_xlabel('Pulse Phase')
ax.set_ylabel('Pulse Index')
#plt.savefig('/fred/oz002/users/mmiles/SinglePulse/First100_dedisperse.jpeg')
fig.tight_layout()

plt.show()
'''