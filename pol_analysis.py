import numpy as np
import pandas as pd 
import psrchive
import sys
import os
import subprocess as sproc 
import matplotlib.pyplot as plt 

#This is the data that initialises the strong and weak indices so I can separate the polarisations
Main = "/fred/oz002/users/mmiles/SinglePulse/"
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

os.chdir(Main)
'''
fdfs = pd.read_pickle("./Freq_small_df.pkl")
active = np.load("fdata_smaller_scrunched.npy")

bright = np.zeros(len(fdfs))
#limit = 1.32123
limit = 1.49511

bright[fdfs['Fluence']>limit] = 1

#Create dataframes for the strong and weak pulses
strong = fdfs[fdfs['Fluence']>=limit]
weak = fdfs[fdfs['Fluence']<limit]
'''
'''
#Now we need to import the polarisation data
#pol_pulses = "/fred/oz002/users/mmiles/SinglePulse/calibrated_all"
pol_pulses = "/fred/oz002/users/mmiles/SinglePulse/1284_f32p"
pol_stokes = "/fred/oz002/users/mmiles/SinglePulse/1284_f32_stokes"

os.chdir(pol_stokes)
parch = []
bulk = []
#for rawdata in sorted(os.listdir(pol_pulses))[:53001]:
pdmp_snr = []
data = []
i=0
for rawdata in sorted(os.listdir(pol_stokes))[:53000]:
    #if rawdata.endswith('handChange'):
    bulk.append(rawdata)
    
    parchive = psrchive.Archive_load(rawdata)
    #parchive.pscrunch()
    parchive.remove_baseline()
    parchive.dedisperse()
    data_polP = parchive.get_data()
    data.append(data_polP)
    i=i+1
    print('parchive data retrieved:{}'.format(i))

datacheck = [x for x in data if x.shape == (1,4,32,2048)]
data = np.array(datacheck)
data = data[:,0,:,:,:]

np.save('/fred/oz002/users/mmiles/SinglePulse/stokes_data',data)
os.chdir(Main)

data = data[:,:,4:13,:]
np.save('/fred/oz002/users/mmiles/SinglePulse/stokes_data_f4_13',data)

data = np.sum(data,axis=2)
np.save('/fred/oz002/users/mmiles/SinglePulse/stokes_data_F',data)

datastokesI = data[:,0,:]
np.save('/fred/oz002/users/mmiles/SinglePulse/data_stokesI_F',datastokesI)

labels = ['Fluence','bins','sigma','snr','Baseline Noise','IP_Fluence','test_nopulse','snr_off','off_p_sigma']
dataframe = []
pol_nopulse = []
for polp in datastokesI:
    S = sum(polp[1450:1550])
    N = 1550-1450
    IP_Fluence = sum(polp[510:710])
    lower = polp[:1450]
    upper = polp[1550:]
    using = np.concatenate((lower,upper),axis=None)
    pol_nopulse.append(using)
    ave = sum(using)/len(using)
    using = np.subtract(using,ave)
    using = np.square(using)
    total = sum(using)
    sigma = (total/len(using))**0.5
    snr = (S-(N*ave))/(sigma*np.sqrt(N))
    non_pulse_S = sum(polp[1150:1250])
    non_pulse_s2 = sum(polp[200:300])
    window_test = non_pulse_S-non_pulse_s2
    op1 = polp[:1150]
    op2 = polp[1250:1450]
    op3 = polp[1550:]
    using2 = np.concatenate((op1,op2,op3),axis=None)
    ave2 = (sum(using2)/len(using2))
    using2 = np.subtract(using2,ave2)
    using2 = np.square(using2)
    total2 = sum(using2)
    sigma2 = (total2/len(using2))**0.5
    snr_offpulse = (non_pulse_S-(N*ave2))/(sigma2*np.sqrt(N))
    d = [S, N, sigma, snr, non_pulse_S, IP_Fluence, window_test, snr_offpulse, sigma2]
    dataframe.append(d)

df = pd.DataFrame(dataframe,columns=labels)
df.to_pickle("/fred/oz002/users/mmiles/SinglePulse/pol_stokesI_df.pkl")
'''

df = pd.read_pickle('pol_stokesI_df.pkl')
os.chdir(Main)
limit = 1.37501
#limit = 1.49511

active = np.load('stokes_data_F.npy')

strong = active[df['snr']>=limit]
weak = active[df['snr']<limit]

strongprofs = np.sum(strong, axis=0)
weakprofs = np.sum(weak,axis=0)

strongI, strongQ, strongU, strongV = strongprofs[0], strongprofs[1], strongprofs[2], strongprofs[3]
weakI, weakQ, weakU, weakV = weakprofs[0], weakprofs[1], weakprofs[2], weakprofs[3]


strongIbaseline = sum(strongI[:400])/len(strongI[:400])
strongQbaseline = sum(strongQ[:400])/len(strongQ[:400])
strongUbaseline = sum(strongU[:400])/len(strongU[:400])
strongVbaseline = sum(strongV[:400])/len(strongV[:400])

weakIbaseline = sum(weakI[:400])/len(weakI[:400])
weakQbaseline = sum(weakQ[:400])/len(weakQ[:400])
weakUbaseline = sum(weakU[:400])/len(weakU[:400])
weakVbaseline = sum(weakV[:400])/len(weakV[:400])

strongIb = strongI - strongIbaseline
strongQb = strongQ - strongQbaseline
strongUb = strongU - strongUbaseline
strongVb = strongV - strongVbaseline

weakIb = weakI - weakIbaseline
weakQb = weakQ - weakQbaseline
weakUb = weakU - weakUbaseline
weakVb = weakV - weakVbaseline

PAstrong = (0.5*np.arctan(strongUb/strongQb))*(180/np.pi)
PAweak = (0.5*np.arctan(weakUb/weakQb))*(180/np.pi)

raise a
fig, ax1,ax2 = plt.subplots(2,1) 

ax1.plot

ax2.plot(strongIb, label='I')
ax2.plot(strongQb, label='Q')
ax2.plot(strongUb, label='U')
ax2.plot(strongVb, label='V')

ax.legend()
ax.set_title('strong')
#ax.set_xlim(1425,1575)

fig.show()

fig, ax = plt.subplots() 

ax.plot(weakIb, label='I')
ax.plot(weakQb, label='Q')
ax.plot(weakUb, label='U')
ax.plot(weakVb, label='V')

ax.legend()
ax.set_title('weak')
#ax.set_xlim(1425,1575)
fig.show()
