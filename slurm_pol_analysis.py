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
#Now we need to import the polarisation data
pol_pulses = "/fred/oz002/users/mmiles/SinglePulse/pol_pulses"
pol_weak = "/fred/oz002/users/mmiles/SinglePulse/pol_weak2"
pol_strong = "/fred/oz002/users/mmiles/SinglePulse/pol_strong2"
'''
parch = []

parch_weak = []
parch_strong = []
os.chdir(pol_weak)
for rawdata_weak in sorted(os.listdir(pol_weak)):
    try:
        archiveweak = psrchive.Archive_load(rawdata_weak)
        parch_weak.append(archiveweak)
    except RuntimeError:
        pass

os.chdir(pol_strong)
for rawdata_strong in sorted(os.listdir(pol_strong)):
    try:
        archivestrong = psrchive.Archive_load(rawdata_strong)
        parch_strong.append(archivestrong)
    except RuntimeError:
        pass

'''
os.chdir(pol_pulses)
parch = []
bulk = []
for rawdata in sorted(os.listdir(pol_pulses))[:53001]:
    bulk.append(rawdata)
    try:
        parchive = psrchive.Archive_load(rawdata)
        parch.append(parchive)
    except RuntimeError:
        pass

data = []
for parchives in parch:
    parchives.pscrunch()
    parchives.remove_baseline()
    parchives.dedisperse()
    parchives.fscrunch()
    data_polP = parchives.get_data()
    data.append(data_polP)

useabledata = []

for entries in data:
    entries = entries[:,0,:,:]
    useabledata.append(entries)

useabledata = np.array(useabledata)
useabledata = useabledata[:,0,:]
useabledata = useabledata[:,0,:]

labels = ['Fluence']
dataframe = []
for polp in useabledata:
    S = sum(polp[1350:1550])
    d = [S]
    dataframe.append(d)

df = pd.DataFrame(dataframe,columns=labels)
df.to_pickle("/fred/oz002/users/mmiles/SinglePulse/pol_fluence.pkl")

os.chdir(Main)
limit = 1.49511

strong = df[df['Fluence']>=limit]
weak = df[df['Fluence']<limit]


pol_strong = [bulk[x] for x in strong.index]
pol_weak = [bulk[x] for x in weak.index]


os.chdir(pol_pulses)

for rawdata_s in pol_strong:
    p = sproc.Popen('cp '+rawdata_s+' ../pol_strong2',shell=True)
    p.wait()

for rawdata_w in pol_weak:
    p = sproc.Popen('cp '+rawdata_w+' ../pol_weak2',shell = True)
    p.wait()


#This is just to create the numpy data files

weakpol_data = []
strongpol_data = []
for weak in pol_weak:
    weak.remove_baseline()
    weak.dedisperse()
    weak.fscrunch()
    weakdata_pol = weak.get_data()
    weakpol_data.append(weakdata_pol)

for strong in pol_strong:
    strong.remove_baseline()
    strong.dedisperse()
    strong.fscrunch()
    strongdata_pol = strong.get_data()
    strongpol_data.append(strongdata_pol)

weak1 = []
weak2 = []
weak3 = []
weak4 = []
strong1 = []
strong2 = []
strong3 = []
strong4 = []

for strongentries in strongpol_data:
    strongentries = strongentries[0,:,0,:]
    first = strongentries[0]
    second = strongentries[1]
    third = strongentries[2]
    fourth = strongentries[3]

    strong1.append(first)
    strong2.append(second)
    strong3.append(third)
    strong4.append(fourth)

strong1 = np.array(strong1)
strong2 = np.array(strong2)
strong3 = np.array(strong3)
strong4 = np.array(strong4)

for weakentries in weakpol_data:
    weakentries = weakentries[0,:,0,:]
    first = weakentries[0]
    second = weakentries[1]
    third = weakentries[2]
    fourth = weakentries[3]

    weak1.append(first)
    weak2.append(second)
    weak3.append(third)
    weak4.append(fourth)

weak1 = np.array(weak1)
weak2 = np.array(weak2)
weak3 = np.array(weak3)
weak4 = np.array(weak4)

os.chdir(Main)
np.save("pol1weak",weak1)
np.save("pol2weak",weak2)
np.save("pol3weak",weak3)
np.save("pol4weak",weak4)
np.save("pol1strong",strong1)
np.save("pol2strong",strong2)
np.save("pol3strong",strong3)
np.save("pol4strong",strong4)

#pol_df = pd.DataFrame(pol_data)


weak1 = np.load('pol1weak.npy')
weak2 = np.load('pol2weak.npy')
weak3 = np.load('pol3weak.npy')
weak4 = np.load('pol4weak.npy')

strong1 = np.load('pol1strong.npy')
strong2 = np.load('pol2strong.npy')
strong3 = np.load('pol3strong.npy')
strong4 = np.load('pol4strong.npy')

weak1profile = np.sum(weak1,axis=0)/len(weak1)
weak1baseline = sum(weak1profile[:400])/len(weak1profile[:400])
weak2profile = np.sum(weak2,axis=0)/len(weak2)
weak2baseline = sum(weak2profile[:400])/len(weak2profile[:400])
weak3profile = np.sum(weak3,axis=0)/len(weak3)
weak3baseline = sum(weak3profile[:400])/len(weak3profile[:400])
weak4profile = np.sum(weak4,axis=0)/len(weak4)
weak4baseline = sum(weak4profile[:400])/len(weak4profile[:400])

strong1profile = np.sum(strong1,axis=0)/len(strong1)
strong1baseline = sum(strong1profile[:400])/len(strong1profile[:400])
strong2profile = np.sum(strong2,axis=0)/len(strong2)
strong2baseline = sum(strong2profile[:400])/len(strong2profile[:400])
strong3profile = np.sum(strong3,axis=0)/len(strong3)
strong3baseline = sum(strong3profile[:400])/len(strong3profile[:400])
strong4profile = np.sum(strong4,axis=0)/len(strong4)
strong4baseline = sum(strong4profile[:400])/len(strong4profile[:400])

fig, ax = plt.subplots()
plt.plot(weak1profile-weak1baseline,label='weak1')
plt.plot(weak2profile-weak2baseline,label='weak2')
plt.plot(weak3profile-weak3baseline,label='weak3')
plt.plot(weak4profile-weak4baseline,label='weak4')

plt.plot(strong1profile-strong1baseline,label='strong1')
plt.plot(strong2profile-strong2baseline,label='strong2')
plt.plot(strong3profile-strong3baseline,label='strong3')
plt.plot(strong4profile-strong4baseline,label='strong4')

plt.legend()

fig.tight_layout()
plt.show()
