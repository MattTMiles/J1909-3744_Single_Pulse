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

fdfs = pd.read_pickle("./Freq_small_df.pkl")
active = np.load("fdata_smaller_scrunched.npy")

bright = np.zeros(len(fdfs))
#limit = 1.32123
limit = 1.49511

bright[fdfs['Fluence']>limit] = 1

#Create dataframes for the strong and weak pulses
strong = fdfs[fdfs['Fluence']>=limit]
weak = fdfs[fdfs['Fluence']<limit]

#Now we need to import the polarisation data
pol_pulses = "/fred/oz002/users/mmiles/SinglePulse/pol_pulses"
parch = []
os.chdir(pol_pulses)
for rawdata in sorted(os.listdir(pol_pulses))[:53001]:
    try:
        archive = psrchive.Archive_load(rawdata)
        parch.append(archive)
    except RuntimeError:
        pass

pol_data = []
for archives in parch:
    archives.remove_baseline()
    archives.dedisperse()
    data_pol = archives.get_data()
    pol_data.append(data_pol)

pol_data = np.array(pol_data)
os.chdir(Main)
np.save('pol_data',pol_data)

os.chdir(pol_pulses)
#Now separate the pol files into weak and strong
pol_strong = [parch[x] for x in strong.index]
pol_weak = [parch[x] for x in weak.index]

for rawdata_s in sorted(pol_strong):
    p = sproc.Popen('cp '+rawdata_s+' ../pol_strong',shell=True)
    p.wait()

for rawdata_w in sorted(pol_weak):
    p = sproc.Popen('cp '+rawdata_w+' ../pol_weak',shell = True)
    p.wait()



