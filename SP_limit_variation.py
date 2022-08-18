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
import os
import psrchive


Main = "/fred/oz002/users/mmiles/SinglePulse/"
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

snr_folder = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window'
os.chdir(Main)
'''
try:
    os.mkdir('strong125')
    os.mkdir('strong1')
    os.mkdir('strong075')
    os.mkdir('strong175')
    os.mkdir('strong200')
    os.mkdir('strong225')

    os.mkdir('weak125')
    os.mkdir('weak1')
    os.mkdir('weak075')
    os.mkdir('weak175')
    os.mkdir('weak200')
    os.mkdir('weak225')
except:
    pass
'''
strong125_dir = "/fred/oz002/users/mmiles/SinglePulse/strong125"
strong1_dir = "/fred/oz002/users/mmiles/SinglePulse/strong1"
strong075_dir = "/fred/oz002/users/mmiles/SinglePulse/strong075"
strong175_dir = "/fred/oz002/users/mmiles/SinglePulse/strong175"
strong200_dir = "/fred/oz002/users/mmiles/SinglePulse/strong200"
strong225_dir = "/fred/oz002/users/mmiles/SinglePulse/strong225"

weak125_dir = "/fred/oz002/users/mmiles/SinglePulse/weak125"
weak1_dir = "/fred/oz002/users/mmiles/SinglePulse/weak1"
weak075_dir = "/fred/oz002/users/mmiles/SinglePulse/weak075"
weak175_dir = "/fred/oz002/users/mmiles/SinglePulse/weak175"
weak200_dir = "/fred/oz002/users/mmiles/SinglePulse/weak200"
weak225_dir = "/fred/oz002/users/mmiles/SinglePulse/weak225"
'''

#Sorting data portion
fdfs = pd.read_pickle("./Freq_small_df.pkl")


active = np.load("fdata_smaller_scrunched.npy")

limit125 = 1.25
limit1 = 1.00
limit075 = 0.75

limit175 = 1.75
limit200 = 2.00
limit225 = 2.25

strong125 = fdfs[fdfs['snr']>=limit125]
weak125 = fdfs[fdfs['snr']<limit125]

strong1 = fdfs[fdfs['snr']>=limit1]
weak1 = fdfs[fdfs['snr']<limit1]

strong075 = fdfs[fdfs['snr']>=limit075]
weak075 = fdfs[fdfs['snr']<limit075]

strong175 = fdfs[fdfs['snr']>=limit175]
weak175 = fdfs[fdfs['snr']<limit175]

strong200 = fdfs[fdfs['snr']>=limit200]
weak200 = fdfs[fdfs['snr']<limit200]

strong225 = fdfs[fdfs['snr']>=limit225]
weak225 = fdfs[fdfs['snr']<limit225]


bulk = []
os.chdir(source)
for rawdata in sorted(os.listdir(source))[:53001]:
    bulk.append(rawdata)


strongbulk125 = [bulk[x] for x in strong125.index]
weakbulk125 = [bulk[x] for x in weak125.index]

strongbulk1 = [bulk[x] for x in strong1.index]
weakbulk1 = [bulk[x] for x in weak1.index]

strongbulk075 = [bulk[x] for x in strong075.index]
weakbulk075 = [bulk[x] for x in weak075.index]

strongbulk175 = [bulk[x] for x in strong175.index]
weakbulk175 = [bulk[x] for x in weak175.index]

strongbulk200 = [bulk[x] for x in strong200.index]
weakbulk200 = [bulk[x] for x in weak200.index]

strongbulk225 = [bulk[x] for x in strong225.index]
weakbulk225 = [bulk[x] for x in weak225.index]


for sp in sorted(strongbulk125):
    os.symlink(os.path.join(source,sp), os.path.join(strong125_dir,sp))

for sp in sorted(strongbulk1):
    os.symlink(os.path.join(source,sp), os.path.join(strong1_dir,sp))

for sp in sorted(strongbulk075):
    os.symlink(os.path.join(source,sp), os.path.join(strong075_dir,sp))

for sp in sorted(strongbulk175):
    os.symlink(os.path.join(source,sp), os.path.join(strong175_dir,sp))

for sp in sorted(strongbulk200):
    os.symlink(os.path.join(source,sp), os.path.join(strong200_dir,sp))

for sp in sorted(strongbulk225):
    os.symlink(os.path.join(source,sp), os.path.join(strong225_dir,sp))


for sp in sorted(weakbulk125):
    os.symlink(os.path.join(source,sp), os.path.join(weak125_dir,sp))

for sp in sorted(weakbulk1):
    os.symlink(os.path.join(source,sp), os.path.join(weak1_dir,sp))

for sp in sorted(weakbulk075):
    os.symlink(os.path.join(source,sp), os.path.join(weak075_dir,sp))

for sp in sorted(weakbulk175):
    os.symlink(os.path.join(source,sp), os.path.join(weak175_dir,sp))

for sp in sorted(weakbulk200):
    os.symlink(os.path.join(source,sp), os.path.join(weak200_dir,sp))

for sp in sorted(weakbulk225):
    os.symlink(os.path.join(source,sp), os.path.join(weak225_dir,sp))
'''

#timing portion

limit_dir = '/fred/oz002/users/mmiles/SinglePulse/limit_tims'
for directory in os.listdir(limit_dir):
    if directory == 'strong075':
        p=sproc.Popen('pat')