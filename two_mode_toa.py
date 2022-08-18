import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import psrchive
import re
import bilby
from decimal import *

#Current dir
current = os.getcwd()
#Master directory
os.chdir('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-DD')

#Subint specific epochs
#epochs = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm/epochs_temp.npy')
strongarch = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/strong_Ft128/all.Ft128_strong')
allarch = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/all_Ft128/all.Ft128')


'''
allepochs = []
for arch in allarch:
    epoch = arch.get_epoch()
    allepochs.append(epoch.strtempo())

strongepochs = []
for arch in strongarch:
    epoch = arch.get_epoch()
    strongepochs.append(epoch.strtempo())'''

allepochs = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/pat_runs/all_epochs.npy')
strongepochs = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/pat_runs/strong_epochs.npy')

all_readlines = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/pat_runs/allepochs_readlines.npy')
strong_readlines = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/pat_runs/strongepochs_readlines.npy')

#Subint specific periods
#Ps = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm/periods_temp.npy')

AllPs = []
for arch in allarch:
    P = arch.get_folding_period()
    AllPs.append(P)
AllPs = np.array(AllPs)

StrongPs = []
for arch in strongarch:
    P = arch.get_folding_period()
    StrongPs.append(P)
StrongPs = np.array(StrongPs)

#Subint specific centre frequencies
centre_freqs = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm/centre_freqs.npy')


os.chdir(current)

def toa_return(phases,suparch,epochs,bins):

    toas = []

    for i, phase in enumerate(phases): 
        #toa = suparch[i].get_epoch() + psrchive.MJD(((-phase/2048)*suparch[i].get_folding_period())/(3600*24.)) 
        toa = Decimal(epochs[i]) + Decimal(((-phase/bins)*suparch[i].get_folding_period())/(3600*24.))
        #toaprint = "{:.50f}".format(toa.intday()+toa.fracday()) 
        toaprint = "{:.50f}".format(toa) 
        toas.append(toaprint)

    return toas

def toa_error_return(errors, suparch,bins):
    
    uncertainties = []

    for i, err in enumerate(errors):
        unc = (err/bins)*suparch[i].get_folding_period()*1e6

        uncertainties.append(unc)

    return uncertainties


def timfile_maker(toas,uncertainties,names):
    master = []
    names = [str(names)]*len(toas)
    master = np.vstack((names[:len(toas)],centre_freqs[:len(toas)]))
    master = np.vstack((master,toas))
    master = np.vstack((master,uncertainties[:len(toas)]))
    
    #List of 'meerkat' to add to tim
    meerkat = ['meerkat']*len(toas)
    master = np.vstack((master,meerkat))

    masterdf = pd.DataFrame(master.T)


    return masterdf

#Saving code
'''
master.to_csv(r'name.tim', header=None, index=None, sep=' ', mode='a')
'''

# Code to extract the phases and errors properly
# Run in ipython in the dir containing the subint dirs
'''
home = os.getcwd() 
      key = 'alldata' 
      dir_files = os.listdir(home) 
      dir_files = [ x for x in dir_files if x.startswith(key) ] 
      dir_files.sort(key=lambda f: int(re.sub('\D', '', f))) 
      phases=[] 
      errors=[] 
       
      for dir in dir_files: 
          if dir.startswith(key): 
              os.chdir(dir) 
              print(dir) 
              try: 
                  results =bilby.result.read_in_result(filename='dynesty_result.json') 
                  sp = results.get_one_dimensional_median_and_error_bar('all_phase').median 
                  print(sp) 
                  se = results.get_one_dimensional_median_and_error_bar('all_phase').plus 
                  print(se) 
                  phases.append(sp) 
                  errors.append(se) 
              except OSError: 
                  os.chdir(home) 
              os.chdir(home)
'''
'''
home = os.getcwd()  
key = 'J1103'  
dir_files = os.listdir(home)  
dir_files = [ x for x in dir_files if x.startswith(key) ]  
dir_files.sort(key=lambda f: int(re.sub('\D', '', f)))  
early_phases=[]  
early_errors=[]  
late_phases=[] 
late_errors=[] 
      
for dir in dir_files:  
    if dir.startswith(key):  
        os.chdir(dir)  
        print(dir)  
        try:  
            results =bilby.result.read_in_result(filename='dynesty_result.json')  
            ep = results.get_one_dimensional_median_and_error_bar('early_phase').median  
            print(ep)  
            ee = results.get_one_dimensional_median_and_error_bar('early_phase').plus  
            print(ee)  
            early_phases.append(ep)  
            early_errors.append(ee) 
            lp = results.get_one_dimensional_median_and_error_bar('late_phase').median 
            print(lp)    
            le = results.get_one_dimensional_median_and_error_bar('late_phase').plus  
            print(le) 
            late_phases.append(lp) 
            late_errors.append(le) 
        except OSError:  
            os.chdir(home)  
        os.chdir(home) 
'''