#This has been made to group the single pulses into archives consisting of 1024
# consecutive pulses

import numpy as np
import scipy
import os
import subprocess as sproc 
import pandas as pd 
import itertools
import glob
import sys

Weak = '/fred/oz002/users/mmiles/SinglePulse/Weak_data2'
Strong = '/fred/oz002/users/mmiles/SinglePulse/Strong_data2'
All = '/fred/oz002/users/mmiles/SinglePulse/bulk_data2'

#Choose which data needs to be grouped

#Need to chooseeither 'Weak', 'Strong', or 'All'
activedir = os.chdir(All)

#Creates a filelist to iterate over. 
# It seems to work better than os.listdir, not sure why
filelist = glob.glob('*.scr')
filelist = sorted(filelist)
# This is the process that rolls through and grabs packages of 1024 pulses
dict1 = {}

def grouper(S,n):
    iterator = iter(S)
    while True:
        items = list(itertools.islice(iterator, n))
        if len(items) == 0:
            break
        yield items

for i, file1 in enumerate(grouper(filelist, 1024)):
    key1 = i
    group1 = dict1.get(key1,[])
    group1.append(file1)
    dict1[key1] = group1

for key1 in list(dict1.keys()):
    active1 = dict1[key1]
    active1 = str(active1)[1:-1]
    active1 = active1.replace(',','')
    active1 = active1.replace("'","")
    active1 = str(active1)[1:-1]
    active1 = active1.replace("'","")
    key1 = str(key1)
    os.system('psradd -o added_'+key1+'.raw '+active1)
    os.system('mv added_'+key1+'.raw ../tempo2_all_new')
