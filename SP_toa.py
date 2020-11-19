#This is just a quick script to send off in a batch file
import os
import pandas as pd 
import numpy as np 
import subprocess as sproc 

Maindir = '/fred/oz002/users/mmiles/SinglePulse'
os.chdir(Maindir)

timingdir = '/fred/oz002/users/mmiles/SinglePulse/timfiles'
os.chdir(timingdir)

#Specify the timing that should happen

p = sproc.Popen('pat -A FDM -f tempo2 -P -s ../J1909-3744.std ../Strong_data/pulse* > J1909-3744.mspcensusstd_strongdata_tim', shell=True)
p.wait()