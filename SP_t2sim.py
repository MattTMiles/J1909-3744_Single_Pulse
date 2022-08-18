#!/usr/bin/env python

#Basic imports
import os
import sys
import subprocess
import shlex
import argparse
import numpy as np
import glob
import pandas as pd
import math as mt
import shutil
from shutil import copyfile

#Wrapper for tempo2
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import matplotlib.pyplot as plt


#Argument parsing
parser = argparse.ArgumentParser(description="Jitter analysis")
parser.add_argument("-par", dest="par", help="Normal ephemeris")
parser.add_argument("-tim", dest="tim", help="Filtered tim file")
parser.add_argument("-avgpar", dest="avgpar", help="Average par (with AVERAGERES)")
parser.add_argument("-output", dest="output", help="Output directory to store the results")
args = parser.parse_args()


par = str(args.par)
tim_file = str(args.tim)
avgpar = str(args.avgpar)
sim_results = str(args.output)

start_dir = os.getcwd()

id = 0 #Accept as argument
while id < 1000:

    #Creating a simulation_ID directory - unique to every run
    sim_dir = os.path.join(sim_results,"sim_{0}".format(id))
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
        print ("sim_{0} directory created".format(id))
    else:
        shutil.rmtree(sim_dir)
        os.makedirs(sim_dir)
        print ("Deleted and created sim_{0} directory".format(id))



    #Use tempo2 createRealisation to simulate identical tim files to the real data
    print("Running form-ideal")

    path,name = os.path.split(tim_file)
    temp = os.path.join(sim_dir,name)
    copyfile(tim_file,temp)
    tim_file = temp

    #os.chdir(sim_dir)
    try:
        t2_formIdeal = "tempo2 -gr formIdeal -f {0} {1} -npsr 1 -nobs 100000".format(par,tim_file)
        print(t2_formIdeal)
        args_t2_formIdeal = shlex.split(t2_formIdeal)
        proc_t2_formIdeal = subprocess.Popen(args_t2_formIdeal)
        proc_t2_formIdeal.wait()
        print ("T2 formIdeal successful")
    except (RuntimeError, TypeError, NameError):
        print ("T2 formIdeal  failed. Quitting")
        sys.exit()

    formIdeal_tim = glob.glob(os.path.join(sim_dir,"*.sim"))[0]
    if os.path.exists(formIdeal_tim):
        try:
            print ("Running addGaussian")
            t2_addGauss = "tempo2 -gr addGaussian -f {0} {1} -npsr 1 -nobs 100000".format(par,formIdeal_tim)
            args_t2_addGauss = shlex.split(t2_addGauss)
            proc_t2_addGauss = subprocess.Popen(args_t2_addGauss)
            proc_t2_addGauss.wait()
            print ("T2 addGaussian successful")
        except (RuntimeError, TypeError, NameError):
            print ("T2 addGaussian failed. Quitting")
            sys.exit()
    else:
        print ("Could not find {0}".format(formIdeal_tim))
        sys.exit()


    addGauss = glob.glob(os.path.join(sim_dir,"*addGauss"))[0]
    if os.path.exists(addGauss):
        try:
            print ("Running createRealisation")
            t2_createReal = "tempo2 -gr createRealisation -f {0} -corr {1} -npsr 1 -nobs 100000".format(formIdeal_tim,addGauss)
            args_t2_createReal = shlex.split(t2_createReal)
            proc_t2_createReal = subprocess.Popen(args_t2_createReal)
            proc_t2_createReal.wait()
            print ("T2 createRealisation successful")
        except (RuntimeError, TypeError):
            print ("T2 createRealisation failed. Quitting.")
            sys.exit()
    else:
        print ("Could not find {0}".format(addGauss))
        sys.exit()

    sim_tim = glob.glob(os.path.join(sim_dir,"*.real"))[0]
    if os.path.exists(sim_tim):
        print ("White noise ToAs simulated successfully")
        print ("Creating frequency averaged timing residuals using the simulated ToAs")
        try:
            #No need for frequency averaging as all have been F scrunched
            #t2 = "tempo2 -f {0} {1} -nofit -fit F0 -fit DM -npsr 1 -nobs 100000".format(avgpar,sim_tim)
            #t2 = 'tempo2 -f '+avgpar+' -nofit -npsr 1 -nobs 100000 -output general2 -s "{sat} {post} {err}\n" -outfile '+sim_dir+'/avg.dat '+sim_tim
            
            #proc_init = subprocess.Popen('tempo2 -f '+avgpar+' -nofit -npsr 1 -nobs 100000 -output general2 -s "{sat} {post} {err}\n" -outfile '+sim_dir+'/avg.dat '+sim_tim, shell=True)
            #'tempo2 -output general2 -s "{sat} {post} {err}\n" -outfile residual.dat -f  ../J1909-3744_avg.par -npsr 1 -nobs 100000 ../*tim'
            #args_t2 = shlex.split(t2)
            proc_t2 = subprocess.Popen('tempo2 -f '+avgpar+' -nofit -npsr 1 -nobs 100000 -output general2 -s "{sat} {post} {err}\n" -outfile '+sim_dir+'/avg.dat '+sim_tim, shell=True)
            proc_t2.wait()
            print ("Tempo2 simulation run successful")
        except (RuntimeError, TypeError, NameError):
            print ("Tempo2 run on simulation failed. Quitting")
            sys.exit()
    
    id = id+1

os.chdir(start_dir)
#Initialises the not simulated avg.dat file must be run from the directory where these exist
proc_init = subprocess.Popen('tempo2 -f '+avgpar+' -nofit -npsr 1 -nobs 100000 -output general2 -s "{sat} {post} {err}\n" -outfile avg.dat '+tim_file, shell=True)
proc_init.wait()