#This script is for plotting toas and etc for the SP project
import os
import subprocess as sproc 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

maindir = '/fred/oz002/users/mmiles/SinglePulse'
os.chdir(maindir)

timdir = '/fred/oz002/users/mmiles/SinglePulse/timfiles'
os.chdir(timdir)

#List the timing files
weak_weak_05 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.weakstd_weakdata05sec_tim'
weak_weak_1 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.weakstd_weakdata1sec_tim'
strong_strong_05 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.strongstd_strongdata05sec_tim'
strong_strong_1 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.strongstd_strongdata1sec_tim'
strong_strong_4 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.strongstd_strongdata4sec_tim'
strong_strong_20 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.strongstd_strongdata20sec_tim'
all_strong_05 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.allstd_strongdata05sec_tim'
all_strong_1 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.allstd_strongdata1sec_tim'
all_strong_4 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.allstd_strongdata4sec_tim'
all_strong_20 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.allstd_strongdata20sec_tim'
all_weak_05 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.allstd_weakdata05sec_tim'
all_weak_1 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.allstd_weakdata1sec_tim'
weak_256 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.weakstd_256data_tim'
strong_256 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.strongstd_256data_tim'
mspcensus_256 = '/fred/oz002/users/mmiles/SinglePulse/timfiles/J1909-3744.mspstd_256data_tim'
#portrait timing files
p_all_strong_05 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_allstd_strongdata05_tim'
p_all_strong_1  = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_allstd_strongdata1_tim'
p_all_weak_05 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_allstd_weakdata0.5_tim'
p_all_weak_1 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_allstd_weakdata1_tim'
p_strong_strong_05 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_strongstd_strongdata05_tim'
p_strong_strong_1 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_strongstd_strongdata1_tim'
p_weak_weak_05 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_weakstd_weakdata0.5_tim'
p_weak_weak_1 = '/fred/oz002/users/mmiles/SinglePulse/portrait_timfiles/J1909-3744.portrait_weakstd_weakdata1_tim'

#tim loading
tim_weak_weak_05 = np.loadtxt(weak_weak_05, usecols=3, skiprows=1)
tim_weak_weak_1 = np.loadtxt(weak_weak_1, usecols=3, skiprows=1)
tim_strong_strong_05 = np.loadtxt(strong_strong_05, usecols=3, skiprows=1)
tim_strong_strong_1 = np.loadtxt(strong_strong_1, usecols=3, skiprows=1)
tim_strong_strong_4 = np.loadtxt(strong_strong_4, usecols=3, skiprows=1)
tim_all_strong_05 = np.loadtxt(all_strong_05, usecols=3, skiprows=1)
tim_all_strong_1 = np.loadtxt(all_strong_1, usecols=3, skiprows=1)
tim_all_strong_4 = np.loadtxt(all_strong_4, usecols=3, skiprows=1)
tim_all_weak_05 = np.loadtxt(all_weak_05, usecols=3, skiprows=1)
tim_all_weak_1 = np.loadtxt(all_weak_1, usecols=3, skiprows=1)
tim_weak_256 = np.loadtxt(weak_256, usecols=3, skiprows=1)
tim_strong_256 = np.loadtxt(strong_256, usecols=3, skiprows=1)
tim_mspcensus_256 = np.loadtxt(mspcensus_256, usecols=3, skiprows=1)

#portrait tims
ptim_all_strong_05 = np.loadtxt(p_all_strong_05, usecols=3, skiprows=1)
ptim_all_strong_1 = np.loadtxt(p_all_strong_1, usecols=3, skiprows=1)
ptim_all_weak_05 = np.loadtxt(p_all_weak_05, usecols=3, skiprows=1)
ptim_all_weak_1 = np.loadtxt(p_all_weak_1, usecols=3, skiprows=1)
ptim_strong_strong_05 = np.loadtxt(p_strong_strong_05, usecols=3, skiprows=1)
ptim_strong_strong_1 = np.loadtxt(p_strong_strong_1, usecols=3, skiprows=1)
ptim_weak_weak_05 = np.loadtxt(p_weak_weak_05, usecols=3, skiprows=1)
ptim_weak_weak_1 = np.loadtxt(p_weak_weak_1, usecols=3, skiprows=1)

fig, ax = plt.subplots()

ax.scatter(tim_all_strong_1, tim_strong_strong_1, s=0.5, zorder=0, c='tab:blue', label='strong data')
ax.scatter(tim_all_weak_1, tim_weak_weak_1, s=0.5, zorder=0, c='tab:green', label='weak data')
ax.set_title('1s subintegrations')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Total\/ portrait \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Modal\/ portrait \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()



'''
fig, ax = plt.subplots()

ax.scatter(tim_all_strong_05, tim_strong_strong_05, s=0.5, zorder=0, c='tab:blue', label='strong data 0.5s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Total\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Strong\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()

ax.scatter(tim_all_strong_1, tim_strong_strong_1, s=0.5, zorder=0, c='tab:blue', label='strong data 1s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Total\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Strong\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()

ax.scatter(tim_all_weak_1, tim_weak_weak_1, s=0.5, zorder=0, c='tab:blue', label='weak data 1s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Total\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Weak\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()

ax.scatter(tim_all_weak_05, tim_weak_weak_05, s=0.5, zorder=0, c='tab:blue', label='weak data 0.5s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Total\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Weak\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()


fig, ax = plt.subplots()

ax.scatter(tim_mspcensus_256, tim_strong_256, s=0.5, zorder=0, c='tab:blue', label='256s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Mspcensus\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Strong\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()

ax.scatter(tim_mspcensus_256, tim_weak_256, s=0.5, zorder=0, c='tab:blue', label='256s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Mspcensus\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Weak\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()

ax.scatter(tim_all_strong_4, tim_strong_strong_4, s=0.5, zorder=0, c='tab:blue', label='4s ints')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathrm{Total\/ profile \/timing}\/ (\mu s)$')
ax.set_ylabel(r'$\mathrm{Strong\/ profile \/timing}\/ (\mu s)$')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
ax.legend()
fig.tight_layout()
'''
plt.show()