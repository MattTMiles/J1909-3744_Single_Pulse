import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os

#tscrunch data
xarray = [2,4,8,16,32,64,128,256]
strong = [4.629,3.214,2.280,1.557,1.045,0.760,0.560,0.386]
weak = [200.010,120.304,54.150,10.922,3.314,2.369,1.683,1.136]
total = [20.636,8.227,2.822,2.085,1.496,1.065,0.742,0.547]

#reg_timing = [1.443,1.321,1.326,1.264,1.219,1.103,1.005,1.008]

fig, ax = plt.subplots()

ax.plot(xarray,strong,'o-', c='tab:blue', linewidth=1, label='strong mode')
ax.plot(xarray,weak,'o-', c='tab:green', linewidth=1,  label='weak mode')
ax.plot(xarray,total,'o-', c='tab:orange', linewidth=1,  label='total single pulses')
ax.set_xlabel('factor of t-scrunching')
ax.set_ylabel(r'$\mathrm{weighted\/ RMS}\/ (\mu s)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Weighted RMS vs t-scrunching factors')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.legend()

fig.tight_layout()

#Integration length comparison
length = [1,2,3,5,10,15,20,25]
strongint = [0.242,0.240,0.240,0.158,0.127,0.162,0.097,0.107]
weakint = [0.651,0.622,0.613,0.371,0.350,0.323,0.322,0.301]
allint = [0.295,0.293,0.292,0.149,0.113,0.145,0.097,0.088]
regint = [0.326,0.326,0.326,0.273,0.282,0.282,0.172,0.186]


fig, ax = plt.subplots()

ax.plot(length,strongint,'o-', c='tab:blue', linewidth=1, label='strong mode')
ax.plot(length,weakint,'o-', c='tab:green', linewidth=1,  label='weak mode')
ax.plot(length,allint,'o-', c='tab:orange', linewidth=1,  label='total single pulses')
ax.plot(length,regint,'o-', c='tab:red', linewidth=1, label='Regular timing data')
ax.set_xlabel('Integration Length (s)')
ax.set_ylabel(r'$\mathrm{weighted\/ RMS}\/ (\mu s)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Weighted RMS vs Integration length')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.legend()

fig.tight_layout()
plt.show()