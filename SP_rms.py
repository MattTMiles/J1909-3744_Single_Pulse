import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os

#New tscrunch data
xarray = [2,4,8,16,32,64,128,256,512]
xarray2 = [2,4,8,16,32,64,128,256,512,1024]
strong = [6.995,3.580,2.673,1.890,1.334,0.936,0.697,0.488,0.347,0.260]
strong_culled = [np.nan,np.nan,2.673,1.890,1.334,0.936,0.697,0.488,0.347,0.260]

strong075 = [11.236,4.062,2.718,1.96,1.405,0.978,0.716,0.496,0.384,0.284]
strong100 = [9.185,3.606,2.69,1.954,1.382,0.977,0.701,0.5,0.352,0.248]
strong125 = [8.175,3.725,2.66,1.907,1.37,0.949,0.681,0.464,0.336,0.251]
strong175 = [6.248,3.543,2.614,1.882,1.328,0.923,0.661,0.477,0.329,0.246]
strong200 = [5.592,3.498,2.566,1.846,1.287,0.905,0.622,0.454,0.318,0.259]
strong225 = [4.977,3.451,2.544,1.799,1.26,0.908,0.647,0.463,0.338,0.243]

weak075 = [878.2,943.6,952.5,854.8,638.2,536.4,332.1,80.03,1.6,1.080]
weak100 = [875.4,849.6,912.5,538.3,420.3,384.6,38,3.06,2.46,0.756]
weak125 = [882.5,710.9,613.3,782.7,297.3,92.46,40.31,2.47,2.06,1.328]
weak175 = [783.81,714.11,551.98,299.28,113.15,5.59,3.23,2.33,1.39,0.86]
weak200 = [613.59,468.8,339.62,203.5,99.53,4.99,2.94,1.87,1.15,0.78]
weak225 = [646.5,545.9,274.75,170.7,37.99,3.92,2.52,1.54,0.92,0.655]

strong075_culled = [np.nan,4.062,2.718,1.96,1.405,0.978,0.716,0.496,0.384,0.284]
strong100_culled = [9.185,3.606,2.69,1.954,1.382,0.977,0.701,0.5,0.352,0.248]
strong125_culled = [8.175,3.725,2.66,1.907,1.37,0.949,0.681,0.464,0.336,0.251]
strong175_culled = [6.248,3.543,2.614,1.882,1.328,0.923,0.661,0.477,0.329,0.246]
strong200_culled = [5.592,3.498,2.566,1.846,1.287,0.905,0.622,0.454,0.318,0.259]
strong225_culled = [4.977,3.451,2.544,1.799,1.26,0.908,0.647,0.463,0.338,0.243]

weak075_culled = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,1.6,1.080]
weak100_culled = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,3.06,2.46,0.756]
weak125_culled = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2.47,2.06,1.328]
weak175_culled = [np.nan,np.nan,np.nan,np.nan,np.nan,5.59,3.23,2.33,1.39,0.86]
weak200_culled = [np.nan,np.nan,np.nan,np.nan,np.nan,4.99,2.94,1.87,1.15,0.78]
weak225_culled = [np.nan,np.nan,np.nan,np.nan,np.nan,3.92,2.52,1.54,0.92,0.655]

weak = [855.089,873.317,867.465,870.967,755.178,922.167,639.964,938.707,663.321]
total = [20.636,8.227,2.822,2.085,1.496,1.065,0.742,0.547,0.380]
total_culled = [np.nan,np.nan,2.822,2.085,1.496,1.065,0.742,0.547,0.380,0.291]

#reg_timing = [1.443,1.321,1.326,1.264,1.219,1.103,1.005,1.008]

fig, ax = plt.subplots()

ax.plot(xarray2,strong_culled,'o-', linewidth=1, label='Strong Mode')
#ax.plot(xarray,weak,'o-', linewidth=1,  label='weak_actual')
#ax.plot(xarray2,strong075_culled,'o-', linewidth=1, label='strong 0.75')
#ax.plot(xarray2,weak075_culled,'o-', linewidth=1,  label='weak 0.75')
#ax.plot(xarray2,strong100_culled,'o-', linewidth=1, label='strong 1.00')
#ax.plot(xarray2,weak100_culled,'o-', linewidth=1,  label='weak 1.00')
#ax.plot(xarray2,strong125_culled,'o-', linewidth=1, label='strong 1.25')
#ax.plot(xarray2,weak125_culled,'o-', linewidth=1,  label='weak 1.25')
#ax.plot(xarray2,strong175_culled,'o-', linewidth=1, label='strong 1.75')
#ax.plot(xarray2,weak175_culled,'o-', linewidth=1,  label='weak 1.75')
#ax.plot(xarray2,strong200_culled,'o-', linewidth=1, label='strong 2.00')
#ax.plot(xarray2,weak200_culled,'o-', linewidth=1,  label='weak 2.00')
#ax.plot(xarray2,strong225_culled,'o-', linewidth=1, label='strong 2.25')
#ax.plot(xarray2,weak225_culled,'o-', linewidth=1,  label='weak 2.25')

ax.plot(xarray2,total_culled,'o-', linewidth=1,  label='Total Single Pulses')
ax.set_xlabel('Factor of time averaging')
ax.set_ylabel(r'$\mathrm{weighted\/ RMS}\/ (\mu s)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Weighted RMS vs degree of time averaging')
#ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.legend(prop={'size':6})

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