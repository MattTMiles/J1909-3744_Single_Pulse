from __future__ import division
import numpy as np
from numpy.lib.npyio import loadtxt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.gridspec as gridspec
import bilby

from scipy import signal
from scipy.signal import convolve, deconvolve
import sys
from pylab import hist, diag
import scipy.integrate as integrate
from scipy.special import gamma, factorial
from astropy.timeseries import LombScargle

import os



#Change to the dir where everything is
os.chdir('/fred/oz002/users/mmiles/SinglePulse')

## Set consistent fonts
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(24)
#font.set_style('italic')

#df= pd.read_pickle("./Freq_small_df.pkl")
df = pd.read_pickle("./pol_df.pkl")

#active = np.load("fdata_smaller_scrunched.npy")
active = np.load('pscrunch_data_F.npy')

snr = df['snr']
'''
## Histogram

fig, ax = plt.subplots()

ax.hist(snr, bins=50, density=True, alpha=.6, color = 'dimgray')
ax.set_xlabel('S/N')
ax.set_ylabel('Probability Density')

ax.set_xlim(-5,25)
ax.set_ylim(0,0.08)

ax.axvline(linewidth=2, x=-5, color='black')
ax.axvline(linewidth=2, x=25, color='black')
ax.axhline(linewidth=2, y=0, color='black')
ax.axhline(linewidth=2, y=0.08, color='black')

#fig.tight_layout()

fig.savefig('paper_plots/snr_histogram.pdf',dpi=1200)
fig.show()
'''

'''
## Waterfall plot

fig, ax = plt.subplots()

#a = plt.axes([.25,.325,.2,.35])
#a.imshow(active[400:600,1350:1650], cmap='afmhot', aspect='auto', interpolation='none', origin='lower')


#x = [5,10,15,20]
#y = [10,15,20,25]

#axins1 = zoomed_inset_axes(ax, zoom=5, loc=6)
#axins1.set_xlim()

Z = active[:1000]
extent = (0, 2048, 0, 1000)
#Z2 = np.zeros((150,150))
ny, nx = Z.shape
#Z2[30:30+ny, 30:30+nx] = Z
colmap = plt.cm.get_cmap('afmhot').reversed()
positions = [0,409.6,819.2,1228.8,1638.4,2048]
labels = ['0','0.2','0.4','0.6','0.8','1']
ax.imshow(Z, extent=extent, origin='lower', cmap=colmap, aspect='auto', interpolation='none')
ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
ax.set_xlabel('Pulse Phase')
ax.set_ylabel('Pulse Number')

axins =ax.inset_axes([0.15,0.15,0.4,0.7])
plt.setp(list(axins.spines.values()), linewidth = 2) 

axins.imshow(Z, extent=extent, origin='lower', cmap=colmap, aspect='auto', interpolation='none')

x1, x2, y1, y2 =  1350, 1650, 400, 600
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')

axins.xaxis.set_major_locator(NullLocator())
axins.yaxis.set_major_locator(NullLocator())

#ax.indicate_inset_zoom(axins, edgecolor='green',linewidth=3)
box, cbin = ax.indicate_inset_zoom(axins, edgecolor='black',linewidth=2)
plt.setp(box, linewidth=3, edgecolor="black")
for c in cbin:
    c.linewidth = 3

#fig.tight_layout()

ax.axvline(linewidth=2, x=0, color='black')
ax.axvline(linewidth=2, x=2048, color='black')
ax.axhline(linewidth=2, y=0, color='black')
ax.axhline(linewidth=2, y=1000, color='black')

fig.show()
fig.savefig('paper_plots/1000_SP.pdf',dpi=1200)
'''

'''
## Waterfall plot with profile

prof = np.sum(active,axis=0)/len(active)
bl = np.average(active[:100])
prof = prof-bl

fig, [ax1,ax2] = plt.subplots(2,1, figsize=(8,15),sharex=True, gridspec_kw={'hspace':0,'height_ratios': [1.3, 5]})

font = 16

ax1.plot(prof,color='dimgray')
ypositions = [prof.min(),prof.max()/2,prof.max()]
ylabels = ['0','0.5','1']
ax1.yaxis.set_major_locator(ticker.FixedLocator(ypositions))
ax1.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))
ax1.set_ylim(prof.min()-0.05,prof.max()+0.02)
ax1.set_xlim(0,len(prof))
ax1.set_ylabel('Intensity \n(arb. units)',fontsize=font)

axins2 =ax1.inset_axes([0.2,0.4,0.2,0.5])
plt.setp(list(axins2.spines.values()), linewidth = 2) 

axins2.plot(prof,color='dimgray')

x12, x22, y12, y22 =  500, 720, prof.min()-0.002, prof.min()+0.015
axins2.set_xlim(x12, x22)
axins2.set_ylim(y12, y22)

axins2.xaxis.set_major_locator(NullLocator())
axins2.yaxis.set_major_locator(NullLocator())

box2, cbin2 = ax1.indicate_inset_zoom(axins2, edgecolor='black',linewidth=2)
plt.setp(box2, linewidth=1, edgecolor="black")
for c1 in cbin2:
    c1.linewidth = 1


Z = active[:1000]
extent = (0, 2048, 0, 1000)
ny, nx = Z.shape
colmap = plt.cm.get_cmap('afmhot').reversed()

positions = [0,409.6,819.2,1228.8,1638.4,2048]
labels = ['0','0.2','0.4','0.6','0.8','1']

ax2.imshow(Z, extent=extent, origin='lower', cmap=colmap, aspect='auto', interpolation='none')
ax2.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax2.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
ax2.set_xlabel('Pulse Phase',fontsize=font)
ax2.set_ylabel('Pulse Number',fontsize=font)

axins =ax2.inset_axes([0.15,0.15,0.4,0.7])
plt.setp(list(axins.spines.values()), linewidth = 2) 

axins.imshow(Z, extent=extent, origin='lower', cmap=colmap, aspect='auto', interpolation='none')

x1, x2, y1, y2 =  1350, 1650, 400, 600
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.xaxis.set_major_locator(NullLocator())
axins.yaxis.set_major_locator(NullLocator())

box, cbin = ax2.indicate_inset_zoom(axins, edgecolor='black',linewidth=2)
plt.setp(box, linewidth=3, edgecolor="black")
for c in cbin:
    c.linewidth = 3

ax1.tick_params(axis='both',labelsize=font)
ax2.tick_params(axis='both',labelsize=font)


ax1.axvline(linewidth=2, x=0, color='black')
ax1.axvline(linewidth=2, x=len(prof), color='black')
ax1.axhline(linewidth=2, y=prof.min()-0.05, color='black')
ax1.axhline(linewidth=2, y=prof.max()+0.02, color='black')

ax2.axvline(linewidth=2, x=0, color='black')
ax2.axvline(linewidth=2, x=2048, color='black')
ax2.axhline(linewidth=2, y=0, color='black')
ax2.axhline(linewidth=2, y=1000, color='black')

fig.tight_layout()
fig.show()
fig.savefig('paper_plots/1000_SP_w_prof.pdf',dpi=1200)
'''


'''
## Deconvolved plot

fdfs = pd.read_pickle("./pol_df.pkl")
Edata = fdfs["snr"]

def gaussian(x,mu,sigma):
    return np.exp(-0.5*(np.abs((x-mu)/sigma)**2))

#Convolved, normalised gaussian function
def model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]
    for unit in x:

        C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
        C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))
        sigma_noise = 1.1
        result = integrate.quad(lambda xdash: (1/(np.sqrt(2*np.pi*(sigma_noise**2))))*np.exp(-0.5*((xdash**2)/(sigma_noise**2)))*((f*(C1*np.exp(-0.5*np.abs(((unit-xdash)-mu1)/sigma1)**alpha1)))+((1-f)*C2*np.exp(-0.5*np.abs(((unit-xdash)-mu2)/sigma2)**alpha2))),x.min(),x.max())[0]

        resultbin.append(result)

    a = np.asarray(resultbin)
    return a

#This is the deconvolved version of the above.
def dc_model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):
    
    C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
    C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))

    pulse_PDF = (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))+((1-f)*C2*np.exp(-0.5*np.abs((x-mu2)/sigma2)**alpha2))
    return pulse_PDF

def gausscomp1(x,f,mu,sigma,alpha):
    C1 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return f*C1*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

def gausscomp2(x,f,mu,sigma,alpha):
    C2 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return (1-f)*C2*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)
    #return C2*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

Bilby_params = (0.15,0.60,0.49,2,8.03,6.54,4.06)
#Bilby_params = (0.15,0.60,0.49,2,0.03,0.54,1.06)

fig,ax = plt.subplots()
E_y,E_x,E_=ax.hist(Edata, 50, alpha=.3,color="dimgray", label='Single Pulse S/N', density=True)


#ax.hist()
E_x = (E_x[1:]+E_x[:-1])/2
min_x = E_x.min()
max_x = E_x.max()
xarray = np.linspace(min_x,max_x,200)

convolved_model = model(xarray, *Bilby_params)
deconvolved_model = dc_model(xarray, *Bilby_params)
noise_model = 1/(1.1*np.sqrt(2*np.pi))*np.exp(-0.5*np.abs(xarray/1.1)**2)

reconvolve = np.convolve(deconvolved_model, noise_model, "full")
xarray2 = np.linspace(min_x,max_x,len(reconvolve))
dx = xarray[1]-xarray[0]
reconvolve = (reconvolve/(max(reconvolve)+sys.float_info[3]))
convolved_modelnorm = model(xarray2, *Bilby_params)

log_rep = np.log10(xarray)
log_rep2 = np.log10(xarray2)

gauss1 = gausscomp1(xarray,Bilby_params[0],Bilby_params[1],Bilby_params[2],Bilby_params[3])
gauss2 = gausscomp2(xarray,Bilby_params[0],Bilby_params[4],Bilby_params[5],Bilby_params[6])

loggauss1 = gausscomp1(log_rep,Bilby_params[0],Bilby_params[1],Bilby_params[2],Bilby_params[3])
loggauss2 = gausscomp2(log_rep,Bilby_params[0],Bilby_params[4],Bilby_params[5],Bilby_params[6])

logconvolved = model(log_rep2, *Bilby_params)
logdeconvolved = dc_model(log_rep, *Bilby_params)

ax.plot(xarray2,convolved_modelnorm,c="tab:purple", label= "Model")
ax.plot(xarray,deconvolved_model,c = "tab:green", label= "Deconvolved Model")
#ax.plot(xarray,gauss1, c = "tab:orange", label = "Weak mode")
#ax.plot(xarray, gauss2, c = "tab:blue",label = "Strong Mode")

#ax.plot(xarray2, logconvolved,c="tab:purple", label= "Modellog")
#ax.plot(xarray, logdeconvolved,c = "tab:green", label= "Deconvolved Modellog")
ax.plot(xarray, loggauss1, c = "tab:orange", label = "Weak mode ($\log$ S/N)")
ax.plot(xarray, loggauss2, c = "tab:blue",label = "Strong Mode ($\log$ S/N)")


#ax.plot(xarray2,convolved_modelnorm,c="tab:purple", label= "Model")
#ax.plot(xarray,deconvolved_model,c = "tab:green", label= "Deconvolved Model")
#ax.plot(xarray,gauss1, c = "tab:orange", label = "Weak mode")
#ax.plot(xarray, gauss2, c = "tab:blue",label = "Strong Mode")

ax.set_xlim(-5,23)
ax.set_ylim(0.005,0.16)
#ax.axvline(linewidth=2, x=-5, color='black')
#ax.axvline(linewidth=2, x=23, color='black')
#ax.axhline(linewidth=2, y=0, color='black')
#ax.axhline(linewidth=2, y=0.16, color='black')

#ax.set_yscale('log')
#ax.set_xscale('log')

ax.set_xlabel("S/N")
ax.set_ylabel("Probability Density")
ax.legend(prop={'size':10})
fig.savefig('paper_plots/snr_overlay.pdf',dpi=1200)
fig.show()
'''
'''
## New equation deconvolved plot

os.chdir('/fred/oz002/users/mmiles/SinglePulse/')

fdfs = pd.read_pickle("./pol_df.pkl")
Edata = fdfs["snr"]

def model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]

    C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
    C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))

    #Normalisation integral
    Cbig = integrate.quad(lambda x: \
        (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))\
            +((1-f)*C2*np.exp((-0.5*np.abs((x-mu2)/sigma2)**alpha2)))\
                ,0,x.max())[0]
    

    #Numerical integration
    for unit in x:

        sigma_noise = 1.1
        result = integrate.quad(lambda xdash: \
            (1/(np.sqrt(2*np.pi*(sigma_noise**2))))*np.exp(-0.5*(((unit-xdash)**2)/(sigma_noise**2)))*\
                ((f*C1*np.exp(-0.5*np.abs((xdash-mu1)/sigma1)**alpha1))\
                    +((1-f)*C2*np.exp(-0.5*np.abs((xdash-mu2)/sigma2)**alpha2)))\
                        ,0,x.max())[0]

        #resultC = result/Cbig
        resultbin.append(result)

    a = np.asarray(resultbin)
    a = a/Cbig
    return a

def dc_model(x,f,mu1,sigma1,alpha1,mu2,sigma2,alpha2):

    resultbin =[]
    Call = []

    C1 = alpha1/((2**(1+(1/alpha1)))*sigma1*gamma(1/alpha1))
    C2 = alpha2/((2**(1+(1/alpha2)))*sigma2*gamma(1/alpha2))

    #Normalisation integral
    Cbig = integrate.quad(lambda x: \
        (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))\
            +((1-f)*C2*np.exp((-0.5*np.abs((x-mu2)/sigma2)**alpha2)))\
                ,0,x.max())[0]


    result = (f*C1*np.exp(-0.5*np.abs((x-mu1)/sigma1)**alpha1))\
            +((1-f)*C2*np.exp(-0.5*np.abs((x-mu2)/sigma2)**alpha2))
        #print(result)
        #resultbin.append(result)

    
    resultbin = np.asarray(resultbin)
    result = result/Cbig
    
    a = result
    return a, Cbig

def gausscomp1(x,f,mu,sigma,alpha):
    C1 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return f*C1*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

def gausscomp2(x,f,mu,sigma,alpha):
    C2 = alpha/((2**(1+(1/alpha)))*sigma*gamma(1/alpha))
    return (1-f)*C2*np.exp(-0.5*np.abs((x-mu)/sigma)**alpha)

Bilby_params = (0.16,0.13,0.28,2,7.28,7.29,4.11)

fig,ax = plt.subplots()

E_y,E_x,E_=hist(Edata, 50, alpha=.3,color="dimgray", label='Single Pulse S/N', density=True)
E_x = (E_x[1:]+E_x[:-1])/2
min_x = E_x.min()
max_x = E_x.max()
xarray = np.linspace(min_x,max_x,200)

convolved_model = model(xarray, *Bilby_params)

Null_bilby = (0.32,0,1.1,2,8.44,5.99,3.30)

model2 = model(xarray, *Null_bilby)

deconvolved_model, Cbig = dc_model(xarray, *Bilby_params)

heavi_x = np.heaviside(xarray,1)
#heavi_x[51]=0.565

noise_model = (1/(1.1*np.sqrt(2*np.pi)))*np.exp(-0.5*(np.abs(xarray/1.1)**2))

heavi_dc = heavi_x*deconvolved_model

gauss1 = gausscomp1(xarray,Bilby_params[0],Bilby_params[1],Bilby_params[2],Bilby_params[3])/Cbig
gauss1 = heavi_x*gauss1
gauss2 = gausscomp2(xarray,Bilby_params[0],Bilby_params[4],Bilby_params[5],Bilby_params[6])/Cbig
gauss2 = heavi_x*gauss2


ax.plot(xarray[xarray>=0],heavi_dc[xarray>=0],c = "tab:green",linewidth=1.5, label= "Deconvolved Model")
ax.axvline(linewidth=1.5,c = "tab:green",ymax=heavi_dc[xarray>0][0]/0.32)

ax.plot(xarray,convolved_model,c="tab:purple",linewidth=1.5, label= "Model")
#ax.plot(xarray,model2,c="tab:green",linewidth=1.5, label= "Null Model")

ax.plot(xarray[xarray>=0],gauss1[xarray>=0], c = "tab:orange",linewidth=1.5, label = "Weak mode")
ax.axvline(linewidth=1.5,c = "tab:orange",ymax=gauss1[xarray>=0][0]/0.32)

ax.plot(xarray[xarray>=0], gauss2[xarray>=0], c = "tab:blue",linewidth=1.5, label = "Strong Mode")
ax.axvline(linewidth=1.5,c = "tab:blue",ymax=gauss2[xarray>=0][0]/0.32)

#ax.axvline(linewidth=2, x=1.375, color='black',linestyle='--')

ax.set_xlabel("S/N")
ax.set_ylabel("Probability Density")
ax.legend(prop={'size':10})

ax.set_xlim(-5,23)
ax.set_ylim(0,0.32)
ax.axvline(linewidth=2, x=-5, color='black')

ax.axvline(linewidth=2, x=23, color='black')
ax.axhline(linewidth=2, y=0, color='black')
ax.axhline(linewidth=2, y=0.32, color='black')
ax.grid(False)

fig.show()
fig.savefig('paper_plots/heavi_snr_overlay.pdf',dpi=1200)
'''
'''
## Scaled Pulse Profiles

active = np.load("fdata_smaller_scrunched.npy")
limit = 1.49511
fdfs = pd.read_pickle("./Freq_small_df.pkl")

strong_array = active[fdfs['snr']>=limit]
weak_array = active[fdfs['snr']<limit]

strong_profile = np.sum(strong_array,axis=0)/len(strong_array)
weak_profile = np.sum(weak_array,axis=0)/len(weak_array)

weak_baseline = sum(weak_profile[:400])/len(weak_profile[:400])
strong_baseline = sum(strong_profile[:400])/len(strong_profile[:400])

scaled_strong = strong_profile - strong_baseline
scaled_weak = weak_profile - weak_baseline

fig, axs = plt.subplots() 
axs.set_xlim(1400,1600)
axs.plot(scaled_strong/(max(scaled_strong)),label = 'Scaled Strong Profile')
axs.plot(scaled_weak/(max(scaled_weak)), label = 'Scaled Weak Profile')
axs.set_xlim(1400,1600)
axs.set_ylim(-0.5,1.1)
#axs.set_title('Scaled comparison')
axs.set_xlabel('Phase bins')
axs.set_ylabel('Normalised Flux')
axs.legend(prop={'size':10}, loc=1)
axs.grid(False)
axs.axvline(linewidth=1, x=1450,color='black',linestyle='--')
axs.axvline(linewidth=1, x=1550,color='black',linestyle='--')

axs.axvline(linewidth=2, x=1400, color='black')
axs.axvline(linewidth=2, x=1600, color='black')
axs.axhline(linewidth=2, y=-0.5, color='black')
axs.axhline(linewidth=2, y=1.1, color='black')

#axs.set_aspect(50)
#fig.tight_layout()
fig.show()
fig.savefig('paper_plots/scaled_profiles.pdf',dpi=1200)
'''
'''
## Tempo2 residuals

strongres = "/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/strong_residuals.dat"
weakres = "/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/weak_residuals.dat"

strongsimdir = "/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-DD/"
strong_simultaneousres = "/fred/oz002/users/mmiles/SinglePulse/Jitter_check/bilby_runs/alldata_smoothed_weakStrongTemplatePython_TMtoa_Gauss_FDM_dyn/strong_simultaneous.dat"

strongtime = (np.loadtxt(strongres, usecols = 0) - 58892.128521291588651)*86400
strongdata = np.loadtxt(strongres, usecols = 1)*1e6
strongerr = np.loadtxt(strongres, usecols = 2)

weaktime = (np.loadtxt(weakres, usecols = 0) - 58892.128521291588651)*86400
weakdata = (np.loadtxt(weakres, usecols = 1) - 9.2590662802399e-06)*1e6
weakerr = np.loadtxt(weakres, usecols = 2)

strongtime_simul = (np.loadtxt(strong_simultaneousres, usecols = 0) - 58892.128521291588651)*86400
strongdata_simul = np.loadtxt(strong_simultaneousres, usecols = 1)*1e6
strongdata_simul_up = np.load(strongsimdir+'/str_unc_up.npy')
strongdata_simul_low = np.load(strongsimdir+'/str_unc_low.npy')

fig, [ax1,ax2] = plt.subplots(2,1,figsize=(8,9))

ax1.errorbar(strongtime, strongdata, yerr=strongerr, color='mediumblue',label='Strong Mode',alpha=0.6,marker='o', linestyle='None')
ax1.errorbar(weaktime, weakdata, yerr=weakerr, color='tab:orange',label='Weak Mode',marker='o', linestyle='None')

ax1.set_xlabel('Observation Length (s)')
ax1.set_ylabel(r'Prefit Residual ($\mu s$)')

ax1.set_xlim(-5,165)
ax1.set_ylim(-27.5,12.5)
ax1.legend(loc=1,prop={'size':12})
ax1.grid(False)

#ax.axvline(linewidth=2, x=-5, color='black')
#ax.axvline(linewidth=2, x=165, color='black')
#ax.axhline(linewidth=2, y=-27.5, color='black')
#ax.axhline(linewidth=2, y=12.5, color='black')
#ax.set_yscale('log')


ax2.errorbar(strongtime, strongdata, yerr=strongerr, color='mediumblue',label='Individually Timed',alpha=0.6,marker='o', linestyle='None')
ax2.errorbar(strongtime_simul,strongdata_simul,yerr=[strongdata_simul_low,strongdata_simul_up],color='crimson',label='Simultaneously Timed',alpha=0.4,marker='o', linestyle='None')

ax2.set_xlabel('Observation Length (s)')
ax2.set_ylabel(r'Prefit Residual ($\mu s$)')

ax2.set_xlim(-5,165)
ax2.set_ylim(-3.5,3.5)
ax2.legend(loc=1,prop={'size':12})
ax2.grid(False)

fig.tight_layout()
fig.show()
os.chdir('/fred/oz002/users/mmiles/SinglePulse')
fig.savefig('paper_plots/tempo_residuals.pdf',dpi=1200)
'''
'''
## Timing precision

xarray = [2,4,8,256,32,64,128,256,512]
xarray2 = [8,16,32,64,128,256,512,1024]


strong = [6.995,3.580,2.673,1.890,1.334,0.936,0.697,0.488,0.347,0.260]
strong_culled = [2.673,1.890,1.334,0.936,0.697,0.488,0.347,0.260]

strongpath = "/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tempo2_strong"
strongtim8 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft8_tim',skiprows=1,usecols=3)
strongerr8 = (np.std(strongtim8))/np.sqrt(len(strongtim8))
strongtim16 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft16_tim',skiprows=1,usecols=3)
strongerr16 = (np.std(strongtim16))/np.sqrt(len(strongtim16))
strongtim32 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft32_tim',skiprows=1,usecols=3)
strongerr32 = (np.std(strongtim32))/np.sqrt(len(strongtim32))
strongtim64 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft64_tim',skiprows=1,usecols=3)
strongerr64 = (np.std(strongtim64))/np.sqrt(len(strongtim64))
strongtim128 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft128_tim',skiprows=1,usecols=3)
strongerr128 = (np.std(strongtim128))/np.sqrt(len(strongtim128))
strongtim256 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft256_tim',skiprows=1,usecols=3)
strongerr256 = (np.std(strongtim256))/np.sqrt(len(strongtim256))
strongtim512 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft512_tim',skiprows=1,usecols=3)
strongerr512 = (np.std(strongtim512))/np.sqrt(len(strongtim512))
strongtim1024 = np.loadtxt(strongpath+'/J1909-3744.strongstd_strongdata_Ft1024_tim',skiprows=1,usecols=3)
strongerr1024 = (np.std(strongtim1024))/np.sqrt(len(strongtim1024))
strong_errs = [strongerr8, strongerr16, strongerr32, strongerr64, strongerr128, strongerr256, strongerr512, strongerr1024]

strong075 = [11.236,4.062,2.718,1.96,1.405,0.978,0.7256,0.496,0.384,0.284]
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

strong075_culled = [np.nan,4.062,2.718,1.96,1.405,0.978,0.7256,0.496,0.384,0.284]
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

weak = [855.089,873.317,867.465,870.967,755.178,922.2567,639.964,938.707,663.321]
total = [20.636,8.227,2.822,2.085,1.496,1.065,0.742,0.547,0.380]
total_culled = [2.822,2.085,1.496,1.065,0.742,0.547,0.380,0.291]

totalpath = "/fred/oz002/users/mmiles/SinglePulse/tempo2_all_new"
totaltim8 = np.loadtxt(totalpath+'/J1909-3744.allstd_t8F_tim',skiprows=1,usecols=3)
totalerr8 = (np.std(totaltim8))/np.sqrt(len(totaltim8))
totaltim16 = np.loadtxt(totalpath+'/J1909-3744.allstd_t16F_tim',skiprows=1,usecols=3)
totalerr16 = (np.std(totaltim16))/np.sqrt(len(totaltim16))
totaltim32 = np.loadtxt(totalpath+'/J1909-3744.allstd_t32F_tim',skiprows=1,usecols=3)
totalerr32 = (np.std(totaltim32))/np.sqrt(len(totaltim32))
totaltim64 = np.loadtxt(totalpath+'/J1909-3744.allstd_t64F_tim',skiprows=1,usecols=3)
totalerr64 = (np.std(totaltim64))/np.sqrt(len(totaltim64))
totaltim128 = np.loadtxt(totalpath+'/J1909-3744.allstd_t128F_tim',skiprows=1,usecols=3)
totalerr128 = (np.std(totaltim128))/np.sqrt(len(totaltim128))
totaltim256 = np.loadtxt(totalpath+'/J1909-3744.allstd_t256F_tim',skiprows=1,usecols=3)
totalerr256 = (np.std(totaltim256))/np.sqrt(len(totaltim256))
totaltim512 = np.loadtxt(totalpath+'/J1909-3744.allstd_t512F_tim',skiprows=1,usecols=3)
totalerr512 = (np.std(totaltim512))/np.sqrt(len(totaltim512))
totaltim1024 = np.loadtxt(totalpath+'/J1909-3744.allstd_t1024F_tim',skiprows=1,usecols=3)
totalerr1024 = (np.std(totaltim1024))/np.sqrt(len(totaltim1024))
total_errs = [totalerr8, totalerr16, totalerr32, totalerr64, totalerr128, totalerr256, totalerr512, totalerr1024]
#reg_timing = [1.443,1.321,1.326,1.264,1.219,1.103,1.005,1.008]

#Integration length comparison
length = [1,2,3,5,10,15,20,25]
strongint = [0.242,0.240,0.240,0.158,0.127,0.162,0.097,0.107]
weakint = [0.651,0.622,0.613,0.371,0.350,0.323,0.322,0.301]
allint = [0.295,0.293,0.292,0.149,0.113,0.145,0.097,0.088]
regint = [0.326,0.326,0.326,0.273,0.282,0.282,0.172,0.186]

fig, ax = plt.subplots()

#x.plot(xarray2,strong_culled,'o-', color = "tab:blue", linewidth=1, label='Strong Mode')
ax.errorbar(xarray2,strong_culled, yerr = strong_errs, color = "tab:blue",marker='o', linewidth=1, label='Strong Mode')
#ax.plot(xarray2,total_culled,'o-', color = "dimgray",linewidth=1, label='Total Single Pulses')
ax.errorbar(xarray2,total_culled, yerr = total_errs, color = "dimgray",marker='o',linewidth=1, label='Total Single Pulses')
ax.set_xlabel('Time averaged pulses per data point')
ax.set_ylabel(r'$\mathrm{RMS}\/ (\mu s)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(6,1200)
ax.set_ylim(0.2,3.1)
#ax.set_title('RMS vs degree of time averaging')
#ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.legend(prop={'size':10})
ax.axvline(linewidth=2, x=6, color='black')
ax.axvline(linewidth=2, x=1200, color='black')
ax.axhline(linewidth=2, y=0.2, color='black')
ax.axhline(linewidth=2, y=3.1, color='black')
#ax.set_aspect(5)
#fig.tight_layout()
fig.savefig('paper_plots/timing_rms.pdf',dpi=1200)
fig.show()

'''
'''
## Jitter comparison
Tsub1024 = 3.0179
Tsub512 = 1.509
Tsub256 = 0.7545
Tsub128 = 0.377
Tsub64 = 0.188625
Tsub32 = 0.0943125
Tsub16 = 0.04715625
Tsub8 = 0.023574125
Tsub4 = 0.0117890625
Tsub2 = 0.00589453125

xtsub =[Tsub1024,Tsub512,Tsub256,Tsub128,Tsub64,Tsub32,Tsub16,Tsub8,Tsub4,Tsub2]
jdata = pd.read_pickle('jitter_data.pkl')
jdata = np.array(jdata)
jdata = jdata[0,:]
jstrong = jdata[1::2]
jall = jdata[0::2]
jerror = pd.read_pickle('jitter_error.pkl')

#strong_err = np.array(jerror.iloc[1])
#all_err = np.array(jerror.iloc[0])

strong_err = np.load('actualjiterr_strong.npy') 
all_err = np.load('actualjiterr_all.npy')

fig, ax = plt.subplots()

ax.set_xlabel('Subintegration time (s)')
ax.set_ylabel('Implied Jitter in 1hr (ns)')
#ax.set_title('Jitter noise vs integration time')
ax.set_xscale('log')
ax.set_yscale('log')

#ax.plot(xtsub,jall_culled,'o-', c='dimgray', linewidth=1,label='All pulses')
#ax.plot(xtsub,jstrong_culled,'o-', c='tab:blue', linewidth=1,label='Strong pulses')

ax.errorbar(xtsub[:6],jall[:6]*10**9, yerr = all_err[:6]*10**9,c='dimgray', marker='o',label='All pulses')
ax.errorbar(xtsub[:7],jstrong[:7]*10**9, yerr = strong_err[:7]*10**9, c='tab:blue', marker='o',label='Strong pulses')

#ax.plot(xtsub,jweak,'o-', c='tab:orange', linewidth=1,label='Weak pulses')
#Adityas values
#plt.axhline(9*10**-9, c='tab:purple', linewidth=1, linestyle='--', label='Previous jitter value')
ax.fill_between(xtsub,6,12, hatch='\\',facecolor='None',edgecolor='orchid',alpha=0.7, label='Parthasarathy et al. (2021)',zorder=-10)
ax.fill_between(xtsub,9.4,7.8, hatch='|',facecolor='None',edgecolor='lightskyblue',alpha=1, label='Shannon et al. (2014)',zorder=-10)
ax.fill_between(xtsub,13.5,14.5, hatch='//',facecolor='None',edgecolor='lightslategrey',alpha=0.7, label='Lam et al. (2019)',zorder=-10)
#ax.yaxis.set_major_locator(ticker.FixedLocator(1*10**-8))

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(10)
#Ryans values
#plt.axhline(8.6*10**-9, c='tab:purple', linewidth=1, linestyle='--', label='Previous jitter value')
#ax.fill_between(xtsub,7.8*10**-9,9.4*10**-9, color='tab:purple', alpha=0.3)
ax.legend(prop={'size':10})
ax.set_ylim(5,15)
ax.set_xlim(0.004,4)
ax.axvline(linewidth=2, x=0.004, color='black')
ax.axvline(linewidth=2, x=4, color='black')
ax.axhline(linewidth=2, y=5, color='black')
ax.axhline(linewidth=2, y=15, color='black')
ax.grid(False)
#plt.ticklabel_format(axis='both',style='plain')
fig.savefig('/fred/oz002/users/mmiles/SinglePulse/paper_plots/jitter_comparison.pdf',dpi=1200)
fig.show()
'''
'''
## Stokes Parameters
Main = "/fred/oz002/users/mmiles/SinglePulse/"
source = "/fred/oz002/users/mmiles/SinglePulse/SinglePulse_data"

df = pd.read_pickle('pol_stokesI_df.pkl')
os.chdir(Main)
limit = 1.37501
#limit=0.668

active = np.load('stokes_data_F.npy')

strong = active[df['snr']>=limit]
weak = active[df['snr']<limit]

strongprofs = np.sum(strong, axis=0)
weakprofs = np.sum(weak,axis=0)

strongI, strongQ, strongU, strongV = strongprofs[0], strongprofs[1], strongprofs[2], strongprofs[3]
weakI, weakQ, weakU, weakV = weakprofs[0], weakprofs[1], weakprofs[2], weakprofs[3]

strongIbaseline = sum(strongI[:400])/len(strongI[:400])
strongQbaseline = sum(strongQ[:400])/len(strongQ[:400])
strongUbaseline = sum(strongU[:400])/len(strongU[:400])
strongVbaseline = sum(strongV[:400])/len(strongV[:400])

weakIbaseline = sum(weakI[:400])/len(weakI[:400])
weakQbaseline = sum(weakQ[:400])/len(weakQ[:400])
weakUbaseline = sum(weakU[:400])/len(weakU[:400])
weakVbaseline = sum(weakV[:400])/len(weakV[:400])

strongI = strongI - strongIbaseline
strongQ = strongQ - strongQbaseline
strongU = strongU - strongUbaseline
strongV = strongV - strongVbaseline

weakI = weakI - weakIbaseline
weakQ = weakQ - weakQbaseline
weakU = weakU - weakUbaseline
weakV = weakV - weakVbaseline

PAstrong = (0.5*np.arctan(strongU/strongQ))*(180/np.pi)
PAsplot = [x if x in PAstrong[1445:1525] else np.nan for x in PAstrong]
PAweak = (0.5*np.arctan(weakU/weakQ))*(180/np.pi)
PAwplot = [x if x in PAweak[1456:1502] else np.nan for x in PAweak]
delta_strong = strongU[1445:1525]/strongQ[1445:1525]
delta_weak = weakU[1456:1502]/weakQ[1456:1502]

gs = gridspec.GridSpec(2,2,hspace=0.3,height_ratios=[1,5])
plt.figure(figsize=(9,7))

#fig, (ax1, (ax3, ax4)) = plt.subplots(2,1,sharex=False, gridspec_kw={'height_ratios': [1, 5]}) 
ax1 = plt.subplot(gs[0,:])
ax3 = plt.subplot(gs[1,0])
ax4 = plt.subplot(gs[1,1])


positions = [1425,1500,1575]
labels = ['0.70','0.73','0.77']

ax1.plot(PAsplot, color='tab:blue',label='Strong Mode')
ax1.set_xlim(1425,1575)
ax1.set_ylim(-45,45)
ax1.set_ylabel(r'P.A. ($\degree$)')

PApositions = [-45,0,45]
PAlabels = ['-45','0','-45']
ax1.yaxis.set_major_locator(ticker.FixedLocator(PApositions))
ax1.yaxis.set_major_formatter(ticker.FixedFormatter(PAlabels))
ax1.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax1.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

#ax1.xaxis.set_major_locator(NullLocator())

ax1.axvline(linewidth=2, x=1425, color='black')
ax1.axvline(linewidth=2, x=1575, color='black')
ax1.axhline(linewidth=2, y=45, color='black')
ax1.axhline(linewidth=2, y=-45, color='black')

ax3.plot(strongI, label='I',color='dimgray')
ax3.plot(strongQ, label='Q',color='red')
ax3.plot(strongU, label='U',color='magenta')
ax3.plot(strongV, label='V',color='blue')

#ax.legend()
#ax.set_title('strong')
ax3.set_xlim(1425,1575)
ax3.set_ylim(-15000,34000)

ax3.text(.5,.9,'Strong Mode',
        fontweight='bold',
        horizontalalignment='center',
        transform=ax3.transAxes)



ypositions = [strongI.min(),strongI.max()/2,strongI.max()]
ylabels = ['0.0','0.5','1.0']
ax3.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax3.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
ax3.yaxis.set_major_locator(ticker.FixedLocator(ypositions))
ax3.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))

#ax3.xaxis.set_major_locator(NullLocator())
#ax3.yaxis.set_major_locator(NullLocator())
ax3.axvline(linewidth=2, x=1425, color='black')
ax3.axvline(linewidth=2, x=1575, color='black')
ax3.axhline(linewidth=2, y=-15000, color='black')
ax3.axhline(linewidth=2, y=34000, color='black')
ax3.set_ylabel('Intensity \n(arb. units)')
ax3.set_xlabel('Pulse Phase')

ax1.plot(PAwplot, color='tab:orange',label='Weak Mode')
ax1.set_xlim(1425,1575)
ax1.set_ylim(-45,45)
ax1.legend(prop={'size':12})
ax1.set_xlabel('Pulse Phase')
#ax2.xaxis.set_major_locator(NullLocator())
#ax2.yaxis.set_major_locator(NullLocator())

#ax2.axvline(linewidth=2, x=1425, color='black')
#ax2.axvline(linewidth=2, x=1575, color='black')
#ax2.axhline(linewidth=2, y=45, color='black')

ax4.plot(weakI, label='I',color='dimgray')
ax4.plot(weakQ, label='Q',color='red')
ax4.plot(weakU, label='U',color='magenta')
ax4.plot(weakV, label='V',color='blue')

ax4.legend(prop={'size':12})
#ax.set_title('weak')
ax4.set_xlim(1425,1575)
ax4.set_ylim(-260,420)
#ax4.set_ylim(-150,180)

ax4.text(.5,.9,'Weak Mode',
        fontweight='bold',
        horizontalalignment='center',
        transform=ax4.transAxes)

ax4.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax4.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
ax4.yaxis.set_major_locator(NullLocator())


ax4.axvline(linewidth=2, x=1425, color='black')
ax4.axvline(linewidth=2, x=1575, color='black')
ax4.axhline(linewidth=2, y=-260, color='black')
ax4.axhline(linewidth=2, y=420, color='black')
ax4.set_xlabel('Pulse Phase')

ax1.grid(False)
ax3.grid(False)
ax4.grid(False)

#plt.tight_layout()

#plt.savefig('/fred/oz002/users/mmiles/SinglePulse/paper_plots/StokesModes.pdf',dpi=1200)
plt.show()
'''
'''
## Lomb-Scargle

fluence = df['Fluence'].values
x = np.linspace(1,len(fluence),len(fluence))
freq = np.linspace(0,0.5,26500)

noise = df['Baseline Noise'].values


ls = LombScargle(x,fluence,normalization='psd')
power = ls.power(freq)
power = power/fluence.var()
fap_pulse = ls.false_alarm_probability(np.nanmax(power),method='naive',samples_per_peak=1,nyquist_factor=1)

lsnoise = LombScargle(x,noise, normalization='psd')
powernoise = lsnoise.power(freq)
powernoise = powernoise/noise.var()
fap_noise = lsnoise.false_alarm_probability(np.nanmax(powernoise),method='naive',samples_per_peak=1,nyquist_factor=1)

#power = LombScargle(x, fluence,normalization='psd').power(freq)

fig, ax =plt.subplots()
ax.plot(freq, power)
ax.set_xlabel('frequency')
ax.set_ylabel('Lomb-Scargle Power')
#ax.set_xlim(-50,2050)
#ax.set_ylim(-.00002,0.00042)

#ax.axvline(linewidth=2, x=-50, color='black')
#ax.axvline(linewidth=2, x=2050, color='black')
#ax.axhline(linewidth=2, y=-.00002, color='black')
#ax.axhline(linewidth=2, y=-.00042, color='black')

fig.show()
'''
'''
# Two-mode Timing
def fft_rotate(data, bins):
    """Return data rotated by 'bins' places to the left. The
       rotation is done in the Fourier domain using the Shift Theorem.
​
       Inputs:
            data: A 1-D numpy array to rotate.
            bins: The (possibly fractional) number of phase bins to rotate by.
​
       Outputs:
            rotated: The rotated data.
    """
    freqs = np.arange(data.size/2+1, dtype=np.float)
    phasor = np.exp(complex(0.0, 2.0*np.pi) * freqs * bins / float(data.size))
    return np.fft.irfft(phasor*np.fft.rfft(data))

nbins = 2048
x = np.linspace(0, 1, nbins)

weak_template = np.load('/fred/oz002/users/mmiles/SinglePulse/weak_prof_smoothed.npy')
strong_template = np.load('/fred/oz002/users/mmiles/SinglePulse/strong_prof_smoothed.npy')

weak_template = fft_rotate(weak_template,weak_template.argmax()-strong_template.argmax())

datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/all_Ft128/all.Ft128')
datas.remove_baseline()
datas.dedisperse()
datas = datas.get_data()
datas = datas[:,0,0,:]

def avg_profile_model(x, weak_amp, weak_phase, strong_amp, strong_phase):
 
    weak_mode = fft_rotate(weak_amp * strong_amp * weak_template / np.max(weak_template), strong_phase + weak_phase)
    strong_mode = fft_rotate(strong_amp * strong_template / np.max(strong_template), strong_phase)
    
    #ratio = (np.max(weak_mode)/np.max(strong_mode))*0.15
    #ratio = (np.max(weak_mode)/np.max(strong_mode))
    return strong_mode + weak_mode

i=28
sa = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-final/str_amps.npy')
sp = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-final/str_phases_slurm.npy') 
wa = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-final/weak_amps.npy')
wp = np.load('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-final/weak_phases_slurm.npy') 


strong = fft_rotate(sa[i] * strong_template / np.max(strong_template), sp[i])
weak = fft_rotate(wa[i] * sa[i] * weak_template / np.max(weak_template), sp[i] + wp[i])

fig,ax = plt.subplots() 

ax.plot(datas[i],label='Folded Profile Data')
ax.plot(strong,label='Strong Component') 
ax.plot(weak,label='Weak Component')
ax.legend()
ax.set_xlim(1300,1700)
ax.set_xlabel('Pulse Phase')
ax.set_ylabel('Intensity \n(arb. units)')

positions = [1300,1500,1700]
labels = ['0.63','0.73','0.83']
ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

ypositions = [strong.min(),datas[i].max()/2,datas[i].max()]
ylabels = ['0.0','0.5','1.0']

ax.yaxis.set_major_locator(ticker.FixedLocator(ypositions))
ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))

ax.grid('False')

fig.tight_layout()
fig.show()
fig.savefig('/fred/oz002/users/mmiles/SinglePulse/paper_plots/TwoMode_Temps.pdf',dpi=1200)
'''

## Bilby Corner Plot

bilby_dir = '/fred/oz002/users/mmiles/SinglePulse/outdir_Positive_Pulses_nlive_2000'

os.chdir(bilby_dir)

result = bilby.result.read_in_result(filename='linear_regression_unknown_noise_result.json')

plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
kwargs2 = dict( 
                 bins=50, smooth=0.9, label_kwargs=dict(fontsize=26), 
                 title_kwargs=dict(fontsize=20), color='#0072C1', 
                 truth_color='tab:orange', quantiles=[0.16, 0.84], 
                 levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), 
                 plot_density=False, plot_datapoints=True, fill_contours=True, 
                 max_n_ticks=3, hist_kwargs=dict(density=True))

fig,ax = plt.subplots(figsize=(15,20))
fig = result.plot_corner(parameters=['f','mu1','sigma1','mu2','sigma2','alpha2'],**kwargs2,display=True)

#fig.tight_layout()
fig.show()
fig.savefig('/fred/oz002/users/mmiles/SinglePulse/paper_plots/bilby_corner.pdf',dpi=1000)
