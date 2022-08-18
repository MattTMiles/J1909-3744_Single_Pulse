#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 21:48:07 2021
​
@author: dreardon
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby
import psrchive
import os


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


"""
Block of code to generate simulated data and templates for two modes
"""

# Pulse phase array
#nbins = 1024
nbins = 2048
x = np.linspace(0, 1, nbins)

# parameters for generating simulated Gaussian templates:
strong_phase_sim = 0
strong_amp_sim = 1
strong_width = 0.02
weak_width = 0.05
noise = 0.0012

# parametrisation of weak mode
weak_amp_sim = 0.1  # amplitude of weak mode as a fraction of strong
weak_phase_sim = 0.05  # phase offset of weak mode relative to strong

# Generate simulated templates for weak and strong mode
#weak_template = strong_amp_sim * weak_amp_sim * \
#    np.exp(-(x - 0.5)**2 / (2*weak_width**2))
#strong_template = strong_amp_sim * \
#    np.exp(-(x - 0.5)**2 / (2*strong_width**2))

weak_template = np.load('/fred/oz002/users/mmiles/SinglePulse/weak_prof_smoothed.npy')
strong_template = np.load('/fred/oz002/users/mmiles/SinglePulse/strong_prof_smoothed.npy')

weak_template = fft_rotate(weak_template,weak_template.argmax()-strong_template.argmax())

#strong_template = fft_rotate(strong_template,
#                             strong_phase_sim * nbins)
#weak_template = fft_rotate(weak_template,
#                           (strong_phase_sim + weak_phase_sim) * nbins)

# Form average profile
#avg_profile = weak_template + strong_template

# Generate a simulated dataset by adding Gaussian noise
#data = avg_profile + np.random.normal(loc=0, scale=noise, size=np.shape(x))
#datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/all_Ft128/all.Ft128')

#All data
#datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/all_Ft128/all.Ft128')

#Long term data
#datas = psrchive.Archive_load('/fred/oz002/users/mmiles/templates/2D_Templates/J1909-3744/timing_256/all.TpF_256')

#Strong data
datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/strong_Ft128/all.Ft128_strong')

datas.remove_baseline()
datas.dedisperse()
datas = datas.get_data()
datas = datas[:,0,0,:]
data = datas[0]


for i, data in enumerate(datas):
# Bilby likelihood
    
    os.system('sbatch ~/soft/DR/two_mode_slurm.sh '+str(i))


'''for i in [16,184,20,227,24,26]:
    os.system('sbatch ~/soft/SP/two_mode_slurm.sh '+str(i))'''

