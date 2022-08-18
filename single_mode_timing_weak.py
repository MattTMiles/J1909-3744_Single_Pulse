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
avg_template = np.load('/fred/oz002/users/mmiles/SinglePulse/avg_profile.npy')
#weak_template = np.load('/fred/oz002/users/mmiles/SinglePulse/weak_prof_smoothed.npy')
#strong_template = np.load('/fred/oz002/users/mmiles/SinglePulse/strong_profile_correct.npy')

#weak_template = fft_rotate(weak_template,weak_template.argmax()-strong_template.argmax())

#strong_template = fft_rotate(strong_template,
#                             strong_phase_sim * nbins)
#weak_template = fft_rotate(weak_template,
#                           (strong_phase_sim + weak_phase_sim) * nbins)

# Form average profile
#avg_profile = weak_template + strong_template

# Generate a simulated dataset by adding Gaussian noise
#data = avg_profile + np.random.normal(loc=0, scale=noise, size=np.shape(x))
datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/all_Ft128/all.Ft128')
datas.remove_baseline()
datas.dedisperse()
datas = datas.get_data()
datas = datas[:,0,0,:]
data = datas[0]

'''
# Plot the data and model
plt.figure(figsize=(9, 6))
plt.plot(x, data)
plt.plot(x, strong_template)
plt.plot(x, weak_template)
plt.xlabel('Pulse phase (arb)')
plt.ylabel('Amplitude / (Strong-mode amplitude)')
plt.legend(['Data', 'Strong-mode template', 'Weak-mode template'])
plt.show()
'''

"""
Block of code to fit amplitude and phase offset of each mode to average data
"""


# Define Bilby-compatible model
'''
def avg_profile_model(x, weak_amp, weak_phase, strong_amp, strong_phase):
    """
    Model for the average profile given two modes
    NOTE: The templates must already be defined in this script
    """
    weak_mode = weak_amp * fft_rotate(weak_template, weak_phase)
    strong_mode = strong_amp * fft_rotate(strong_template, strong_phase)
    return strong_mode + weak_mode
'''
def avg_profile_model(x, avg_amp, avg_phase):
 
    #weak_mode = fft_rotate(weak_amp * weak_template / np.max(weak_template), weak_phase)
    #strong_mode = fft_rotate(strong_amp * strong_template / np.max(strong_template), strong_phase)
    avg = fft_rotate(avg_amp * avg_template / np.max(avg_template), avg_phase)
    #ratio = (np.max(weak_mode)/np.max(strong_mode))*0.15
    #ratio = (np.max(weak_mode)/np.max(strong_mode))
    return avg

#data = datas[0]
#weak_phases = []
#weak_phases_upper = []
#weak_phases_lower = []
#strong_phases = []
#strong_phases_upper = []
#strong_phases_lower = []

avg_phases = []
avg_phases_upper = []
avg_phases_lower = []
avg_amp = []

for i, data in enumerate(datas):
# Bilby likelihood
    likelihood = bilby.likelihood.GaussianLikelihood(x, data,
                                                    avg_profile_model, noise)

    priors = dict()
    #
    #priors['weak_amp'] = bilby.core.prior.Gaussian(mu=0.116654865, sigma=0.031803366, name='weak_amp')
    priors['avg_amp'] = bilby.core.prior.Uniform(0, 100, 'avg_amp')
    priors['avg_phase'] = bilby.core.prior.Uniform(-nbins/2, nbins/2,
                                                    'avg_phase')
    #priors['weak_phase'] = bilby.core.prior.Gaussian(mu=6.434, sigma=2.7353889403452913, name='weak_phase')
    #priors['weak_phase'] = bilby.core.prior.Uniform(-14, 0,
    #                                                'weak_phase')                                               
    #priors['strong_amp'] = bilby.core.prior.Uniform(0, 10, 'strong_amp')
    #priors['strong_phase'] = bilby.core.prior.Uniform(-nbins/2, nbins/2,
    #                                                'strong_phase')
    #priors['ratio'] = bilby.core.prior.Gaussian(mu=0.116654865, sigma=0.031803366, name='ratio')



    results = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label='dynesty',
        nlive=50, verbose=True, resume=False,
        outdir='single-mode-timing-slurm-avg')

    #wp = results.get_one_dimensional_median_and_error_bar('weak_phase').median
    #wp_upper = results.get_one_dimensional_median_and_error_bar('weak_phase').plus
    #wp_lower = results.get_one_dimensional_median_and_error_bar('weak_phase').minus
    #sp = results.get_one_dimensional_median_and_error_bar('strong_phase').median
    #sp_upper = results.get_one_dimensional_median_and_error_bar('strong_phase').plus
    #sp_lower = results.get_one_dimensional_median_and_error_bar('strong_phase').minus

    ap = results.get_one_dimensional_median_and_error_bar('avg_phase').median
    ap_upper = results.get_one_dimensional_median_and_error_bar('avg_phase').plus
    ap_lower = results.get_one_dimensional_median_and_error_bar('avg_phase').minus
    aa = results.get_one_dimensional_median_and_error_bar('avg_amp').median

    
    #weak_phases.append(wp)
    #weak_phases_upper.append(wp_upper)
    #weak_phases_lower.append(wp_lower)
    #strong_phases.append(sp)
    #strong_phases_upper.append(sp_upper)
    #strong_phases_lower.append(sp_lower)

    avg_phases.append(ap)
    avg_phases_upper.append(ap_upper)
    avg_phases_lower.append(ap_lower)
    avg_amp.append(aa)

    os.chdir('/fred/oz002/users/mmiles/SinglePulse/single-mode-timing-slurm-avg')
    #np.save('str_phases_slurm_temp',strong_phases)
    #np.save('str_error_upper_slurm_temp',strong_phases_upper)
    #np.save('str_error_lower_slurm_temp',strong_phases_lower)

    np.save('avg_phases_slurm_temp',avg_phases)
    np.save('avg_error_upper_slurm_temp',avg_phases_upper)
    np.save('avg_error_lower_slurm_temp',avg_phases_lower)
    np.save('avg_amp_slurm_temp',avg_amp)

    results.save_posterior_samples(filename="isub_posterior_{}".format(i))

#np.save('str_phases_slurm',strong_phases)
#np.save('str_error_upper_slurm',strong_phases_upper)
#np.save('str_error_lower_slurm',strong_phases_lower)

np.save('avg_phases_slurm',avg_phases)
np.save('avg_error_upper_slurm',avg_phases_upper)
np.save('avg_error_lower_slurm',avg_phases_lower)
np.save('avg_amp_slurm',avg_amp)


