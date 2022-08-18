import numpy as np
import matplotlib.pyplot as plt
import bilby
import psrchive
import os
import sys

i = sys.argv[1]

nbins = 2048
x = np.linspace(0, 1, nbins)

noise = 0.0012


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


#Unsmoothed versions
weak_template = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/bilby_runs/weak_profile_zero.npy')
strong_template = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/bilby_runs/strong_profile.npy')

#Rotate towards strong
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



data = datas[int(i)]
rfft_data = np.fft.rfft(data)
rfft_weak_template = np.fft.rfft(weak_template)
rfft_strong_template = np.fft.rfft(strong_template)

def phasor_scale_fft(rfft_data, bins):
    """
    Add a phase gradient to the rotated FFT input
    """
    freqs = np.arange(rfft_data.size, dtype=np.float)
    phasor = np.exp(complex(0.0, 2.0*np.pi) * freqs * bins / float(2*(rfft_data.size - 1)))
    return phasor*rfft_data

def avg_profile_model_fdm_fast(x, weak_amp, weak_phase, strong_amp, strong_phase):
    """
    Model for the average profile given two modes
    NOTE: The templates must already be defined in this script
    """
    weak_mode = phasor_scale_fft(weak_amp*rfft_weak_template, weak_phase)
    strong_mode = phasor_scale_fft(strong_amp*rfft_strong_template, strong_phase)
    return np.concatenate((np.real(strong_mode + weak_mode), np.imag(strong_mode + weak_mode)))

data_fdm = np.concatenate((np.real(rfft_data), np.imag(rfft_data)))
noise_fdm = noise*np.sqrt(len(data_fdm)/2)

likelihood = bilby.likelihood.GaussianLikelihood(x, data_fdm,
                                                avg_profile_model_fdm_fast, noise_fdm)

priors = dict()
priors['weak_amp'] = bilby.core.prior.Gaussian(mu=0.116654865, sigma=0.031803366, name='weak_amp')
#priors['weak_amp'] = bilby.core.prior.Uniform(0, 1, 'weak_amp')
#priors['weak_phase'] = bilby.core.prior.Uniform(-nbins/8, nbins/8,
#                                                'weak_phase')
priors['weak_phase'] = bilby.core.prior.Gaussian(mu=6.434, sigma=2.205031088120973, name='weak_phase')
#priors['weak_phase'] = bilby.core.prior.Uniform(-14, 0,
#                                                'weak_phase')                                               
priors['strong_amp'] = bilby.core.prior.Uniform(0, 10, 'strong_amp')
priors['strong_phase'] = bilby.core.prior.Uniform(-nbins/2, nbins/2,
                                                'strong_phase')
#priors['ratio'] = bilby.core.prior.Gaussian(mu=0.116654865, sigma=0.031803366, name='ratio')



results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler='dynesty', label='dynesty',
    nlive=1000, verbose=True, resume=False, npool=16,
    outdir='two-mode-timing-unsmoothed-Gauss_{}'.format(i))

