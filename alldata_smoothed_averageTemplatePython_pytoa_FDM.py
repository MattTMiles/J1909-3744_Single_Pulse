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



#Smoothed psradded strong template
average_template = np.load('/fred/oz002/users/mmiles/SinglePulse/avg_profile.npy')

#Alldata
datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/all_Ft128/all.Ft128')
#Strongdata
#datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/strong_Ft128/all.Ft128_strong')


datas.remove_baseline()
datas.dedisperse()
datas = datas.get_data()
datas = datas[:,0,0,:]


def average_profile_model(x, ave_amp, ave_phase):
 
    average_mode = fft_rotate(ave_amp * average_template / np.max(average_template), ave_phase)

    return average_mode


data = datas[int(i)]
rfft_data = np.fft.rfft(data)
rfft_ave_template = np.fft.rfft(average_template)
#rfft_strong_template = np.fft.rfft(strong_template)

def phasor_scale_fft(rfft_data, bins):
    """
    Add a phase gradient to the rotated FFT input
    """
    freqs = np.arange(rfft_data.size, dtype=np.float)
    phasor = np.exp(complex(0.0, 2.0*np.pi) * freqs * bins / float(2*(rfft_data.size - 1)))
    return phasor*rfft_data

def avg_profile_model_fdm_fast(x, ave_amp, ave_phase, strong_amp, strong_phase):
    """
    Model for the average profile given two modes
    NOTE: The templates must already be defined in this script
    """
    ave_mode = phasor_scale_fft(ave_amp*rfft_ave_template, ave_phase)
    return np.concatenate((np.real(ave_mode), np.imag(ave_mode)))

data_fdm = np.concatenate((np.real(rfft_data), np.imag(rfft_data)))
noise_fdm = noise*np.sqrt(len(data_fdm)/2)


likelihood = bilby.likelihood.GaussianLikelihood(x, data_fdm,
                                                avg_profile_model_fdm_fast, noise_fdm)

priors = dict()
#priors['weak_amp'] = bilby.core.prior.Gaussian(mu=0.116654865, sigma=0.031803366, name='weak_amp')
#priors['weak_amp'] = bilby.core.prior.Uniform(0, 1, 'weak_amp')
#priors['weak_phase'] = bilby.core.prior.Uniform(-nbins/8, nbins/8,
#                                                'weak_phase')
#priors['weak_phase'] = bilby.core.prior.Gaussian(mu=6.434, sigma=2.205031088120973, name='weak_phase')
#priors['weak_phase'] = bilby.core.prior.Uniform(-14, 0,
#                                                'weak_phase')                                               
priors['ave_amp'] = bilby.core.prior.Uniform(0, 10, 'ave_amp')
priors['ave_phase'] = bilby.core.prior.Uniform(-nbins/2, nbins/2,
                                                'ave_phase')
#priors['ratio'] = bilby.core.prior.Gaussian(mu=0.116654865, sigma=0.031803366, name='ratio')



results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler='dynesty', label='dynesty',
    nlive=1000, verbose=True, resume=False, npool=16,
    outdir='alldata-ave-template-timing_FDM_{}'.format(i))
