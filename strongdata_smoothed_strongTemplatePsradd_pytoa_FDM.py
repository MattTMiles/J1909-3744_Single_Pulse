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
strong_template = np.load('/fred/oz002/users/mmiles/SinglePulse/Jitter_check/bilby_runs/J1909-3744_psradd_strong.npy')

#Alldata
#datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/tempo2_all_new/all.Ft128')
#Strongdata
datas = psrchive.Archive_load('/fred/oz002/users/mmiles/SinglePulse/strong_Ft128/all.Ft128_strong')


datas.remove_baseline()
datas.dedisperse()
datas = datas.get_data()
datas = datas[:,0,0,:]


def strong_profile_model(x, strong_amp, strong_phase):
 
    strong_mode = fft_rotate(strong_amp * strong_template / np.max(strong_template), strong_phase)

    return strong_mode


data = datas[int(i)]
rfft_data = np.fft.rfft(data)
#rfft_weak_template = np.fft.rfft(weak_template)
rfft_strong_template = np.fft.rfft(strong_template)

def phasor_scale_fft(rfft_data, bins):
    """
    Add a phase gradient to the rotated FFT input
    """
    freqs = np.arange(rfft_data.size, dtype=np.float)
    phasor = np.exp(complex(0.0, 2.0*np.pi) * freqs * bins / float(2*(rfft_data.size - 1)))
    return phasor*rfft_data

def avg_profile_model_fdm_fast(x, strong_amp, strong_phase):
    """
    Model for the average profile given two modes
    NOTE: The templates must already be defined in this script
    """
    #weak_mode = phasor_scale_fft(weak_amp*rfft_weak_template, weak_phase)
    strong_mode = phasor_scale_fft(strong_amp*rfft_strong_template, strong_phase)
    return np.concatenate((np.real(strong_mode), np.imag(strong_mode)))

data_fdm = np.concatenate((np.real(rfft_data), np.imag(rfft_data)))
noise_fdm = noise*np.sqrt(len(data_fdm)/2)

likelihood = bilby.likelihood.GaussianLikelihood(x, data_fdm,
                                                avg_profile_model_fdm_fast, noise_fdm)

priors = dict()
                                         
priors['strong_amp'] = bilby.core.prior.Uniform(0, 10, 'strong_amp')
priors['strong_phase'] = bilby.core.prior.Uniform(-nbins/2, nbins/2,
                                                'strong_phase')



results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler='dynesty', label='dynesty',
    nlive=1000, verbose=True, resume=False, npool=16,
    outdir='smoothed-strong-mode-timing_{}'.format(i))

wp = results.get_one_dimensional_median_and_error_bar('weak_phase').median
wa = results.get_one_dimensional_median_and_error_bar('weak_amp').median
wp_upper = results.get_one_dimensional_median_and_error_bar('weak_phase').plus
wp_lower = results.get_one_dimensional_median_and_error_bar('weak_phase').minus
sp = results.get_one_dimensional_median_and_error_bar('strong_phase').median
sa = results.get_one_dimensional_median_and_error_bar('strong_amp').median
sp_upper = results.get_one_dimensional_median_and_error_bar('strong_phase').plus
sp_lower = results.get_one_dimensional_median_and_error_bar('strong_phase').minus

weak_phases.append(wp)
weak_amps.append(wa)
weak_phases_upper.append(wp_upper)
weak_phases_lower.append(wp_lower)
strong_phases.append(sp)
strong_amps.append(sa)
strong_phases_upper.append(sp_upper)
strong_phases_lower.append(sp_lower)

#os.chdir('/fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-uniform-pool_{}'.format(i))
np.save('str_phases_slurm_temp',strong_phases)
np.save('str_amps_temp',strong_amps)
np.save('str_error_upper_slurm_temp',strong_phases_upper)
np.save('str_error_lower_slurm_temp',strong_phases_lower)

np.save('weak_phases_slurm_temp',weak_phases)
np.save('weak_amps_temp',weak_amps)
np.save('weak_error_upper_slurm_temp',weak_phases_upper)
np.save('weak_error_lower_slurm_temp',weak_phases_lower)
results.save_posterior_samples(filename="isub_posterior_{}".format(i))