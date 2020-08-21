from __future__ import division
import numpy as np
from scipy import signal
from scipy.signal import convolve
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import bilby


def Gaussian(x, mean, sigma):
    """ Returns a Gaussian distribution """
    norm = 1/np.sqrt(2 * np.pi * sigma**2) 
    return norm * np.exp(-(((x - mean)**2)/(2*sigma**2)))


def log_normal(x, mu, sigma):
   f_x = np.array([])
   g_x = np.array([])
   for i in range(0, len(x)):
       if x[i] <= 0.0:
           f_x = np.append(f_x, 0.0)
           g_x = np.append(g_x, 0.0)
       else:
           f_x = np.append(f_x, 1/(sigma * x[i] * np.sqrt(2*np.pi)))
           g_x = np.append(g_x, np.exp(-((np.log(x[i]) - mu)**2 / (2*sigma**2))))
   return f_x * g_x


def concolved_distribution(x, mu, sigma):
    gauss = Gaussian(x, 0, 1)
    log_n = log_normal(x, mu, sigma)

    dist = convolve(gauss, log_n, "same")

    return dist/(max(dist)+sys.float_info[3])


class FitLikelihood(bilby.likelihood.Likelihood):
    def __init__(self, x, y, func, sigma=None):
        """
        Parameters
        ----------
        x, y: array_like
            Data to be analysed.
        """

        self.x = x
        self.y = y
        self.func = func
        self.parameters = dict(Amp=None, mean=None, var=None, 
            sigma=None)

    def log_likelihood(self):

        self.sigma = self.parameters["sigma"]

        self.model = self.func(self.x, self.parameters["mean"], 
            self.parameters["var"])

        self.residual = self.y - (self.parameters["Amp"] * self.model)
  
        ln_like = np.sum(- (self.residual / self.sigma)**2 / 2 -
            np.log(2 * np.pi * self.sigma**2) / 2)
        return ln_like


if __name__ == "__main__":
    
    outpath = "../outdir/histogram_fitting/"    

    data = np.genfromtxt("../single_pulse_fluxes.txt", dtype=float,
        skip_header=8)

    flux = data[:,-2] * 1.4

    hist = plt.hist(flux, bins=100)
    
    priors = dict()
    priors["Amp"] = bilby.prior.Uniform(1, 1000, r"$A$")
    priors["mean"] = bilby.prior.Uniform(0, 8, r"$\mu$")
    priors["var"] = bilby.prior.Uniform(0, 4, r"$\sigma$")
    priors["sigma"] = bilby.prior.Uniform(1e-5, 500, r"$\varepsilon$")

    likelihood = FitLikelihood(x=hist[1][:-1], y=hist[0], 
        func=concolved_distribution)

    result = bilby.sampler.run_sampler(likelihood=likelihood, priors=priors,
        sampler="dynesty", nlive=1024, outdir=outpath, plot=False, 
        label="fit_data")

    result.plot_corner(dpi=100)
