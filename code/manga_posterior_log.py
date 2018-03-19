""" A fantastic python code to determine the quenched SFH parameters of galaxies using emcee (http://dan.iel.fm/emcee/current/). This file contains all the functions needed to determine the mean SFH parameters of a population.
    
    np.B. The data files .ised_ASCII contain the extracted bc03 models and have a 0 in the origin at [0,0]. The first row contains the model ages (from the second column) - data[0,1:]. The first column contains the model lambda values (from the second row) - data[1:,0]. The remaining data[1:,1:] are the flux values at each of the ages (columns, x) and lambda (rows, y) values 
    """

import numpy as np
import scipy as S
import pylab as P
import emcee
import time
import os
import matplotlib.image as mpimg
from astropy.cosmology import Planck15
from scipy.stats import kde
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d
from itertools import product
import sys
from emcee.autocorr import AutocorrError
from scipy.interpolate import interp2d


font = {'family':'serif', 'size':16}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='medium')

pad_zmet = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.012, 0.015, 0.019, 0.024, 0.03])
pad_zmetsol = 0.019

pad_solmet = pad_zmet/pad_zmetsol
zmets = (np.linspace(0, 10, 11)*2 + 1).astype(int)
zsolmets = pad_solmet[(zmets-1).astype(int)] 

#ages = Planck15.age(10**np.linspace(-0.824, -2.268, 25))[-7:-5].value
time_steps = Planck15.age(10**np.linspace(-0.824, -2.268, 15)).value
tqs = np.flip(13.805 - 10**(np.linspace(7, 10.14, 50))/1e9, axis=0)
taus = 10**np.linspace(6, 9.778, 50)/1e9

with np.load('../../data/iteration_number_em_indx_par_pool_mapped.npz') as idxs:
    un, unidx = np.unique(idxs['idx'], return_index=True)

with np.load('../../data/em_indx_par_pool_mapped.npz') as orig_pred:
    pred = orig_pred['lookup'][unidx, 0, :].reshape(len(tqs), len(taus), len(time_steps), len(zmets), 8)
# with np.load('em_indx_err_par_pool_mapped.npz') as orig_pred_err:
#     pred_err = orig_pred_err['lookuperr'][unidx, :].reshape(len(tqs), len(taus), len(time_steps), len(zmets), 8)


print('interpolating, maybe go grab a drink, we`ll be here a while...')

f = RegularGridInterpolator((tqs, taus, time_steps, zsolmets), pred, method='linear', bounds_error=False, fill_value=None)
#f_err = RegularGridInterpolator((tqs, taus, time_steps, pad_solmet[zmets]), pred_err, method='linear', bounds_error=False, fill_value=None)


def predict_spec_one(theta, age):

    pred = f([theta[1], theta[2], age, theta[0]])[0]
    ha_pred = pred[0]
    oii_pred = pred[1]
    d4000_pred = pred[7]
    hb_pred = pred[2]
    hdA_pred = pred[6]
    mgfe_pred = np.sqrt( pred[3] * ( 0.72*pred[4] + 0.28*pred[5] )  )  
    
    return np.array([ha_pred, oii_pred, d4000_pred, hb_pred, hdA_pred, mgfe_pred])


def lookup_col_one(theta, age):
    pred = f([theta[1], theta[2], age, theta[0]])[0]

    ha_pred = pred[0]
    oii_pred = pred[1]
    d4000_pred = pred[7]
    hb_pred = pred[2]
    hdA_pred = pred[6]
    mgfe_pred = np.sqrt( pred[3] * ( 0.72*pred[4] + 0.28*pred[5] )  )  
    
    return ha_pred, oii_pred, d4000_pred, hb_pred, hdA_pred, mgfe_pred

    
def lnlike_one(theta, ha, e_ha, oii, e_oii, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age):
    """ Function for determining the likelihood of ONE quenching model described by theta = [tq, tau] for all the galaxies in the sample. Simple chi squared likelihood between predicted and observed colours of the galaxies. 
    
    :theta:
    An array of size (1,2) containing the values [tq, tau] in Gyr.
    
    :tq:
    The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time.
    
    :tau:
    The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5.
    
    :ha, hd, oii, d4000, nad:
    spectral inputs Halpha, Hdelta, OII [3727.092], D4000, NaD

    :e_ha, e_hd, e_oii, e_d4000, e_nad:
    error on measurement of spectral inputs Halpha, Hdelta, OII [3727.092], D4000, NaD

    :age:
    Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr.
    
    RETURNS:
    Array of same shape as :age: containing the likelihood for each galaxy at the given :theta:
        """
    #Z, tq, tau = theta
    ha_pred, oii_pred, d4000_pred, hb_pred, hdA_pred, mgfe_pred = lookup_col_one(theta, age)
    return np.nansum([-0.5*np.log(2*np.pi*e_ha**2)-0.5*((ha-ha_pred)**2/e_ha**2) , -0.5*np.log(2*np.pi*e_oii**2)-0.5*((oii-oii_pred)**2/e_oii**2), -0.5*np.log(2*np.pi*e_d4000**2)-0.5*((d4000-d4000_pred)**2/e_d4000**2), -0.5*np.log(2*np.pi*e_hb**2)-0.5*((hb-hb_pred)**2/e_hb**2), -0.5*np.log(2*np.pi*e_hdA**2)-0.5*((hdA-hdA_pred)**2/e_hdA**2), -0.5*np.log(2*np.pi*e_mgfe**2)-0.5*((mgfe-mgfe_pred)**2/e_mgfe**2)])

# Overall likelihood function combining prior and model
def lnprob(theta, ha, e_ha, oii, e_oii, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age):
    """Overall posterior function combiningin the prior and calculating the likelihood. Also prints out the progress through the code with the use of n. 
        
        :theta:
        An array of size (1,4) containing the values [tq, tau] for both smooth and disc galaxies in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Can be either for smooth or disc galaxies.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Can be either for smooth or disc galaxies.
        
        :ha, hd, oii, d4000, nad:
        spectral inputs Halpha, Hdelta, OII [3727.092], D4000, NaD

        :e_ha, e_hd, e_oii, e_d4000, e_nad:
        error on measurement of spectral inputs Halpha, Hdelta, OII [3727.092], D4000, NaD
    
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (N,1) or (N,).
        
        RETURNS:
        Value of the posterior function for the given :theta: value.
        
        """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_one(theta, ha, e_ha, oii, e_oii, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age)

def lnprior(theta):
    """ Function to calcualted the prior likelihood on theta values given the inital w values assumed for the mean and standard deviation of the tq and tau parameters. Defined ranges are specified - outside these ranges the function returns -np.inf and does not calculate the posterior probability. 
        
        :theta: 
        An array of size (1,4) containing the values [tq, tau] for both smooth and disc galaxies in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Can be either for smooth or disc galaxies.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Can be either for smooth or disc galaxies.
        
        RETURNS:
        Value of the prior at the specified :theta: value.
        """
    Z, tq, tau = theta
    if 0.003 <= tq <= 13.807108309208775 and 0.001 <= tau <= 3.9 and 0.001 <= Z <= 1.6:
        return 0.0
    elif 0.003 <= tq <= 13.807108309208775 and 3.9 < tau <= 4.0 and 0.001 <= Z <= 1.6:
        return 2*(np.exp(3.9) - np.exp(tau))
    else:
        return -np.inf

def sample(ndim=3, nwalkers=100, nsteps=100, burnin=500, start=[1.0, 13.0, 1.0], ha=np.nan, e_ha=np.nan, oii=np.nan, e_oii=np.nan, d4000=np.nan, e_d4000=np.nan, hb=np.nan, e_hb=np.nan, hdA=np.nan, e_hdA=np.nan, mgfe=np.nan, e_mgfe=np.nan, age=Planck15.age(0).value, ID=0):
    """ Function to implement the emcee EnsembleSampler function for the sample of galaxies input. Burn in is run and calcualted fir the length specified before the sampler is reset and then run for the length of steps specified. 
        
        :ndim:
        The number of parameters in the model that emcee must find. In this case it always 2 with tq, tau.
        
        :nwalkers:
        The number of walkers that step around the parameter space. Must be an even integer number larger than ndim. 
        
        :nsteps:
        The number of steps to take in the final run of the MCMC sampler. Integer.
        
        :burnin:
        The number of steps to take in the inital burn-in run of the MCMC sampler. Integer. 
        
        :start:
        The positions in the tq and tau parameter space to start for both disc and smooth parameters. An array of shape (1,4).
        
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (N,1) or (N,).
        
        :id:
        ID number to specify which galaxy this run is for.
        
        :ra:
        right ascension of source, used for identification purposes
        
        :dec:
        declination of source, used for identification purposes
        
        RETURNS:
        :samples:
        Array of shape (nsteps*nwalkers, 4) containing the positions of the walkers at all steps for all 4 parameters.
        :samples_save:
        Location at which the :samples: array was saved to. 
        
        """

    print('emcee running...')
    p0 = [start + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4, args=(ha, e_ha, oii, e_oii, d4000, e_d4000, hb, e_hb, hdA, e_hdA, mgfe, e_mgfe, age))
    """ Burn in run here..."""
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    lnp = sampler.flatlnprobability
    np.savez('lnprob_burnin_'+str(ID)+'.npz', lnp=lnp)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = 'samples_burn_in_'+str(ID)+'.npz'
    np.savez(samples_save, samples=samples)
    sampler.reset()
    print('Burn in complete...')
    """ Main sampler run here..."""
    sampler.run_mcmc(pos, nsteps)
    lnpr = sampler.flatlnprobability
    np.savez('lnprob_run_'+str(ID)+'.npz', lnp=lnpr)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = 'samples_'+str(ID)+'.npz'
    np.savez(samples_save, samples=samples)
    print('Main emcee run completed.')
    sampler.reset()
    try:
        acor = sampler.get_autocorr_time(c=1)
        return samples, samples_save, sampler.acceptance_fraction, acor
    except AutocorrError:
        return samples, samples_save, sampler.acceptance_fraction, [np.nan, np.nan, np.nan]


#Define function to plot the walker positions as a function of the step
def walker_plot(samples, nwalkers, limit, id):
    """ Plotting function to visualise the steps of the walkers in each parameter dimension for smooth and disc theta values. 
        
        :samples:
        Array of shape (nsteps*nwalkers, 4) produced by the emcee EnsembleSampler in the sample function.
        
        :nwalkers:
        The number of walkers that step around the parameter space used to produce the samples by the sample function. Must be an even integer number larger than ndim.
        
        :limit:
        Integer value less than nsteps to plot the walker steps to. 
        
        :id:
        ID number to specify which galaxy this plot is for.
        
        RETURNS:
        :fig:
        The figure object
        """
    s = samples.reshape(nwalkers, -1, 2)
    s = s[:,:limit, :]
    fig = P.figure(figsize=(8,5))
    ax1 = P.subplot(2,1,1)
    ax2 = P.subplot(2,1,2)
    for n in range(len(s)):
        ax1.plot(s[n,:,0], 'k')
        ax2.plot(s[n,:,1], 'k')
    ax1.tick_params(axis='x', labelbottom='off')
    ax2.set_xlabel(r'step number')
    ax1.set_ylabel(r'$t_{quench}$')
    ax2.set_ylabel(r'$\tau$')
    P.subplots_adjust(hspace=0.1)
    save_fig = 'walkers_steps_'+str(int(id))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf'
    fig.savefig(save_fig)
    return fig


