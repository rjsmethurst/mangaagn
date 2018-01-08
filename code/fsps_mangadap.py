import os
import time
import numpy as np
from scipy import signal

from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import fsps

from mangadap.util.instrument import spectrum_velocity_scale

from mangadap.drpfits import DRPFits
from mangadap.par.obsinput import ObsInputPar
from mangadap.proc.spatialbinning import RadialBinningPar, RadialBinning
from mangadap.proc.spectralstack import SpectralStackPar, SpectralStack

from mangadap.proc.templatelibrary import TemplateLibrary

from mangadap.proc.ppxffit import PPXFFit
from mangadap.proc.stellarcontinuummodel import StellarContinuumModelBitMask

from mangadap.proc.emissionlinemodel import EmissionLineModelBitMask
from mangadap.proc.elric import Elric
from mangadap.par.emissionlinedb import EmissionLineDB
from astropy import constants as con
from astropy import units as un

from mangadap.proc.spectralindices import SpectralIndices
from mangadap.par.absorptionindexdb import AbsorptionIndexDB
from mangadap.par.bandheadindexdb import BandheadIndexDB

from scipy import interpolate
from astropy.cosmology import Planck15 

from itertools import product
from tqdm import tqdm

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

pad_zmet = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.012, 0.015, 0.019, 0.024, 0.03])
pad_zmetsol = 0.019

pad_solmet = pad_zmet/pad_zmetsol

def expsfh(tq, tau, time):
    """ This function when given an array of [tq, tau] values will calcualte the SFR at all times. First calculate the sSFR at all times as defined by Peng et al. (2010) - then the SFR at the specified time of quenching, tq and set the SFR at this value  at all times before tq. Beyond this time the SFR is an exponentially declining function with timescale tau. 
        
        INPUT:
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Shape (1, M, N). 
        
        :tq: 
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Shape (1, M, N). 
        
        e.g:
        
        tqs = np.linspace(0, 14, 10)
        taus = np.linspace(0, 4, 5)
        
        tq = tqs.reshape(1,-1,1).repeat(len(taus), axis=2)
        tau = taus.reshape(1,1,-1).repeat(len(tqs), axis=1)
        
        :time:
        An array of time values at which the SFR is calcualted at each step. Shape (T, 1, 1)
        
        RETURNS:
        :sfr:
        Array of the same dimensions of time containing the sfr at each timestep. Shape (T, M, N). 
        """
    ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
    c = np.apply_along_axis(lambda a: a.searchsorted(3.0), axis = 0, arr = time)    
    ssfr[:c.flatten()[0]] = np.interp(3.0, time.flatten(), ssfr.flatten())
    c_sfr = np.interp(tq, time.flatten(), ssfr.flatten())*(1E10)/(1E9)
    ### definition is for 10^10 M_solar galaxies and per gyr - convert to M_solar/year ###
    sfr = np.ones_like(time)*c_sfr
    mask = time <= tq
    sfrs = np.ma.masked_array(sfr, mask=mask)
    times = np.ma.masked_array(time-tq, mask=mask)
    sfh = sfrs*np.exp(-times/tau)
    return sfh.data

def gauss(a, u, s, x):
    return a * np.exp(- (((x - u)**2) /(s**2)))    

def burstsfh(tb, Hb, time):
    return gauss(Hb, tb, 0.1, time)

def constsfh(csfr, time):
    return np.ones_like(time)*csfr

def normsfh(tq, sigma, time):
    ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
    c = np.apply_along_axis(lambda a: a.searchsorted(3.0), axis = 0, arr = time)
    ssfr[:c.flatten()[0]] = np.interp(3.0, time.flatten(), ssfr.flatten())
    c_sfr = np.interp(tq, time.flatten(), ssfr.flatten())*(1E10)/(1E9)
    return gauss(c_sfr, tq, sigma, time)

def dubnormsfh(tq, sigmasf, sigmaq, time):
    ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
    c = np.apply_along_axis(lambda a: a.searchsorted(3.0), axis = 0, arr = time)
    ssfr[:c.flatten()[0]] = np.interp(3.0, time.flatten(), ssfr.flatten())
    c_sfr = np.interp(tq, time.flatten(), ssfr.flatten())*(1E10)/(1E9)
    maskq = time <= tq
    masksf = time >= tq
    #sfrs = np.ma.masked_array(sfr, mask=mask)
    sfhsf = np.ma.masked_array(gauss(c_sfr, tq, sigmasf, time), mask=masksf)
    sfhq = np.ma.masked_array(gauss(c_sfr, tq, sigmaq, time), mask=maskq)
    return sfhsf.filled(0.0)+sfhq.filled(0.0)

if __name__ == '__main__':
    t = time.clock()

    # age = np.append(np.linspace(0.01, 13, 60), np.linspace(13.025, 14.0, 40)).reshape(-1,1,1)
    # tqs = np.append(np.linspace(0.01, 12.0, 30), np.linspace(12.07, 14.0, 20))
    # taus = np.linspace(0.001, 6, 50)

    age = np.flip(13.805 - 10**(np.linspace(7, 10.14, 100))/1e9, axis=0).reshape(-1,1,1)
    tqs = np.flip(13.805 - 10**(np.linspace(7, 10.14, 50))/1e9, axis=0)
    taus = 10**np.linspace(6, 9.778, 50)/1e9

    tq = tqs.reshape(1,-1,1).repeat(len(taus), axis=2)
    tau = taus.reshape(1,1,-1).repeat(len(tqs), axis=1)

    sfr = expsfh(tq, tau, age).reshape(age.shape[0], -1) 
    #x = np.linspace(0, 14, 200).reshape(-1,1,1)
    #sfr = burstsfh(4.0, 10.0, x) + burstsfh(7.5, 3.0, x) + burstsfh(11.3, 11.0, x) + burstsfh(13.5, 2.0, x)


    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0, zmet=18, sfh=3, dust_type=2, dust2=0.2, 
        add_neb_emission=True, add_neb_continuum=False, imf_type=1, add_dust_emission=True, min_wave_smooth=3600, max_wave_smooth=7500, sigma_smooth=True)

    time_steps = Planck15.age(10**np.linspace(-0.824, -2.268, 30)).reshape(-1,1,1).value
    #time_steps = np.linspace(0.2, 14, 7)
    zmets = np.arange(22)+1.
    #zmets = [4,7,10,13,16,19,22]
    #zmets= [19]
    #zmets = np.append(np.array([1, 7, 10]), np.linspace(13, 22, 10))

    #bins = numpy.linspace(0, 1, 11)
    #mid_bins = bins[:-1]+(numpy.diff(bins)/2.)

    #You need the template spectra.  For now just use the MILESHC library:
    tpl = TemplateLibrary('MILESHC',
                            match_to_drp_resolution=False,
                            velscale_ratio=1,    
                            spectral_step=1e-4,
                            log=True,
                            directory_path='.',
                            processed_file='mileshc.fits',
                            clobber=True)

    # Instantiate the object that does the fitting
    contbm = StellarContinuumModelBitMask()
    ppxf = PPXFFit(contbm)

    abs_db = AbsorptionIndexDB('EXTINDX')
    band_db = BandheadIndexDB('BHBASIC')
    indx_names = np.hstack([abs_db.data['name'], band_db.data['name']])

    specm = EmissionLineModelBitMask()
    elric = Elric(specm)
    emlines  = EmissionLineDB("ELPFULL")

    c = con.c.to(un.km/un.s).value

    # if os.path.isfile('emission_line_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy'):
    #     eml = list(np.load('emission_line_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy'))
    # else:
    #     eml = []

    # if os.path.isfile('abs_index_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy'):
    #     idm = list(np.load('abs_index_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy'))
    # else:
    #     idm = []

    manga_wave = np.load('manga_wavelengths_AA.npy')
    

    #for n in range(0, sfr.shape[1]):
    #for n in range(485,486):
    def get_fluxes(sfr):
        #print(str(100*(n/sfr.shape[1]))+'% of the way through')

        sp.set_tabular_sfh(age = age, sfr=sfr)

        def sm_spec(spec):
            return sp.smoothspec(fsps_wave, spec, 77.)

        def time_spec(params):
            #print(age)
            sps = sp.get_spectrum(tage=params[0], zmet=params[1])[1]
            #print(sps.shape)
            return sps

        fsps_wave = sp.get_spectrum()[0]
        fsps_spect = np.array(list(map(time_spec, list(product(time_steps, zmets)))))

        fsps_specs = np.array(list(map(sm_spec, fsps_spect)))

        #fsps_spec = fsps_specs[-24:]
        fsps_spec = fsps_specs
        #spec_ages = (10**(sp.log_age)/1E9)[-24:]
        #spec_ages = age[1:]
        #num_wave = len(fsps_wave)

        f = interpolate.interp1d(fsps_wave, fsps_spec)
        fsps_flux = f(manga_wave)

        return fsps_flux 

    def measure_spec(fsps_flux):
        if fsps_flux.shape[0] == len(fsps_wave):
            fsps_flux = fsps_flux.reshape(1,-1)
        else:
            pass
        nspec = fsps_flux.shape[0]

            # Provide the guess redshift and guess velocity dispersion
        guess_redshift = numpy.full(nspec, 0.0003, dtype=float)
        guess_dispersion = numpy.full(nspec, 77.0, dtype=float)

        # Perform the fits and construct the models
        model_wave, model_flux, model_mask, model_par  = ppxf.fit(tpl['WAVE'].data.copy(), tpl['FLUX'].data.copy(), fsps_wave, fsps_flux, np.ones_like(fsps_flux), guess_redshift, guess_dispersion, iteration_mode='none', velscale_ratio=1, degree=8, mdegree=-1, moments=2, quiet=True)
        em_model_wave, em_model_flux, em_model_base, em_model_mask, em_model_fit_par, em_model_eml_par = elric.fit(fsps_wave, fsps_flux, emission_lines=emlines, ivar=np.ones_like(fsps_flux), sres=np.ones_like(fsps_flux), continuum=model_flux, guess_redshift = model_par['KIN'][:,0]/c, guess_dispersion=model_par['KIN'][:,1], base_order=1)
        indx_measurements = SpectralIndices.measure_indices(absdb=abs_db, bhddb=band_db, wave=fsps_wave, flux=fsps_flux, ivar=np.ones_like(fsps_flux), mask=None, redshift=model_par['KIN'][:,0]/c, bitmask=None)

        plt.close('all')
        plt.cla()
        plt.clf()

        return em_model_eml_par, indx_measurements

    fsps_wave_z0 = manga_wave
    fsps_wave = fsps_wave_z0*(1+ (100.0*(un.km/un.s)/con.c.to(un.km/un.s)) ).value

    fluxes = np.array(list(map(get_fluxes, tqdm(list(sfr.T)[0])))).reshape(-1, len(fsps_wave))
    #np.save('spectrum_all_star_formation_rates_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy', fluxes)
    #fluxes = np.load('spectrum_all_star_formation_rates_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy')

    f = fluxes.reshape(len(time_steps), len(zmets), len(tqs), len(taus),-1)

    if os.path.isfile('../data/emission_line_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'.npy'):
        em_par = np.load('../data/emission_line_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'.npy')
    else:
        em_par = []

    if os.path.isfile('../data/abs_index_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'.npy'):
        indx_par = np.load('../data/abs_index_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'.npy')
    else:
        indx_par = []


    istart = len(em_par)/22
    jstart = len(em_par) % 22

    for i in range(0, istart):#f.shape[0]):
        for j in range(0, jstart):#f.shape[1]):
            meas = measure_spec(f[i,j,:,:,:].reshape(-1, len(fsps_wave)))
            em_par.append(meas[0].reshape(len(tqs), len(taus)))
            indx_par.append(meas[1].reshape(len(tqs), len(taus)))
            np.save('../data/emission_line_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'.npy', em_par)
            np.save('../data/abs_index_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'.npy', indx_par)
    #np.save('many_burst_sfh.npy', sfr[:,0,0])

    #meas = measure_spec(fluxes)

    #np.save('../data/emission_line_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy', meas[0])
    #np.save('../data/abs_index_params_log_scale_tq_'+str(len(tqs))+'_tau_'+str(len(taus))+'_to_'+str(len(time_steps))+'_Z_'+str(len(zmets))+'_'+'.npy', meas[1])
    #np.save('many_burst_sfh.npy', sfr[:,0,0])
    #np.save('emission_line_params_many_burst_mu_12.6_s_1.5.npy', meas[0])
    #np.save('abs_index_params_manyburst_mu_12.6_s_1.5.npy', meas[1])


# ews =  np.zeros(len(eml)*len(zmets)*len(time_steps)*len(emlines.data['name'])).reshape(len(time_steps)*len(zmets), len(eml), len(emlines.data['name']))
# idms =  np.zeros(len(idm)*len(zmets)*len(time_steps)*len(indx_names)).reshape(len(time_steps)*len(zmets),len(idm),len(indx_names))

# for n in range(len(eml)):
#     ews[:,n,:] = eml[n]['EW']
#     idms[:,n,:] =idm[n]['INDX']

# ewss = ews.reshape(len(time_steps), len(zmets), len(tqs), len(taus), len(emlines.data['name']))
# idmss = idms.reshape(len(time_steps), len(zmets), len(tqs), len(taus), len(indx_names))

# Mgb = idmss[:,:,:, np.where(indx_names=='Mgb')[0][0]]
# Fe5270 = idmss[:, :,:, np.where(indx_names=='Fe5270')[0][0]]
# Fe5335 = idmss[:, :,:, np.where(indx_names=='Fe5335')[0][0]]
# MgFe = np.sqrt(Mgb * ( 0.72*Fe5270 + 0.28*Fe5335)  )

# hahdoii = ewss[:,:,:,:,[7,11,14]]
# d4nadmgfe = np.append(idmss[:,:,:,:,[39, 40, 18]], MgFe.reshape(MgFe.shape[0], MgFe.shape[1], MgFe.shape[2], 1), axis=-1)

# tz = np.array(list(product(time_steps, zmets)))
# zmetss = tz[:,1].reshape(len(time_steps),len(zmets))
# timess = tz[:,0].reshape(len(time_steps),len(zmets))

# # has = hahdoii[:,:,0,0,0].reshape(len(time_steps),len(zmets))
# # hds = hahdoii[:,:,0,0,1].reshape(len(time_steps),len(zmets))
# # oiis = hahdoii[:,:,0,0,2].reshape(len(time_steps),len(zmets))

# # d4s = d4nadmgfe[:,:,0,0,0].reshape(len(time_steps),len(zmets))
# # dn4s = d4nadmgfe[:,:,0,0,1].reshape(len(time_steps),len(zmets))
# # nads = d4nadmgfe[:,:,0,0,2].reshape(len(time_steps),len(zmets))
# # mgfes = d4nadmgfe[:,:,0,0,3].reshape(len(time_steps),len(zmets))


# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, has[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel(r'EW[H$\alpha$]')
# # plt.savefig('ha_change_metallicity.pdf')

# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, hds[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel(r'EW[H$\delta$]')
# # plt.savefig('hd_change_metallicity.pdf')

# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, oiis[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel(r'EW[OII]')
# # plt.savefig('oii_change_metallicity.pdf')

# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, d4s[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel('D4000')
# # plt.savefig('d4000_change_metallicity.pdf')

# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, dn4s[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel('Dn4000')
# # plt.savefig('dn4000_change_metallicity.pdf')

# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, nads[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel('NaD')
# # plt.savefig('nad_change_metallicity.pdf')

# # plt.figure()
# # for n in range(len(time_steps)):
# #     plt.plot(pad_solmet, mgfes[n,:], label='%.1f' % timess[n,0]+' Gyr')
# # plt.legend(frameon=False)
# # plt.xlabel(r'$Z/Z_{\odot}$')
# # plt.ylabel('MgFe')
# # plt.savefig('mgfe_change_metallicity.pdf')

# # Now plot with time on x axis and Z different coloured lines

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], has[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel(r'EW[H$\alpha$]')
# # plt.savefig('ha_change_time_Z.pdf')

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], hds[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel(r'EW[H$\delta$]')
# # plt.savefig('hd_change_time_Z.pdf')

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], oiis[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel(r'EW[OII]')
# # plt.savefig('oii_change_time_Z.pdf')

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], d4s[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel('D4000')
# # plt.savefig('d4000_change_time_Z.pdf')

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], dn4s[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel('Dn4000')
# # plt.savefig('dn4000_change_time_Z.pdf')

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], nads[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel('NaD')
# # plt.savefig('nad_change_time_Z.pdf')

# # plt.figure()
# # for n in [1,3,5,7,9,11,13,15,17,19,21]:
# #     plt.plot(timess[:,0], mgfes[:,n], label='%.2f' % pad_solmet[n]+r' $Z/Z_{\odot}$')
# # plt.legend(frameon=False, fontsize=10)
# # plt.xlabel(r'$t$ $\rm{[Gyr]}$')
# # plt.ylabel('MgFe')
# # plt.savefig('mgfe_change_time_Z.pdf')

# # np.save('fsps_look_up_ha_hd_oii_50_time_50_tq_50_tau.npy', ewss)
# # np.save('fsps_look_up_d4000_nad_mgfe_50_time_50_tq_50_tau.npy', idmss)

# # from mpl_toolkits.axes_grid1 import make_axes_locatable


# # for k in range(len(time_steps)):
# #     plt.figure(figsize=(40,6))
# #     ax1 = plt.subplot(161)
# #     cbar = ax1.imshow(D4000[k,:,:].T, origin='lower', vmax=2.8, vmin=1.5, aspect='auto', extent=(3,14,0,4), interpolation='nearest', cmap=plt.cm.viridis_r)
# #     cb = plt.colorbar(cbar, ax=ax1)
# #     cb.set_label(r'$\rm{D4000}$')
# #     ax1.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #     ax1.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #     ax1 = plt.subplot(164)
# #     cbar = ax1.imshow(Halpha[k,:,:].T, vmax=70, vmin=0, origin='lower', aspect='auto', extent=(3,14,0,4), interpolation='nearest')
# #     cb = plt.colorbar(cbar, ax=ax1)
# #     cb.set_label(r'$\rm{EW}[H\alpha]$')
# #     ax1.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #     ax1.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #     ax1 = plt.subplot(165)
# #     cbar = ax1.imshow(Hdelta[k,:,:].T, vmax=2.5, vmin=0.5, origin='lower', aspect='auto', extent=(3,14,0,4), interpolation='nearest')
# #     cb = plt.colorbar(cbar, ax=ax1)
# #     cb.set_label(r'$\rm{EW}[H\delta]$')
# #     ax1.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #     ax1.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #     ax1 = plt.subplot(166)
# #     cbar = ax1.imshow(ewss[k,:,:,np.where(emlines.data['restwave']==3727.092)[0][0]].T, vmin=13, vmax=0, origin='lower', aspect='auto', extent=(3,14,0,4), interpolation='nearest')
# #     cb = plt.colorbar(cbar, ax=ax1)
# #     cb.set_label(r'$\rm{EW}[OII]$')
# #     ax1.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #     ax1.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #     ax1 = plt.subplot(162)
# #     cbar = ax1.imshow(idmss[k,:,:,np.where(indx_names=='NaD')[0][0]].T, vmax=3.4, vmin=2.3, origin='lower', aspect='auto', extent=(3,14,0,4), interpolation='nearest')
# #     cb = plt.colorbar(cbar, ax=ax1)
# #     cb.set_label(r'$\rm{NaD}$')
# #     ax1.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #     ax1.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #     ax1 = plt.subplot(163)
# #     Mgb = idmss[k,:,:, np.where(indx_names=='Mgb')[0][0]]
# #     Fe5270 = idmss[k, :,:, np.where(indx_names=='Fe5270')[0][0]]
# #     Fe5335 = idmss[k, :,:, np.where(indx_names=='Fe5335')[0][0]]
# #     MgFe = np.sqrt(Mgb * ( 0.72*Fe5270 + 0.28*Fe5335)  )
# #     cbar = ax1.imshow(MgFe.T, origin='lower', aspect='auto', extent=(3,14,0,4), interpolation='nearest')
# #     cb = plt.colorbar(cbar, ax=ax1)
# #     cb.set_label(r'$\rm{MgFe}$')
# #     ax1.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #     ax1.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #     plt.tight_layout()
# #     plt.savefig('../figures/fsps_mangadap_D4000_NaD_MgFe_halpha_hdelta_OII_t_'+str(time_steps[k])+'.pdf')
# #     plt.clf()
# #     plt.close('all')

# # # plt.figure(figsize=(6,6))
# # # ax1 = plt.subplot(111)
# # # ax1.plot(spec_ages, em_model_eml_par['EW'][:,np.where(emlines.data['name']=='Ha')].flatten(), color='k', label=r'$H\alpha$', marker='x')
# # # ax1.plot(spec_ages, em_model_eml_par['EW'][:,np.where(emlines.data['name']=='Hdel')].flatten(), color='r', label=r'$H\delta$', marker='x')
# # # ax1.plot(spec_ages, em_model_eml_par['EW'][:,np.where(emlines.data['name']=='Hb')].flatten(), color='g', label=r'$H\beta$', marker='x')
# # # ax1.plot(spec_ages, em_model_eml_par['EW'][:,np.where(emlines.data['name']=='Hgam')].flatten(), color='m', label=r'$H\gamma$', marker='x')
# # # ax1.plot(spec_ages, em_model_eml_par['EW'][:,np.where(emlines.data['restwave']==3727.092)].flatten(), color='y', label=r'$[OII]$', marker='x')
# # # ax1.set_xlabel(r'$R/R_e$')
# # # ax1.set_ylabel(r'$\rm{EW}$')
# # # ax2 = ax1.twinx()
# # # ax2.plot(spec_ages, indx_measurements['INDX'][:,np.where(indx_names=='D4000')].flatten(), color='b', label=r'$\rm{D4000}$', marker='x')
# # # #ax2.plot(mid_bins, MgFe, color='b')
# # # #ax2.plot(mid_bins, FeH, color='purple')

# # # ax2.set_ylabel(r'$\rm{D4000}$', color='b')
# # # #ax2.set_ylabel(r'$[MgFe]$', color='b')
# # # ax1.legend(frameon=False)
# # # plt.show()

# from mpl_toolkits.axes_grid1 import make_axes_locatable

# zsolmets = pad_solmet[(zmets-1).astype(int)] 
# for i in range(len(zmets)):
#     for k in range(len(time_steps)):
#         m = 0
#         n = 0
#         plt.figure(figsize=(32,24))
#         for j in range(len(indx_names)):
#             ax = plt.subplot2grid((6,7), (m, n))
#             cb = ax.pcolor(tq[0], tau[0], idmss[k,i,:,:,j], vmin=np.min(idmss[:,:,:,:,j]), vmax=np.max(idmss[:,:,:,:,j]))
#             ax.set_xlim(np.min(tq), np.max(tq))
#             ax.set_ylim(np.min(tau), np.max(tau))

#             the_divider = make_axes_locatable(ax)
#             cax = the_divider.append_axes("right", size="5%", pad=0.2)

#             cbar = plt.colorbar(cb, cax=cax)
#             cbar.set_label(indx_names[j], labelpad=-2)
#             if m == 7:
#               ax.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
#             else:
#               pass
#             if n == 0:
#               ax.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
#             else:
#               pass
#             ax.minorticks_on()
#             if n%7 == 6:
#               n = 0
#               m+=1
#             else:
#               n+=1
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.1)
#         plt.savefig('../figures/fsps_mangadap_meas_indexes_t_'+str(time_steps[k])+'_Z_'+str(zsolmets[i])+'.pdf')
#         plt.clf()
#         plt.close('all')



# zsolmets = pad_solmet[(zmets-1).astype(int)]
# for i in range(len(tqs)):
#     for k in range(len(taus)):
#         m = 0
#         n = 0
#         plt.figure(figsize=(32,24))
#         for j in range(len(indx_names)):
#             ax = plt.subplot2grid((6,7), (m, n))
#             cb = ax.pcolor(zsolmets, time_steps, idmss[:,:,i,k,j], vmin=np.min(idmss[:,:,:,:,j]), vmax=np.max(idmss[:,:,:,:,j]))
#             ax.set_ylim(np.min(time_steps), np.max(time_steps))
#             ax.set_xlim(np.min(zsolmets), np.max(zsolmets))

#             the_divider = make_axes_locatable(ax)
#             cax = the_divider.append_axes("right", size="5%", pad=0.2)

#             cbar = plt.colorbar(cb, cax=cax)
#             cbar.set_label(indx_names[j], labelpad=-2)
#             if m == 7:
#               ax.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
#             else:
#               pass
#             if n == 0:
#               ax.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
#             else:
#               pass
#             ax.minorticks_on()
#             if n%7 == 6:
#               n = 0
#               m+=1
#             else:
#               n+=1
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.1)
#         plt.savefig('../figures/fsps_mangadap_meas_indexes_tq_'+str(tqs[i])+'_tau_'+str(taus[k])+'.pdf')
#         plt.clf()
#         plt.close('all')

# zsolmets = pad_solmet[(zmets-1).astype(int)]
# for i in range(len(zmets)):
#     for k in range(len(time_steps)):
#         m = 0
#         n = 0
#         plt.figure(figsize=(27,16))
#         for j in range(len(emlines.data['name'])):
#             print(j, m,n)
#             ax = plt.subplot2grid((4,6), (m, n))
#             cb = ax.pcolor(tq[0], tau[0], ewss[k,i,:,:,j], vmin=np.max(np.append(0, np.median(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)])-3*np.std(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)]))), vmax=np.min(np.append(np.max(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)]), np.median(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)])+3*np.std(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)]))))
#             ax.set_xlim(np.min(tq), np.max(tq))
#             ax.set_ylim(np.min(tau), np.max(tau))

#             the_divider = make_axes_locatable(ax)
#             cax = the_divider.append_axes("right", size="5%", pad=0.2)

#             cbar = plt.colorbar(cb, cax=cax)
#             cbar.set_label(r'$\rm{EW}$ $[$'+emlines.data['name'][j]+r'$]$', labelpad=-2)
#             if m == 6:
#               ax.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
#             else:
#               pass
#             if n == 0:
#               ax.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
#             else:
#               pass
#             ax.minorticks_on()
#             if n%6 == 5:
#               n = 0
#               m+=1
#             else:
#               n+=1
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.1)
#         plt.savefig('../figures/fsps_mangadap_EWs_t_'+str(time_steps[k])+'_Z_'+str(zsolmets[i])+'.pdf')
#         plt.clf()
#         plt.close('all')



# zsolmets = pad_solmet[(zmets-1).astype(int)]
# for i in range(len(tqs)):
#     for k in range(len(taus)):
#         m = 0
#         n = 0
#         plt.figure(figsize=(32,24))
#         for j in range(len(emlines.data['name'])):
#             ax = plt.subplot2grid((6,7), (m, n))
#             cb = ax.pcolor(time_steps, zsolmets, ewss[:,:,i,k,j], vmin=np.max(np.append(0, np.median(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)])-3*np.std(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)]))), vmax=np.min(np.append(np.max(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)]), np.median(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)])+3*np.std(ewss[:,:,:,:,j][np.where(ewss[:,:,:,:,j]<1E10)]))))
#             ax.set_xlim(np.min(time_steps), np.max(time_steps))
#             ax.set_ylim(np.min(zsolmets), np.max(zsolmets))

#             the_divider = make_axes_locatable(ax)
#             cax = the_divider.append_axes("right", size="5%", pad=0.2)

#             cbar = plt.colorbar(cb, cax=cax)
#             cbar.set_label(r'$\rm{EW}$ $[$'+emlines.data['name'][j]+r'$]$', labelpad=-2)
#             if m == 7:
#               ax.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
#             else:
#               pass
#             if n == 0:
#               ax.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
#             else:
#               pass
#             ax.minorticks_on()
#             if n%7 == 6:
#               n = 0
#               m+=1
#             else:
#               n+=1
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.1)
#         plt.savefig('../figures/fsps_mangadap_EWs_tq_'+str(tqs[i])+'_tau_'+str(taus[k])+'.pdf')
#         plt.clf()
#         plt.close('all')


# # for k in range(len(time_steps)):
# #     m = 0
# #     n = 0
# #     plt.figure(figsize=(32, 16))
# #     for j in range(len(emlines.data['name'])):
# #         print(m, n)
# #         ax = plt.subplot2grid((4,7), (m, n))
# #         cb = ax.pcolor(tq[0], tau[0], ewss[k,:,:,j])
# #         ax.set_xlim(np.min(tq), np.max(tq))
# #         ax.set_ylim(np.min(tau), np.max(tau))

# #         the_divider = make_axes_locatable(ax)
# #         cax = the_divider.append_axes("right", size="5%", pad=0.2)

# #         cbar = plt.colorbar(cb, cax=cax)
# #         cbar.set_label(r'$\rm{EW}$ $[$'+emlines.data['name'][j]+r'$]$', labelpad=-2)
# #         if m == 3:
# #           ax.set_xlabel(r'$t_q$ $\rm{[Gyr]}$')
# #         else:
# #           pass
# #         if n == 0:
# #           ax.set_ylabel(r'$\tau$ $\rm{[Gyr]}$')
# #         else:
# #           pass
# #         ax.minorticks_on()
# #         if n%7 == 6:
# #           n = 0
# #           m+=1
# #         else:
# #           n+=1
# #     plt.tight_layout()
# #     plt.subplots_adjust(hspace=0.1)
# #     plt.savefig('../figures/fsps_mangadap_meas_EWs_t_'+str(time_steps[k])+'.pdf')
# #     plt.clf()
# #     plt.close('all')

# for k in range(len(tqs)):
#     for i in range(len(taus)):
#         plt.figure(figsize=(5, 5))
#         ax = plt.subplot(111)
#         # for j in range(len(emlines.data['name'])):
#         #     ax.plot(time_steps, ewss[:,k,i,j], label=r'$\rm{'+emlines.data['name'][j]+r'}$')
#         ax.plot(time_steps, ewss[:,k,i,np.where(emlines.data['name']=='Ha')[0][0]]/np.nanmax(ewss[:,k,i,np.where(emlines.data['name']=='Ha')[0][0]]), label=r'$\rm{H}\alpha$')
#         ax.plot(time_steps, ewss[:,k,i,np.where(emlines.data['name']=='Hdel')[0][0]]/np.nanmax(ewss[:,k,i,np.where(emlines.data['name']=='Hdel')[0][0]]), label=r'$\rm{H}\delta$')
#         ax.plot(time_steps, ewss[:,k,i,np.where(emlines.data['restwave']==3727.092)[0][0]]/np.nanmax(ewss[:,k,i,np.where(emlines.data['restwave']==3727.092)[0][0]]), label=r'$\rm{[OII]}$')
#         ax.set_ylabel(r'$\rm{EW}$')
#         ax2 = ax.twinx()
#         ax2.plot(time_steps, idmss[:,k,i,np.where(indx_names=='D4000')[0][0]]/np.nanmax(idmss[:,k,i,np.where(indx_names=='D4000')[0][0]]), color='b', label=r'$\rm{D4000}$', marker='x')
#         ax2.plot(time_steps, idmss[:,k,i,np.where(indx_names=='NaD')[0][0]]/np.nanmax(idmss[:,k,i,np.where(indx_names=='NaD')[0][0]]), color='purple', linestyle='dashed', label=r'$\rm{NaD}$', marker='x')
#         Mgb = idmss[:,k, i, np.where(indx_names=='Mgb')[0][0]]
#         Fe5270 = idmss[:,k, i, np.where(indx_names=='Fe5270')[0][0]]
#         Fe5335 = idmss[:,k, i, np.where(indx_names=='Fe5335')[0][0]]
#         MgFe = np.sqrt(Mgb * ( 0.72*Fe5270 + 0.28*Fe5335)  )
#         ax2.plot(time_steps, MgFe/np.max(MgFe), color='black', linestyle='dashed', label=r'$\rm{MgFe}$', marker='x')
#         ax2.set_ylabel(r'$\rm{Index}$', color='b')

#         h1, l1 = ax.get_legend_handles_labels()
#         h2, l2 = ax2.get_legend_handles_labels()
#         ax.legend(h1+h2, l1+l2, frameon=False)
        
#         ax.set_xlabel(r'$t$ $\rm{[Gyr]}$')
#         ax.minorticks_on()

#         plt.tight_layout()
#         plt.savefig('../figures/fsps_mangadap_EWs_indexes_with_time_tq_'+str(tqs[k])+'_'+str(taus[i])+'.pdf')
#         plt.clf()
#         plt.close('all')