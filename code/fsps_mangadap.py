# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from scipy import signal

from argparse import ArgumentParser

import multiprocessing

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import ctypes


#import fsps

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
from tqdm import tqdm, trange

from multiprocessing import Lock
global l
l = Lock()


np.set_printoptions(suppress=True, precision=4)

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

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

# def gauss(a, u, s, x):
#     return a * np.exp(- (((x - u)**2) /(s**2)))    

# def burstsfh(tb, Hb, time):
#     return gauss(Hb, tb, 0.1, time)

# def constsfh(csfr, time):
#     return np.ones_like(time)*csfr

# def normsfh(tq, sigma, time):
#     ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
#     c = np.apply_along_axis(lambda a: a.searchsorted(3.0), axis = 0, arr = time)
#     ssfr[:c.flatten()[0]] = np.interp(3.0, time.flatten(), ssfr.flatten())
#     c_sfr = np.interp(tq, time.flatten(), ssfr.flatten())*(1E10)/(1E9)
#     return gauss(c_sfr, tq, sigma, time)

# def dubnormsfh(tq, sigmasf, sigmaq, time):
#     ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
#     c = np.apply_along_axis(lambda a: a.searchsorted(3.0), axis = 0, arr = time)
#     ssfr[:c.flatten()[0]] = np.interp(3.0, time.flatten(), ssfr.flatten())
#     c_sfr = np.interp(tq, time.flatten(), ssfr.flatten())*(1E10)/(1E9)
#     maskq = time <= tq
#     masksf = time >= tq
#     #sfrs = np.ma.masked_array(sfr, mask=mask)
#     sfhsf = np.ma.masked_array(gauss(c_sfr, tq, sigmasf, time), mask=masksf)
#     sfhq = np.ma.masked_array(gauss(c_sfr, tq, sigmaq, time), mask=maskq)
#     return sfhsf.filled(0.0)+sfhq.filled(0.0)




if __name__ == "__main__":
    # t = time.clock()

    # age = np.append(np.linspace(0.01, 13, 60), np.linspace(13.025, 14.0, 40)).reshape(-1,1,1)
    # tqs = np.append(np.linspace(0.01, 12.0, 30), np.linspace(12.07, 14.0, 20))
    # taus = np.linspace(0.001, 6, 50)
    pad_zmet = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.012, 0.015, 0.019, 0.024, 0.03])
    pad_zmetsol = 0.019

    pad_solmet = pad_zmet/pad_zmetsol

    age = np.flip(13.805 - 10**(np.linspace(7, 10.14, 100))/1e9, axis=0).reshape(-1,1,1)
    tqs = np.flip(13.805 - 10**(np.linspace(7, 10.14, 50))/1e9, axis=0)
    taus = 10**np.linspace(6, 9.778, 50)/1e9
    #tqs = np.array([12])
    #taus = np.array([2, 4])

    #tq = tqs.reshape(1,-1,1).repeat(len(taus), axis=2)
    #tau = taus.reshape(1,1,-1).repeat(len(tqs), axis=1)

    #sfr = expsfh(tq, tau, age).reshape(age.shape[0], -1) 
    # #x = np.linspace(0, 14, 200).reshape(-1,1,1)
    ##sfr = burstsfh(4.0, 10.0, x) + burstsfh(7.5, 3.0, x) + burstsfh(11.3, 11.0, x) + burstsfh(13.5, 2.0, x)


    # sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0, zmet=18, sfh=3, dust_type=2, dust2=0.2, 
    # 	add_neb_emission=True, add_neb_continuum=False, imf_type=1, add_dust_emission=True, min_wave_smooth=3600, max_wave_smooth=7500, sigma_smooth=True)

    time_steps = Planck15.age(10**np.linspace(-0.824, -2.268, 15)).reshape(-1,1,1).value
    # #time_steps = np.linspace(0.2, 14, 7)
    # #zmets = np.arange(22)+1.
    zmets = (np.linspace(0, 10, 11)*2 + 1).astype(int)
    # #zmets = [4,7,10,13,16,19,22]
    #zmets= [21]
    #zmets = np.append(np.array([1, 7, 10]), np.linspace(13, 22, 10))

    #bins = numpy.linspace(0, 1, 11)
    #mid_bins = bins[:-1]+(numpy.diff(bins)/2.)

    #You need the template spectra.  For now just use the MILESHC library:
    tpl = TemplateLibrary("MILESHC",
                            match_to_drp_resolution=False,
                            velscale_ratio=1,    
                            spectral_step=1e-4,
                            log=True,
                            directory_path=".",
                            processed_file="mileshc.fits",
                            clobber=True)

    # Instantiate the object that does the fitting
    contbm = StellarContinuumModelBitMask()
    ppxf = PPXFFit(contbm)

    abs_db = AbsorptionIndexDB(u"EXTINDX")
    band_db = BandheadIndexDB(u"BHBASIC")
    global indx_names
    indx_names = np.hstack([abs_db.data["name"], band_db.data["name"]])

    specm = EmissionLineModelBitMask()
    elric = Elric(specm)
    global emlines
    emlines  = EmissionLineDB(u"ELPFULL")

    c = con.c.to(un.km/un.s).value

    # if os.path.isfile("emission_line_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+"_"+".npy"):
    #     eml = list(np.load("emission_line_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+"_"+".npy"))
    # else:
    #     eml = []

    # if os.path.isfile("abs_index_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+"_"+".npy"):
    #     idm = list(np.load("abs_index_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+"_"+".npy"))
    # else:
    #     idm = []

    manga_wave = np.load("manga_wavelengths_AA.npy")
    

    #for n in range(0, sfr.shape[1]):
    #for n in range(485,486):
    def get_fluxes(sfr):
        #print(str(100*(n/sfr.shape[1]))+"% of the way through")

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
        guess_redshift = np.full(nspec, 0.0001, dtype=float)
        guess_dispersion = np.full(nspec, 80.0, dtype=float)
    
        # Perform the fits and construct the models
        model_wave, model_flux, model_mask, model_par  = ppxf.fit(tpl["WAVE"].data.copy(), tpl["FLUX"].data.copy(), fsps_wave, fsps_flux, np.sqrt(fsps_flux), guess_redshift, guess_dispersion, iteration_mode="none", velscale_ratio=1, degree=8, mdegree=-1, moments=2, quiet=True)
        em_model_wave, em_model_flux, em_model_base, em_model_mask, em_model_fit_par, em_model_eml_par = elric.fit(fsps_wave, fsps_flux, emission_lines=emlines, ivar=np.abs(1/(fsps_flux+0.0000001)), sres=np.ones_like(fsps_flux), continuum=model_flux, guess_redshift = model_par["KIN"][:,0]/c, guess_dispersion=model_par["KIN"][:,1], base_order=1, quiet=True)
        indx_measurements = SpectralIndices.measure_indices(absdb=abs_db, bhddb=band_db, wave=fsps_wave, flux=fsps_flux, ivar=np.abs(1/(fsps_flux+0.0000001)), mask=None, redshift=model_par["KIN"][:,0]/c, bitmask=None)
    
        plt.close("all")
        plt.cla()
        plt.clf()
    
        return em_model_eml_par, indx_measurements

    def meas_non_parallel(i):
        meas = measure_spec(fluxes[i])
        #print(meas[0]["EW"])
        # em_par[i] = meas[0]
        # print(em_par[i])
        # indx_par[i] = meas[1]
        # if len(em_par) < 1:
        #     em_par = meas[0].reshape(1, -1)
        # else:
        #     em_par = np.append(np.array(em_par), meas[0].reshape(1,-1), axis=0)
        # if len(indx_par) < 1:
        #     indx_par = meas[1].reshape(1, -1)
        # else:
        #     indx_par = np.append(np.array(indx_par), meas[1].reshape(1,-1), axis=0)    
        return meas
        #np.save("../data/emission_line_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy", em_par)        
        #np.save("../data/abs_index_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy", indx_par)


    # def shared_array(shape, locked):
    #     """
    #     Form a shared memory numpy array.
        
    #     http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
    #     """
        
    #     shared_array_base = multiprocessing.Array(ctypes.c_double, shape[0]*shape[1], lock=locked)
    #     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    #     shared_array = shared_array.reshape(*shape)
    #     return shared_array

    fsps_wave_z0 = manga_wave
    fsps_wave = fsps_wave_z0*(1+ (100.0*(un.km/un.s)/con.c.to(un.km/un.s)) ).value
    print('about to load the fluxes file...')
    if os.path.isfile("spectrum_all_star_formation_rates_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy"):
        fluxes = np.load("spectrum_all_star_formation_rates_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy")
    else:
        fluxes = np.array(list(map(get_fluxes, tqdm(list(sfr.T))))).reshape(-1, len(fsps_wave))
        np.save("spectrum_all_star_formation_rates_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy", fluxes)
   

    # lock_em = multiprocessing.Lock()
    # lock_indx = multiprocessing.Lock()

    # if os.path.isfile("../data/emission_line_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy"):
    #     em_par = list(np.load("../data/emission_line_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy"))
    # else:
    #     em_par = shared_array((len(fluxes),1), lock_em)

    # if os.path.isfile("../data/abs_index_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy"):
    #     indx_par = list(np.load("../data/abs_index_params_log_scale_tq_"+str(len(tqs))+"_tau_"+str(len(taus))+"_to_"+str(len(time_steps))+"_Z_"+str(len(zmets))+".npy"))
    # else:
    #     indx_par = shared_array((len(fluxes),1), lock_indx)

    # Form a shared array and a lock, to protect access to shared memory.
    
    def meas_parallel(i):#, def_param=(lock_em, lock_indx, em_par, indx_par)):
        #print("Is running on ",multiprocessing.current_process().name)
        print(str(100*(i/412500.)), "% of the way through \r")
        eml, idm = measure_spec(shared_fluxes[i])
        emls = eml["EW"][:, np.logical_or(emlines['name']=='Ha', emlines['restwave']==3727.092)].reshape(1,-1)
        idms =idm["INDX"][:,np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))].reshape(1,-1)

        emls_err = eml["EWERR"][:, np.logical_or(emlines['name']=='Ha', emlines['restwave']==3727.092)].reshape(1,-1)
        idms_err =idm["INDXERR"][:,np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))].reshape(1,-1)
        
        lu = np.append(emls, idms, axis=1).reshape(1, 1, 8) # Halpha 0th,  OII 1st, Hbeta 2nd, MgB 3rd, Fe5270 4th, Fe5335 5th, HDeltaA 6th, D4000 7th
        lu_err = np.append(emls_err, idms_err, axis=1).reshape(1,-1) # Halpha 0th,  OII 1st, Hbeta 2nd, MgB 3rd, Fe5270 4th, Fe5335 5th, HDeltaA 6th, D4000 7th
        print("current time, ", time.time(), " for iteration: ", str(i))
        with l:
            if os.path.isfile("../data/em_indx_par_pool_mapped.npz"):
                with np.load("../data/em_indx_par_pool_mapped.npz") as ei_lu:
                    np.savez("../data/em_indx_par_pool_mapped.npz", lookup=np.append(ei_lu["lookup"], lu, axis=0))
                with np.load("../data/em_indx_err_par_pool_mapped.npz") as ei_lu_err:
                    np.savez("../data/em_indx_err_par_pool_mapped.npz", lookuperr=np.append(ei_lu_err["lookuperr"], lu_err, axis=0))
                with np.load("../data/iteration_number_em_indx_par_pool_mapped.npz") as iter_num:
                    np.savez("../data/iteration_number_em_indx_par_pool_mapped.npz", idx=np.append(iter_num["idx"], [i], axis=0))
            else:
                np.savez("../data/em_indx_par_pool_mapped.npz", lookup=lu)
                np.savez("../data/em_indx_err_par_pool_mapped.npz", lookuperr=lu_err)
                np.savez("../data/iteration_number_em_indx_par_pool_mapped.npz", idx=[i])

        return i

    print('creating shared array...')
    shared_array_base = multiprocessing.Array(ctypes.c_double, fluxes.shape[0]*fluxes.shape[1])
    shared_fluxes = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_fluxes = shared_fluxes.reshape(fluxes.shape[0], fluxes.shape[1])

    st = time.time()
    for i in range(len(shared_fluxes)):
        shared_fluxes[i,:] = fluxes[i,:]
    print("indexing took ", (time.time()-st)/60., " minutes")

    del fluxes



    print("shared_fluxes is ", shared_fluxes.nbytes, " bytes")
    
    st = time.time()
    num_cores = multiprocessing.cpu_count()
    print("number of cores ", num_cores)
    pool = multiprocessing.Pool(processes=8)
    i = len(shared_fluxes)
	# js = np.linspace(0., 412499., 412500.)
	# with np.load("../data/iteration_number_em_indx_par_pool_mapped.npz") as iter_num:
	#    jdxs = np.where(np.in1d(js, iter_num['idx'], invert=True))[0]
    results = pool.map(meas_parallel, range(0, i), chunksize=100)
    # print("starting from ", jdxs[0], " going to ", jdxs[-1])
    # results = pool.map(meas_parallel, jdxs, chunksize=10)
    #results = pool.map(meas_parallel, range(i), chunksize=1000)
    
 #    #print("saving results for ", str(i)," iterations now at time...", time.time())
 #    #print("size of results array for ", str(i)," iterations", results.nbytes)
 #    #np.save("../data/em_indx_par_pool_mapped_"+str(i)+"_.npy", results)
    print("results took ", (time.time() - st)/60., " minutes with normal pool \n")



