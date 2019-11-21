import os
import time
import numpy 
import numpy as np

from astropy.table import Table, Column
from astropy import constants as con

from argparse import ArgumentParser

import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt

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
from mangadap.par.spectralfeaturedb import SpectralFeatureDBDef

import requests
from requests.auth import HTTPBasicAuth

from scipy.interpolate import interp1d

import sys
sys.path.append('/Users/smethurst/Projects/mangaagn/snitch/snitch/')

from snitch import *
import functions
from astropy.cosmology import Planck15

from tqdm import tqdm, trange

import corner

c = con.c.to(un.km/un.s).value

sdss_verify = open('../sdss_user_pass.txt')
up = sdss_verify.readlines()

top_level_url='https://data.sdss.org/sas/mangawork/manga/spectro/redux/MPL-6/'

top_level_url_par='https://data.sdss.org/sas/mangawork/manga/spectro/analysis/MPL-6/SPX-GAU-MILESHC/'


font = {'family':'serif', 'size':16}
plt.rc('font', **font)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('axes', labelsize='medium', lw=1, facecolor='None', edgecolor='k')
plt.rc('text', usetex=True)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------


def corner_plot(s, labels, extents, bf, truth):
    """ Plotting function to visualise the gaussian peaks found by the sampler function. 2D contour plots of tq against tau are plotted along with kernelly smooth histograms for each parameter.
        
        :s:
         Array of shape (#, 2) for either the smooth or disc produced by the emcee EnsembleSampler in the sample function of length determined by the number of walkers which resulted at the specified peak.
        
        :labels:
        List of x and y axes labels i.e. disc or smooth parameters
        
        :extents:
        Range over which to plot the samples, list shape [[xmin, xmax], [ymin, ymax]]
        
        :bf:
        Best fit values for the distribution peaks in both tq and tau found from mapping the samples. List shape [(tq, poserrtq, negerrtq), (tau, poserrtau, negerrtau)]
        :truth:
        Known t and tau values - generally for test data
        
        RETURNS:
        :fig:
        The figure object
        """
    x, y = s[:,0], s[:,1]
    fig = P.figure(figsize=(6.25,6.25))
    ax2 = P.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])
    corner.hist2d(x, y, ax=ax2, bins=100, extent=extents, plot_contours=True)
    ax2.axvline(x=bf[0][0], linewidth=1)
    ax2.axhline(y=bf[1][0], linewidth=1)
    ax2.axvline(x=truth[0], c='r', linewidth=1)
    ax2.axhline(y=truth[1], c='r', linewidth=1)
    [l.set_rotation(45) for l in ax2.get_xticklabels()]
    [j.set_rotation(45) for j in ax2.get_yticklabels()]
    ax2.tick_params(axis='x', labeltop='off')
    ax1 = P.subplot2grid((3,3), (0,0),colspan=2)
    #ax1.hist(x, bins=100, range=(extents[0][0], extents[0][1]), normed=True, histtype='step', color='k')
    den = kde.gaussian_kde(x[np.logical_and(x>=extents[0][0], x<=extents[0][1])])
    pos = np.linspace(extents[0][0], extents[0][1], 750)
    ax1.plot(pos, den(pos), 'k-', linewidth=1)
    ax1.axvline(x=bf[0][0], linewidth=1)
    ax1.axvline(x=bf[0][0]+bf[0][1], c='b', linestyle='--')
    ax1.axvline(x=bf[0][0]-bf[0][2], c='b', linestyle='--')
    ax1.set_xlim(extents[0][0], extents[0][1])
    ax1.axvline(x=truth[0], c='r', linewidth=1)
    #    ax12 = ax1.twiny()
    #    ax12.set_xlim((extent[0][0], extent[0][1])
    #ax12.set_xticks(np.array([1.87, 3.40, 6.03, 8.77, 10.9, 12.5]))
    #ax12.set_xticklabels(np.array([3.5, 2.0 , 1.0, 0.5, 0.25, 0.1]))
    #    [l.set_rotation(45) for l in ax12.get_xticklabels()]
    #    ax12.tick_params(axis='x', labelbottom='off')
    #    ax12.set_xlabel(r'$z$')
    ax1.tick_params(axis='x', labelbottom='off', labeltop='off')
    ax1.tick_params(axis='y', labelleft='off')
    ax3 = P.subplot2grid((3,3), (1,2), rowspan=2)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='y', labelleft='off')
    #ax3.hist(y, bins=100, range=(extents[1][0], extents[1][1]), normed=True, histtype='step',color='k', orientation ='horizontal')
    den = kde.gaussian_kde(y[np.logical_and(y>=extents[1][0], y<=extents[1][1])])
    pos = np.linspace(extents[1][0], extents[1][1], 750)
    ax3.plot(den(pos), pos, 'k-', linewidth=1)
    ax3.axhline(y=bf[1][0], linewidth=1)
    ax3.axhline(y=bf[1][0]+bf[1][1], c='b', linestyle='--')
    ax3.axhline(y=bf[1][0]-bf[1][2], c='b', linestyle='--')
    ax3.set_ylim(extents[1][0], extents[1][1])
    ax3.axhline(y=truth[1], c='r', linewidth=1)
    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.subplots_adjust(hspace=0.0)
    return fig


def walker_plot(samples, nwalkers, limit, truth, ID):
    """ Plotting function to visualise the steps of the walkers in each parameter dimension for smooth and disc theta values. 
        
        :samples:
        Array of shape (nsteps*nwalkers, 4) produced by the emcee EnsembleSampler in the sample function.
        
        :nwalkers:
        The number of walkers that step around the parameter space used to produce the samples by the sample function. Must be an even integer number larger than ndim.
        
        :limit:
        Integer value less than nsteps to plot the walker steps to. 
        
        RETURNS:
        :fig:
        The figure object
        """
    s = samples.reshape(nwalkers, -1, 3)
    s = s[:,:limit, :]
    fig = P.figure(figsize=(8,12))
    ax1 = P.subplot(3,1,1)
    ax2 = P.subplot(3,1,2)
    ax3 = P.subplot(3,1,3)
    ax1.plot(s[:,:,0].T, 'k')
    ax1.axhline(truth[0], color='r', alpha=0.9)
    ax2.plot(s[:,:,1].T, 'k')
    ax2.axhline(truth[1], color='r', alpha=0.9)
    ax3.plot(s[:,:,2].T, 'k')
    ax3.axhline(truth[2], color='r', alpha=0.9)
    ax1.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='x', labelbottom='off')
    ax3.set_xlabel(r'step number')
    ax1.set_ylabel(r'$Z$')
    ax2.set_ylabel(r'$t_{quench}$')
    ax3.set_ylabel(r'$\tau$')
    P.subplots_adjust(hspace=0.1)
    #save_fig = './test_precision_mosaic/walkers_steps_'+str(ID)+'_log.pdf'
    #fig.savefig(save_fig)
    return fig

def scipyinterp(xnew, x, y, fill_value="extrapolate"):
    fnew = interp1d(x, y, fill_value=np.nan, bounds_error=False)
    return fnew(xnew)

#mpl6agn = Table.read('../../data/unique_EM_ALLWISEAGN_RASSDSSAGN_2XMMi_Xray_limited_1042_AGN_rembold_type2_with_mpa_jhu_measurement_plus_extra_dominika_AGN_BPT.fits', format='fits')
ellison_list = Table.read('/Users/smethurst/Projects/mangaagn/code/ellison/ellison_list.csv')

mid_bins = []
emls = []
idms = []
emls_error = []
idms_error = []
emls_mask = []
idms_mask = []
masks= []


Z_mcmc = Column(name='inferred Z', data = np.nan*np.empty([len(ellison_list), 15]))
Z_p_mcmc = Column(name='inferred Z plus error', data = np.nan*np.empty([len(ellison_list), 15]))
Z_m_mcmc = Column(name='inferred Z minus error', data = np.nan*np.empty([len(ellison_list), 15]))


tq_mcmc = Column(name='inferred dtq', data = np.nan*np.empty([len(ellison_list), 15]))
tq_p_mcmc = Column(name='inferred dtq plus error', data = np.nan*np.empty([len(ellison_list), 15]))
tq_m_mcmc = Column(name='inferred dtq minus error', data = np.nan*np.empty([len(ellison_list), 15]))

tau_mcmc = Column(name='inferred tau', data = np.nan*np.empty([len(ellison_list), 15]))
tau_p_mcmc = Column(name='inferred tau plus error', data = np.nan*np.empty([len(ellison_list), 15]))
tau_m_mcmc = Column(name='inferred tau minus error', data = np.nan*np.empty([len(ellison_list), 15]))

nllstartdtq = Column(name='nll_start_dtq', data = np.nan*np.empty([len(ellison_list), 15]))
nllstarttau = Column(name='nll_start_tau', data = np.nan*np.empty([len(ellison_list), 15]))
nllstartZ = Column(name='nll_start_Z', data = np.nan*np.empty([len(ellison_list), 15]))

bin_col = Column(name='bins', data = np.nan*np.empty([len(ellison_list), 15]))
mask_col = Column(name='mask', data = np.zeros([len(ellison_list), 15]))

ew_col = Column(name='emls', data = np.nan*np.empty([len(ellison_list), 15, 1]))
ew_mask_col = Column(name='emls_mask', data = np.nan*np.empty([len(ellison_list), 15, 1]))
ew_error_col = Column(name='emls_error', data = np.nan*np.empty([len(ellison_list), 15, 1]))
idms_col = Column(name='idms', data = np.nan*np.empty([len(ellison_list), 15, 8]))
idms_mask_col = Column(name='idms_mask', data = np.nan*np.empty([len(ellison_list), 15, 8]) )
idms_error_col = Column(name='idms_error', data = np.nan*np.empty([len(ellison_list), 15, 8]) )


ellison_list.add_columns([bin_col, mask_col, ew_col, ew_mask_col, ew_error_col, idms_col, idms_mask_col, idms_error_col, Z_mcmc, Z_p_mcmc, Z_m_mcmc, tq_mcmc, tq_p_mcmc, tq_m_mcmc, tau_mcmc, tau_p_mcmc, tau_m_mcmc, nllstartdtq, nllstarttau, nllstartZ])

nll = lambda *args: -lnprob(*args)
from scipy.optimize import minimize, basinhopping

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, hard))

nwalkers = 100 # number of monte carlo chains
nsteps= 100 # number of steps in the monte carlo chain
opstart = [0.5, np.log10(0.5), 1] # starting place of all the chains
burnin = 500 # number of steps in the burn in phase of the monte carlo chain
ndim = 3

#ages = Planck15.age(mpl6agn['nsa_z']).value

define_em_db = SpectralFeatureDBDef(key='USEREM',
                              file_path='../elpsnitch.par')
emlines  = EmissionLineDB(u"USEREM", emldb_list=define_em_db)
define_abs_db = SpectralFeatureDBDef(key='USERABS',
                              file_path='../extindxsnitch.par')
abs_db = AbsorptionIndexDB(u"USERABS", indxdb_list=define_abs_db)
band_db = BandheadIndexDB(u"BHBASIC")
indx_names = np.hstack([abs_db.data["name"], band_db.data["name"]])

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))

spxinfo = Table.read('../../data/spaxel_PCA_VAC.fits')
start = 0

if start > 0:
    ellison_list = Table.read('ellisonagn_inferred_snitch_linear_interp_deltatq.fits')
else:
    pass

for n in trange(start, len(ellison_list)):
    if os.path.isfile('../../data/manga-'+str(ellison_list['plateifu'][n])+'-LOGCUBE.fits.gz') is False:
        r = requests.get(top_level_url+str(ellison_list['plateifu'][n].split('-')[0])+'/stack/manga-'+str(ellison_list['plateifu'][n])+'-LOGCUBE.fits.gz', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)
        if r.ok:
             with open('../../data/manga-'+str(ellison_list['plateifu'][n])+'-LOGCUBE.fits.gz', 'wb') as file:
                 file.write(r.content) 
    else:
        pass

    if os.path.isfile('../../data/manga-'+str(ellison_list['plateifu'][n])+'-LOGRSS.fits.gz') is False:
        s = requests.get(top_level_url+str(ellison_list['plateifu'][n].split('-')[0])+'/stack/manga-'+str(ellison_list['plateifu'][n])+'-LOGRSS.fits.gz', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)
        if s.ok:
            with open('../../data/manga-'+str(ellison_list['plateifu'][n])+'-LOGRSS.fits.gz', 'wb') as sfile:
                sfile.write(s.content)
        else:
            pass

    if os.path.isfile('../../data/mangadap-'+str(ellison_list['plateifu'][n])+'-LOGCUBE-input.par') is False:
        p = requests.get(top_level_url_par+str(ellison_list['plateifu'][n].split('-')[0])+'/'+str(ellison_list['plateifu'][n].split('-')[1])+'/ref/mangadap-'+str(ellison_list['plateifu'][n])+'-LOGCUBE-input.par', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)
        if p.ok:
             with open('../../data/mangadap-'+str(ellison_list['plateifu'][n])+'-LOGCUBE-input.par', 'wb') as file:
                 file.write(p.content) 

    else:
        pass

    print('starting analysis of ', str(ellison_list[n]['plateifu']), ' which is ', str(100*(n/len(ellison_list))),'% of the way through')
    drpf = DRPFits(ellison_list['plateifu'][n].split('-')[0], ellison_list['plateifu'][n].split('-')[1], 'CUBE', read=True, directory_path='../../data/')
    print(drpf.file_path())
    x,y = drpf.mean_sky_coordinates(offset=True)
    r = np.sqrt(x**2+y**2)

    if os.path.isfile('../../data/manga-'+str(ellison_list['plateifu'][n])+'-LOGCUBE.fits.gz'):

        par_file = '../../data/mangadap-'+str(ellison_list['plateifu'][n])+'-LOGCUBE-input.par'

        obsp = ObsInputPar.from_par_file(par_file)


        if '19' in ellison_list['plateifu'][n].split('-')[1]:
            nbins = 7
        elif '37' in ellison_list['plateifu'][n].split('-')[1]:
            nbins = 9
        elif '61' in ellison_list['plateifu'][n].split('-')[1]:
            nbins = 11
        elif '91' in ellison_list['plateifu'][n].split('-')[1]:
            nbins = 13
        else:
            nbins = 15

        mask = np.append(np.zeros(nbins), np.ones(15-nbins))
        if 'BPT' in ellison_list['survey'][n]:
            pass
        else:
            mask[0] = 1
        masks.append(mask.astype(bool))
        ellison_list['mask'][n] = mask

        # ellison_list['bins'][n].mask = mask
        # ellison_list['inferred Z'][n].mask = mask
        # ellison_list['inferred Z plus error'][n].mask = mask
        # ellison_list['inferred Z minus error'][n].mask = mask
        # ellison_list['inferred tq'][n].mask = mask
        # ellison_list['inferred tq plus error'][n].mask = mask
        # ellison_list['inferred tq minus error'][n].mask = mask
        # ellison_list['inferred tau'][n].mask = mask
        # ellison_list['inferred tau plus error'][n].mask = mask
        # ellison_list['inferred tau minus error'][n].mask = mask

        if (r ==0).any():
            rbin = RadialBinning(par=RadialBinningPar(center=[0.0,0.0], pa=obsp['pa'], ell=obsp['ell'], radius_scale=obsp['reff'], radii=[np.nanmin(r[np.nonzero(r)]), -1, nbins], log_step=True))
        else:
            rbin = RadialBinning(par=RadialBinningPar(center=[0.0,0.0], pa=obsp['pa'], ell=obsp['ell'], radius_scale=obsp['reff'], radii=[2*np.nanmin(r), -1, nbins], log_step=True))
        binid = rbin.bin_index(x,y)
        r, theta = rbin.sma_coo.polar(x, y)

        binwgt = np.ones(len(binid))
        print(len(binwgt))
        print(x, y)

        stringtest = ellison_list[n]['plateifu']
        while len(stringtest) < 11:
            stringtest+=' '

        spxifu = spxinfo[spxinfo['plateifu'] == stringtest]

        if len(binwgt) == len(spxifu):
            #print(spxifu)
            #remove spaxels run through PCA analysis which are flagged as broadline AGN or starburst
            binwgt[np.logical_or(spxifu['class']<1,spxifu['class'] == 3)] = 0
        else:
            pass


        bins = numpy.linspace(np.log10(rbin.par['radii'][0]), np.log10(rbin.par['radii'][1]), nbins+1)
        mbins = 10**(bins[:-1]+(numpy.diff(bins)/2.))
        if len(mbins) < 15:
            ellison_list['bins'][n] = np.append(mbins, np.nan*np.ones(15-len(mbins)))
        else:
            ellison_list['bins'][n] = mbins

        mid_bins.append(mbins)

        stack = SpectralStack()
        #stack_wave, stack_flux, stack_sdev, stack_npix, stack_ivar, stack_sres, stack_covar = stack.stack_DRPFits(drpf, binid, binwgt=np.ones(len(binid)), par=SpectralStackPar('mean', False, None, 'channels', SpectralStack.parse_covariance_parameters('channels', '11'), None))
        stack_wave, stack_flux, stack_sdev, stack_npix, stack_ivar, stack_sres, stack_covar = stack.stack_DRPFits(drpf, binid, binwgt=binwgt, par=SpectralStackPar('mean', False, None, 'channels', SpectralStack.parse_covariance_parameters('channels', '11'), None))

        redshift = obsp['vel']/c

        abs_names=[r'H$\beta$', r'$MgB$', r'$Fe5270$', r'$Fe5335$', r'$H\delta_A$', r'D4000', r'Dn4000', r'TiO']
        plt.figure(figsize=(8,5))
        ax = plt.subplot(111)
        for m in range(stack_flux.shape[0]):
            if len(stack_flux[m,:].compressed()) == 0:
                pass
            else:
                label = '%.3f' % mbins[m]
                ax.plot(stack_wave[~stack_flux[m,:].mask]/(1+redshift), stack_flux[m,:].compressed(), label=str(label)+'$ R/R_e$')
        ys = ax.get_ylim()
        ax.set_xlabel(r'$\rm{Wavelength}$ $[\AA]$')
        ax.set_ylabel(r'$\rm{Arbitrary}$ $\rm{Flux}$')
        ax.axvline(emlines['restwave'][3], c='0.5', linestyle='dashed', linewidth=0.5)
        ax.text(emlines['restwave'][3]+2, 30, r'$\rm{EW}$[H$\alpha$]', fontsize='x-small')
        ax.fill_betweenx(y=ys, x1 = band_db['blueside'][1][0], x2 = band_db['blueside'][1][1], facecolor='0.5', alpha=0.3)
        ax.fill_betweenx(y=ys, x1 = band_db['redside'][1][0], x2 = band_db['redside'][1][1], facecolor='0.5', alpha=0.3)
        for m in range(len(abs_db.data)):
            ax.axvline((abs_db['primary'][m][0]+abs_db['primary'][m][1])/2., c='0.5', linestyle='dashed', linewidth=0.5)
            ax.text(abs_db['primary'][m][1], ys[-1]-(0.1 + 0.1*float(m)*ys[-1]),  abs_names[m], fontsize='x-small')
        ax.set_xlim(3500, 7000)
        ax.legend(frameon=False, fontsize='x-small')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
        plt.tight_layout()
        plt.savefig('figures/stacked_wave_for_'+ellison_list[n]['plateifu']+'_rre_bin.pdf')

        em_model_eml_par, indx_measurements = functions.measure_spec(stack_flux, errors=stack_sdev, ivar=stack_ivar, sres=stack_sres)

        emls= em_model_eml_par["EW"][:, np.where(emlines['name']=='Ha')].reshape(1,-1,1)
        idms = indx_measurements["INDX"].reshape(1, -1,8)
        
        emls_error = em_model_eml_par["EWERR"][:, np.where(emlines['name']=='Ha')].reshape(1,-1,1)
        idms_error = indx_measurements["INDXERR"].reshape(1, -1,8)

        emls_mask = em_model_eml_par['MASK'][:, np.where(emlines['name']=='Ha')].reshape(1,-1,1).astype(bool)
        idms_mask = indx_measurements["MASK"].reshape(1,-1,8).astype(bool)

        np.save(str(ellison_list['plateifu'][n])+'_ellison_measured_emls.npy', emls)
        np.save(str(ellison_list['plateifu'][n])+'_ellison_measured_idms.npy', idms)

        np.save(str(ellison_list['plateifu'][n])+'_ellison_measured_emls_error.npy', emls_error)
        np.save(str(ellison_list['plateifu'][n])+'_ellison_measured_idms_error.npy', idms_error)

        np.save(str(ellison_list['plateifu'][n])+'_ellison_measured_emls_mask.npy', emls_mask)
        np.save(str(ellison_list['plateifu'][n])+'_ellison_measured_idms_mask.npy', idms_mask)

        emls[0][np.logical_or(emls[0] == 0.0, emls_mask[0]==True)] = np.nan
        idms[0][np.logical_or(idms[0] == 0.0, idms_mask[0]==True)] = np.nan

        ellison_list['emls'][n][:len(emls[0]), :] = np.array(emls)[0]
        ellison_list['emls_mask'][n][:len(emls[0]), :] = np.array(emls_mask)[0]
        ellison_list['emls_error'][n][:len(emls[0]), :] = np.array(emls_error)[0]

        ellison_list['idms'][n][:len(emls[0]), :]= np.array(idms)[0]
        ellison_list['idms_mask'][n][:len(emls[0]), :] = np.array(idms_mask)[0]
        ellison_list['idms_error'][n][:len(emls[0]), :] = np.array(idms_error)[0]

        mask[:len(emls[0])] += emls_mask[0][:,0]
        mask[:len(emls[0])] += np.sum(idms_mask[0][:,:-1], axis=1).astype(bool)
        ellison_list['mask'][n]=mask

        start_bh = np.zeros(nbins*ndim).reshape(-1,ndim)


        for j in range(idms.shape[1]):
            mgfe = np.sqrt(idms[0][j,1] * ( 0.72*idms[0][j,2] + 0.28*idms[0][j,3]))
            if np.all(np.isnan(idms[0][j,:])):
                ellison_list['inferred Z'][n, j] = np.nan
                ellison_list['inferred Z plus error'][n, j] = np.nan
                ellison_list['inferred Z minus error'][n, j] = np.nan
                ellison_list['inferred dtq'][n, j] = np.nan
                ellison_list['inferred dtq plus error'][n, j] = np.nan
                ellison_list['inferred dtq minus error'][n, j] = np.nan
                ellison_list['inferred tau'][n, j] = np.nan
                ellison_list['inferred tau plus error'][n, j] = np.nan
                ellison_list['inferred tau minus error'][n, j] = np.nan
            else:
                opstart = [0.5, np.log10(0.5), 1]
                if ellison_list['mask'][n][j]:
                    pass
                else:
                    for k in range(len(idms[0][j])):
                        idms_error[0][j,k] = 0.5*idms[0][j,k]
                    emls_error[0][j,0] = 0.5*emls[0][j,0]
                        
                    emcee_values, nllstart = snitch(emls[0][j,0], emls_error[0][j,0], idms[0][j,6], idms_error[0][j,6], idms[0][j,0], idms_error[0][j,0],idms[0][j,4], idms_error[0][j,4], mgfe, 0.1*mgfe, redshift=redshift, ident=ellison_list['plateifu'][n]+'_'+str(format(mbins[j], '.2f')), opstart=opstart)
                    dtq_mcmc, tau_mcmc, Z_mcmc = emcee_values
                    ellison_list['inferred Z'][n, j] = Z_mcmc[0]
                    ellison_list['inferred Z plus error'][n, j] = Z_mcmc[1]
                    ellison_list['inferred Z minus error'][n, j] = Z_mcmc[2]
                    ellison_list['inferred dtq'][n, j] = dtq_mcmc[0]
                    ellison_list['inferred dtq plus error'][n, j] = dtq_mcmc[1]
                    ellison_list['inferred dtq minus error'][n, j] = dtq_mcmc[2]
                    ellison_list['inferred tau'][n, j] = tau_mcmc[0]
                    ellison_list['inferred tau plus error'][n, j] = tau_mcmc[1]
                    ellison_list['inferred tau minus error'][n, j] = tau_mcmc[2]
                    ellison_list['nll_start_dtq'][n,j] = nllstart[0]
                    ellison_list['nll_start_tau'][n,j] = nllstart[1]
                    ellison_list['nll_start_Z'][n,j] = nllstart[2]

            plt.close('all')

        ellison_list.write('ellisonagn_inferred_snitch_linear_interp_deltatq.fits', format='fits', overwrite=True)

    else:
        print('nothing for ', str(ellison_list[n]['plateifu']))
            #print('Best fit [Z, t, tau] values found by starpy for input parameters are : [', Zs_mcmc[0], tqs_mcmc[0], taus_mcmc[0], ']'

        #np.save('figures/bins_for_plotting_mpl6agn_stack'+mpl6agn['plateifu'][n]+'.npy', mid_bins)

redshift = np.nan*np.ones(len(ellison_list))
reff = np.nan*np.ones(len(ellison_list))
for n in trange(start, len(ellison_list)):
    if os.path.isfile('../../data/manga-'+str(ellison_list['plateifu'][n])+'-LOGCUBE.fits.gz'):

        par_file = '../../data/mangadap-'+str(ellison_list['plateifu'][n])+'-LOGCUBE-input.par'

        obsp = ObsInputPar.from_par_file(par_file)
        redshift[n] = obsp['vel']/c
        reff[n] = obsp['reff']
#deltatq = (Planck15.age(redshift).reshape(391,1)*np.ones((1, 15))).value - np.array(ellison_list['inferred tq'].flatten()).reshape(391,15)

kpcarcsec = Planck15.kpc_proper_per_arcmin(redshift).to(un.kpc/un.arcsecond) 
mangascale = 0.5 * un.arcsec/un.pixel
kpcbins = ellison_list['bins']*(reff*un.pixel*mangascale*kpcarcsec).reshape(-1,1)*(np.ones(15).reshape(1,15))
kpc_bins = Column(name='kpc_bins', data=kpcbins, unit=un.kpc)
ellison_list.add_column(kpc_bins)



tqmask = (((ellison_list['inferred dtq plus error'].flatten() + ellison_list['inferred dtq minus error'].flatten())/2.).data > 1).reshape(391,15)
tqmask2 = (((ellison_list['inferred dtq plus error'].flatten() + ellison_list['inferred dtq minus error'].flatten())/2.).data < 0.).reshape(391,15)
tqmask3 = (((ellison_list['inferred dtq'].flatten()).data > 14)).reshape(391,15)
tqmask4 = (((ellison_list['inferred dtq'].flatten()).data < 0.)).reshape(391,15)
#tqmask5 = (deltatq < 0)

taumask = (((ellison_list['inferred tau plus error'].flatten() + ellison_list['inferred tau minus error'].flatten())/2.).data > 1).reshape(391,15)
taumask2 = (((ellison_list['inferred tau plus error'].flatten() + ellison_list['inferred tau minus error'].flatten())/2.).data < 0.0).reshape(391,15)
taumask3 = (((ellison_list['inferred tau'].flatten()).data > 1)).reshape(391,15)
taumask4 = (((ellison_list['inferred tau'].flatten()).data < -6.)).reshape(391,15)

Zmask = (((ellison_list['inferred Z plus error'].flatten() + ellison_list['inferred Z minus error'].flatten())/2.).data > 1).reshape(391,15)
Zmask2 = (((ellison_list['inferred Z plus error'].flatten() + ellison_list['inferred Z minus error'].flatten())/2.).data < 0.0).reshape(391,15)
#Zmask3 = (((ellison_list['inferred Z'].flatten()).data > 1.5)).reshape(391,15)
#Zmask4 = (((ellison_list['inferred Z'].flatten()).data < 0)).reshape(391,15)


masks = tqmask +tqmask2 +tqmask3 +tqmask4 + taumask + taumask2 + taumask3 + taumask4 + Zmask + Zmask2

#av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_bins = np.nanmedian(kpcbins, axis=0)
av_i = np.empty((n+1, len(av_bins)))
labels = ['inferred Z', 'inferred dtq', 'inferred tau']
lims = [[-1,2.5],[-1, 15],[-3, 0.5]]
ax_labels = [r'$\overline{Z}$',r'$\Delta\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
for k in range(len(labels)):
    plt.figure()
    for m in range(n+1):
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['kpc_bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], color='k', alpha=0.1)
        try:
            av_i[m,:] = scipyinterp(av_bins, ellison_list['kpc_bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], fill_value='extrapolate=')
        except(ValueError):
            pass
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R [kpc]$')
    plt.ylabel(ax_labels[k])
    #plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_'+labels[k]+'_with_kpc_radius_ellison_findagetheninterp.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

# #av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
# av_bins = np.nanmedian(kpcbins, axis=0)
# av_i = np.empty((n+1, len(av_bins)))
# lims = [[-1,14]]
# ax_labels = [r'$\Delta t_q$ $\rm{[Gyr]}$']
# for k in range(len(ax_labels)):
#     plt.figure()
#     for m in range(n+1):
#         #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#         plt.plot(ellison_list['kpc_bins'][m][~masks[m]], deltatq[m][~masks[m]], color='k', alpha=0.1)
#         try:
#             av_i[m,:] = scipyinterp(av_bins, ellison_list['kpc_bins'][m][~masks[m]], deltatq[m][~masks[m]], fill_value='extrapolate=')
#         except(ValueError):
#             pass
#     plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R [kpc]$')
#     plt.ylabel(ax_labels[k])
#     #plt.xlim(0, 1.0)
#     plt.ylim(lims[k])
#     plt.tick_params('both', direction='in', top='on', right='on')
#     plt.minorticks_on()
#     try:
#         plt.tight_layout()
#         plt.savefig('figures/median_deltatq_with_kpc_radius_ellison_findagetheninterp.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     except(IndexError):
#         pass
#     plt.close('all')

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.empty((n+1, len(av_bins)))
labels = ['inferred Z', 'inferred tq', 'inferred tau']
lims = [[-1,2.5],[-1, 15],[-3, 0.5]]
ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
for k in range(len(labels)):
    plt.figure()
    for m in range(n+1):
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_'+labels[k]+'_with_radius_ellison_findagetheninterp.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.empty((n+1, len(av_bins)))
lims = [[-1,14]]
ax_labels = [r'$\Delta t_q$ $\rm{[Gyr]}$']
for k in range(len(ax_labels)):
    plt.figure()
    for m in range(n+1):
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_deltatq_with_radius_ellison_findagetheninterp.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')


idx = np.where(ellison_list['survey']=="WISE")[0]

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
labels = ['inferred Z', 'inferred tq', 'inferred tau']
lims = [[-1,2.5],[-1, 15],[-3, 0.5]]
ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
for k in range(len(labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_'+labels[k]+'_with_radius_ellison_WISE.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
lims = [[-1,14]]
ax_labels = [r'$\Delta t_q$ $\rm{[Gyr]}$']
for k in range(len(ax_labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_deltatq_with_radius_ellison_WISE.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

idx = np.where(ellison_list['survey']=="XRAY")[0]

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
labels = ['inferred Z', 'inferred tq', 'inferred tau']
lims = [[-1,2.5],[-1, 15],[-3, 0.5]]
ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
for k in range(len(labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_'+labels[k]+'_with_radius_ellison_XRAY.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
lims = [[-1,14]]
ax_labels = [r'$\Delta t_q$ $\rm{[Gyr]}$']
for k in range(len(ax_labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_deltatq_with_radius_ellison_XRAY.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

idx = np.where(ellison_list['survey']=="BPT")[0]

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
labels = ['inferred Z', 'inferred tq', 'inferred tau']
lims = [[-1,2.5],[-1, 15],[-3, 0.5]]
ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
for k in range(len(labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_'+labels[k]+'_with_radius_ellison_BPT.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
lims = [[-1,14]]
ax_labels = [r'$\Delta t_q$ $\rm{[Gyr]}$']
for k in range(len(ax_labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_deltatq_with_radius_ellison_BPT.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

idx = np.where(ellison_list['survey']=="LERG")[0]

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
labels = ['inferred Z', 'inferred tq', 'inferred tau']
lims = [[-1,2.5],[-1, 15],[-3, 0.5]]
ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
for k in range(len(labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], ellison_list[labels[k]][m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_'+labels[k]+'_with_radius_ellison_LERG.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

av_bins = np.append(np.append(np.linspace(0.01, 0.05, 2), np.linspace(0.1, 0.9, 5)), np.linspace(1, 2, 3))
av_i = np.nan*np.empty((n+1, len(av_bins)))
lims = [[-1,14]]
ax_labels = [r'$\Delta t_q$ $\rm{[Gyr]}$']
for k in range(len(ax_labels)):
    plt.figure()
    for m in idx:
        #mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
        plt.plot(ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], color='k', alpha=0.1)
        av_i[m,:] = scipyinterp(av_bins, ellison_list['bins'][m][~masks[m]], deltatq[m][~masks[m]], fill_value='extrapolate=')
    plt.plot(av_bins, np.nanmedian(av_i[:m+1,:], axis=0), color='r', alpha=0.9)
    plt.xlabel(r'$R/R_e$')
    plt.ylabel(ax_labels[k])
    plt.xlim(0, 1.0)
    plt.ylim(lims[k])
    plt.tick_params('both', direction='in', top='on', right='on')
    plt.minorticks_on()
    try:
        plt.tight_layout()
        plt.savefig('figures/median_deltatq_with_radius_ellison_LERG.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    except(IndexError):
        pass
    plt.close('all')

# mpl6agn = Table.read('/Users/smethurst/Projects/mangaagn/data/mpl6agn_snitch_inferred_simard_bt_ratio.fits', format='fits') 

# Z_mcmc = Column(name='inferred Z', data = np.nan*np.empty([len(ellison_list), 15]))
# Z_p_mcmc = Column(name='inferred Z plus error', data = np.nan*np.empty([len(ellison_list), 15]))
# Z_m_mcmc = Column(name='inferred Z minus error', data = np.nan*np.empty([len(ellison_list), 15]))


# tq_mcmc = Column(name='inferred tq', data = np.nan*np.empty([len(ellison_list), 15]))
# tq_p_mcmc = Column(name='inferred tq plus error', data = np.nan*np.empty([len(ellison_list), 15]))
# tq_m_mcmc = Column(name='inferred tq minus error', data = np.nan*np.empty([len(ellison_list), 15]))

# tau_mcmc = Column(name='inferred tau', data = np.nan*np.empty([len(ellison_list), 15]))
# tau_p_mcmc = Column(name='inferred tau plus error', data = np.nan*np.empty([len(ellison_list), 15]))
# tau_m_mcmc = Column(name='inferred tau minus error', data = np.nan*np.empty([len(ellison_list), 15]))

# start = 0

# mbins = []
# masks = []
# for n in trange(start, len(ellison_list)):
#     files = glob.glob('inferred_SFH_parameters_ID_'+mpl6agn['plateifu'][n]+'*.npy')
#     j=0
#     bins = []
#     for file in files:
#         #if time.gmtime(os.path.getmtime(file)).tm_mon >= 3:

#         params = np.load(file)
            
#         Z_mcmc[n,j] = params[0, 0]
#         Z_p_mcmc[n,j] = params[0,1]
#         Z_m_mcmc[n,j] = params[0, 2]

#         tq_mcmc[n,j] = params[1,0]
#         tq_p_mcmc[n,j] = params[1,1]
#         tq_m_mcmc[n,j] = params[1,2]

#         tau_mcmc[n,j] = params[2,0]
#         tau_p_mcmc[n,j] = params[2,1]
#         tau_m_mcmc[n,j] = params[2,2]
#         j+=1
#         bins.append(float(file.split('_')[-1].split('.npy')[0]))
#     mask = np.append(np.zeros(len(bins)), np.ones(15-len(bins))).astype(bool)
#     masks.append(mask)
#     mbins.append(bins)

# masks = np.array(masks)

# tqmask = (((mpl6agn['inferred tq plus error'].flatten() + mpl6agn['inferred tq minus error'].flatten())/2.).data > 1.).reshape(360,15)
# taumask = (((mpl6agn['inferred tau plus error'].flatten() + mpl6agn['inferred tau minus error'].flatten())/2.).data > 0.6).reshape(360,15)
# masks += tqmask + taumask  

# mpl6agn['inferred Z'].mask += masks
# mpl6agn['inferred Z plus error'].mask += masks
# mpl6agn['inferred Z minus error'].mask += masks
# mpl6agn['inferred tq'].mask += masks
# mpl6agn['inferred tq plus error'].mask += masks
# mpl6agn['inferred tq minus error'].mask += masks
# mpl6agn['inferred tau'].mask += masks
# mpl6agn['inferred tau plus error'].mask += masks
# mpl6agn['inferred tau minus error'].mask += masks

# deltatq = (Planck15.age(mpl6agn['z']).reshape(360,1)*np.ones((1, 15))).value - mpl6agn['inferred tq']


# av_bins = np.append(np.append(np.linspace(0.01, 0.1, 3), np.linspace(0.1, 1, 5)), np.linspace(1, 2, 3))

# av_dtq = np.empty((len(mbins), len(av_bins)))
# av_tau = np.empty((len(mbins), len(av_bins)))
# av_Z = np.empty((len(mbins), len(av_bins)))

# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\overline{\Delta t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# ylims = [[-1.5, np.log10(1.75)],[-2, np.log10(14)],[-4.5, 0.65]]
# for k in range(len(labels)):
#     for m in range(len(mbins)):
#         if (np.any(Planck15.age(mpl6agn[m]['nsa_z']).value - tq_mcmc[m,:][~masks[m]] < 0)) or (np.any(Planck15.age(mpl6agn[m]['nsa_z']).value - tq_mcmc[m,:][~masks[m]] > 20)):
#             print('before',m,  masks[m])
#             masks[m][np.where(Planck15.age(mpl6agn[m]['nsa_z']).value - tq_mcmc[m,:][~masks[m]] < 0)] = True
#             print('after', m, masks[m])
#     plt.figure()
#     if k == 1:
#         for m in range(len(mbins)):
#             #mbins =  mbins[m][~mask]
#             plt.plot(np.array(mbins[m])[~masks[m][:len(mbins[m])]], np.log10(Planck15.age(mpl6agn[m]['nsa_z']).value - tq_mcmc[m,:][~masks[m]]), color='k', alpha=0.1)
#             if len(np.array(mbins[m])[~masks[m][:len(mbins[m])]]) > 0:
#                 av_dtq[m,:] = np.interp(av_bins, np.array(mbins[m])[~masks[m][:len(mbins[m])]], Planck15.age(mpl6agn[m]['nsa_z']).value - tq_mcmc[m,:][~masks[m]])
#         av_dtq[np.where(np.logical_and(av_dtq > 100, av_dtq < -100))] = np.nan
#         plt.plot(av_bins, np.log10(np.nanmedian(av_dtq[:m+1,:], axis=0)), color='r', alpha=0.9)
#     elif k == 2:
#         for m in range(len(mbins)):
#             #mbins =  mbins[m][~mask]
#             plt.plot(np.array(mbins[m])[~masks[m][:len(mbins[m])]], tau_mcmc[m,:][~masks[m]], color='k', alpha=0.1)
#             if len(np.array(mbins[m])[~masks[m][:len(mbins[m])]]) > 0:
#                 av_tau[m,:] = np.interp(av_bins, np.array(mbins[m])[~masks[m][:len(mbins[m])]], 10**tau_mcmc[m,:][~masks[m]])
#         av_tau[np.where(np.logical_and(av_tau > 100, av_tau < -100))] = np.nan
#         plt.plot(av_bins, np.log10(np.nanmedian(av_tau[:m+1,:], axis=0)), color='r', alpha=0.9)
#     else:
#         for m in range(len(mbins)):
#             #mbins =  mbins[m][~mask]
#             plt.plot(np.array(mbins[m])[~masks[m][:len(mbins[m])]], np.log10(Z_mcmc[m,:][~masks[m]]), color='k', alpha=0.1)
#             if len(np.array(mbins[m])[~masks[m][:len(mbins[m])]]) > 0:
#                 av_Z[m,:] = np.interp(av_bins, np.array(mbins[m])[~masks[m][:len(mbins[m])]], Z_mcmc[m,:][~masks[m]])
#         plt.plot(av_bins, np.log10(np.nanmedian(av_Z[:m+1,:], axis=0)), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 2.0)
#     plt.ylim(ylims[k])
#     plt.tight_layout()
#     plt.savefig('../figures/median_'+labels[k]+'_with_radius_mpl6agn_masked_still_SF_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close('all')

# #just the BPT Rembold sources 
# idx = np.where(mpl6agn['survey']=="BPT_rembold  ")[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\Delta\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     if k != 1:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     else:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_BPT_rembold_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# idx = np.where(mpl6agn['survey']=="BPT_Dominika ")[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\Delta \overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     if k!=1:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     else:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_BPT_Dominika_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# idx = np.where(np.logical_or(mpl6agn['survey']=="BPT_Dominika ", mpl6agn['survey']=="BPT_rembold  "))[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\Delta \overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     if k !=1:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     else:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_BPT_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# idx = np.where(mpl6agn['survey']=="2XMMi        ")[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\Delta \overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     if k!=1:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     else:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_2XMMi_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# idx = np.where(mpl6agn['survey']=="ALLWISEAGN   ")[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     for m in idx:
#         mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#         plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#         av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_ALLWISEAGN_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# idx = np.where(mpl6agn['survey']=="EdelsonMalkan")[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\Delta \overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     if k !=1:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     else:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_EdelsonMalkan_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()


# idx = np.where(mpl6agn['survey']=="RASSDSSAGN   ")[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     for m in idx:
#         mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#         plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#         av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     plt.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/median_'+labels[k]+'_with_radius_mpl6agn_RASSDSSAGN_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# idx = np.where(np.logical_or(mpl6agn['survey']=="RASSDSSAGN   ", np.logical_or(mpl6agn['survey']=="EdelsonMalkan", mpl6agn['survey']=="2XMMi        ")))[0]
# #av_i = np.empty((len(mid_bins), len(av_bins)))
# labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
# ax_labels = [r'$\overline{Z}$',r'$\Delta \overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
# for k in range(len(labels)):
#     plt.figure()
#     av_i = np.empty((len(mid_bins), len(av_bins)))
#     if k!=1:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#     else:
#         for m in idx:
#             mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#             plt.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#             av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#     plt.plot(av_bins, np.nanmean(av_i[idx,:], axis=0), color='r', alpha=0.9)
#     plt.xlabel(r'$R/R_e$')
#     plt.ylabel(ax_labels[k])
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig('figures/mean_'+labels[k]+'_with_radius_mpl6agn_xray_masked_big_error.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#     plt.close()

# r_absmag = np.array(mpl6agn['nsa_elpetro_absmag'])[:,-3]
# maglim = np.percentile(r_absmag, [0, 33, 66, 100])
# maglabels = [r'Brightest Galaxies', r'Middle brightness', r'Faintest galaxies']
# for j in range(1, 4):
#     idx = np.where(np.logical_and(r_absmag > maglim[j-1], r_absmag < maglim[j]))[0]
#     #av_i = np.empty((len(mid_bins), len(av_bins)))
#     labels = ['inferred Z', 'inferred delta tq', 'inferred tau']
#     ax_labels = [r'$\overline{Z}$',r'$\Delta \overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
#     for k in range(len(labels)):
#         plt.figure()
#         ax = plt.subplot(111)
#         av_i = np.empty((len(mid_bins), len(av_bins)))
#         if k != 1:
#             for m in idx:
#                 mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#                 ax.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#                 av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#         else:
#             for m in idx:
#                 mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#                 ax.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#                 av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#         ax.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#         ax.text(0.1, 0.9, maglabels[j-1], fontsize=12, transform=ax.transAxes)
#         ax.set_xlabel(r'$R/R_e$')
#         ax.set_ylabel(ax_labels[k])
#         ax.set_xlim(0, 1.0)
#         plt.tight_layout()
#         plt.savefig('figures/mean_'+labels[k]+'_with_radius_mpl6agn_elpetro_abs_rmag_gtr_'+str(maglim[j-1])+'_lt_'+str(maglim[j])+'.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#         plt.close()


# deltatq = (Planck15.age(mpl6agn['z']).reshape(360,1)*np.ones((1, 15))).value - mpl6agn['inferred tq']

# btr = mpl6agn['(B/T)r']
# maglim = np.nanpercentile(btr, [0, 33, 66, 100])
# maglabels = [r'Disk domianted', r'Disk+Bulge', r'Bulge dominated']
# for j in range(1, 4):
#     idx = np.where(np.logical_and(btr > maglim[j-1], btr < maglim[j]))[0]
#     #av_i = np.empty((len(mid_bins), len(av_bins)))
#     labels = ['inferred Z', 'inferred tq', 'inferred tau']
#     ax_labels = [r'$\overline{Z}$',r'$\Delta\overline{t_q}$',r'$\log_{10}$ $\overline{\tau}$']
#     for k in range(len(labels)):
#         plt.figure()
#         ax = plt.subplot(111)
#         av_i = np.empty((len(mid_bins), len(av_bins)))
#         if k != 1:
#             for m in idx:
#                 mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
#                 ax.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
#                 av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
#         else:
#             for m in idx:
#                 mbins =  mid_bins[m][np.invert(deltatq[m].mask[:len(mid_bins[m])].astype(bool))]
#                 ax.plot(mbins, np.ma.compressed(deltatq[m]), color='k', alpha=0.1)
#                 av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(deltatq[m]))
#                 #ax.set_ylim(-0.2, 0.5)
#         ax.plot(av_bins, np.nanmedian(av_i[idx,:], axis=0), color='r', alpha=0.9)
#         ax.text(0.1, 0.9, maglabels[j-1], fontsize=12, transform=ax.transAxes)
#         ax.set_xlabel(r'$R/R_e$')
#         ax.set_ylabel(ax_labels[k])
#         ax.set_xlim(0, 1.0)
#         plt.tight_layout()
#         if k != 1:
#             plt.savefig('figures/mean_'+labels[k]+'_with_radius_mpl6agn_BTr_gt'+str(maglim[j-1])+'_lt_'+str(maglim[j])+'.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#         else:
#             plt.savefig('figures/mean_deltatq_with_radius_mpl6agn_BTr_gt'+str(maglim[j-1])+'_lt_'+str(maglim[j])+'.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
#         plt.close()

