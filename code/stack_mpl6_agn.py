import os
import time
import numpy 
import numpy as np

from astropy.table import Table, Column

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

import requests
from requests.auth import HTTPBasicAuth

import sys
sys.path.append('/Users/becky/Projects/mangaagn/snitch/snitch/')

from snitch import *
from functions import *
from astropy.cosmology import Planck15

from tqdm import tqdm, trange

import corner

sdss_verify = open('sdss_user_pass.txt')
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
    ax1.axhline(truth[0], color='r')
    ax2.plot(s[:,:,1].T, 'k')
    ax2.axhline(truth[1], color='r')
    ax3.plot(s[:,:,2].T, 'k')
    ax3.axhline(truth[2], color='r')
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

mpl6agn = Table.read('../data/unique_EM_ALLWISEAGN_RASSDSSAGN_2XMMi_Xray_limited_1042_AGN_rembold_type2_with_mpa_jhu_measurement.fits', format='fits')

mid_bins = []
emls = []
idms = []
emls_error = []
idms_error = []
masks= []

Z_mcmc = Column(name='inferred Z', data = np.empty([len(mpl6agn), 11]))
Z_p_mcmc = Column(name='inferred Z plus error', data = np.empty([len(mpl6agn), 11]))
Z_m_mcmc = Column(name='inferred Z minus error', data = np.empty([len(mpl6agn), 11]))


tq_mcmc = Column(name='inferred tq', data = np.empty([len(mpl6agn), 11]))
tq_p_mcmc = Column(name='inferred tq plus error', data = np.empty([len(mpl6agn), 11]))
tq_m_mcmc = Column(name='inferred tq minus error', data = np.empty([len(mpl6agn), 11]))

tau_mcmc = Column(name='inferred tau', data = np.empty([len(mpl6agn), 11]))
tau_p_mcmc = Column(name='inferred tau plus error', data = np.empty([len(mpl6agn), 11]))
tau_m_mcmc = Column(name='inferred tau minus error', data = np.empty([len(mpl6agn), 11]))

mpl6agn.add_columns([Z_mcmc, Z_p_mcmc, Z_m_mcmc, tq_mcmc, tq_p_mcmc, tq_m_mcmc, tau_mcmc, tau_p_mcmc, tau_m_mcmc])

nll = lambda *args: -lnprob(*args)
from scipy.optimize import minimize, basinhopping

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, hard))

nwalkers = 100 # number of monte carlo chains
nsteps= 100 # number of steps in the monte carlo chain
opstart = [0.7, 9.0, 1.25] # starting place of all the chains
burnin = 500 # number of steps in the burn in phase of the monte carlo chain
ndim = 3

ages = Planck15.age(mpl6agn['nsa_z']).value

define_em_db = SpectralFeatureDBDef(key='USEREM',
                              file_path='elpsnitch.par')
emlines  = EmissionLineDB(u"USEREM", emldb_list=define_em_db)
define_abs_db = SpectralFeatureDBDef(key='USERABS',
                              file_path='extindxsnitch.par')
abs_db = AbsorptionIndexDB(u"USERABS", indxdb_list=define_abs_db)
band_db = BandheadIndexDB(u"BHBASIC")
indx_names = np.hstack([abs_db.data["name"], band_db.data["name"]])

for n in trange(len(mpl6agn)):
    if os.path.isfile('../data/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGCUBE.fits.gz'):
    #     pass
    # else: 
    #     r = requests.get(top_level_url+str(mpl6agn[n]['plate'])+'/stack/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGCUBE.fits.gz', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)
    #     if r.ok:
    #         with open('../data/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGCUBE.fits.gz', 'wb') as file:
    #             file.write(r.content) 
    #     s = requests.get(top_level_url+str(mpl6agn[n]['plate'])+'/stack/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGRSS.fits.gz', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)
    #     if s.ok:
    #         with open('../data/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGRSS.fits.gz', 'wb') as sfile:
    #             sfile.write(s.content)
    #     p = requests.get(top_level_url_par+str(mpl6agn[n]['plate'])+'/'+str(mpl6agn[n]['ifudsgn'].strip())+'/ref/mangadap-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGCUBE-input.par', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)         
    #     if p.ok:
    #         with open('../data/mangadap-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGCUBE-input.par', 'wb') as pfile:
    #             pfile.write(p.content)
    #parser = ArgumentParser()

    #parser.add_argument('plt', type=str, help='plate')
    #parser.add_argument('ifudsgn', type=str, help='ifudsgn')
    #parser.add_argument('obs', type=str, help='Observational parameters file')

    #arg = parser.parse_args()
        if os.path.isfile('../data/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGRSS.fits.gz') is False:
            s = requests.get(top_level_url+str(mpl6agn[n]['plate'])+'/stack/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGRSS.fits.gz', auth=HTTPBasicAuth(up[0].rstrip('\n'), up[1].rstrip('\n')), stream=True)
            if s.ok:
                with open('../data/manga-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGRSS.fits.gz', 'wb') as sfile:
                    sfile.write(s.content)
        else:
            pass

        print('starting analysis of ', str(mpl6agn[n]['plateifu']), ' which is ', str(100*(n/len(mpl6agn))),'% of the way through')
        drpf = DRPFits(mpl6agn[n]['plate'], mpl6agn[n]['ifudsgn'].strip(), 'CUBE', read=True, directory_path='../data/')
        print(drpf.file_path())
        x,y = drpf.mean_sky_coordinates(offset=True)

        par_file = '../data/mangadap-'+str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'-LOGCUBE-input.par'

        obsp = ObsInputPar.from_par_file(par_file)

        if '19' in mpl6agn[n]['ifudsgn'].strip():
            nbins = 3
        elif '37' in mpl6agn[n]['ifudsgn'].strip():
            nbins = 5
        elif '61' in mpl6agn[n]['ifudsgn'].strip():
            nbins = 7
        elif '91' in mpl6agn[n]['ifudsgn'].strip():
            nbins = 9
        else:
            nbins = 11

        mask = np.append(np.zeros(nbins), np.ones(11-nbins))
        if mpl6agn['survey'][n]=='BPT_rembold  ':
            pass
        else:
            mask[0] = 1
        masks.append(mask)

        mpl6agn['inferred Z'][n].mask = mask
        mpl6agn['inferred Z plus error'][n].mask = mask
        mpl6agn['inferred Z minus error'][n].mask = mask
        mpl6agn['inferred tq'][n].mask = mask
        mpl6agn['inferred tq plus error'][n].mask = mask
        mpl6agn['inferred tq minus error'][n].mask = mask
        mpl6agn['inferred tau'][n].mask = mask
        mpl6agn['inferred tau plus error'][n].mask = mask
        mpl6agn['inferred tau minus error'][n].mask = mask

        rbin = RadialBinning(par=RadialBinningPar(center=[0.0,0.0], pa=obsp['pa'], ell=obsp['ell'], radius_scale=obsp['reff'],
                                                  radii=[0.0, -1, nbins], log_step=False))
        binid = rbin.bin_index(x,y)
        r, theta = rbin.sma_coo.polar(x, y)

        bins = numpy.linspace(0, rbin.par['radii'][1]/rbin.par['radius_scale'], nbins+1)
        mid_bins.append((bins[:-1]+(numpy.diff(bins)/2.)))


        stack = SpectralStack()
        stack_wave, stack_flux, stack_sdev, stack_npix, stack_ivar, stack_sres, stack_covar = stack.stack_DRPFits(drpf, binid, par=SpectralStackPar('mean', False, None, 'channels', SpectralStack.parse_covariance_parameters('channels', '11'), None))

        bins = numpy.linspace(0, rbin.par['radii'][1]/rbin.par['radius_scale'], nbins+1)
        mid_bins.append((bins[:-1]+(numpy.diff(bins)/2.)))

        
        em_model_eml_par, indx_measurements = measure_spec(stack_flux, errors=stack_sdev, ivar=stack_ivar, sres=stack_sres)

        emls.append(em_model_eml_par["EW"][:, np.where(emlines['name']=='Ha')].reshape(-1,1))
        idms.append(indx_measurements["INDX"].reshape(-1,8))
        
        emls_error.append(em_model_eml_par["EWERR"][:, np.where(emlines['name']=='Ha')].reshape(-1,1))
        idms_error.append(indx_measurements["INDXERR"].reshape(-1,8))

        np.save(str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'_mpl6agn_measured_emls.npy', emls)
        np.save(str(mpl6agn[n]['plate'])+'-'+str(mpl6agn[n]['ifudsgn'].strip())+'_mpl6agn_measured_idms.npy', idms)

        av_bins = np.linspace(0, 1.0, 15)
        #av_emls = np.empty((len(av_bins), len(emls)))
        labels=[r'H$\alpha$', r'$[OII]$']
        lims=[[-1, 80],[-1, 25]]
        for k in range(1):
            plt.figure()
            plt.plot(mid_bins[n], emls[n][:,k], color='k', alpha=0.1)
            av_emls = np.interp(av_bins, mid_bins[n], emls[n][:,k])
            plt.plot(av_bins, av_emls, color='r')
            plt.xlabel(r'$R/R_e$')
            plt.xlim(0,1)
            plt.ylim(lims[k])
            plt.ylabel(labels[k])
            plt.tight_layout()
            plt.savefig('change_in_'+labels[k]+'_with_radius_mpl6agn.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

        labels=[r'H$\beta$', r'$MgB$', r'$Fe5270$', r'$Fe5335$', r'$H\delta_A$', r'D4000', r'Dn4000', r'TiO']
        lims=[[-2, 4],[0,0],[0,0],[0,0],[-3, 5],[-0.5, 2.5],[-0.5, 2.5]]
        #av_idms = np.empty((len(mid_bins), len(av_bins)))
        indx = [0, 4, 6]
        for k in indx:
            plt.figure()
            plt.plot(mid_bins[n], idms[n][:,k], color='k', alpha=0.1)
            av_idms = np.interp(av_bins, mid_bins[n], idms[n][:,k])
            plt.plot(av_bins, av_idms, color='r')
            plt.xlabel(r'$R/R_e$')
            plt.xlim(0,1)
            plt.ylim(lims[k])
            plt.ylabel(labels[k])
            plt.tight_layout()
            plt.savefig('change_in_'+labels[k]+'_with_radius_mpl6agn.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

        av_mgfe = np.empty((len(mid_bins), len(av_bins)))
        plt.figure()
        mgfe = np.sqrt(idms[n][:,1] * ( 0.72*idms[n][:,2] + 0.28*idms[n][:,3]))
        plt.plot(mid_bins[n], mgfe, color='k', alpha=0.1)
        av_mgfe = np.interp(av_bins, mid_bins[n], mgfe)
        plt.plot(av_bins, av_mgfe, color='r')
        plt.xlabel(r'$R/R_e$')
        plt.ylabel(r'$\rm{MgFe}$')
        plt.xlim(0,1)
        plt.ylim(-1, 10)
        plt.tight_layout()
        plt.savefig('change_in_mgfe_with_radius_mpl6agn.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

        start_bh = np.zeros(nbins*ndim).reshape(-1,ndim)


        for j in range(nbins):
            mgfe = np.sqrt(idms[n][j,1] * ( 0.72*idms[n][j,2] + 0.28*idms[n][j,3]))
            Zs_mcmc, tq_mcmc, tau_mcmc = snitch(emls[n][j,0], emls_error[n][j,0], idms[n][j,6], idms_error[n][j,6], idms[n][j,0], idms_error[n][j,0],idms[n][j,4], idms_error[n][j,4], mgfe, 0.1*mgfe, redshift=mpl6agn['nsa_z'][n], ident=mpl6agn['plateifu'][n]+'_'+str(format(mid_bins[n][j], '.2f')) )

            
            mpl6agn['inferred Z plus error'][n, j] = Zs_mcmc[1]
            mpl6agn['inferred Z minus error'][n, j] = Zs_mcmc[2]
            mpl6agn['inferred tq'][n, j] = tqs_mcmc[0]
            mpl6agn['inferred tq plus error'][n, j] = tqs_mcmc[1]
            mpl6agn['inferred tq minus error'][n, j] = tqs_mcmc[2]
            mpl6agn['inferred tau'][n, j] = taus_mcmc[0]
            mpl6agn['inferred tau plus error'][n, j] = taus_mcmc[1]
            mpl6agn['inferred tau minus error'][n, j] = taus_mcmc[2]

            mpl6agn.write('mpl6agn_inferred_snitch.fits', format='fits', overwrite=True)

            plt.close('all')
            print('Best fit [Z, t, tau] values found by starpy for input parameters are : [', Zs_mcmc[0], tqs_mcmc[0], taus_mcmc[0], ']')
            
        labels = ['inferred Z', 'inferred tq', 'inferred tau']
        ax_labels = [r'$\overline{Z}$',r'$\overline{t_q}$',r'$\overline{\tau}$']
        for k in range(len(labels)):
            plt.figure()
            av_i = np.empty((len(mid_bins), len(av_bins)))
            for m in range(len(mid_bins)):
                mbins =  mid_bins[m][np.invert(mpl6agn[labels[k]][m].mask[:len(mid_bins[m])].astype(bool))]
                plt.plot(mbins, np.ma.compressed(mpl6agn[labels[k]][m]), color='k', alpha=0.1)
                av_i[m,:] = np.interp(av_bins, mbins, np.ma.compressed(mpl6agn[labels[k]][m]))
            plt.plot(av_bins, np.median(av_i[:m+1,:], axis=0), color='r')
            plt.xlabel(r'$R/R_e$')
            plt.ylabel(ax_labels[k])
            plt.xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig('median_'+labels[k]+'_with_radius_mpl6agn'+mpl6agn['plateifu'][n]+'.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
            plt.close()

        np.save('bins_for_plotting_mpl6agn_stack'+mpl6agn['plateifu'][n]+'.npy', mid_bins)

    else:
        print('nothing for ', str(mpl6agn[n]['plateifu']))



