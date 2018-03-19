from manga_posterior_log import *
from astropy.cosmology import Planck15
import numpy as np
import sys
import corner
import matplotlib.pyplot as plt
import sys

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
    #ax2.axvline(x=truth[0], c='r', linewidth=1)
    #ax2.axhline(y=truth[1], c='r', linewidth=1)
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
    #ax1.axvline(x=truth[0], c='r', linewidth=1)
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
    #ax3.axhline(y=truth[1], c='r', linewidth=1)
    #P.tight_layout()
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
    #ax1.axhline(truth[0], color='r')
    ax2.plot(s[:,:,1].T, 'k')
    #ax2.axhline(truth[1], color='r')
    ax3.plot(s[:,:,2].T, 'k')
    #ax3.axhline(truth[2], color='r')
    ax1.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='x', labelbottom='off')
    ax3.set_xlabel(r'step number')
    ax1.set_ylabel(r'$Z$')
    ax2.set_ylabel(r'$t_{quench}$')
    ax3.set_ylabel(r'$\tau$')
    P.subplots_adjust(hspace=0.1)
    #save_fig = './test_log_one/walkers_steps_'+str(ID)+'_log.pdf'
    #fig.savefig(save_fig)
    return fig

age = 13.64204

time_steps = Planck15.age(10**np.linspace(-0.824, -2.268, 20)).reshape(-1,1,1).value

emlinesdata = np.load('emlines_data.npy')
indx_names = np.load('index_names.npy')

consteml = np.load('emission_line_params_constsfh_3.npy')
constidm = np.load('abs_index_params_constsfh_3.npy')

constews =  np.zeros(len(consteml)*len(emlinesdata['name'])).reshape(len(consteml), len(emlinesdata['name']))
constidms =  np.zeros(len(constidm)*len(indx_names)).reshape(len(constidm),len(indx_names))

for n in range(len(consteml)):
    constews[n,:] = consteml[n]['EW']
    constidms[n,:] = constidm[n]['INDX']

const_ews_preds = constews[:, np.logical_or(emlinesdata['name']=='Ha', emlinesdata['restwave']==3727.092)] # Halpha 0th and OII 1st 
const_ind_preds = constidms[:, np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))]

const_preds = np.append(const_ews_preds, const_ind_preds, axis=1)[-5]

bursteml = np.load('emission_line_params_burstsfh_tb_12.1_Hb_10.npy')
burstidm = np.load('abs_index_params_burstsfh_tb_12.1_Hb_10.npy')

burstews =  np.zeros(len(bursteml)*len(emlinesdata['name'])).reshape(len(bursteml), len(emlinesdata['name']))
burstidms =  np.zeros(len(burstidm)*len(indx_names)).reshape(len(burstidm),len(indx_names))

for n in range(len(bursteml)):
    burstews[n,:] = bursteml[n]['EW']
    burstidms[n,:] = burstidm[n]['INDX']

burst_ews_preds = burstews[:, np.logical_or(emlinesdata['name']=='Ha', emlinesdata['restwave']==3727.092)] # Halpha 0th and OII 1st 
burst_ind_preds = burstidms[:, np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))]

burst_preds = np.append(burst_ews_preds, burst_ind_preds, axis=1)[-5]


manybeml = np.load('emission_line_params_many_burst_mu_12.6_s_1.5.npy')
manybidm = np.load('abs_index_params_many_burst_mu_12.6_s_1.5.npy')

manybews =  np.zeros(len(manybeml)*len(emlinesdata['name'])).reshape(len(manybeml), len(emlinesdata['name']))
manybidms =  np.zeros(len(manybidm)*len(indx_names)).reshape(len(manybidm),len(indx_names))

for n in range(len(manybeml)):
    manybews[n,:] = manybeml[n]['EW']
    manybidms[n,:] = manybidm[n]['INDX']

manyb_ews_preds = manybews[:, np.logical_or(emlinesdata['name']=='Ha', emlinesdata['restwave']==3727.092)] # Halpha 0th and OII 1st 
manyb_ind_preds = manybidms[:, np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))]

manyb_preds = np.append(manyb_ews_preds, manyb_ind_preds, axis=1)[-5]


lognormeml = np.load('emission_line_params_lognormsfh_mu_log12.6_s_0.3.npy')
lognormidm = np.load('abs_index_params_lognormsfh_mu_log12.6_s_0.3.npy')

lognormews =  np.zeros(len(lognormeml)*len(emlinesdata['name'])).reshape(len(lognormeml), len(emlinesdata['name']))
lognormidms =  np.zeros(len(lognormidm)*len(indx_names)).reshape(len(lognormidm),len(indx_names))

for n in range(len(lognormeml)):
    lognormews[n,:] = lognormeml[n]['EW']
    lognormidms[n,:] = lognormidm[n]['INDX']

lognorm_ews_preds = lognormews[:, np.logical_or(emlinesdata['name']=='Ha', emlinesdata['restwave']==3727.092)] # Halpha 0th and OII 1st 
lognorm_ind_preds = lognormidms[:, np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))]

lognorm_preds = np.append(lognorm_ews_preds, lognorm_ind_preds, axis=1)[-5]


normeml = np.load('emission_line_params_normsfh_mu_12.6_s_1.5.npy')
normidm = np.load('abs_index_params_normsfh_mu_12.6_s_1.5.npy')

normews =  np.zeros(len(normeml)*len(emlinesdata['name'])).reshape(len(normeml), len(emlinesdata['name']))
normidms =  np.zeros(len(normidm)*len(indx_names)).reshape(len(normidm),len(indx_names))

for n in range(len(normeml)):
    normews[n,:] = normeml[n]['EW']
    normidms[n,:] = normidm[n]['INDX']

norm_ews_preds = normews[:, np.logical_or(emlinesdata['name']=='Ha', emlinesdata['restwave']==3727.092)] # Halpha 0th and OII 1st 
norm_ind_preds = normidms[:, np.logical_or(np.logical_or(indx_names=='D4000', indx_names=='Fe5270'), np.logical_or(np.logical_or(indx_names=='Fe5335', indx_names=='HDeltaA'),  np.logical_or(indx_names=='Hb', indx_names=='Mgb')))]

norm_preds = np.append(norm_ews_preds, norm_ind_preds, axis=1)[-5]

order = ['constant', 'burst', 'many burst', 'log-normal', 'normal']

preds = np.vstack([const_preds, burst_preds, manyb_preds, lognorm_preds, norm_preds])

spec = np.zeros(len(preds)*6).reshape(len(preds),6)
spec[:,0] = preds[:,0]
spec[:,1] = preds[:,1]
spec[:,2] = preds[:,7]
spec[:,3] = preds[:,2]
spec[:,4] = preds[:,6]
spec[:,5] = np.sqrt( preds[:,3] * ( 0.72*preds[:,4] + 0.28*preds[:,5] )  )

errs = np.array([0.008, 0.24, 0.003, 0.88, 0.14, 0.02]).reshape(1,spec.shape[1]) #these are mean errors from mangadap-7495-12704-LOGCUBE

specerr = errs*spec

# Define parameters needed for emcee 
nwalkers = 100 # number of monte carlo chains
nsteps= 100 # number of steps in the monte carlo chain
opstart = [1.0, 7.5, 1.75] # starting place of all the chains
burnin = 500 # number of steps in the burn in phase of the monte carlo chain
ndim = 3

specs = np.append(spec, np.zeros(len(spec)*9).reshape(len(spec),9), axis=-1).reshape(len(spec),-1)


nll = lambda *args: -lnprob(*args)
from scipy.optimize import minimize, basinhopping

start_bh = np.zeros(len(opstart)*len(spec)).reshape(len(spec),len(opstart))


for n in range(len(spec)):
    result_bh = basinhopping(nll, opstart, minimizer_kwargs={"args": (spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age), "method":'Nelder-Mead'}, stepsize=1)
   #result_nm = minimize(nll, opstart, args=(spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age), method='Nelder-Mead')
    if "successfully" in result_bh.message[0]:
        start_bh[n,:] = result_bh['x']
    else:
        start_bh[n,:] = np.array(opstart)


#The rest calls the emcee module which is initialised in the sample function of the posterior file. 
for n in range(1,len(spec)):
    print('starting run number: ', n,' which is the ', order[n], ' model')
    samples, samples_save, af, act = sample(ndim, nwalkers, nsteps, burnin, list(start_bh[n,:]), spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age, n)

    lnp = np.load('./test_diff_SFH/lnprob_run_'+str(n)+'.npy').reshape(nwalkers, nsteps)
    lk = np.mean(lnp, axis=1)
    idxs = np.argsort(-lk)
    slk = -lk[idxs]
    cluster_idx = np.argmax(np.diff(slk) >  10000*np.diff(slk)[0]/ (np.linspace(1, len(slk)-1, len(slk)-1)-1))+1
    if cluster_idx > 1:
        lnps = slk[:cluster_idx]
        samples = samples.reshape(nwalkers, nsteps, ndim)[idxs,:,:][:cluster_idx,:,:].reshape(-1,ndim)
    else:
        pass
                
    Z_mcmc, tq_mcmc, tau_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0)))
    specs[n, 6] = Z_mcmc[0]
    specs[n, 7] = Z_mcmc[1]
    specs[n, 8] = Z_mcmc[2]
    specs[n, 9] = tq_mcmc[0]
    specs[n, 10] = tq_mcmc[1]
    specs[n, 11] = tq_mcmc[2]
    specs[n, 12] = tau_mcmc[0]
    specs[n, 13] = tau_mcmc[1]
    specs[n, 14] = tau_mcmc[2]

    #fig = corner_plot(samples, labels = [r'$ t_{quench}$', r'$ \tau$'], extents=[[np.min(samples[:,0]), np.max(samples[:,0])],[np.min(samples[:,1]),np.max(samples[:,1])]], bf=[tq_mcmc, tau_mcmc])
    fig = corner_plot(s=samples[:,1:], labels=[r'$t_q$', r'$\tau$'], extents=[[0, 13.8], [0, 4]], bf=[(specs[n,8], specs[n,9], specs[n,10]), (specs[n,11], specs[n,12], specs[n,13])], truth=None)
    fig.savefig('./test_diff_SFH/starpy_output_testing_one_pruning_'+str(n)+'_'+order[n]+'.png')

    fig = corner.corner(samples, labels=[r'$Z$', r'$t_q$', r'$\tau$'], quantiles=([0.16, 0.5, 0.84]))    
    fig.savefig('./test_diff_SFH/starpy_output_corner_testing_one_pruning_'+str(n)+'_'+order[n]+'.png')

    fig = walker_plot(samples, nwalkers, -1, np.nan, n)
    fig.savefig('./test_diff_SFH/walkers_steps_pruning_'+str(n)+'_'+order[n]+'.pdf')

    burninload = np.load('./test_diff_SFH/samples_burn_in_'+str(n)+'.npy')
    fig = walker_plot(burninload, nwalkers, -1, np.nan, n)
    fig.savefig('./test_diff_SFH/walkers_steps_burn_in_wihtout_pruning_'+str(n)+'_'+order[n]+'.pdf')

    plt.close('all')
    print('Best fit [Z, t, tau] values found by starpy for ', order[n], ' are : [', Z_mcmc[0], tq_mcmc[0], tau_mcmc[0], ']')

np.save('test_manga_starpy_results_log.npy', specs)

ages = np.flip(13.805 - 10**(np.linspace(7, 10.14, 100))/1e9, axis=0)

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

x = np.linspace(0, 14, 200)

csfr = constsfh(3.0, x)
bsfh = burstsfh(12.1, 10, x)

lnsfh = normsfh(np.log10(12.6e9), 0.3, np.log10(x*1e9))
nsfh = normsfh(12.6, 1.5, x)
mbsfh = burstsfh(4.0, 10.0, x) + burstsfh(7.5, 3.0, x) + burstsfh(11.3, 11.0, x) + burstsfh(13.5, 2.0, x)

asfhs = list([csfr, bsfh, mbsfh, lnsfh, nsfh])

sfhs = np.zeros(len(specs)*len(ages)).reshape(len(specs), len(ages))
for n in range(len(specs)):
    sfhs[n,:] = expsfh(specs[n,9], specs[n,12], ages)


for n in range(len(sfhs)):
    plt.figure()
    plt.plot(ages, (sfhs[n,:])/(np.max(sfhs[n,:])), color='b', label='inferred')
    plt.plot(x, (asfhs[n])/(np.max(asfhs[n])), color='k', label=order[n])
    plt.legend(frameon=False)
    plt.xlabel('time [Gyr]')
    plt.ylabel(r'normalised SFR')
    plt.ylim(0,1.1)
    plt.tight_layout()
    plt.savefig('./test_diff_SFH/compare_actual_inferred_different_SFHs_'+order[n]+'.png')



