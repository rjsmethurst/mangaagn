from manga_posterior_PT import *
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
    #P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.subplots_adjust(hspace=0.0)
    return fig


def walker_plot(samples, nwalkers, nsteps, limit, truth, ID):
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
    s = samples.reshape(-1, nsteps, 3)
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
    #save_fig = './test_PT/walkers_steps_'+str(ID)+'.pdf'
    #fig.savefig(save_fig)
    return fig


age = 12.8

tq = (12.8 - 12.2) * np.random.random_sample(25) + 12.2

tau = (0.3- 0.01) * np.random.random_sample(25) + 0.01

Z = (1.26 - 0.5) * np.random.random_sample(25) + 0.5


#k = np.array([list(product(Z, tq, tau))])
k = np.array([Z, tq, tau]).T


spec = np.zeros(len(k)*6).reshape(len(k), 6)

for j in range(len(k)):
    spec[j,:] = predict_spec_one([k[j,0], k[j,1], k[j,2]], age)


errs = np.array([0.008, 0.24, 0.003, 0.88, 0.88, 0.02]).reshape(1,spec.shape[1]) #these are mean errors from mangadap-7495-12704-LOGCUBE

specerr = errs*spec

# Define parameters needed for emcee
ntemps = 10 
nwalkers = 200 # number of monte carlo chains
nsteps= 100 # number of steps in the monte carlo chain
opstart = [1.0, 12.6, 0.1] # starting place of all the chains
burnin = 500 # number of steps in the burn in phase of the monte carlo chain
ndim = 3

specs = np.append(spec, np.zeros(len(spec)*9).reshape(len(spec),9), axis=-1).reshape(len(spec),-1)


nll = lambda *args: -lnprob(*args)
from scipy.optimize import minimize

start = np.zeros_like(k)

for n in range(len(spec)):
    result = minimize(nll, opstart, args=(spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age), method='Nelder-Mead')
    if result['success']==True :
        start[n,:] = result['x']
    else:
        start[n,:] = np.array(opstart)

plt.figure()
plt.scatter(k[:,0], start[:,0])
plt.plot(np.linspace(0, 1.5, 10), np.linspace(0,1.5, 10), color='r')
plt.savefig('./test_PT/Z_opt_minimize_start_point.pdf')

plt.figure()
plt.scatter(k[:,1], start[:,1])
plt.plot(np.linspace(12.2, 12.8, 10), np.linspace(12.2,12.8, 10), color='r')
plt.savefig('./test_PT/tq_opt_minimize_start_point.pdf')

plt.figure()
plt.scatter(k[:,2], start[:,2])
plt.plot(np.linspace(0, 0.3, 10), np.linspace(0,0.3, 10), color='r')
plt.savefig('./test_PT/tau_opt_minimize_start_point.pdf')


#The rest calls the emcee module which is initialised in the sample function of the posterior file. 
for n in range(0, 5):
    print('starting run number: ', n)
    samples, samples_save = sample(ntemps, ndim, nwalkers, nsteps, burnin, list(start[n,:]), spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age, n)
    samples = samples.reshape(-1, ndim)
    #samples = np.load('./test_PT/samples_'+str(n)+'.npy').reshape(-1,ndim)
    Z_mcmc, tq_mcmc, tau_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0)))
    specs[n, 5] = Z_mcmc[0]
    specs[n, 6] = Z_mcmc[1]
    specs[n, 7] = Z_mcmc[2]
    specs[n, 8] = tq_mcmc[0]
    specs[n, 9] = tq_mcmc[1]
    specs[n, 10] = tq_mcmc[2]
    specs[n, 11] = tau_mcmc[0]
    specs[n, 12] = tau_mcmc[1]
    specs[n, 13] = tau_mcmc[2]

    #fig = corner_plot(samples, labels = [r'$ t_{quench}$', r'$ \tau$'], extents=[[np.min(samples[:,0]), np.max(samples[:,0])],[np.min(samples[:,1]),np.max(samples[:,1])]], bf=[tq_mcmc, tau_mcmc])
    fig = corner_plot(s=samples[:,1:], labels=[r'$t_q$', r'$\tau$'], extents=[[0, 13.8], [0, 4]], bf=[(specs[n,8], specs[n,9], specs[n,10]), (specs[n,11], specs[n,12], specs[n,13])], truth=[k[n,1], k[n,2]])
    fig.savefig('./test_PT/starpy_output_test_'+str(n)+'.png')

    fig = corner.corner(samples, labels=[r'$Z$', r'$t_q$', r'$\tau$'], truths=[k[n,0], k[n,1], k[n,2]], quantiles=([0.16, 0.5, 0.84]))    
    fig.savefig('./test_PT/starpy_output_corner_test_Temp_0_only_'+str(n)+'.png')

    fig = walker_plot(samples, nwalkers, nsteps, -1, k[n], n)
    fig.savefig('./test_PT/walkers_steps_'+str(n)+'.pdf')

    burninload = np.load('./test_PT/samples_burn_in_'+str(n)+'.npy').reshape(-1,ndim)
    fig = walker_plot(burninload, nwalkers, burnin, -1, k[n], n)
    fig.savefig('./test_PT/walkers_steps_burn_in_'+str(n)+'.pdf')

    plt.close('all')
    print('Best fit [Z, t, tau] values found by starpy for input parameters are : [', Z_mcmc[0], tq_mcmc[0], tau_mcmc[0], ']')
    print('Actual [Z, t, tau] values known parameters are : [', k[n,0], k[n,1], k[n,2], ']')


np.save('test_manga_starpy_results.npy', specs)

def place_image(ax, im):
    ax.imshow(im)
    ax.tick_params(axis='x', labelbottom='off', labeltop='off', bottom='off', top='off')
    ax.tick_params(axis='y', labelleft='off', labelright='off', left='off', right='off')

n=0
for j in range(len(Z)):
    a = np.arange(25).reshape(5,5)
    fig, axes = P.subplots(nrows=5, ncols=5, figsize=(100,100), edgecolor='None')
    for row in axes:
        for ax in row:
            im = mpimg.imread('./test_PT/starpy_output_test_'+str(n)+'.png')
            place_image(ax, im)
            n+=1
    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.subplots_adjust(hspace=0.0)
    P.savefig('./test_PT/mosaic_spec_starpy_test_'+str(Z[j])+'.pdf')



