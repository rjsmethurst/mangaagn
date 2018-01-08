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
    #save_fig = './test_log_one/walkers_steps_'+str(ID)+'_log.pdf'
    #fig.savefig(save_fig)
    return fig


age = 13.63

#tq = (13.715 - 3.0) * np.random.random_sample(25) + 3.0
#tq = np.array([13.2, 13.3, 13.4, 13.5, 13.6])
#tau = (3.5- 0.01) * np.random.random_sample(25) + 0.01
#tau = np.array([0.025, 0.075, 0.125, 0.175, 0.225])
Z = (1.23 - 0.1) * np.random.random_sample(25) + 0.1
#Z = np.array([0.79, 0.84, 0.89, 0.94, 0.99])
tq = np.linspace(13.249, 13.249, 1)
tau =  np.linspace(0.312, 0.312, 1)

k = np.array([list(product(Z, tq, tau))]).reshape(25,3)
#k = np.array([Z, tq, tau]).T
#k = np.append(Z.reshape(-1,1), np.array(list(product(tq, tau))), axis=1)


spec = np.zeros(len(k)*6).reshape(len(k), 6)

for j in range(len(k)):
    spec[j,:] = predict_spec_one([k[j,0], k[j,1], k[j,2]], age)


errs = np.array([0.008, 0.24, 0.003, 0.88, 0.14, 0.02]).reshape(1,spec.shape[1]) #these are mean errors from mangadap-7495-12704-LOGCUBE

specerr = errs*spec

# Define parameters needed for emcee 
nwalkers = 100 # number of monte carlo chains
nsteps= 100 # number of steps in the monte carlo chain
opstart = [0.85, 7.5, 1.75] # starting place of all the chains
burnin = 500 # number of steps in the burn in phase of the monte carlo chain
ndim = 3

specs = np.append(spec, np.zeros(len(spec)*9).reshape(len(spec),9), axis=-1).reshape(len(spec),-1)


nll = lambda *args: -lnprob(*args)
from scipy.optimize import minimize, basinhopping

start_bh = np.zeros_like(k)
start_nm = np.zeros_like(k)


for n in range(len(spec)):
    result_bh = basinhopping(nll, opstart, minimizer_kwargs={"args": (spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age), "method":'Nelder-Mead'}, stepsize=1)
   #result_nm = minimize(nll, opstart, args=(spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age), method='Nelder-Mead')
    if "successfully" in result_bh.message[0]:
        start_bh[n,:] = result_bh['x']
    else:
        start_bh[n,:] = np.array(opstart)

    # if result_nm['success']==True :
    #     start_nm[n,:] = result_nm['x']
    # else:
    #     start_nm[n,:] = np.array(opstart)

plt.figure()
plt.scatter(k[:,0], start_bh[:,0])
plt.plot(np.linspace(0.75, 1.05, 10), np.linspace(0.75,1.05, 10), color='r')
plt.savefig('./test_log_one/Z_opt_minimize_start_bh_point.pdf')

# plt.figure()
# plt.scatter(k[:,1], start_bh[:,1])
# plt.plot(np.linspace(0, 13.8, 10), np.linspace(0,13.8, 10), color='r')
# plt.savefig('./test_log_one/tq_opt_minimize_start_bh_point.pdf')

# plt.figure()
# plt.scatter(k[:,2], start_bh[:,2])
# plt.plot(np.linspace(0, 4.0, 10), np.linspace(0,4.0, 10), color='r')
# plt.savefig('./test_log_one/tau_opt_minimize_start_bh_point.pdf')

# plt.figure()
# plt.scatter(k[:,0], start_bh[:,0], color='b', label='basinhopping')
# plt.scatter(k[:,0], start_nm[:,0], color='r', label='Nelder-Mead')
# for n in range(len(k[:,0])):
#     plt.plot([k[n,0], k[n,0]], [start_bh[n,0], start_nm[n,0]], color='k', alpha=0.3)
# plt.plot(np.linspace(0.75, 1.05, 10), np.linspace(0.75,1.05, 10), color='k')
# plt.legend(frameon=False)
# plt.savefig('./Z_opt_minimize_basinhopping_vs_NM_log.pdf')

# plt.figure()
# plt.scatter(k[:,1], start_bh[:,1], color='b', label='basinhopping')
# plt.scatter(k[:,1], start_nm[:,1], color='r', label='Nelder-Mead')
# for n in range(len(k[:,1])):
#     plt.plot([k[n,1], k[n,1]], [start_bh[n,1], start_nm[n,1]], color='k', alpha=0.3)
# plt.plot(np.linspace(3.0, 13.8, 10), np.linspace(3.0,13.8, 10), color='k')
# plt.legend(frameon=False)
# plt.savefig('./tq_opt_minimize_basinhopping_vs_NM_log.pdf')

# plt.figure() 
# plt.scatter(k[:,2], start_bh[:,2], color='b', label='basinhopping')
# plt.scatter(k[:,2], start_nm[:,2], color='r', label='Nelder-Mead')
# for n in range(len(k[:,2])):
#     plt.plot([k[n,2], k[n,2]], [start_bh[n,2], start_nm[n,2]], color='k', alpha=0.3)
# plt.plot(np.linspace(0, 4.0, 10), np.linspace(0, 4.0, 10), color='k')
# plt.legend(frameon=False)
# plt.savefig('./tau_opt_minimize_basinhopping_vs_NM_log.pdf')

# md_bh_Z = (np.abs((k[:,0]-start_bh[:,0])))
# md_nm_Z = (np.abs((k[:,0]-start_nm[:,0])))

# md_bh_tq = (np.abs((k[:,1]-start_bh[:,1])))
# md_nm_tq = (np.abs((k[:,1]-start_nm[:,1])))

# md_bh_tau = (np.abs((k[:,2]-start_bh[:,2])))
# md_nm_tau = (np.abs((k[:,2]-start_nm[:,2])))

# plt.figure()
# plt.hist(md_bh_Z, color='b', label='basinhopping', histtype='step')
# plt.hist(md_nm_Z, color='r', label='Nelder-Mead', histtype='step')
# plt.legend(frameon=False)
# plt.savefig('./Z_range_diff_from_known_opt_minimize_basinhopping_vs_NM_log.pdf')


# plt.figure()
# plt.hist(md_bh_tq, color='b', label='basinhopping', histtype='step')
# plt.hist(md_nm_tq, color='r', label='Nelder-Mead', histtype='step')
# plt.legend(frameon=False)
# plt.savefig('./tq_range_diff_from_known_opt_minimize_basinhopping_vs_NM_log.pdf')
# plt.figure()

# plt.hist(md_bh_tau, color='b', label='basinhopping', histtype='step')
# plt.hist(md_nm_tau, color='r', label='Nelder-Mead', histtype='step')
# plt.legend(frameon=False)
# plt.savefig('./tau_range_diff_from_known_opt_minimize_basinhopping_vs_NM_log.pdf')

# print(md_bh_Z, md_nm_Z)
# print(md_bh_tq, md_nm_tq)
# print(md_bh_tau, md_nm_tau)

#The rest calls the emcee module which is initialised in the sample function of the posterior file. 
for n in range(1, len(spec)):
    print('starting run number: ', n)
    samples, samples_save, af, act = sample(ndim, nwalkers, nsteps, burnin, list(start_bh[n,:]), spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], spec[n,5], specerr[n,5], age, n)

    # if np.nanmean(af) > 0.5 or np.nanmean(af) < 0.25 or np.mean(af)==np.nan:
    #     Z_mcmc, tq_mcmc, tau_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0)))
    #     fig = corner_plot(s=samples[:,1:], labels=[r'$t_q$', r'$\tau$'], extents=[[0, 13.8], [0, 4]], bf=[(tq_mcmc[0], tq_mcmc[1], tq_mcmc[2]), (tau_mcmc[0], tau_mcmc[1], tau_mcmc[2])], truth=[k[n,1], k[n,2]])
    #     fig.savefig('./test_log_one/starpy_output_first_run_af_not_right_testing_one_'+str(n)+'_log.png')

    #     fig = corner.corner(samples, labels=[r'$Z$', r'$t_q$', r'$\tau$'], truths=[k[n,0], k[n,1], k[n,2]], quantiles=([0.16, 0.5, 0.84]))    
    #     fig.savefig('./test_log_one/starpy_output_corner_first_run_af_not_right_testing_one_'+str(n)+'_log.png')

    #     fig = walker_plot(samples, nwalkers, -1, k[n], n)
    #     fig.savefig('./test_log_one/walkers_steps_first_run_af_not_right_'+str(n)+'_log.pdf')

    #     burninload = np.load('./test_log_one/samples_burn_in_'+str(n)+'.npy')
    #     fig = walker_plot(burninload, nwalkers, -1, k[n], n)
    #     fig.savefig('./test_log_one/walkers_steps_burn_in_first_run_af_not_right_'+str(n)+'_log.pdf')

    #     plt.close('all')
    #     print('running again...')
        
    #     samples, samples_save, af, act = sample(ndim, nwalkers*2, nsteps, burnin, list([Z_mcmc[0], tq_mcmc[0], tau_mcmc[0]]), spec[n,0], specerr[n,0], spec[n,1], specerr[n,1], spec[n,2], specerr[n,2], spec[n,3], specerr[n,3], spec[n,4], specerr[n,4], age, n)
    # else:
    #     pass

    # clustering pruning
    #samples = np.load('./test_log_one/samples_'+str(n)+'.npy')
lnp = np.load('./test_log_one/lnprob_run_'+str(n)+'.npy').reshape(nwalkers, nsteps)
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
    fig.savefig('./test_log_one/starpy_output_testing_one_pruning_'+str(n)+'_log_BH.png')

fig = corner.corner(samples, labels=[r'$Z$', r'$t_q$', r'$\tau$'], truths=[k[n,0], k[n,1], k[n,2]], truth_color='r', quantiles=([0.16, 0.5, 0.84]))    
fig.savefig('./test_log_one/starpy_output_corner_testing_one_pruning_'+str(n)+'_log_BH.png')

    fig = walker_plot(samples, nwalkers, -1, k[n], n)
    fig.savefig('./test_log_one/walkers_steps_pruning_'+str(n)+'_log_BH.pdf')

    burninload = np.load('./test_log_one/samples_burn_in_'+str(n)+'.npy')
    fig = walker_plot(burninload, nwalkers, -1, k[n], n)
    fig.savefig('./test_log_one/walkers_steps_burn_in_wihtout_pruning_'+str(n)+'_log_BH.pdf')

    plt.close('all')
    print('Best fit [Z, t, tau] values found by starpy for input parameters are : [', Z_mcmc[0], tq_mcmc[0], tau_mcmc[0], ']')
    print('Actual [Z, t, tau] values known parameters are : [', k[n,0], k[n,1], k[n,2], ']')
    print('DELTA [Z, t, tau] values are: [', k[n,0]-Z_mcmc[0], k[n,1]-tq_mcmc[0], k[n,2]-tau_mcmc[0],']')

    print('Mean acceptance fraction ', np.nanmean(af))
    print('Auto correlation time ', act)

np.save('test_manga_starpy_results_log.npy', specs)

def place_image(ax, im):
    ax.imshow(im)
    ax.tick_params(axis='x', labelbottom='off', labeltop='off', bottom='off', top='off')
    ax.tick_params(axis='y', labelleft='off', labelright='off', left='off', right='off')

n=0
for j in range(1):
    a = np.arange(25).reshape(5,5)
    fig, axes = P.subplots(nrows=5, ncols=5, figsize=(100,100), edgecolor='None')
    for row in axes:
        for ax in row:
            im = mpimg.imread('./test_log_one/starpy_output_testing_one_pruning_'+str(n)+'_log_BH.png')
            place_image(ax, im)
            n+=1
    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.subplots_adjust(hspace=0.0)
    P.savefig('./test_log_one/mosaic_spec_starpy_testing_one_log_BH.pdf')



