from __future__ import division
from common import *
import scipy.stats
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys

cm_in_inch = 0.394
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.figsize'] = (19 * cm_in_inch, 30 * cm_in_inch)
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['lines.linewidth'] = .5
mpl.rcParams['figure.subplot.hspace'] = .3


#
# ls and ms to plot for
#
seed = np.random.RandomState(34)

def format_ax(ax, l, m):
    ax.set_title(r'$\ell=%d,m=%d$' % (l, m))
    ax.set_xticklabels([])
    ax.set_yticklabels([])    

ls = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 37, 40])
tmp = []

##I, K = 12, 7
## ms = []
## lm = []
## for l in [2, 3, 4, 5, 6, 7, 8, 9]:
##     for m in range(-l, l+1):
##         lm.append((l, m))
##         if len(lm) >= I*K:
##             break
##     if len(lm) >= I*K:
##         break


def load_samples(filename):
    with locked_h5file(filename, 'r') as f:
        samples = f['samples']
        return samples[...].T

def process(name, is_microkelvin):
    samples = load_samples(out_dir('samples_%s.h5' % name))
    samples = samples[:(lmax+1)**2, :]
    if is_microkelvin:
        samples *= 1e-6
    
    mean_samples = np.mean(samples, axis=1)
    print mean_samples.shape, lmin, lmax
    mean_samples = harmonic_sphere_map(mean_samples, 0, lmax, is_complex=False)
    mean_delta = harmonic_sphere_map(mean_samples.view(np.ndarray) -
                                     mean_direct.view(np.ndarray),
                                     0, lmax, is_complex=False)

    with out_dir:
        px = mean_samples.to_pixel(Nside_plot)
        px.map2gif('mean_%s.gif' % name, title='mean_%s' % name)
        mean_delta.to_pixel(Nside_plot).map2gif('meandelta_%s.gif' % name,
                                                title='meandelta_%s' % name)

    print samples.shape, mean_direct.shape
    samples -= mean_direct[:, None]
    sigma = np.sqrt(Sigma_direct.diagonal())
    samples /= sigma[:, None]

    print 'Making plots'
    x = np.linspace(-3, 3, 100)
    fig = plt.figure()
    for i in range(0, I*K):
        l, m = lm[i]; idx = l**2 + l + m
        subz = samples[idx, :]
#        print l, m, np.mean(subz), np.std(subz), subz.shape
        ax = fig.add_subplot(I, K, i+1)
        format_ax(ax, l, m)
        ax.hist(subz, bins=15, normed=True, histtype='step')
        ax.plot(x, scipy.stats.norm.pdf(x), 'r-')
        #ax.set_xlim([-3, 3])
        fig.savefig(out_dir('var_%s.ps' % name))
    print 'Making stats'
    vars = np.zeros((lmax+1)**2, np.double)
    means = np.zeros((lmax+1)**2, np.double)
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            idx = l**2 + l + m
            subz = samples[idx, :]
            means[idx] = np.mean(subz)
            vars[idx] = np.var(subz)
    import h5py
    with h5py.File(out_dir('stats_%s.h5' % name)) as f:
        f.create_dataset('means', data=means)
        f.create_dataset('vars', data=vars)


#
# Load data
#

print 'Loading direct results'
loadvars(globals(), out_dir('direct.pickle'))

if os.path.exists(out_dir('samples_commander.h5')):
    print 'Processing Commander data'
    process('commander', is_microkelvin=True)
    
if os.path.exists(out_dir('samples_pycmb.h5')):
    print 'Processing pycmb data'
    process('pycmb', is_microkelvin=False)
    

#
# Transform data. Sigma_direct has lmin=0 so need to slice it
#

# Output mean map
    

## # Transform, plots
## use_cholesky = False
## if use_cholesky:
##     print 'Finding chol'
##     Sigma_L = np.linalg.cholesky(Sigma_direct.numpy()[4:,4:])
##     print 'done'
##     samples = np.linalg.solve(Sigma_L, samples)
## else:
