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

I, K = 12, 7
ls = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 37, 40])
tmp = []
ms = []
lm = []

for l in [2, 3, 4, 5, 6, 7, 8, 9]:
    for m in range(-l, l+1):
        lm.append((l, m))
        if len(lm) >= I*K:
            break
    if len(lm) >= I*K:
        break


#
# Load data
#

print 'Loading samples'
samples = load_samples()
print 'Loading direct results'
loadvars(globals(), out_dir('direct.pickle'))

#
# Transform data. Sigma_direct has lmin=0 so need to slice it
#
assert lmin == 2
samples -= mean_direct[4:, None]

use_cholesky = False
if use_cholesky:
    print 'Finding chol'
    Sigma_L = np.linalg.cholesky(Sigma_direct.numpy()[4:,4:])
    print 'done'
    samples = np.linalg.solve(Sigma_L, samples)
else:
    sigma = np.sqrt(Sigma_direct.diagonal()[4:])
    samples /= sigma[:, None]
    

x = np.linspace(-3, 3, 100)
fig = plt.figure()
for i in range(0, I*K):
    l, m = lm[i]; idx = l**2 + l + m - lmin**2
    subz = samples[idx, :]
    print l, m, np.mean(subz), np.std(subz), subz.shape
    ax = fig.add_subplot(I, K, i+1)
    format_ax(ax, l, m)
    ax.hist(subz, bins=15, normed=True, histtype='step')
    ax.plot(x, scipy.stats.norm.pdf(x), 'r-')
    ax.set_xlim([-3, 3])
fig.savefig(out_dir('var.ps'))


