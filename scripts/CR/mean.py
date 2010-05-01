from __future__ import division
from common import *
import matplotlib.pyplot as plt
import matplotlib as mpl

with out_dir:
    loadvars(globals(), 'direct.pickle')
        
sampler = ConstrainedSignalSampler(
    model=model,
    observations=[obs],
    verbosity=3,
    lprecond=lprecond,
    seed=12,
    lmax=lmax)            

mean_cg = sampler.find_mean()

mean_direct = harmonic_sphere_map(mean_direct[4:], lmin, lmax, is_complex=False)

with out_dir:
    plot(mean_cg, 'mean_cg')
    plot(mean_direct, 'mean_direct')
    plot(mean_cg - mean_direct, 'mean_delta')
    plot(obs.load_temperature('ring'), 'observation')

