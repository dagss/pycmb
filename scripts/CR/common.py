from __future__ import division
from cmb import *
import numpy as np
from numpy import pi
import cPickle
import re
import os
from mpi4py import MPI

def dumpf(obj, filename):
    with file(filename, 'w') as f:
        cPickle.dump(obj, f, protocol=2)

def loadf(filename):
    with file(filename) as f:
        return cPickle.load(f)

def dumpvars(vs, patterns, filename):
    objs = {}
    for key in vs.keys():
        for pattern in patterns:
            if re.match(pattern, key):
                objs[key] = vs[key]
    dumpf(objs, filename)

def loadvars(globaldict, filename):
    globaldict.update(loadf(filename))

def plot(x, title, ax=None):
    from cmb.maps import _PixelSphereMap
    if not isinstance(x, _PixelSphereMap):        
        M = harmonic_sphere_map(x, lmin, lmax, is_complex=False).to_pixel(Nside_plot)
    else:
        M = x
    M.map2gif('%s.gif' % title, title=title)

out_dir = working_directory('$RESULT_PATH/CR/5', create=MPI.COMM_WORLD.Get_rank() == 0)
samples_filename = out_dir('CR_test_samples.h5')



Nside = 32
Nside_plot = 64
Npix = nside2npix(Nside)
lmin = 2
if pycmb_debug:# and False:
    lbeamstart = 30
    lmax = 40
    lmax_sim = 50
    lprecond = 30    
else:
    lbeamstart = 45
    lmax = 70
    lmax_sim = 80
    lprecond = 55


#
# Make beam
#
beam = make_cosine_beam(0, lbeamstart, lmax)


with working_directory('$WMAP_PATH'):
    model = IsotropicCmbModel(
        #        power_spectrum='wmap_lcdm_sz_lens_wmap5_cl_v3.dat'
        power_spectrum='comm_init_cls_flat.dat'
    )

    obsprop = CmbObservationProperties(
        Nside=Nside,
        #sigma0=3.319e-3,
        #n_obs=('wmap_da_forered_iqumap_r9_7yr_V1_v4.fits', 1, 'N_OBS'),
        beam=beam,# 'wmap_V1_ampl_bl_7yr_v4.txt',
        #mask=('wmap_temperature_analysis_mask_r9_7yr_v4.fits', 1, 'TEMPERATURE'),
        mask_downgrade_treshold=.5,
        seed=22,
        uniform_rms=np.sqrt(Npix / np.sqrt(4*np.pi) * 1e-12)
    )

#
# Simulate the known signal and corresponding observation
#
signal, [obs] = model.simulate_observations(lmin=lmin, lmax=lmax_sim,
                                            properties=[obsprop],
                                            seed=45)

def load_samples():
    with locked_h5file(samples_filename, 'r') as f:
        samples = f['samples']
        return samples[...].T
