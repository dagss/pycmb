from __future__ import division
#
# Make a set of maps for different beams and noise, to
# verify that the noise-and-beam application looks reasonable
#


from cmb import *
from healpix import *
import matplotlib.pyplot as plt
import logging

logging.basicConfig()

lmax = 50
lprecond = 50
Nside = 64
J = 4

Npix = nside2npix(Nside)
ipix = np.arange(0, Npix)
pixdir = pix2vec_ring(Nside, ipix)

outdir = working_directory('out')

#
# Unscaled noise: A dipole along the y axis
#
dipdir = np.array([0, 1, 0], dtype=np.double)

unscaled_rms = (1.2 + np.dot(pixdir, dipdir)) / 2
unscaled_rms = pixel_sphere_map(unscaled_rms, Nside=Nside)


#
# Mask: Band *above* equator
#
mask = ((pixdir[:,2] < 0) | (pixdir[:,2] > .5))
mask = pixel_sphere_map(mask.astype(np.double), Nside=Nside)



#
# Flat Cls
#
model = IsotropicCmbModel(
    power_spectrum='power_spectrum.dat'
)


#
# Use two different noise levels and two different beams
#

def doit(name, beam, rms):
    obsprop = CmbObservationProperties(
        Nside=Nside,
        beam=beam,
        rms=rms,
        mask=mask
    )
    
    signal, [obs] = model.simulate_observations(2, lmax,
                                                properties=[obsprop],
                                                seed=45)
    signal.to_pixel(Nside).map2gif('signal-%s.gif' % name, title='%s - signal'%name)
    obs.load_temperature().map2gif('%s-obs.gif' % name, title='%s - observation' % name)

    sampler = ConstrainedSignalSampler(
        model=model,
        observations=[obs],
        lprecond=lprecond,
        seed=83,
        lmax=lmax)
    
    for j in range(J):
        signal, [map] = sampler.sample_unmasked()
        map.map2gif('%s-%d.gif' % (name, j), title='%s - sample %d' % (name, j))



with outdir:
    mask.map2gif('mask.gif', title='mask')
    unscaled_rms.map2gif('unscaledrms.gif', title='unscaled rms')
    for beamname, lbeam in [
            ('lobeam', 20),
            ('hibeam', 40)
            ]:
        beam = make_cosine_beam(0, lbeam, lmax)
        for noisename, noiselevel in [
                ('lonoise', 1e-5),
                ('hinoise', 1e-4)
                ]:
            rms = noiselevel * unscaled_rms
            doit('%s-%s' % (beamname, noisename), beam, rms)
