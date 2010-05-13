from __future__ import division
#
# Mask out half of the sky, do CRs, and check that resulting power spectrums
# are reasonable, indicating that beam and noise was correctly reapplied
#


from cmb import *
from healpix import *
import matplotlib.pyplot as plt
import logging

logging.basicConfig()

lbeam = 35
lmax = 50
lnoise = 40 # point at which noise is 3000uK^2
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
#dipdir = np.array([0, 1, 0], dtype=np.double)

#unscaled_rms = (1.2 + np.dot(pixdir, dipdir)) / 2
#unscaled_rms = pixel_sphere_map(unscaled_rms, Nside=Nside)


#
# Mask: Horizontal band
#
mask = ((pixdir[:,2] < -0.8) | (pixdir[:,2] > .6))
mask = pixel_sphere_map(mask.astype(np.double), Nside=Nside)



#
# Flat Cls
#
model = IsotropicCmbModel(
    power_spectrum='power_spectrum.dat'
)

beam = make_cosine_beam(0, lbeam, lmax)

noise_power = 3000e-12 * 2*np.pi / (lnoise * (lnoise+1))
sigma_sh = 1e-2

obsprop = CmbObservationProperties(
    Nside=Nside,
    beam=beam,
#    rms=rms,
    mask=mask,
    uniform_rms=np.sqrt(Npix / (4*np.pi) * noise_power),
    seed=14
)
    
signal, [obs] = model.draw_observations(2, lmax,
                                        properties=[obsprop],
                                        seed=45)

sampler = ConstrainedSignalSampler(
    model=model,
    observations=[obs],
    lprecond=lprecond,
    seed=83,
    lmax=lmax)

with outdir:
    signal.to_pixel(Nside).map2gif('signal.gif', title='signal')
    obs.load_temperature().map2gif('obs.gif', title='observation')

    fig = plt.gcf()
    fig.clear()

    ax = fig.add_subplot(1,1,1)
#    plot_power_spectrum(signal.power_spectrum() * 1e12, ax=ax, title='signal',
#                        ylim=ylim)

    smoothed_signal = obsprop.load_beam_transfer_matrix(2, lmax) * signal
    plot_power_spectrum(smoothed_signal.power_spectrum() * 1e12, ax=ax)

    rmspowerspectrum = ClArray(noise_power, 2, 2*lmax)
    line = plot_power_spectrum(rmspowerspectrum * 1e12, ax=ax)[0]
    line.set_color('red')

    # Then, draw 5 signals and plot their power spectrums
    for i in range(5):
        signal, [map] = sampler.sample_unmasked()
        s = map.to_harmonic(2, 2*lmax)
        line = plot_power_spectrum(s.power_spectrum() * 1e12, ax=ax)[0]
        line.set_linestyle('dashed')
        line.set_color('black')

    fig.savefig('plot.ps')
    ax.set_yscale('log')
    fig.savefig('logplot.ps')

    
##     for j in range(J):
##         signal, [map] = sampler.sample_unmasked()
##         map.map2gif('%s-%d.gif' % (name, j), title='%s - sample %d' % (name, j))


    plt.draw()
## with outdir:
##     mask.map2gif('mask.gif', title='mask')
##     unscaled_rms.map2gif('unscaledrms.gif', title='unscaled rms')
##     for beamname, lbeam in [
##             ('lobeam', 20),
##             ('hibeam', 40)
##             ]:
##         beam = make_cosine_beam(0, lbeam, lmax)
##         for noisename, noiselevel in [
##                 ('lonoise', 1e-5),
##                 ('hinoise', 1e-4)
##                 ]:
##             rms = noiselevel * unscaled_rms
##             doit('%s-%s' % (beamname, noisename), beam, rms)



