from __future__ import division
from common import *
import cmb.noise

S = model.load_covariance(0, lmax)
P, L = S.cholesky()

Ninv_map = obsprop.load_Ninv_map_mutable('ring')
Ninv = cmb.noise.inverse_pixel_noise_to_harmonic_matmul(0, lmax, Ninv_map)
Ninv = as_matrix(Ninv, copy=False)

A = load_temperature_pixel_window_matrix(Nside, 0, lmax)
A *= beam.as_matrix()
A_Ninv_A = A * Ninv * A

B = DenseMatrix(np.eye(S.shape[0]) +
                (P.H * L.H * P * A_Ninv_A * P.H * L * P).numpy())
Binv = B.inverse()

Sigma_direct = P.H * L * P * Binv * P.H * L.H * P

d = obs.load_temperature_mutable('ring')
#d.remove_multipoles_inplace(2, obs.properties.load_mask('ring'))

Ninv_d = Ninv_map * d * Npix / 4 / pi
A_Ninv_d = A * Ninv_d.to_real_harmonic(0, lmax)
mean_direct = Sigma_direct * A_Ninv_d


with out_dir:
    dumpvars(globals(), ['Sigma_direct', 'mean_direct'], 'direct.pickle')

