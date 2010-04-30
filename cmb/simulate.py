import numpy as np
import maps

def simulate_cmb_map(lmin, lmax, S, rms_map, beam, state=np.random):
    z = maps.simulate_harmonic_sphere_maps(lmin, lmax, state=state).to_real()

    # Simulate map
    P, L = S.cholesky()
    xclean = P.H * (L * (P * (beam * z)))

    # Add noise
    xpix = xclean.to_complex().to_pixel(rms_map.Nside)
    xpix += rms_map * state.standard_normal(size=rms_map.shape)

    # Return both noise map in pixel space and clean map in harmonic space
    return xpix, xclean
