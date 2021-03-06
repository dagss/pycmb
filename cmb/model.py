from __future__ import division
##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

import numpy as np
from utils import as_random_state

class CmbModel(object):
    def default_lmin(self, Nside):
        return max([2, self.lmin])

    def default_lmax(self, Nside):
        return min([3*Nside-1, self.lmax])

    def draw_observations(self, lmin=None, lmax=None, properties=None,
                             seed=None):
        """
        Draws observations. The result is a signal and a list of
        simulated CmbObservation instances each corresponding to
        a set of observation properties.

        One must provide an lmax for the simulation.
        """
        import healpix.resources
        from isotropic import ClArray
        from observation import load_temperature_pixel_window_matrix, CmbObservation
        from maps import random_real_harmonic_sphere_map

        try:
            iter(properties)
        except TypeError:
            properties = [properties]
        
        random_state = as_random_state(seed)
        Nside = min([x.Nside for x in properties])
        if lmin is None: lmin = self.default_lmin(Nside)
        if lmax is None: lmax = self.default_lmax(Nside)

        # Simulate signal with the covariance of the model
        S = self.load_covariance(lmin, lmax)
        P, L = S.cholesky()
        z = random_real_harmonic_sphere_map(lmin, lmax, state=random_state)

        signal = P.H * (L * (P * z))
        # Produce observations
        observations = []
        for prop in properties:
            # First, smooth the signal with the beam
            smoothed_signal = signal
            smoothed_signal = prop.load_beam_transfer_matrix(lmin, lmax) * signal
            # Add the pixel window
            pixwin = load_temperature_pixel_window_matrix(prop.Nside, lmin, lmax)
            smoothed_signal = pixwin * smoothed_signal
            # Convert to pixel space and add noise
            map = smoothed_signal.to_pixel(Nside=prop.Nside)
            assert map.pixel_order == 'ring'
            rms = prop.load_rms('ring', include_added_noise=False)
            map += random_state.normal(scale=rms)
            observations.append(CmbObservation(temperature=map, properties=prop))
        
        return signal, observations

    def load_covariance(self, lmin, lmax, dtype=np.double):
        raise NotImplementedError()
    
