from __future__ import division

##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

import os
import numpy as np
from cmbtypes import *

from isotropic import ClArray

# Should look into Pygr (or supporting technology) for
# replacing this

wmap_lcdm_sz_lens_dtype = [('l', np.int),
                           ('lcdm', real_dtype),
                           ('sz', real_dtype),
                           ('lens', real_dtype)]

def debug(*args):
    import logging
    import traceback
    tb = traceback.extract_stack(limit=2)
    call_line = "%s:%d (%s)" % tb[0][0:3]
    logging.debug("%s\n%r\n" % (call_line, args))

class CmbData:
    
    def __init__(self, data_path):
        self.data_path = data_path

    def path(self, file):
        return os.path.join(self.data_path, file)

    def Cl_fid(self, file='wmap_lcdm_sz_lens_wmap5_cl_v3.dat'):
        """
        Returns an IsotropicCovar instance.
        """
        import covar
        data = np.loadtxt(self.path(file), dtype=wmap_lcdm_sz_lens_dtype)
        l = data['l']
        if np.any(l != np.ogrid[2:data.shape[0]+2]):
            raise RuntimeError("Unexpected data in data file")
        lcdm = data['lcdm'] * 2*np.pi / (l*(l+1))
        return ClArray(lcdm, 2, 2 + lcdm.shape[0] - 1)

    def wmap_beam(self, band, year=7, ver=4, format='matrix'):
        assert format in ('array', 'matrix')
        path = self.path('wmap_%s_ampl_bl_%dyr_v%d.txt' % (band, year, ver))
        record_dtype = np.dtype([('l', np.int), ('beam', np.double)])
        data = np.loadtxt(path, dtype=record_dtype)
        ls = data['l']
        lmin, lmax = ls[0], ls[-1]
        if np.any(ls != np.arange(lmin, lmax + 1)):
            raise NotImplementedError('l\'s not stored consecutively in beam file')
        beam = data['beam']
        return ClArray(beam.copy(), lmin, lmax)

    def wmap_map(self, band, T=False, Q=False, U=False, N_obs=False,
                 counts=False, foreground_reduced=False,
                 year=7, dtype=np.float64):
        raise NotImplementedError() # see observation.py instead
        if not band in ('Ka1', 'Q1', 'Q2', 'V1', 'V2', 'W1', 'W2', 'W3', 'W4'):
            raise ValueError()
        assert foreground_reduced and not Q and not U

        mask = (T, Q, U, N_obs)
        fields = [x for x, inc in zip(
                     ('TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION', 'N_OBS'),
                     mask)
                  if inc]
        key = 'wmap_%d_%s_forered' % (year, band)
        
        import maps
        
        filename = 'wmap_da_forered_iqumap_r9_%dyr_%s_v4.fits' % (year, band)
        return maps.sphere_maps_from_fits(self.path(filename), fields)

    def wmap_temperature_mask(self, dtype=np.float64):
        filename = self.path('wmap_temperature_analysis_mask_r9_7yr_v4.fits')
        import pyfits
        import maps
        
        hdulist = pyfits.open(self.path(filename))
        try:
            data = hdulist[1].data.view(np.ndarray)
            mapdata = data['N_OBS'].ravel().astype(dtype)
        finally:
            hdulist.close()
        out = maps.pixel_sphere_map(mapdata, nested=True)
        return out

    
default_cmbdata = CmbData('/mn/corcaroli/d1/dagss/data/wmap')
#default_cmbdata = CmbData('/home/dagss/cmb/data')
default = default_cmbdata
