from __future__ import division

##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

import numpy as np
from mapdatautils import broadcast_l_to_lm, isotropic_real_covar
from model import CmbModel

__all__ = ['ClArray', 'IsotropicCmbModel']

#Cl_scale = 2*np.pi / l / (l+1)

def debug(*args):
    import logging
    import traceback
    tb = traceback.extract_stack(limit=2)
    call_line = "%s:%d (%s)" % tb[0][0:3]
    logging.debug("%s\n%r\n" % (call_line, args))

class ClArray(np.ndarray):
    """
    An array to represent Cl values. Everything is like a normal array, and slices return arrays,
    but by_l provides slicing by l values, with zero padding.

    EXAMPLES::
    
        >>> Cl = ClArray([1,2,3], lmin=10); Cl
        Power spectrum defined over l=10..12:
        [1 2 3]
        >>> Cl[1:]
        array([2, 3])
        >>> [Cl.by_l[l] for l in range(9, 14)]
        [0, 1, 2, 3, 0]
        >>> Cl.by_l[8:15]
        Power spectrum defined over l=8..14:
        [ 0.  0.  1.  2.  3.  0.  0.]
        >>> Cl.by_l[11:12]
        Power spectrum defined over l=11..11:
        [ 2.]

        >>> ClArray(1, 10, 12)
        Power spectrum defined over l=10..12:
        [ 1.  1.  1.]
    """

    __array_priority__ = 10

    def __new__(subtype, data, lmin=0, lmax=None):
        lmin = int(lmin)
        if np.isscalar(data):
            if lmax is None:
                raise ValueError('lmax must be provided for scalar values')
            obj = np.ones(lmax - lmin + 1) * data
        else:
            obj = np.asarray(data)
        obj = obj.view(ClArray)
        if obj.ndim != 1:
            raise ValueError('Only 1D Cl arrays allowed')
        if lmax is None:
            lmax = obj.shape[0] + lmin - 1
        else:
            lmax = int(lmax)
            if obj.shape[0] != lmax - lmin + 1 or lmin > lmax:
                raise ValueError('Illegal lmin/lmax')
        obj.lmin = lmin
        obj.lmax = lmax
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.lmin = getattr(obj, 'lmin', 0)
            lmax_guess = self.shape[0] + self.lmin - 1
            self.lmax = getattr(obj, 'lmax', lmax_guess)
         
    def __repr__(self):
        return ("l-array defined over l=%d..%d:\n%s" %
            (self.lmin, self.lmax, self.view(np.ndarray)))

    def __getitem__(self, idx):
        return self.view(np.ndarray)[idx]

    def __getslice__(self, i, j):
        return self.view(np.ndarray)[i:j]

    def get_by_l(self, l):
        if l < self.lmin or l > self.lmax:
            return 0
        else:
            return self[l - self.lmin]

    def slice_by_l(self, lmin, lmax):
        before = np.zeros(max(self.lmin - lmin, 0), dtype=self.dtype)
        minlmin = max(self.lmin, lmin)
        after = np.zeros(max(lmax - self.lmax, 0), dtype=self.dtype)
        maxlmax = min(self.lmax, lmax)
        Cl = self[minlmin - self.lmin:maxlmax - self.lmin + 1]
        out = np.concatenate((before, Cl, after))
        return ClArray(out, lmin=lmin, lmax=lmax)

    by_l = property(lambda self: ClArrayAccessByEll(self))

    def as_matrix(self):
        """
        Returns the l array as an lm-indexed diagonal matrix, with repeated elements for
        each m in [-l, l].

        EXAMPLES::

            >>> ClArray([1,2], 1, 2).as_matrix()
            8 by 8 diagonal matrix of float64:
            [ 1.  0.  0.  0.  0.  0.  0.  0.]
            [ 0.  1.  0.  0.  0.  0.  0.  0.]
            [ 0.  0.  1.  0.  0.  0.  0.  0.]
            [ 0.  0.  0.  2.  0.  0.  0.  0.]
            [ 0.  0.  0.  0.  2.  0.  0.  0.]
            [ 0.  0.  0.  0.  0.  2.  0.  0.]
            [ 0.  0.  0.  0.  0.  0.  2.  0.]
            [ 0.  0.  0.  0.  0.  0.  0.  2.]
        """
        return isotropic_real_covar(self.astype(np.double))

    def broadcast_to_lm(self):
        """
        >>> ClArray([1,2], 3, 4).broadcast_to_lm()
        array([1, 1, 1, 1, 2, 2, 2, 2, 2])
        """
        assert False # is now ambigious with respect to packing, check what uses it
        return broadcast_l_to_lm(self, lmin=self.lmin)
        
class ClArrayAccessByEll(object):
    def __init__(self, base):
        self.base = base
        
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            raise ValueError("1D indexing only")
        elif isinstance(idx, slice):
            if idx.step not in (1, None):
                raise ValueError("Step must be 1")
            return self.base.slice_by_l(lmin=idx.start, lmax=idx.stop-1)
        else:
            return self.base.get_by_l(idx)


## def power_spectrum_from_fits(f, dtype=np.double):
##     msg = 'Unexpected FITS data for power spectrum'
##     import pyfits
##     if isinstance(f, str):
##         f = pyfits.open(f)
##     if len(f) != 1:
##         raise ValueError(msg)
##     data = f[0].data
##     if data.shape[0:3] != (2, 1, 1):
##         raise ValueError(msg)
##     ls = data[0, 0, 0, :]
##     Cls = data[1, 0, 0, :].astype(dtype)
##     if np.any(ls != np.arange(ls.shape[0])):
##         raise NotImplementedError('Only contiguous Cls supported')
##     return ClArray(Cls, 0, Cls.shape[0]-1)

def as_power_spectrum(desc):
    """
    Loads the file and returns a ClArray.
    """
    if isinstance(desc, ClArray):
        return desc
    elif not isinstance(desc, str):
        raise TypeError('Please provide a filename')


    # First attempt as FITS file
#    import pyfits
#    try:
#        f = pyfits.open(desc)
#    except IOError:
#        pass # not a fits file; fall back to text loading below
#    else:
#        return power_spectrum_from_fits(f)

    # Load as text file
    data = np.loadtxt(desc, dtype=[('l', np.int),
                                   ('cl', np.double)])
    l = data['l']
    if np.any(l != np.arange(2, data.shape[0] + 2)):
        raise RuntimeError("Unexpected data in data file")
    Cl = data['cl'] * 2*np.pi / (l*(l+1)) * 1e-12 # microkelvin**2
    return ClArray(Cl, 2, 2 + Cl.shape[0] - 1)

class IsotropicCmbModel(CmbModel):
    def __init__(self, power_spectrum):
        self._power_spectrum = as_power_spectrum(power_spectrum)
        self.lmin = self._power_spectrum.lmin
        self.lmax = self._power_spectrum.lmax

    def load_covariance(self, lmin, lmax, dtype=np.double):
        Cl = self._power_spectrum.by_l[lmin:lmax + 1].astype(dtype)
        return isotropic_real_covar(Cl)
        
    def get_power_spectrum(self):
        return self._power_spectrum.copy()

    def get_power_spectrum_mutable(self):
        return self._power_spectrum.copy()

    def plot(self, ax=None, lmax=None, scale=True):
        import maps
        cl = self.get_power_spectrum().by_l[self.lmin, lmax+1]
        maps.plot_power_spectrum(cl, ax=ax, title='Model power spectrum')

    def get_scale(self, lmax):
        l = np.arange(0, lmax + 1)
        scale = (l * (l+1)) / (2*np.pi) * 1e12
        return l, scale
        
