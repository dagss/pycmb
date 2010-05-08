from __future__ import division
import numpy as np
import sys
from healpix import npix2nside, nside2npix
import healpix
from cmbtypes import complex_dtype, real_dtype
import mapdatautils
from mapdatautils import (
    py_l_to_lm as l_to_lm,
    py_lm_to_idx_brief as lm_to_idx_brief,
    py_lm_to_idx_full as lm_to_idx_full,
    py_lm_count_full as lm_count_full,
    py_lm_count_brief as lm_count_brief,
    broadcast_l_to_lm, m_range_lm)
from mapdatautils import py_l_to_lm as l_to_lm, num_alm_to_lmax
import operator
import healpix.resources

healpix_res = healpix.resources.get_default()

#default_plot_mode = 'interactive' # or 'file'
#default_plot_output = '%SCRATCH/%s.png'

def debug(*args):
    import logging
    import traceback
    tb = traceback.extract_stack(limit=2)
    call_line = "%s:%d (%s)" % tb[0][0:3]
    logging.debug("%s\n%r\n" % (call_line, args))

def log(*args):
    print >>sys.stderr, args

def is_real_dtype(dtype):
    return dtype == np.float32 or dtype == np.float64

def is_complex_dtype(dtype):
    return dtype == np.complex64 or dtype == np.complex128

def to_complex_dtype(dtype):
    if is_complex_dtype(dtype):
        return dtype
    if not is_real_dtype(dtype):
        return ValueError("dtype must be floating point")
    if dtype == np.float32:
        return np.complex64
    elif dtype == np.float64:
        return np.complex128

def to_real_dtype(dtype):
    if is_real_dtype(dtype):
        return dtype
    if not is_complex_dtype(dtype):
        return ValueError("dtype must be floating point")
    if dtype == np.complex64:
        return np.float32
    elif dtype == np.complex128:
        return np.float64

def angles_to_vector(theta, phi):
    from numpy import cos, sin
    x = cos(phi) * sin(theta)
    y = sin(phi) * sin(theta)
    z = cos(theta)
    return x, y, z

class NDArraySubclassBase(np.ndarray):
    """
    Make it easier to subclass ndarray in a particular way:
     - By default the array behaves just like np.ndarray
     - However one can whitelist operations and have them return
       the given subclass

    For instance, have a+b return a subclass view, but np.dot(a, b)
    and a[...] return an object behaving like ndarray.

    This wasn't easy to do with NumPy alone -- so we have a flag,
    self._enable_subclass, which controls whether we should enable
    any extra features/data or not.
    """
    _allow_elementwise = True
    _allow_copy = True
    _force_data_order = None

    _enable_subclass = False 

    def __new__(cls, data, *args, **kw):
        if cls._force_data_order is not None:
            obj = np.asarray(data, order=cls._force_data_order)
        else:
            obj = np.asarray(data)
        obj = obj.view(cls)
        obj.set_properties(*args, **kw)
        obj._enable_subclass = True
        return obj

    def __array_wrap__(self, out, ctx=None):
        # Called before a ufunc is called.
        # self is the input array with highest __array_priority__.
        # out is the result buffer of the ufunc
        if not self._enable_subclass:
            return out
        if ctx is not None:
            ufunc, params, domain = ctx
            if (isinstance(ufunc, np.ufunc) and self._allow_elementwise):
                self._check_ufunc(ufunc, params, domain)
                out = out.view(type(self))
                out._enable_subclass = True
                self.copy_properties_to(out)
        return out

    def _check_ufunc(self, ufunc, params, domain):
        pass # override to raise any incompatability errors

    def copy(self, order=None):
        if not self._allow_copy or not self._enable_subclass:
            if order is None:
                order = 'C'
            return self.view(np.ndarray).copy(order)
        else:
            if self._force_data_order is not None:
                if order is None:
                    order = self._force_data_order
                elif order != self._force_data_order:
                    raise ValueError("Cannot make %s-ordered copies of this object (please do obj.view(np.ndarray).copy())" % order)
            copy = np.ndarray.copy(self, order)
            copy._enable_subclass = True
            self.copy_properties_to(copy)
            return copy

    def __repr__(self):
        if not self._enable_subclass:
            typename = type(self).__name__
            r = np.ndarray.__repr__(self)
            r = r.replace(typename, '%s pretending to be array' % typename)
            return r
        else:
            return self._repr()

    def _repr(self):
        return np.ndarray.__repr__(self)

    def _check_not_pretending(self):
        if not self._enable_subclass:
            raise TypeError('method not enabled on property-less instance')

    
#   Do not allow slicing etc. yet, so no need for finalize
#   def __array_finalize__(self, obj):
#        debug('finalize', obj)
#        return obj.view(np.ndarray)
#        return obj.view(np.ndarray)

class TestSubclass(NDArraySubclassBase):
    """
    >>> T = TestSubclass([[1,2,3]], myattr=4)
    >>> T
    <TestSubclass attr=4>
    >>> T + 3
    <TestSubclass attr=4>
    >>> T + T
    <TestSubclass attr=4>
    >>> np.sin(T)
    <TestSubclass attr=4>
    >>> T.mean(axis=0)
    array([ 1.,  2.,  3.])
    >>> np.dot(T, T.T)
    TestSubclass pretending to be array([[14]])
    >>> T[:,1:]
    TestSubclass pretending to be array([[2, 3]])
    >>> T.copy()
    <TestSubclass attr=4>
    >>> T.copy('F')
    Traceback (most recent call last):
        ...
    ValueError: Cannot make F-ordered copies of this object (please do obj.view(np.ndarray).copy())
    >>> np.dot(T, T.T).copy()
    array([[14]])

    >>> T == 2
    <TestSubclass attr=4>
    
    >>> np.all(T == 2)
    False
    """
    _force_data_order = 'C'
    
    def set_properties(self, myattr):
        self.myattr = myattr

    def copy_properties_to(self, other):
        other.myattr = self.myattr

    def _repr(self):
        return "<TestSubclass attr=%r>" % self.myattr

def pixel_sphere_map(data, Nside=None, shape=None, pixel_order=None,
                     nested=None, ring=None):
    """

    

    INPUT:
    
     - ``Nside`` - The resolution of the map, given as the
       :math:`N_{side}` argument. The number of pixels is
       ``12*Nside**2``. Can be given as None to infer from data.
     
     - ``data`` - (default 0) The data. Can be a scalar 
     
     - ``pixel_order`` - (default 'ring') whether the data should be
       ring-ordered.  Set to 'nested' for the nested scheme.
    
    EXAMPLES::

        >>> pixel_sphere_map(0, 4)
        Pixel sphere map (ring-ordered, Nside=4, Npix=192)

        >>> M = pixel_sphere_map(np.arange(0, 49152)); M
        Pixel sphere map (ring-ordered, Nside=64, Npix=49152)
        >>> M.dtype
        dtype('float64')
        >>> M[23]
        23.0

        # Disable this for now >>> M[:,None,None]

        >>> np.sin(M) + 34
        Pixel sphere map (ring-ordered, Nside=64, Npix=49152)

        >>> M2 = pixel_sphere_map(np.hstack((M[:,None], M[:,None], M[:,None]))); M2
        (3,)-stack of pixel sphere maps (ring-ordered, Nside=64, Npix=49152)
        >>> M3 = pixel_sphere_map(np.dstack((M2[:,:,None], M2[:,:,None], M2[:,:,None]))); M3
        (3, 3)-stack of pixel sphere maps (ring-ordered, Nside=64, Npix=49152)


        >>> pixel_sphere_map(1)
        Traceback (most recent call last):
           ...
        ValueError: Cannot infer Nside from data

    """
    if (ring or nested) and pixel_order:
        raise ValueError()
    if nested:
        pixel_order = 'nested'
    elif ring or pixel_order is None:
        pixel_order = 'ring'
        
    if Nside not in (None, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        raise ValueError('Invalid Nside (must be None or power of 2 <= 8192)')
    if np.isscalar(data):
        if shape is None:
            shape = ()
        elif np.isscalar(shape):
            shape = (shape,)
        if Nside is None:
            raise ValueError('Cannot infer Nside from data')
        Npix = nside2npix(Nside)
        data = np.ones((Npix,) + shape, dtype=real_dtype, order='F') * float(data)
    else:
        if shape is not None:
            raise ValueError()
        data = np.asfortranarray(data, dtype=real_dtype)
        Npix = data.shape[0]
        if Nside is not None and nside2npix(Nside) != Npix:
            raise ValueError('Nside given but mismatches data size')
    return _PixelSphereMap(data, pixel_order=pixel_order)

class _PixelSphereMap(NDArraySubclassBase):
    """
    TESTS:

        >>> M = pixel_sphere_map(0, 4, shape=10)
        >>> np.dot(M, M.T) #doctest:+ELLIPSIS
        _PixelSphereMap pretending to be array(...
        >>> M + M
        (10,)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)
        >>> M * M
        (10,)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)

        >>> M + M.to_nested()
        Traceback (most recent call last):
            ...
        ValueError: Incompatible maps (different pixel order)
        
        >>> np.sin(M)
        (10,)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)
        >>> np.mean(M, axis=0)
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    """

    __array_priority__ = 10

    _allow_elementwise = True
    _allow_copy = True
    _force_data_order = 'F'


    def set_properties(self, pixel_order):
        if pixel_order not in ('ring', 'nested'):
            raise ValueError('invalid pixel order')
        self.pixel_order = pixel_order
        if nside2npix(npix2nside(self.shape[0])) != self.shape[0]:
            raise ValueError('invalid shape (must be valid Npix)')

    def copy_properties_to(self, other):
        other.set_properties(self.pixel_order)

    def __reduce__(self):
        """
        >>> M = pixel_sphere_map(0, 4, shape=10)        
        >>> from cPickle import loads, dumps
        >>> loads(dumps(M, 2))        
        (10,)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)
        """
        return (_PixelSphereMap, (
            self.view(np.ndarray),
            self.pixel_order))
                 
    def _repr(self):
        if self.ndim == 1:
            return "Pixel sphere map (%s-ordered, Nside=%d, Npix=%d)" % (
                self.pixel_order, self.Nside, self.Npix)
        else:
            return "%s-stack of pixel sphere maps (%s-ordered, Nside=%d, Npix=%d)" % (
                self.shape[1:], self.pixel_order, self.Nside, self.Npix)

    def _check_ufunc(self, ufunc, params, domain):
        for p in params:
            if isinstance(p, _PixelSphereMap) and p.pixel_order != self.pixel_order:
                raise ValueError("Incompatible maps (different pixel order)")

    def __getitem__(self, indices):
        map_result = False
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) >= 1 and isinstance(indices[0], slice):
            s = indices[0]
            if s.start == s.stop == s.step == None:
                map_result = True
        out = self.view(np.ndarray)[indices]
        if map_result:
            return pixel_sphere_map(out, pixel_order=self.pixel_order)
        else:
            return out

    def get_Npix(self):
        self._check_not_pretending()
        return self.shape[0]
    Npix = property(fget=get_Npix)

    def get_Nside(self):
        self._check_not_pretending()
        return npix2nside(self.shape[0])
    Nside = property(fget=get_Nside)

    def map2gif(self, outfile, bar=True, title=None, col=5,
                max=None, min=None):
        self._check_not_pretending()
        import os
        import tempfile
        if self.ndim != 1:
            raise ValueError("Can only plot single maps, but self.shape is %s" % str(self.shape))
        assert self.ndim == 1
        fd, infile = tempfile.mkstemp(suffix='.fits')
        os.close(fd)
        self.to_fits(infile, map_units='raw')
        if os.path.isfile(outfile):
            os.remove(outfile)
        flags = []
        if title is not None: flags.append('-ttl "%s"' % title)
        if max is not None: flags.append('-max %f' % max)
        if min is not None: flags.append('-min %f' % min)
        if bar: flags.append('-bar .true.')
        
        cmd = ('map2gif %s -col %d -inp "%s" -out "%s"' %
                  (' '.join(flags), col, infile, outfile))
        os.system(cmd)
        os.remove(infile)

    def to_fits(self, filename, map_units='K', output_units='mK'):
        import pyfits
        import os

        assert output_units == 'mK'
        data = self.to_nested()
        if map_units == 'K':
            data *= 1e3
        elif map_units == 'mK':
            pass
        elif map_units == 'uK':
            data *= 1e6
        elif map_units == 'raw':
            pass
        else:
            raise ValueError('Illegal map unit')

        if self.ndim != 1:
            raise NotImplementedError()
        pri = pyfits.PrimaryHDU()
        col = pyfits.Column(name='TEMPERATURE', format='D', unit='mK,thermodynamic', array=data)
        sec = pyfits.new_table(pyfits.ColDefs([col]))
        sec.header.update('PIXTYPE', 'HEALPIX')
        sec.header.update('ORDERING', 'NESTED')
        sec.header.update('NSIDE', data.Nside)
        sec.header.update('FIRSTPIX', 0)
        sec.header.update('LASTPIX', data.Npix)
        if os.path.isfile(filename):
            os.unlink(filename)
        pyfits.HDUList([pri, sec]).writeto(filename)

#        heal.output_map_d(, filename)
#        mapdatautils.output_ringmap(copy, filename)

    def plot(self, ax=None, show=None, title=None, **kw):
        self._check_not_pretending()
        from PIL import Image
        import matplotlib.pyplot as plt
        import os
        import tempfile


        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, title=title)
            if show is None:
                show = True
        elif show:
            raise ValueError("show cannot be set when ax is provided")
        fd, tmpfile = tempfile.mkstemp(suffix='.gif')
        os.close(fd)
        self.map2gif(tmpfile, bar=False)

        minval = np.min(self)
        maxval = np.max(self)

        # rely on HealPix map2gif to have produced the picture;
        # palette indices will correspond to gray-levels and
        # palette index 1 will be background => mapped to NaN
        # (we do all this in order to get a matplotlib colorbar)
        # palette index 3..102 is range of gray values
        imgdata = np.asarray(Image.open(tmpfile))#.convert('RGB'))
        os.remove(tmpfile)
        imgdata = imgdata.copy().astype(np.double)
        imgdata[imgdata == 1] = np.nan
        imgdata = (imgdata - 3) / 100.
        imgdata = minval + imgdata * (maxval - minval)
        img = ax.imshow(imgdata, interpolation='nearest', vmin=minval, vmax=maxval,
                          **kw)
        cbar = ax.figure.colorbar(img, ax=ax, ticks=[minval, maxval],
                                    orientation='horizontal')
 #       from matplotlib.ticker import FormatStrFormatter
        cbar.ax.set_xticklabels(['%.2e' % x for x in (minval, maxval)])
        ax.set_xticks([])
        ax.set_yticks([])
        if show:
            fig.show()
        return img
        
    def estimate_covariance(self):
        self._check_not_pretending()
        from covar import PixelCovar
        data = self.data
        cov_hat = np.dot(data.T, data) / data.shape[0]
        return PixelCovar(cov_hat)

    def to_ring(self):
        self._check_not_pretending()
        if self.pixel_order == 'ring':
            return self
        else:
            out = self.copy()
            out.change_order_inplace('ring')
            return out

    def to_nested(self):
        self._check_not_pretending()
        if self.pixel_order == 'nested':
            return self
        else:
            out = self.copy()
            out.change_order_inplace('nested')
            return out

    def change_order_inplace(self, order, _strict=False):
        """
        Change pixel order in-place.
        """
        self._check_not_pretending()
        if not order in ('ring', 'nested'):
            raise ValueError()
        if order == self.pixel_order:
            return
        out = self.reshape(self.shape[0], -1, order='F')
        if order == 'ring':
            healpix.convert_nest2ring(self.Nside, out)
        else:
            healpix.convert_ring2nest(self.Nside, out)
        if out.flags.owndata:
            if _strict:
                assert False, 'For some reason a temporary buffer was allocated'
            self[...] = out
        self.pixel_order = order

    def reformat_and_discard(self, pixel_order, Nside, _strict=False):
        """
        Return a map with the given pixel order and resolution. It is
        NOT a requirement that self has the same pixel order afterwards.
        If Nside == self.Nside, self will be returned.
        """
        if self.pixel_order == pixel_order and self.Nside == Nside:
            return self
        elif Nside != self.Nside:
            self.change_order_inplace('nested', _strict)
            x = self.change_resolution(Nside)
            if pixel_order == 'ring':
                x.change_order_inplace('ring', _strict)
            return x
        else:
            self.change_order_inplace(pixel_order, _strict)
            return self
            
    def to_harmonic(self, lmin, lmax, use_weights=True,
                    weights_transform=None):
        """
        ::
        
            >>> inpix = pixel_sphere_map(0, 4)
            >>> inpix.to_harmonic(0, 8)
            Brief complex harmonic sphere map with l=0..8 (45 coefficients)

        Convert a uniform map. First elements should be 12*sqrt(4 pi),
        the remaining elements 0::
        
            >>> N = pixel_sphere_map(Nside=8, data=12)
            >>> x = N.to_harmonic(0, 16)
            >>> np.allclose(x[0], 12 * np.sqrt(4*np.pi))
            True
            >>> np.allclose(x[1:], 0, atol=1e-1)
            True

        One can convert multiple maps at once:
            >>> N = np.hstack((N[:,None], N[:,None], N[:,None], N[:,None]))
            >>> N = np.dstack((N, N))
            >>> N = pixel_sphere_map(data=N, Nside=8); N
            (4, 2)-stack of pixel sphere maps (ring-ordered, Nside=8, Npix=768)
            >>> x = N.to_harmonic(0, 16); x
            (4, 2)-stack of brief complex harmonic sphere maps with l=0..16 (153 coefficients)
            >>> np.allclose(x[0,:,:], 12 * np.sqrt(4*np.pi))
            True
            >>> np.allclose(x[1:,:,:], 0, atol=.1)
            True



        """
        if self.pixel_order == 'nested':
            return self.to_ring().to_harmonic(lmin, lmax)
        assert self.pixel_order == 'ring'
        self._check_not_pretending()
        dtype = to_complex_dtype(self.dtype)
        # for now, only double...
        assert dtype == np.complex128

        size = self.shape[1:]
        numalm = l_to_lm(lmax + 1)
        alm_matrix_buf = np.empty((lmax + 1, lmax + 1), complex_dtype, order='F')
        alm = np.empty((numalm,) + size, complex_dtype, order='F')
        if use_weights:
            weight_ring_temperature = healpix_res.weight_ring(Nside=self.Nside)[0,:]
            if weights_transform is not None:
                weight_ring_temperature = weights_transform(weight_ring_temperature)
        else:
            weight_ring_temperature = np.ones(2 * self.Nside)

        for mapidx in np.ndindex(*size):
            idx = (slice(None),) + mapidx
            map = self[idx]
            mapdatautils.map2alm(lmax, map, weight_ring_temperature, out=alm_matrix_buf)
            mapdatautils.alm_complexmatrix2complexpacked(alm_matrix_buf, out=alm[idx])

        # Trim away portion up to lmin
        alm = alm[l_to_lm(lmin):,...]
        return _HarmonicSphereMap(alm, lmin, lmax, COMPLEX_BRIEF)

    def to_real_harmonic(self, lmin, lmax, use_weights=True,
                         weights_transform=None):
        return self.to_harmonic(lmin, lmax, use_weights, weights_transform).to_real()

    def change_resolution(self, Nside):
        """
        >>> M = pixel_sphere_map(np.arange(3072)); M
        Pixel sphere map (ring-ordered, Nside=16, Npix=3072)
        >>> np.mean(M)
        1535.5
        >>> P = M.change_resolution(32); P
        Pixel sphere map (ring-ordered, Nside=32, Npix=12288)
        >>> np.mean(P)
        1535.5
        """
        if self.Nside == Nside:
            return self.copy()
        elif self.pixel_order == 'ring':
            return self.to_nested().change_resolution(Nside).to_ring()
        else:
            assert self.pixel_order == 'nested'
            size = self.shape[1:]
            Npix_new = nside2npix(Nside)
            buf = np.empty((Npix_new,) + size, order='F', dtype=np.double)
            for mapidx in np.ndindex(*size):
                idx = (slice(None),) + mapidx
                healpix.sub_udgrade_nest(self[idx], buf[idx])
            return _PixelSphereMap(buf, pixel_order='nested')

    def remove_multipoles_inplace(self, degree, mask):
        """
        Estimates and removes the monopole and/or the dipole. See HEALPix
        remove_dipole. The coefficients for the multipoles removed are
        returned as a tuple.
        """
        if mask.pixel_order != self.pixel_order:
            raise ValueError("Pixel order of mask does not match pixel order of map")
        return healpix.remove_dipole_double(self.Nside, self, self.pixel_order,
                                            degree, mask)
def harmonic_sphere_map(data, lmin=0, lmax=None, is_complex=None, is_brief=None):
    """
    Constructs a sphere map in harmonic space.

    EXAMPLES::
    
        >>> M = harmonic_sphere_map([[0], [1, 3+1j], [1, 1, 1], [1, 1, 1, 1]]); M
        Brief complex harmonic sphere map with l=0..3 (10 coefficients)
        >>> M2 = harmonic_sphere_map([[1, 1, 1, 1]], lmin=3); M2
        Brief complex harmonic sphere map with l=3..3 (4 coefficients)
        
        >>> harmonic_sphere_map([[0], [1 + 1j, 3+1j]])
        Traceback (most recent call last):
            ...
        ValueError: Entries on m=0-positions must be real
        >>> harmonic_sphere_map(np.dstack((M[:,None,None], M[:,None,None])))
        (1, 2)-stack of brief complex harmonic sphere maps with l=0..3 (10 coefficients)
        >>> M = harmonic_sphere_map(2, 0, 100); M
        Brief complex harmonic sphere map with l=0..100 (5151 coefficients)
        >>> np.sin(M) + 34 * M
        Brief complex harmonic sphere map with l=0..100 (5151 coefficients)

    Real coefficients are also supported::

        >>> M = harmonic_sphere_map([[0], [-1, 1, 3], [-1, -2, 3, 4, 5]]); M
        Real harmonic sphere map with l=0..2 (9 coefficients)

    Scalar initialization::

        >>> harmonic_sphere_map(10, 2, 4)
        Brief complex harmonic sphere map with l=2..4 (12 coefficients)
    """
    if np.isscalar(data):
        if is_complex is None:
            is_complex = True
        if lmax is None:
            raise ValueError('Cannot infer lmax from data')
        data = (np.ones(l_to_lm(lmax + 1, is_complex) - l_to_lm(lmin, is_complex),
                        dtype=complex_dtype if is_complex else real_dtype, order='F') *
                float(data))
    elif isinstance(data, list):
        # Need at least two nesting levels of lists
        if (not isinstance(data[0], list) or isinstance(data[0][0], list)):
            raise ValueError("Please provide coefficients as [[a00], [a10, a11], ...]")
        if lmax is None:
            lmax = len(data) + lmin - 1

        
        if is_complex is None:
            # Auto-detect is_complex from data
            if lmin == 0 and len(data) == 1:
                is_complex = True # only a00 given; assume is_complex
            elif lmin == 0:
                is_complex = (len(data[1]) == 2)
            else:
                is_complex = len(data[0]) == lmin + 1
        
        num_coefs = l_to_lm(lmax + 1, is_complex) - l_to_lm(lmin, is_complex)
        buf = np.zeros(num_coefs, dtype=complex_dtype if is_complex else real_dtype, order='F')

        i = 0
        for l, coefs_for_l in zip(range(lmin, lmax+1), data):
            n = l+1 if is_complex else 2*l + 1
            if len(coefs_for_l) != n:
                raise ValueError("Wrong number of coefficients for l=%d" % l)
            if coefs_for_l[0].imag != 0:
                raise ValueError("Entries on m=0-positions must be real")
            buf[i:i+n] = coefs_for_l
            i += n
        data = buf
    else:
        if is_complex is None:
            is_complex = True
        data = np.asfortranarray(data, dtype=complex_dtype if is_complex else real_dtype)
        lmax_wanted = num_alm_to_lmax(l_to_lm(lmin, is_complex) + data.shape[0], is_complex)
        if lmax is not None and lmax != lmax_wanted:
            raise ValueError('lmax given does not match data lmax')
        if lmax is None:
            lmax = lmax_wanted
    if is_brief is None:
        is_brief = is_complex
    if is_complex:
        format = COMPLEX_BRIEF if is_brief else COMPLEX_FULL
    else:
        if is_brief:
            raise ValueError()
        format = REAL_FULL
    return _HarmonicSphereMap(data, lmin=lmin, lmax=lmax, format=format)

class HarmonicSphereMapFormat(object):
    def __init__(self, capdesc, desc, name, is_brief):
        self.is_brief = is_brief
        self.capdesc = capdesc
        self.desc = desc
        self.name = name
        
    def __repr__(self):
        return "<HarmonicSphereMapFormat '%s'>" % self.desc
    
    def __reduce__(self):
        return (_unpickle_HarmonicSphereMapFormat, (self.name,))

def _unpickle_HarmonicSphereMapFormat(name):
    return globals()[name]

COMPLEX_FULL = HarmonicSphereMapFormat('Full complex', 'full complex', 'COMPLEX_FULL', False)
COMPLEX_BRIEF = HarmonicSphereMapFormat('Brief complex', 'brief complex', 'COMPLEX_BRIEF', True)
REAL_FULL = HarmonicSphereMapFormat('Real', 'real', 'REAL_FULL', False)

class _HarmonicSphereMap(NDArraySubclassBase):
    """
    TESTS::
    
        >>> M = harmonic_sphere_map([[1], [2, 3j]])
        >>> M + M
        Brief complex harmonic sphere map with l=0..1 (3 coefficients)
        >>> M + harmonic_sphere_map([[1, 2j, 3j]], lmin=2)
        Traceback (most recent call last):
            ...
        ValueError: Incompatible maps (different lmin/lmax or format)
        >>> M[1:]
        _HarmonicSphereMap pretending to be array([ 2.+0.j,  0.+3.j])
    """


    def set_properties(self, lmin, lmax, format):
        self.lmin = lmin
        self.lmax = lmax
        self.format = format
        if format not in (COMPLEX_FULL, COMPLEX_BRIEF, REAL_FULL):
            raise ValueError()
        num_coefs = lm_count_brief(lmin, lmax) if format.is_brief else lm_count_full(lmin, lmax)
        if num_coefs != self.shape[0]:
            raise ValueError("Wrong number of coefficients for lmin/lmax given")

    def copy_properties_to(self, other):
        other.set_properties(self.lmin, self.lmax, self.format)

    def __reduce__(self):
        """
        >>> M = harmonic_sphere_map(0, 3, 10); M
        Brief complex harmonic sphere map with l=3..10 (60 coefficients)
        >>> from cPickle import loads, dumps
        >>> loads(dumps(M, 2))        
        Brief complex harmonic sphere map with l=3..10 (60 coefficients)
        >>> loads(dumps(harmonic_sphere_map(np.arange(4), 0, 1, is_complex=False)))
        Real harmonic sphere map with l=0..1 (4 coefficients)
        """
        return (_HarmonicSphereMap, (
            self.view(np.ndarray),
            self.lmin, self.lmax, self.format))
                     
    def _check_ufunc(self, ufunc, params, domain):
        for p in params:
            if (isinstance(p, _HarmonicSphereMap) and (p.lmin != self.lmin or p.lmax != self.lmax
                or p.format != self.format)):
                raise ValueError("Incompatible maps (different lmin/lmax or format)")

    def _repr(self):
        if self.ndim == 1:
            return ("%s harmonic sphere map with l=%d..%d "
                    "(%d coefficients)" % (
                    self.format.capdesc, self.lmin, self.lmax, self.shape[0]))
        else:
            return ("%s-stack of %s harmonic sphere maps with l=%d..%d "
                    "(%d coefficients)" % (
                    self.shape[1:], self.format.desc, self.lmin, self.lmax, self.shape[0]))

    def to_array(self):
        return self.view(np.ndarray)

    def to_list(self):
        return self.to_array().tolist()

    def to_full_complex(self):
        """
        ::

            >>> C = harmonic_sphere_map([[12], [2,3 - 1j], [3, 4 -1j, 5 +2j]]); C
            Brief complex harmonic sphere map with l=0..2 (6 coefficients)
            >>> F = C.to_full_complex(); F
            Full complex harmonic sphere map with l=0..2 (9 coefficients)
            >>> F.to_list()
            [(12+0j), (-3-1j), (2+0j), (3-1j), (5-2j), (-4-1j), (3+0j), (4-1j), (5+2j)]
            >>> F.to_full_complex() is F
            True

        Many maps::

            >>> M = C
            >>> M = np.hstack((M[:,None], M[:,None], M[:,None], M[:,None]))
            >>> M = np.dstack((M, M))
            >>> M = harmonic_sphere_map(M, 0, 2); M
            (4, 2)-stack of brief complex harmonic sphere maps with l=0..2 (6 coefficients)
            >>> M.to_full_complex()
            (4, 2)-stack of full complex harmonic sphere maps with l=0..2 (9 coefficients)

        """
        if self.format is COMPLEX_FULL:
            return self
        elif self.format is REAL_FULL:
            return self.to_complex().to_full_complex()
        else:
            size = self.shape[1:]
            Nlm = lm_count_full(self.lmin, self.lmax)
            out = np.empty((Nlm,) + size, complex_dtype, order='F')
            for mapidx in np.ndindex(*size):
                idx = (slice(None),) + mapidx
                mapdatautils.alm_complex_brief2full(self.lmin, self.lmax, self[idx], out[idx])
            return _HarmonicSphereMap(out, self.lmin, self.lmax, COMPLEX_FULL)
        

    def to_complex(self):
        """
        ::
        
            >>> R = harmonic_sphere_map([[12], [-1, 1, 3], [-1, -2, 3, 4, 5]]); R
            Real harmonic sphere map with l=0..2 (9 coefficients)
            >>> C = R.to_complex(); C
            Brief complex harmonic sphere map with l=0..2 (6 coefficients)
            >>> x = C.to_array(); x[[0, 1, 3]] /= np.sqrt(2); x *= np.sqrt(2); x
            array([ 12.+0.j,   1.+0.j,   3.-1.j,   3.+0.j,   4.-2.j,   5.-1.j])

        Many maps::

            >>> M = R
            >>> M = np.hstack((M[:,None], M[:,None], M[:,None], M[:,None]))
            >>> M = np.dstack((M, M))
            >>> M = harmonic_sphere_map(M, 0, 2, is_complex=False); M
            (4, 2)-stack of real harmonic sphere maps with l=0..2 (9 coefficients)
            >>> M.to_complex()
            (4, 2)-stack of brief complex harmonic sphere maps with l=0..2 (6 coefficients)
        
        """
        self._check_not_pretending()
        if self.format is COMPLEX_BRIEF:
            return self
        elif self.format is COMPLEX_FULL:
            raise ValueError('Cannot convert full complex format to brief')
        elif self.format is REAL_FULL:
            size = self.shape[1:]
            Nlm = lm_count_brief(self.lmin, self.lmax)
            out = np.empty((Nlm,) + size, complex_dtype, order='F')
            for mapidx in np.ndindex(*size):
                idx = (slice(None),) + mapidx
                mapdatautils.alm_real2complex(self.lmin, self.lmax, self[idx], out[idx])
            return _HarmonicSphereMap(out, self.lmin, self.lmax, COMPLEX_BRIEF)
        else:
            raise AssertionError()
        
    def to_real(self):
        """
        >>> C = harmonic_sphere_map([[12], [1, 3-1j], [3, 4-2j, 5-1j]]); C
        Brief complex harmonic sphere map with l=0..2 (6 coefficients)
        >>> R = C.to_real(); R
        Real harmonic sphere map with l=0..2 (9 coefficients)
        >>> x = R.to_array(); x[[0, 2, 6]] *= np.sqrt(2); x /= np.sqrt(2); x
        array([ 12.,  -1.,   1.,   3.,  -1.,  -2.,   3.,   4.,   5.])

        Many maps::

            >>> M = C
            >>> M = np.hstack((M[:,None], M[:,None], M[:,None], M[:,None]))
            >>> M = np.dstack((M, M))
            >>> M = harmonic_sphere_map(M, 0, 2); M
            (4, 2)-stack of brief complex harmonic sphere maps with l=0..2 (6 coefficients)
            >>> M.to_real()
            (4, 2)-stack of real harmonic sphere maps with l=0..2 (9 coefficients)
        """
        self._check_not_pretending()
        if self.format is REAL_FULL:
            return self
        elif self.format is COMPLEX_FULL:
            raise ValueError('Cannot convert full complex format to brief')            
        else:
            size = self.shape[1:]
            Nlm = lm_count_full(self.lmin, self.lmax)
            out = np.empty((Nlm,) + size, real_dtype, order='F')
            for mapidx in np.ndindex(*size):
                idx = (slice(None),) + mapidx
                mapdatautils.alm_complex2real(self.lmin, self.lmax, self[idx], out[idx])
            return _HarmonicSphereMap(out, self.lmin, self.lmax, REAL_FULL)

    def to_pixel(self, Nside=None, Npix=None, dtype=None, out=None):
        """
        ::
        
            >>> insh = harmonic_sphere_map(0, 0, 4)
            >>> insh.to_pixel(8)
            Pixel sphere map (ring-ordered, Nside=8, Npix=768)

        Convert a uniform map::        
            >>> shmap = harmonic_sphere_map([[23], [0, 0], [0, 0, 0]])
            >>> pixmap = shmap.to_pixel(8); pixmap
            Pixel sphere map (ring-ordered, Nside=8, Npix=768)
            >>> np.allclose(pixmap, 23.0 / np.sqrt(4*np.pi))
            True

        One can convert multiple maps at once::

            >>> M = shmap
            >>> M = np.hstack((M[:,None], M[:,None], M[:,None], M[:,None]))
            >>> M = np.dstack((M, M))
            >>> M = harmonic_sphere_map(M, 0, 2); M
            (4, 2)-stack of brief complex harmonic sphere maps with l=0..2 (6 coefficients)
            >>> x = M.to_pixel(4); x
            (4, 2)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)
            >>> np.allclose(x, 23.0 / np.sqrt(4*np.pi))
            True

        """
        self._check_not_pretending()
        if self.format is not COMPLEX_BRIEF:
            return self.to_complex().to_pixel(Nside, Npix, dtype, out)
        if Nside is None:
            if Npix is None: raise ValueError("Provide Nside or Npix")
            Nside = npix2nside(Npix)
        if Npix is None:
            Npix = nside2npix(Nside)
        if dtype is None:
            dtype = self.dtype
        dtype = to_real_dtype(dtype)
        # for now, only double...
        assert dtype == np.float64
        
        size = self.shape[1:]
        if out is None:
            out = np.empty((Npix,) + size, dtype, order='F')

        # Todo: Optimize -- have lmin in complexpacked2complexmatrix
        if self.lmin == 0:
            source = self
        else:
            source = np.empty((l_to_lm(self.lmax + 1),) + size, complex_dtype, order='F')
            source[:l_to_lm(self.lmin)] = 0
            source[l_to_lm(self.lmin):] = self

        for mapidx in np.ndindex(*size):
            idx = (slice(None),) + mapidx
            alm_matrix = mapdatautils.alm_complexpacked2complexmatrix(source[idx])
            mapdatautils.alm2map(Nside, alm_matrix, out=out[idx])

        return _PixelSphereMap(out, pixel_order='ring')

    def rotate(self, psi, theta, phi):
        """
        Returns a rotated copy of self. O(lmax^3).

        ::
        
            >>> M = harmonic_sphere_map([[1, 0], [1, 0, 0]], lmin=1)
            >>> R = M.rotate(0, np.pi, 0); R
            Brief complex harmonic sphere map with l=1..2 (5 coefficients)
            >>> np.round(R.to_array(), 4)
            array([-1.+0.j, -0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j])

        One can convert multiple maps at once::

        TODO write tests

        """
        from healpix import rotate_alm_d
        self._check_not_pretending()

        if psi == 0 and theta == 0 and phi == 0:
            return self.copy()
        
        if self.format is not COMPLEX_BRIEF:
            x = self.to_complex().rotate(psi, theta, phi)
            if self.format is REAL_FULL:
                return x.to_real()
            else:
                assert False

        size = self.shape[1:]

        # Todo: Optimize -- have lmin in complexpacked2complexmatrix
        out = np.empty((l_to_lm(self.lmax + 1),) + size, complex_dtype, order='F')
        out[:l_to_lm(self.lmin), ...] = 0
        out[l_to_lm(self.lmin):, ...] = self

        alm_matrix = np.empty((1, self.lmax + 1, self.lmax + 1), complex_dtype, order='F')

        for mapidx in np.ndindex(*size):
            idx = (slice(None),) + mapidx
            mapdatautils.alm_complexpacked2complexmatrix(out[idx], alm_matrix[0,:,:])
            rotate_alm_d(self.lmax, alm_matrix, psi, theta, phi)
            mapdatautils.alm_complexmatrix2complexpacked(alm_matrix[0,:,:], out[idx])
        out = out[l_to_lm(self.lmin):, ...]
        return _HarmonicSphereMap(out, self.lmin, self.lmax, COMPLEX_BRIEF)

    def plot(self, Nside=64, ax=None, title=None):
        self._check_not_pretending()
        Nside = int(Nside)
        self.to_pixel(Nside).plot(ax, title=title)

##     def change_lmax(self, lmax):
##         if lmax == self.lmax:
##             return self.copy()
##         else:
##             numcoefs = l_to_lm(lmax + 1) - l_to_lm(lmin)
##             newdata = np.zeros((numcoefs,) + self.shape[1:], complex_dtype)
##             coefs_to_copy = l_to_lm(min(lmax, self.lmax) + 1)
##             newdata[:coefs_to_copy,...] = self[:coefs_to_copy,...]
##             return _HarmonicSphereMap(newdata, lmax)

    def to_harmonic(self, lmax):
        self._check_not_pretending()
        return self.change_lmax(lmax)

    def to_fits(self, filename, fitsformat='healpix', dataformat=None):
        if self.format is not COMPLEX_BRIEF:
            self.to_complex().to_fits(filename, fitsformat=fitsformat, dataformat=dataformat)
            return
        if fitsformat != 'healpix':
            raise NotImplementedError()
        if dataformat is None:
            if self.dtype == np.cdouble:
                dataformat = 'D'
            elif self.dtype == np.csingle:
                dataformat = 'E'
            else:
                raise NotImplementedError()
            
        import pyfits
        import os
        if self.ndim != 1:
            raise NotImplementedError()
        if os.path.exists(filename):
            os.unlink(filename)

        ls = broadcast_l_to_lm(np.arange(self.lmin, self.lmax + 1), self.lmin)
        ms = m_range_lm(self.lmin, self.lmax)
        index = ls**2 + ls + ms + 1

        cols = pyfits.ColDefs([
            pyfits.Column(name='index=l^2+l+m+1', array=index, format='J'), # 4-byte int
            pyfits.Column(name='alm (real)', array=self.real, format=dataformat),
            pyfits.Column(name='alm (imaginary)', array=self.imag, format=dataformat)])

        tab = pyfits.new_table(cols)
        tab.header.update('MIN-LPOL', self.lmin)
        tab.header.update('MAX-LPOL', self.lmax)
        tab.header.update('MAX-MPOL', self.lmax)

        pyfits.HDUList([pyfits.PrimaryHDU(), tab]).writeto(filename)    


def harmonic_sphere_map_from_fits(filename, extno=1, fitsformat='healpix', dtype=np.cdouble):
    if fitsformat != 'healpix':
        raise NotImplementedError()
    import pyfits
    hdulist = pyfits.open(filename)
    try:
        ext = hdulist[extno]
        lmin = ext.header['MIN-LPOL']
        lmax = ext.header['MAX-LPOL']

        out = harmonic_sphere_map(0, lmin, lmax, is_complex=True)
        ls = broadcast_l_to_lm(np.arange(lmin, lmax + 1), lmin)
        ms = m_range_lm(lmin, lmax)
        running_index = ls**2 + ls + ms + 1
        index = ext.data.field(0).ravel()
        alm_real = ext.data.field(1).ravel()
        alm_imag = ext.data.field(2).ravel()
        # If Cythonizing this we can make it fast anyway and loose the special case
        if np.all(running_index == index):
            out.real = alm_real
            out.imag = alm_imag
        else:
            raise NotImplementedError('Too lazy to implement non-contiguous indices')
    finally:
        hdulist.close()
    return out
    

def dipole_sphere_map(offset, amplitude, theta, phi):
    x, y, z = angles_to_vector(theta, phi)
    c = np.sqrt(2*np.pi / 3) * amplitude
    return harmonic_sphere_map([[offset],
                                [c*np.sqrt(2)*z, -c*(x - 1j*y)]])

def random_real_harmonic_sphere_map(lmin, lmax, size=(), state=np.random):
    if np.isscalar(size):
        size = (size,)
    Nlm = lm_count_full(lmin, lmax)
    return _HarmonicSphereMap(state.standard_normal((Nlm,) + size).astype(real_dtype),
                              lmin, lmax,
                              format=REAL_FULL)

def simulate_harmonic_sphere_maps(lmin, lmax, size=None, state=np.random):
    """
    Simulates standard normal maps in spherical harmonic space.
    
    INPUT:
    
     - ``n`` - The number of maps to simulate

    EXAMPLES::

    >>> simulate_harmonic_sphere_maps(0, 3)
    Brief complex harmonic sphere map with l=0..3 (10 coefficients)
    >>> simulate_harmonic_sphere_maps(2, 3, size=10)
    (10,)-stack of brief complex harmonic sphere maps with l=2..3 (7 coefficients)

    
    """
    if size is None:
        size = ()
    if not isinstance(size, tuple):
        size = (size,)
    real_coefs = (lmax+1)**2
    complex_coefs = l_to_lm(lmax+1)
    lmin_lm = l_to_lm(lmin)
    Zreal = state.standard_normal((real_coefs,) + size).astype(real_dtype)
    Zcomplex = np.empty((complex_coefs,) + size, complex_dtype)
    Zcomplex
    if len(size) > 1:
        raise NotImplementedError()
    # TODO: Have lmin in realpacked2complexpacked
    # For now, just slice away output for 0:lmin
    if size == ():
        mapdatautils.alm_realpacked2complexpacked(Zreal, out=Zcomplex)
    else:
        for i in range(size[0]):
            mapdatautils.alm_realpacked2complexpacked(Zreal[:,i], out=Zcomplex[:,i])
    return _HarmonicSphereMap(Zcomplex[l_to_lm(lmin):, ...], lmin, lmax, COMPLEX_BRIEF)

def simulate_pixel_sphere_maps(Nside, size=None, pixel_order='ring', state=np.random):
    """
    Simulates standard normal maps in pixel space.
    
    INPUT:
    
     - ``Nside`` - Size of maps
     - size (optional)

    EXAMPLES::

    >>> simulate_pixel_sphere_maps(4)
    Pixel sphere map (ring-ordered, Nside=4, Npix=192)
    >>> simulate_pixel_sphere_maps(4, (2,3,4))
    (2, 3, 4)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)

    
    """
    Npix = nside2npix(Nside)
    if size is None:
        size = ()
    elif not isinstance(size, tuple):
        size = (size,)
    # A bit complicated in order to simulate Fortran-ordered array;
    # simulate the reverse shape and then transpose the result
    size = list(size)
    size.reverse()
    size = tuple(size) + (Npix,)
    return _PixelSphereMap(state.standard_normal(size).T, pixel_order=pixel_order)


def sphere_maps_from_fits(path, extno=1, fields=None, dtype=real_dtype):
    """
    Loads a HEALPix pixel sphere map from a FITS file.

    INPUT:

     - fields - If given, is an iterable of keys (names or map indices)
                which should be loaded. If None, all the maps are loaded.
                
    OUTPUT:

    A 2D _PixelSphereMap.

    EXAMPLES::

    Set up example FITS file::
    
        >>> from tempfile import mkstemp
        >>> import os
        >>> fd, path = mkstemp(suffix='.fits'); os.close(fd)
        >>> D = dipole_sphere_map(1, .3, 1, .2).to_pixel(Nside=8)
        >>> D.to_fits(path)

    Load it again::

        >>> M = sphere_maps_from_fits(path)
        >>> M
        (1,)-stack of pixel sphere maps (nested-ordered, Nside=8, Npix=768)
        >>> M = M.to_ring()
        >>> np.all(M[:,0] == D)
        True

        >>> M = sphere_maps_from_fits(path, ['TEMPERATURE'])
        >>> np.all(M.to_ring()[:,0] == D)
        True


    TESTS::

        >>> M = sphere_maps_from_fits(path, ['NONEXISTING'])
        Traceback (most recent call last):
            ...
        ValueError: field named NONEXISTING not found.
    """
    import pyfits
    hdulist = pyfits.open(path)
    try:
        ext = hdulist[extno]
        if not ext.header['ORDERING'] == 'NESTED':
            raise RuntimeError("Unexpected data")
        data = ext.data.view(np.ndarray)
        result = []
        if fields is None:
            fields = data.dtype.names
        if require_single_field and len(data.dtype.names) > 1:
            raise IOError('More than one field was present in the FITS table: %s:%d' %
                          (path, extno))
        # Some FITS files stores interleaved sets of 1024 contiguous bytes...
        # do sanity checks on these.
        blocklen = None
        for idx in range(len(data.dtype)):
            dshape = data.dtype[idx].shape
            if dshape != ():
                if len(dshape) != 1:
                    raise RuntimeError("Unexpected data (2)")
                if blocklen is None:
                    blocklen = dshape[0]
                elif blocklen != dshape[0]:
                    raise RuntimeError("Unexpected data (3)")
        if blocklen is None:
            blocklen = 1
        out = np.empty((data.shape[0] * blocklen, len(fields)), dtype=dtype)
        for field, col in zip(fields, range(len(fields))):
             if isinstance(field, int):
                 if field == 0:
                     raise ValueError('Field indices are 1-based, 0 not allowed')
                 field = data.dtype.names[field - 1]
             out[:,col] = data[field].ravel()
    finally:
        hdulist.close()
    return pixel_sphere_map(out, pixel_order='nested')


try:
    import oomatrix
except ImportError:
    pass
else:
    class SphereMapsOOMatrixBehaviour(object):
        """
        If operated on by square matrices, keep it a map of the same type.
        Otherwise, return a pure np.ndarray.

        TESTS::
        
            >>> import oomatrix
            >>> M = pixel_sphere_map(0, 4, shape=10)
            >>> H = M.to_harmonic(2, 4)

        Sparse matrices are known not to preserve subclass::

            >>> I = oomatrix.identity_matrix(192).sparse_matrix()

        Square matrices preserve type, rectangular do not::
        
            >>> I * M
            (10,)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)
            >>> R = I[:100, :]
            >>> type(R * M)
            <type 'numpy.ndarray'>
            >>> type(M.T * I) # .T removes map-iness
            <type 'numpy.ndarray'>
            >>> M * oomatrix.identity_matrix(10)
            (10,)-stack of pixel sphere maps (ring-ordered, Nside=4, Npix=192)

            >>> I = oomatrix.identity_matrix(12).sparse_matrix()
            >>> I * H
            (10,)-stack of brief complex harmonic sphere maps with l=2..4 (12 coefficients)
            >>> type(I[:8, :] * H)
            <type 'numpy.ndarray'>
            >>> type(H.T * I)
            <type 'numpy.ndarray'>
    
        """
        
        def allocate_out(self, op, inshape, outshape, input, shape, dtype):
            out = np.empty(shape, dtype=dtype)
            if inshape == outshape:
                out = out.view(type(input))
                out._enable_subclass = True
                input.copy_properties_to(out)
            return out

        def wrap_out(self, op, inshape, outshape, input, out):
            if inshape == outshape and input._enable_subclass:
                out = out.view(type(input))
                out._enable_subclass = True
                input.copy_properties_to(out)
            else:
                out = out.view(np.ndarray)
            return out

        def as_array(self, op, inshape, outshape, input, dtype):
            return input

    oomatrix.add_default_data_behaviour([_PixelSphereMap, _HarmonicSphereMap],
                                        SphereMapsOOMatrixBehaviour())
 
