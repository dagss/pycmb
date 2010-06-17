from __future__ import division

##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

import healpix as heal
import numpy as np
from numpy import r_
cimport numpy as np
cimport cython
import sys

from cmbtypes import real_dtype, complex_dtype, index_dtype
from cmbtypes cimport index_t, real_t, complex_t

complexdtype = np.complex

cdef extern from "math.h":
    double fabs(double)

DATA = "/uio/arkimedes/s07/dagss/cmb/"
LCDM_FILE = DATA + "data/wmap_lcdm_sz_lens_wmap5_cl_v3.dat"

def log(*msg):
    sys.stderr.write(repr(msg))
    sys.stderr.write(b'\n')

def load_Cl():
    """
    Return value: (Cl, lmax)
    """
    dtype_lcdm = {'names' : ('l','lcdm','sz','lens'),
                  'formats':(np.int, real_dtype, real_dtype, real_dtype)}
    cl = np.loadtxt(LCDM_FILE, dtype=dtype_lcdm)
    l = cl['l']
    lcdm = cl['lcdm'] * 2*np.pi / (l*(l+1))
    Cl = r_[0, 0, lcdm]
    return Cl, l[-1]

def Cl2S(lmin, lmax, Cl):
    """
    >>> almvar = Cl2S(0, 4, r_[:5])
    >>> list(almvar.astype(int))
    [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    >>> almvar = Cl2S(2, 2, r_[2:3])
    >>> list(almvar.astype(int))
    [2, 2, 2, 2, 2]
    >>> almvar = Cl2S(2, 3, r_[2:4])
    >>> list(almvar.astype(int))
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    """

    # i_lm = l*l + l + m
    
    Sdiag = np.zeros((lmax+1)**2 - lmin**2, real_dtype)
    for l in range(lmin, lmax+1):
        al0_idx = l*l + l - lmin*lmin
#        print al0_idx
        Sdiag[(al0_idx-l):(al0_idx+l+1)] = Cl[l-lmin]
    return Sdiag

def compute_power_spectrum_real(int lmin, int lmax, np.ndarray[real_t] alm):
    """
    
    INPUT:
      alm - packed alm array
    OUTPUT:
      sigma_l
    EXAMPLE:

    >>> lmax = 2
    >>> x = Cl2S(0, lmax, np.array([1, 2, 3], real_dtype))
    >>> print calcsigma(0, lmax, x)
    [ 1.  4.  9.]
    >>> print calcsigma(1, lmax, x[1**2:])
    [ 4.  9.]
    """
    cdef int l, m, idx
    cdef real_t s
    cdef np.ndarray[real_t] out = np.zeros((lmax+1-lmin,), real_dtype)
    for l in range(lmin, lmax+1):
        idx = l*l + l - lmin*lmin
        s = 0
        for m in range(-l, l+1):
            s += alm[idx + m]**2
        out[l - lmin] = s / (2*l+1)
    return out

def output_alm(nside, alm, filename):
    dummy, lmax, mmax = alm.shape
    lmax -= 1
    mmax -= 1
    assert dummy==1, "Polarization not supported yet"
    smax = heal.nside2npix(nside) # map size
    map = np.empty((smax,), dtype=np.float64)
    heal.alm2map_sc_d(nside, lmax, mmax, alm, map)
    output_ringmap(map, filename)

def output_ringmap(map, filename):
    import os
    maps = map[:,np.newaxis]
    nside = heal.npix2nside(map.shape[0])
    if os.path.exists(filename): os.unlink(filename)
    heal.convert_ring2nest_d(nside, maps)
    heal.output_map_d(maps, filename)

cpdef int l2cpos(int l):
    return (l+1)*l//2

#@cython.boundscheck(False)
def alm_real2complex(lmin, lmax, almR_, out_=None):
    """
    >>> x = np.r_[4:16].astype(real_dtype)
    >>> z = alm_real2complex(2, 3, x)
    >>> z *= np.sqrt(2)
    >>> z[[0,3]] /= np.sqrt(2)
    >>> print z
    [  6. +0.j   7. +5.j   8. +4.j  12. +0.j  13.+11.j  14.+10.j  15. +9.j]
    
    """
    cdef np.ndarray[real_t] almR = almR_
    cdef np.ndarray[complex_t] out = out_
    cdef unsigned int l, Cidx, Ridx_l0, m
    cdef complex_t tmp
    cdef double invsqrt2 = 1/np.sqrt(2)
    size = l2cpos(lmax+1) - l2cpos(lmin)
    if out is None:
        out = np.zeros(size, complex_dtype)
    else:
        assert out.shape[0] >= size
    Cidx = 0
    for l in range(lmin, lmax+1):
        Ridx_l0 = l*l + l - lmin*lmin
        tmp.real = almR[Ridx_l0]
        tmp.imag = 0
        out[Cidx] = tmp
        Cidx += 1
        for m in range(1, l+1):
            tmp.real = almR[Ridx_l0 + m] * invsqrt2
            tmp.imag = almR[Ridx_l0 - m] * invsqrt2
            out[Cidx] = tmp
            Cidx += 1
    return out

def alm_complex2real(lmin, lmax, almC_, out_=None):
    """

    lmin=2, lmax=3:
    
    >>> z = np.array(
    ... [  6. +0.j,   7. +5.j,   8. +4.j,  12. +0.j,  13.+11.j,  14.+10.j,  15. +9.j]
    ... ).astype(complex_dtype)
    >>> x = alm_complex2real(2, 3, z)
    >>> x /= np.sqrt(2)
    >>> x[[2,8]] *= np.sqrt(2)
    >>> print x
    [  4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.  15.]

    lmin=0, lmax=1:

    >>> z = np.array([4, 3, 2+1j], complex_dtype)
    >>> x = alm_complex2real(0, 1, z)
    >>> x[[1, 3]] /= np.sqrt(2)
    >>> print x
    [ 4.  1.  3.  2.]
    
    
    """
    cdef np.ndarray[complex_t] almC = almC_
    cdef np.ndarray[real_t] out = out_
    cdef unsigned int l, Cidx, Ridx_l0
    cdef complex_t tmp
    cdef int m
    cdef double sqrt2 = np.sqrt(2)
    size = (lmax+1)**2 - lmin**2
    if out is None:
        out = np.zeros(size, real_dtype)
    else:
        assert out.shape[0] >= size
    Cidx = 0
    for l in range(lmin, lmax+1):
        Ridx_l0 = l*l + l - lmin*lmin
        out[Ridx_l0] = almC[Cidx].real
        Cidx += 1
        for m in range(1, l+1):
            tmp = almC[Cidx]
            out[Ridx_l0 + m] = tmp.real * sqrt2
            out[Ridx_l0 - m] = tmp.imag * sqrt2
            Cidx += 1
    return out


def alm_complex_brief2full(index_t lmin, index_t lmax,
                          np.ndarray[complex_t] arr,
                          np.ndarray[complex_t] out=None):
    """
    Converts an harmonic map given by m>=0-coefficients into a map
    with all -l <= m <= l coefficients (i.e. with redundant data).

    lmin=1, lmax=2:
    
    >>> z = np.array(
    ... [  1. +0.j,   2. +3.j,   4. +0.j,  5.+6j,  7.+8.j],
    ... complex_dtype)
    >>> x = alm_complex_brief2full(1, 2, z)
    >>> print x.real
    [-2.  1.  2.  7. -5.  4.  5.  7.]
    >>> print x.imag
    [ 3.  0.  3. -8.  6.  0.  6.  8.]

    lmin=0, lmax=1:

    >>> z = np.array([4, 3, 2+1j], complex_dtype)
    >>> x = alm_complex_brief2full(0, 1, z)
    >>> print x
    [ 4.+0.j -2.+1.j  3.+0.j  2.+1.j]
    
    """
    cdef index_t arr_l0, out_l0, l, minus_one_to_m
    cdef complex_t coef
    cdef int m
    if out is None:
        out = np.zeros(lm_count_full(lmin, lmax), complex_dtype)
    elif not out.shape[0] >= lm_count_full(lmin, lmax):
        raise ValueError()
    for l in range(lmin, lmax+1):
        arr_l0 = lm_to_idx_brief(l, 0, lmin)
        out_l0 = lm_to_idx_full(l, 0, lmin)
        minus_one_to_m = 1
        out[out_l0] = arr[arr_l0]
        for m in range(1, l+1):
            minus_one_to_m = -minus_one_to_m
            coef = arr[arr_l0 + m]
            out[out_l0 - m] = minus_one_to_m * coef.conjugate()
            out[out_l0 + m] = coef
    return out

def alm_complexpacked2complexmatrix(np.ndarray[complex_t] alm,
                                    np.ndarray[complex_t, ndim=2] out=None,
                                    order='F'):
    """
    >>> y = np.array([1, 2, 3+1j, 4, 5+2j, 6+3j]).astype(complex_dtype)
    >>> M = alm_complexpacked2complexmatrix(y)
    >>> np.round(M, 1)
    array([[ 1.+0.j,  0.+0.j,  0.+0.j],
           [ 2.+0.j,  3.+1.j,  0.+0.j],
           [ 4.+0.j,  5.+2.j,  6.+3.j]])
    """
    # Solve alm.shape[0] == (lmax + 2)(lmax + 1)/2:
    cdef index_t lmax = (-3 + int(np.sqrt(1 + 8 * alm.shape[0]))) // 2
    cdef index_t mmax = lmax
    cdef index_t l, m, alm_idx
    if out is None:
        out = np.zeros((lmax + 1, mmax + 1), dtype=complex_dtype,
                       order=order)
    alm_idx = 0
    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            out[l, m] = alm[alm_idx]
            alm_idx += 1
    return out

def alm_complexmatrix2complexpacked(np.ndarray[complex_t, ndim=2] alm,
                                    np.ndarray[complex_t] out=None):
    """
    >>> M = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j],
    ...               [ 2.+0.j,  3.+1.j,  0.+0.j],
    ...               [ 4.+0.j,  5.+2.j,  6.+3.j]])
    >>> alm_complexmatrix2complexpacked(M)
    array([ 1.+0.j,  2.+0.j,  3.+1.j,  4.+0.j,  5.+2.j,  6.+3.j])

    >>> M = np.array([[ 1.+0.j,  0.+0.j],
    ...               [ 2.+0.j,  3.+1.j],
    ...               [ 4.+0.j,  5.+2.j]])
    >>> alm_complexmatrix2complexpacked(M)
    array([ 1.+0.j,  2.+0.j,  3.+1.j,  4.+0.j,  5.+2.j,  0.+0.j])
    
    """
    cdef index_t lmax = alm.shape[0] - 1
    cdef index_t mmax = alm.shape[1] - 1
    cdef index_t l, m, out_idx
    if out is None:
        out = np.zeros(l_to_lm(lmax + 1), dtype=complex_dtype)
    out_idx = 0
    for l in range(0, lmax + 1):
        for m in range(0, min(l, mmax) + 1):
            out[out_idx + m] = alm[l, m]
        out_idx += (l + 1)
    return out

cdef extern from *:
    ctypedef int ssize_t

def alm_realpacked2complexpacked(np.ndarray[real_t] alm, np.ndarray[complex_t] out=None):
    """
    Converts alm in real, packed vector format to complex, packed format.

    >>> R = np.r_[:16].astype(real_dtype)    
    >>> C = alm_realpacked2complexpacked(R)
    >>> C[[2, 4, 5, 7, 8, 9]] *= np.sqrt(2)
    >>> C.real
    array([  0.,   2.,   3.,   6.,   7.,   8.,  12.,  13.,  14.,  15.])
    >>> C.imag
    array([  0.,   0.,   1.,   0.,   5.,   4.,   0.,  11.,  10.,   9.])
    """
    cdef index_t lmax = <ssize_t>(np.sqrt(alm.shape[0])) - 1
    cdef index_t num_c_alms = l_to_lm(lmax + 1)
    if out is None:
        out = np.empty(num_c_alms, complex_dtype)
    else:
        if out.ndim != 1 or out.shape[0] < num_c_alms:
            raise ValueError("out array to small or wrong ndim")
    
    cdef unsigned l, m, l_real, l_complex
    cdef complex_t tmp
    cdef real_t invsqrt2 = 1 / np.sqrt(2)
    for l in range(lmax + 1):
        l_real = l*l + l
        l_complex = l_to_lm(l)        
        tmp.real = alm[l_real]
        tmp.imag = 0
        out[l_complex] = tmp
        for m in range(1, l+1):
            tmp.real = alm[l_real + m] * invsqrt2
            tmp.imag = alm[l_real - m] * invsqrt2
            out[l_complex + m] = tmp
    return out


def alm_realpacked2complexmatrix(int lmax, np.ndarray[real_t] alm, order='F'):
    """
    Converts alm in real, packed vector format to
    a HealPix complex matrix format.
    >>> y = np.r_[:16].astype(real_dtype)
    >>> print y.astype(np.int)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

    #l=0|   1    |      2       |         3
    
    >>> x = alm_realpacked2complexmatrix(3, y)
    >>> undo_invsq2 = np.ones((4,4)) * np.sqrt(2)
    >>> undo_invsq2[:,0] = 1
    >>> x = x * undo_invsq2
    >>> print np.round(x, 2)
    [[  0. +0.j   0. +0.j   0. +0.j   0. +0.j]
     [  2. +0.j   3. +1.j   0. +0.j   0. +0.j]
     [  6. +0.j   7. +5.j   8. +4.j   0. +0.j]
     [ 12. +0.j  13.+11.j  14.+10.j  15. +9.j]]
    """
    cdef np.ndarray[complex_t, ndim=2] result = np.zeros((lmax+1, lmax+1), complexdtype,
                                                         order=order)
    cdef unsigned l, m, l0
    cdef complex_t tmp
    cdef real_t invsqrt2 = 1 / np.sqrt(2)
    for l in range(lmax + 1):
        l0 = l*l + l
        tmp.real = alm[l0]
        tmp.imag = 0
        result[l, 0]= tmp
        for m in range(1, l+1):
            tmp.real = alm[l0 + m] * invsqrt2
            tmp.imag = alm[l0 - m] * invsqrt2
            result[l, m] = tmp
    return result

def alm_complexmatrix2realpacked(np.ndarray[complex_t, ndim=2] almC, order='F'):
    """
    >>> almC = np.zeros((4,4), dtype=complex_dtype)
    >>> almC.real = np.array(
    ... [[ 1,  0,  0,  0],
    ... [ 2,  3,  0,  0],
    ... [ 4,  5,  6,  0],
    ... [ 7, 8, 9, 10]])
    >>> almC.imag = np.array(
    ... [[ 0,  0,  0,  0],
    ...  [ 0,  1,  0,  0],
    ...  [ 0,  2,  3,  0],
    ...  [ 0,  4,  5,  6]])
    >>> x = alm_complexmatrix2realpacked(almC)
    >>> x /= np.sqrt(2)
    >>> l = np.r_[:4]
    >>> x[l**2 + l] *= np.sqrt(2)
    >>> print x
    [  1.   1.   2.   3.   3.   2.   4.   5.   6.   6.   5.   4.   7.   8.   9.
      10.]

    
    """
    cdef unsigned lmax = almC.shape[0] - 1
    cdef np.ndarray[real_t] alm = np.zeros(((lmax+1)*(lmax+1),), dtype=real_dtype,
                                           order=order)
    cdef unsigned l, m, l0
    cdef complex_t tmp
    cdef real_t sqrt2 = np.sqrt(2)

    for l in range(lmax + 1):
        l0 = l*l + l
        alm[l0] = almC[l, 0].real
        for m in range(1, l+1):
            tmp = almC[l, m]
            alm[l0 + m] = tmp.real * sqrt2
            alm[l0 - m] = tmp.imag * sqrt2
    return alm

def packedalm2map(int Nside, alm):
    """
##     >>> lmax = 10
##     >>> N_side = 1024
##     >>> alm = np.r_[1:(lmax+1)**2+1].astype(real_dtype)
##     >>> map = packedalm2map(N_side, alm)
##     >>> alm2 = map2packedalm(lmax, map)
##     >>> print np.linalg.norm(alm - alm2) < 10**-3
##     True
    
    """
    lmax = np.sqrt(alm.shape[0]) - 1
    almC = alm_realpacked2complexmatrix(lmax, alm)
    return alm2map(Nside, almC)

def map2packedalm(int lmax, map, weight_ring):
    almC = map2alm(lmax, map, weight_ring)
    return alm_complexmatrix2realpacked(almC)

def make_out_arr(out, shape, dtype):
    if out is None:
        return np.zeros(shape, dtype)
    else:
        assert out.shape == shape, "%r != %r" % (out.shape, shape)
        assert out.dtype == dtype
        return out

def alm2map(int Nside, almC, out=None):
    cdef int Npix = heal.nside2npix(Nside)
    cdef int lmax = almC.shape[0] - 1
    cdef int mmax = almC.shape[1] - 1
    map = make_out_arr(out, (Npix,), real_dtype)
    heal.alm2map_sc_d(Nside, lmax, mmax,
                      almC[np.newaxis,...],
                      map)
    if np.any(np.isnan(map)):
        raise RuntimeError("nan")
    return map

#def packedalm2map(int Nside, alm, out=None):
#    alm_realpacked2complexmatrix

def map2alm(int lmax, map, weight_ring, out=None):
    cdef int Nside = heal.npix2nside(map.shape[0])
    cdef int mmax = lmax
    almC = make_out_arr(out, (lmax+1,lmax+1), complex_dtype)
    heal.map2alm_sc_d(Nside, lmax, mmax,
                      map,
                      almC[None,...],
                      weight_ring)
    if np.any(np.isnan(almC)):
        raise RuntimeError("nan")
    return almC

def num_alm_to_lmax(num_alm, is_complex=True):
    if is_complex:
        # Solve num_alm == (lmax + 2)(lmax + 1)/2:
        return int((-3 + int(np.sqrt(1 + 8 * num_alm))) // 2)
    else:
        return int(np.sqrt(num_alm)) - 1

def broadcast_l_to_lm(data_l, lmin=0, out=None):
    """
    Broadcasts one value per l to an array in packed complex ordering.
    TODO: Optimize array access.
    
    >>> list(broadcast_l_to_lm(np.r_[3:7]).astype(int))
    [3, 4, 4, 5, 5, 5, 6, 6, 6, 6]

    >>> list(broadcast_l_to_lm(np.r_[3:5], lmin=3).astype(int))
    [3, 3, 3, 3, 4, 4, 4, 4, 4]

    """
    
    cdef index_t lmax = lmin + data_l.shape[0] - 1
    cdef index_t lmin_lm = l_to_lm(lmin)
    cdef index_t numcoefs = l_to_lm(lmax + 1) - lmin_lm
    cdef index_t l, m
    
    if out is None:
        out = np.zeros(numcoefs, dtype=data_l.dtype)
    elif out.ndim != 1 or out.shape[0] < numcoefs:
        raise ValueError("Invalid out array")
    
    for l in range(lmin, lmax + 1):
        out[(l * (l+1)) // 2 - lmin_lm: ((l+1) * (l+2)) // 2 - lmin_lm] = data_l[l - lmin]
    return out

def broadcast_l_to_lm_full(data_l, lmin=0, out=None):
    """
    Broadcasts one value per l to an alm-array.
    TODO: Optimize array access.
    
    >>> list(broadcast_l_to_lm_full(np.r_[3:7]).astype(int))
    [3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]

    >>> list(broadcast_l_to_lm_full(np.r_[3:5], lmin=3).astype(int))
    [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    """
    
    cdef index_t lmax = lmin + data_l.shape[0] - 1
    cdef index_t numcoefs = (lmax + 1)**2 - lmin**2
    cdef index_t l
    
    if out is None:
        out = np.zeros(numcoefs, dtype=data_l.dtype)
    elif out.ndim != 1 or out.shape[0] < numcoefs:
        raise ValueError("Invalid out array")
    
    for l in range(lmin, lmax + 1):
        out[lm_to_idx_full(l, -l, lmin):lm_to_idx_full(l, l, lmin) + 1] = data_l[l - lmin]
    return out

    
def m_range_lm(index_t lmin, index_t lmax, np.ndarray[np.int64_t] out=None):
    """
    Fills a complex ordered spherical harmonic vector with the m indices.

    TODO: Compute from lmin rather than chopping

    >>> list(m_range_lm(0, 3))
    [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
    """
    cdef index_t l, m
    cdef index_t numcoefs = ((lmax + 1) * (lmax + 2)) // 2

    if out is None:
        out = np.zeros(numcoefs, np.int64)
    elif out.shape[0] < numcoefs:
        raise ValueError("Invalid out array")

    idx = 0
    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            out[idx] = m
            idx += 1
    return out[l_to_lm(lmin):]
    
def m_range_lm_full(index_t lmin, index_t lmax, np.ndarray[np.int64_t] out=None):
    """
    Fills an alm-array with the m indices.

    >>> list(m_range_lm_full(0, 2))
    [0, -1, 0, 1, -2, -1, 0, 1, 2]
    >>> list(m_range_lm_full(3, 4))
    [-3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    """
    cdef index_t l, m, idx_l, idx
    cdef index_t numcoefs = (lmax+1)**2 - lmin**2

    if out is None:
        out = np.zeros(numcoefs, np.int64)
    elif out.shape[0] < numcoefs:
        raise ValueError("Invalid out array")

    idx = 0
    for l in range(lmin, lmax + 1):
        idx_l = lm_to_idx_full(l, 0, lmin)
        for m in range(-l, l + 1):
            out[idx] = m
            idx += 1
    return out

cpdef index_t py_l_to_lm(index_t l, is_complex=True):
    if is_complex:
        return (l * (l+1)) // 2
    else:
        return l*l

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double acos(double)
    double atan2(double y, double x)
    double sqrt(double)

def array_ang2vec(np.ndarray[double, ndim=2] theta_phi):
    cdef index_t i
    cdef double theta, phi, sintheta
    if theta_phi.shape[1] != 2:
        raise ValueError()
    cdef np.ndarray[double, ndim=2] out = np.empty((theta_phi.shape[0], 3), np.double)
    for i in range(theta_phi.shape[0]):
        theta = theta_phi[i, 0]
        phi = theta_phi[i, 1]
        sintheta = sin(theta)
        out[i, 0] = sintheta * cos(phi)
        out[i, 1] = sintheta * sin(phi)
        out[i, 2] = cos(theta)
    return out

def array_vec2ang(np.ndarray[double, ndim=2] vecs):
    cdef index_t i
    cdef double x, y, z, phi, theta
    cdef double twopi = 2 * np.pi
    if vecs.shape[1] != 3:
        raise ValueError()
    cdef np.ndarray[double, ndim=2] out = np.empty((vecs.shape[0], 2), np.double)
    for i in range(vecs.shape[0]):
        x = vecs[i, 0]; y = vecs[i, 1]; z = vecs[i, 2]
        z /= sqrt(x**2 + y**2 + z**2)
        theta = acos(z)
        phi = 0
        if x != 0 or y != 0:
            phi = atan2(y, x) # phi in [-pi, pi]
        if phi < 0:
            phi += twopi # phi in [0, 2pi]
        out[i,0] = theta
        out[i,1] = phi
    return out

cdef class Vector:
    """
    Vector(x, y, z)

    Immutable.
    """
    
    cdef readonly real_t x, y, z

    def __init__(self, real_t x, real_t y, real_t z):
        self.x, self.y, self.z = x, y, z

    def __repr__(self):
        return "Vector(%.2f, %.2f, %.2f)" % (self.x, self.y, self.z)
        
    def get_ang(self):
        return heal.vec2ang(self.x, self.y, self.z)

    cpdef check_unity(self, eps=1e-8):
        if self.x**2 + self.y**2 + self.z**2 - 1 > eps:
            raise ValueError("Vector does not have unit length")

    cpdef Vector rotate_x(self, real_t ang):
        """
        Returns a new vector representing this vector rotated ``ang``
        radians around the X axis

        >>> Vector(0, 0, 1).rotate_x(np.pi)
        Vector(0.00, -0.00, -1.00)
        >>> Vector(1, 1, 0).rotate_x(np.pi/2)
        Vector(1.00, 0.00, 1.00)
        """
        cdef real_t x, y, z, c, s
        c = cos(ang)
        s = sin(ang)
        x = self.x
        y = c*self.y - s*self.z
        z = s*self.y + c*self.z
        return Vector(x, y, z)
        
    cpdef Vector rotate_y(self, real_t ang):
        """
        Returns a new vector representing this vector rotated ``ang``
        radians around the Y axis

        >>> Vector(0, 0, 1).rotate_y(np.pi)
        Vector(0.00, 0.00, -1.00)
        >>> Vector(1, 1, 0).rotate_y(np.pi/2)
        Vector(0.00, 1.00, -1.00)
        >>> Vector(1, 1, 1).rotate_y(np.pi/2)
        Vector(1.00, 1.00, -1.00)
        """
        cdef real_t x, y, z, c, s
        c = cos(ang)
        s = sin(ang)
        x = c*self.x + s*self.z
        y = self.y
        z = -s*self.x + c*self.z
        return Vector(x, y, z)

    cpdef Vector rotate_z(self, real_t ang):
        """
        Returns a new vector representing this vector rotated ``ang``
        radians around the Z axis

        >>> Vector(1, 0, 0).rotate_z(np.pi)
        Vector(-1.00, 0.00, 0.00)
        >>> Vector(0, 1, 1).rotate_z(np.pi/2)
        Vector(-1.00, 0.00, 1.00)
        """
        cdef real_t x, y, z, c, s
        c = cos(ang)
        s = sin(ang)
        x = c*self.x - s*self.y
        y = s*self.x + c*self.y
        z = self.z
        return Vector(x, y, z)

    cpdef Vector rotate_yz(self, real_t theta, real_t phi):
        """
        Returns a new vector representing this vector rotated
        first ``theta``  radians around the Y axis, then ``phi``
        radians around the Z axis.

        Vector(0, 0, 1).rotate_yz(theta, phi) yields a vector
        with coordinates (theta, phi).
        
        >>> Vector(1, 1, 1).rotate_yz(np.pi/2, np.pi/2)
        Vector(-1.00, 1.00, -1.00)
        >>> [round(x,2) for x in Vector(0, 0, 1).rotate_yz(1, 2).get_ang()]
        [1.0, 2.0]
        >>> [round(x,2) for x in Vector(0, 0, 1).rotate_yz(2, 1 + 20*np.pi).get_ang()]
        [2.0, 1.0]
        """
        cdef real_t x, y, z, cp, sp, ct, st
        ct = cos(theta); st = sin(theta)
        cp = cos(phi); sp = sin(phi)
        x = ct*cp*self.x - sp*self.y + st*cp*self.z
        y = ct*sp*self.x + cp*self.y + st*sp*self.z
        z = -st*self.x + ct*self.z
        return Vector(x, y, z)

    cpdef size_t pixidx_ring(self, Nside):
        """
        >>> Vector(0, 0, 1).pixidx_ring(4)
        0
        >>> Vector(1, 1, 1).pixidx_ring(4)
        42
        >>> Vector(0, 0, -1).pixidx_ring(4)
        188
        """
        return heal.vec2pix_ring(Nside, self.x, self.y, self.z)


def matrix_complex_to_real(np.ndarray[complex_t, ndim=2, mode='c'] C,
                           Py_ssize_t lmin, Py_ssize_t lmax,
                           np.ndarray[real_t, ndim=2, mode='c'] R=None):
    """
    Converts a complex matrix to real by doing the transformation R = U^H C U.
    Both matrices are assumed to store rows/columns for -l <= m <= l.

    Identity matrix should go through unchanged::

        >>> I = np.eye(16, dtype=complex_dtype)
        >>> R = matrix_complex_to_real(I, 0, 3)
        >>> np.allclose(R, I)
        True

    The rest has been tested only in tests/noisedense.py for now.
    """
    cdef Py_ssize_t l1, l2, m1, m2, m3, m4, R_idx, idx1, idx2, opp_idx1, opp_idx2
    cdef Py_ssize_t pow1, pow2, pow12, pow0=1, minus_one_to_m1, minus_one_to_m2
    cdef complex_t C_val, tmp, val
    cdef real_t sqrt_one_half = np.sqrt(.5)
        
    if not (C.shape[0] == C.shape[1] == lm_count_full(lmin, lmax)):
        raise ValueError("Invalid shape of C argument")
    Nlm_out = lm_count_full(lmin, lmax)
    if R is not None:
        if not (R.shape[0] == R.shape[1] == Nlm_out):
            raise ValueError("Invalid shape of out argument")
    else:
        R = np.empty((Nlm_out, Nlm_out), dtype=real_dtype)

    cdef np.ndarray[complex_t, ndim=2, mode='c'] T = np.zeros_like(C)

    # Compute T = U C
    for l1 in range(lmin, lmax + 1):
        for m1 in range(-l1, l1+1):
            idx1 = lm_to_idx_full(l1, m1, lmin)
            opp_idx1 = lm_to_idx_full(l1, -m1, lmin)
            minus_one_to_m1 = -1 if m1 % 2 == 1 else 1
            for l2 in range(lmin, lmax + 1):
                for m2 in range(-l2, l2 + 1):
                    idx2 = lm_to_idx_full(l2, m2, lmin)
                    if m1 < 0:
                        val = 1j * sqrt_one_half * (minus_one_to_m1 * C[idx1, idx2]
                                                    - C[opp_idx1, idx2])
                    elif m1 == 0:
                        val = C[idx1, idx2]
                    else:
                        val = (minus_one_to_m1 * C[opp_idx1, idx2]
                               + C[idx1, idx2]) * sqrt_one_half
                    T[idx1, idx2] = val

    # Compute R = T U^H = U C U^H
    for l1 in range(lmin, lmax + 1):
        for m1 in range(-l1, l1+1):
            idx1 = lm_to_idx_full(l1, m1, lmin)
            for l2 in range(lmin, lmax + 1):
                for m2 in range(-l2, l2 + 1):
                    idx2 = lm_to_idx_full(l2, m2, lmin)
                    opp_idx2 = lm_to_idx_full(l2, -m2, lmin)
                    minus_one_to_m2 = -1 if m2 % 2 == 1 else 1

                    if m2 < 0:
                        val = 1j * sqrt_one_half * (T[idx1, opp_idx2]
                                                    - minus_one_to_m2 * T[idx1, idx2])
                    elif m2 == 0:
                        val = T[idx1, idx2]
                    else:
                        val = sqrt_one_half * (T[idx1, idx2]
                                               + minus_one_to_m2 * T[idx1, opp_idx2])
                    R[idx1, idx2] = val.real # val.imag == 0, always

    return R


ctypedef int sparse_idx_t # what is used by by scipy.sparse
sparse_idx_dtype = np.intc # what is used by scipy.sparse
cdef real_t sqrt_one_half = np.sqrt(.5)

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_matrix_complex_to_real(C, index_t lmin, index_t lmax, real_t eps=1e-300):
    """
    Converts a sparse matrix C to real by doing the transformation R = U^H C U.
    
    The conversion happens using coordinate representation -- the elements
    of C are iterated, and the contribution of the final matrix "pushed"
    out (rather than the usual "pull" implementation of matrix multiplication).
    The end-result is a coo matrix with each element stored up to four elements;
    These must be summed over to get the final matrix.

    This strategy may change if it is necasarry for performance reasons to
    work directly with CSC (and if that is indeed faster).

    Basic functionality, lmin=0::
    
        >>> import scipy.sparse as sparse
        >>> I = sparse.eye(16, 16, dtype=complex_dtype)
        >>> R = sparse_matrix_complex_to_real(I, 0, 3)
        >>> np.allclose(R.todense(), np.eye(16))
        True
        >>> R.tocsc().nnz
        16

    With lmin != 0::
    
        >>> I = sparse.eye(15, 15, dtype=complex_dtype)
        >>> R = sparse_matrix_complex_to_real(I, 1, 3)
        >>> np.allclose(R.todense(), np.eye(15))
        True
        >>> R.tocsc().nnz
        15

    oomatrix API::
    
        >>> import oomatrix
        >>> I = oomatrix.identity_matrix(16, dtype=complex_dtype)
        >>> sparse_matrix_complex_to_real(I, 0, 3)
        16 by 16 sparse matrix of float64
    
    """
    use_oomatrix = False
    try:
        import oomatrix
    except ImportError:
        pass
    else:
        if isinstance(C, oomatrix.Matrix):
            C = C.sparse_matrix()._data
            use_oomatrix = True
    C = C.tocoo()
    cdef index_t nnz = C.nnz, N = C.shape[0]
    
    cdef np.ndarray[sparse_idx_t, mode='c'] C_row = C.row, C_col = C.col
    cdef np.ndarray[complex_t, mode='c'] C_data = C.data
    
    # Overallocate R for now (diagonal elements do not take up 4)
    cdef np.ndarray[sparse_idx_t, mode='c'] R_row = np.zeros(4*nnz, dtype=np.intc)
    cdef np.ndarray[sparse_idx_t, mode='c'] R_col = np.zeros(4*nnz, dtype=np.intc)
    cdef np.ndarray[real_t, mode='c'] R_data = np.zeros(4*nnz, dtype=real_dtype)
    cdef index_t R_idx = 0

    # Set up lookup array mapping idx -> (l,m)
    cdef np.ndarray[index_t, mode='c'] ls = broadcast_l_to_lm_full(
        np.arange(lmin, lmax+1, dtype=index_dtype), lmin=lmin)
    cdef np.ndarray[index_t, mode='c'] ms = m_range_lm_full(lmin, lmax)

    cdef complex_t val

    cdef index_t idx, row, col, l1, m1, l2, m2, idx1, idxopp1
    for idx in range(nnz):
        row = C_row[idx]
        col = C_col[idx]
        C_val = C_data[idx]
        l1, m1 = ls[row], ms[row]
        l2, m2 = ls[col], ms[col]
        minus_one_to_m1 = -1 if m1 % 2 == 1 else 1

            
        # We need to produce up to four values in the R matrix from this
        # one value in the C matrix. Let T = U C, and R = U C U^H.
        # We simply compute the 1 or 2 values we scatter to in T, and
        # then call another routine to do further scattering.
        # NOTE the changes in m1 below.
        idx1 = lm_to_idx_full(l1, m1, lmin)
        idxopp1 = lm_to_idx_full(l1, -m1, lmin)
        if m1 < 0:
            R_idx = _scatter_T_elem_to_R( # Same m1
                1j * minus_one_to_m1 * sqrt_one_half * C_val,
                lmin, idx1,
                l2, m2, R_idx, <sparse_idx_t*>R_row.data,
                <sparse_idx_t*>R_col.data, <real_t*>R_data.data, eps)
            R_idx = _scatter_T_elem_to_R( # Opposite m1
                minus_one_to_m1 * sqrt_one_half * C_val,
                lmin, idxopp1,
                l2, m2, R_idx, <sparse_idx_t*>R_row.data,
                <sparse_idx_t*>R_col.data, <real_t*>R_data.data, eps)
        elif m1 == 0:
            R_idx = _scatter_T_elem_to_R(
                C_val,
                lmin, idx1,
                l2, m2, R_idx, <sparse_idx_t*>R_row.data,
                <sparse_idx_t*>R_col.data, <real_t*>R_data.data, eps)
        else:
            R_idx = _scatter_T_elem_to_R( # Same m1
                sqrt_one_half * C_val,
                lmin, idx1,
                l2, m2, R_idx, <sparse_idx_t*>R_row.data,
                <sparse_idx_t*>R_col.data, <real_t*>R_data.data, eps)
            R_idx = _scatter_T_elem_to_R( # Opposite m1
                -1j * sqrt_one_half * C_val,
                lmin, idxopp1,
                l2, m2, R_idx, <sparse_idx_t*>R_row.data,
                <sparse_idx_t*>R_col.data, <real_t*>R_data.data, eps)

    from scipy.sparse import coo_matrix
    R = coo_matrix((R_data[:R_idx], (R_row[:R_idx], R_col[:R_idx])),
                   C.shape)
    if use_oomatrix:
        R = oomatrix.SparseMatrix(R.tocsc())
    return R
    


cdef int _scatter_T_elem_to_R(complex_t T_val, index_t lmin,
                              index_t idx1,
                              index_t l2, index_t m2,
                              index_t R_idx, # idx to start on in R_x arrays
                              sparse_idx_t* R_row,
                              sparse_idx_t* R_col,
                              real_t* R_data,
                              real_t eps) except -1:
    # Internal function for use by sparse_matrix_complex_to_real.
    # Does the final scattering of a single element of T = U C
    # to one or two elements in R = U C U^H.
    # Return value: New R_idx (the input one + 0, 1 or 2).
    # eps: Truncation to zero level

    # We only need to scatter real parts -- the imaginary parts
    # are nonzero, but in the final sum, when summing up scattered
    # elements at the same location, we know it will cancel.
    cdef index_t minus_one_to_m2 = -1 if m2 % 2 == 1 else 1
    cdef real_t val

    if m2 < 0:
        # +m2
        val = (-1j * sqrt_one_half * minus_one_to_m2 * T_val).real
        if fabs(val) > eps:
            R_data[R_idx] = val
            R_row[R_idx] = idx1
            R_col[R_idx] = lm_to_idx_full(l2, m2, lmin)
            R_idx += 1

        # -m2
        val = minus_one_to_m2 * sqrt_one_half * T_val.real
        if fabs(val) > eps:
            R_data[R_idx] = val
            R_row[R_idx] = idx1
            R_col[R_idx] = lm_to_idx_full(l2, -m2, lmin)
            R_idx += 1
    elif m2 == 0:
        # Single element
        val = T_val.real
        if fabs(val) > eps:
            R_data[R_idx] = val
            R_row[R_idx] = idx1
            R_col[R_idx] = lm_to_idx_full(l2, m2, lmin)
            R_idx += 1
    else:
        # +m2
        val = sqrt_one_half * T_val.real
        if fabs(val) > eps:
            R_data[R_idx] = val
            R_row[R_idx] = idx1
            R_col[R_idx] = lm_to_idx_full(l2, m2, lmin)
            R_idx += 1

        # -m2
        val = (1j * sqrt_one_half * T_val).real
        if fabs(val) > eps:
            R_data[R_idx] = val
            R_row[R_idx] = idx1
            R_col[R_idx] = lm_to_idx_full(l2, -m2, lmin)
            R_idx += 1

    return R_idx
        
    

def isotropic_real_covar(np.ndarray[real_t] Cl):
    cdef index_t idx = 0, l, m, lmin = Cl.lmin, lmax = Cl.lmax
    cdef np.ndarray[real_t] buf = np.zeros((lmax + 1)**2 - lmin**2, real_dtype)
    for l in range(lmin, lmax + 1):
        for m in range(-l, l + 1):
            buf[idx] = Cl[l - lmin]
            idx += 1
    import oomatrix
    return oomatrix.DiagonalMatrix(buf)
    

def py_lm_to_idx_brief(l, m, lmin):
    return lm_to_idx_brief(l, m, lmin)

def py_lm_to_idx_full(l, m, lmin):
    return lm_to_idx_full(l, m, lmin)

def py_lm_count_full(lmin, lmax):
    return lm_count_full(lmin, lmax)

def py_lm_count_brief(lmin, lmax):
    return lm_count_brief(lmin, lmax)


