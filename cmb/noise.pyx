#cython: profile=False

from __future__ import division

from cmbtypes cimport complex_t, real_t, index_t
from cmbtypes import complex_dtype, real_dtype, index_dtype
import maps
import numpy as np
cimport numpy as np
cimport cython
from mapdatautils import (m_range_lm_full, broadcast_l_to_lm_full,
                          num_alm_to_lmax, matrix_complex_to_real)
from mapdatautils cimport (lm_count_full, lm_to_idx_full, lm_to_idx_brief, l_to_lm)
from utils cimport imax, alm_fixnegidx
from cython cimport boundscheck, wraparound
from slatec cimport drc3jj_fast
import oomatrix
from healpix import openmp

cdef double pi
from numpy import pi
import threading

from python cimport PyErr_CheckSignals

cdef extern from "stdlib.h":
    index_t indexabs "llabs"(index_t) nogil

#from c.math cimport sqrt, acos, cos, sin, round

cdef extern from "math.h":
    double sqrt(double) nogil
    double acos(double) nogil
    double cos(double) nogil
    double sin(double) nogil
    double round(double) nogil

cdef extern from "complex.h":
    double complex conj(double complex) nogil
    

#cdef extern from "complex.h":
#    pass

cdef do_abort(int arg) with gil:
    import sys
    print 'drc3jj error: %d', arg
    sys.exit(10)

@cython.profile(False)
cdef int drc3jj(index_t l2, index_t l3, index_t m2, index_t m3,
                 index_t* l1min, index_t* l1max, double* thrcof, size_t thrcof_len) nogil:
    # Convenience function for calling drc3jj_fast
    cdef int ier
    cdef double l1min_d, l1max_d
    ier = drc3jj_fast(l2, l3, m2, m3, &l1min_d, &l1max_d, thrcof, thrcof_len)
    if ier != 0:
        do_abort(ier)
    l1min[0] = <index_t>l1min_d
    l1max[0] = <index_t>l1max_d
    return 0

def Ninv_to_harmonic_real_block(index_t lmin, index_t lmax, Ninv_map, Npix=None):
    # Compute complex, then convert
    C = Ninv_to_harmonic_complex_block(lmin, lmax, Ninv_map, Npix)
    R = matrix_complex_to_real(C, lmin, lmax)
    return R

#@cython.wraparound(False)
#@cython.boundscheck(True)
def Ninv_to_harmonic_real_diagonal(index_t lmin, index_t lmax, Ninv, Npix=None):
    """
    >>> Ninv = maps.pixel_sphere_map(12, Nside=32)
    >>> D = Ninv_to_harmonic_real_diagonal(2, 8, Ninv).to_array()
    >>> np.allclose(np.diagonal(D), Ninv.Npix / 4 / np.pi * 12, atol=1e-10)   
    True

#    >>> np.allclose(D, np.diagflat(np.diagonal(D)), atol=1e-3)
#    True

    """
    if Npix is None:
        # Assume Ninv is given in pixel space and get Npix from there
        Npix = Ninv.Npix
    Ninv = Ninv.to_harmonic(0, 2 * lmax + 2)

    cdef index_t i, j, l, m, l3, m3
    cdef index_t l3min_zeros, l3max_zeros, l3min_dia, l3max_dia, l3min_antidia, l3max_antidia
    cdef index_t minus_one_to_m
    cdef np.ndarray[complex_t, mode='c'] alm_brief = Ninv
    cdef real_t C_val_dia, C_val_antidia, l_scale
    cdef index_t bufsize = 2*(lmax+1)
    cdef np.ndarray[double, mode='c'] three_j_symbol_zeros = np.zeros(bufsize, dtype=np.double)
    cdef np.ndarray[double, mode='c'] three_j_symbol_dia = np.zeros(bufsize, dtype=np.double)
    cdef np.ndarray[double, mode='c'] three_j_symbol_antidia = np.zeros(bufsize, dtype=np.double)
    cdef double Npix_by_four_pi = Npix / 4.0 / pi
    cdef double inv_sqrt_four_pi = 1.0 / sqrt(4.0 * pi)
    cdef index_t lmin_lm
#    cdef np.ndarray[double, mode='c'] l_sqrt = np.sqrt(np.arange(2*(2*lmax + 1) + 1, dtype=np.double))

    cdef np.ndarray[real_t, ndim=1] out

    out = np.zeros(lm_count_full(lmin, lmax), dtype=real_dtype)

    cdef double temp

    for l in range(lmin, lmax + 1):
        l_scale = (2*l + 1) * Npix_by_four_pi * inv_sqrt_four_pi
        drc3jj(l, l, 0, 0, &l3min_zeros, &l3max_zeros,
               <double*>three_j_symbol_zeros.data, bufsize)

        # Treat m=0 seperately
        if l == 0:
            # special case, drc3jj returns 0 but value is 1
            # We don't need to worry about this in other places as
            # l == 0 => m == 0
            C_val_dia = alm_brief[lm_to_idx_brief(0, 0, 0)].real
        else:
            C_val_dia = 0
            for l3 in range(l3min_zeros, l3max_zeros + 1):
                C_val_dia += (three_j_symbol_zeros[l3 - l3min_zeros]**2
                              * sqrt(2*l3 + 1)
                              * alm_brief[lm_to_idx_brief(l3, 0, 0)].real)
                
        C_val_dia *= l_scale
        out[lm_to_idx_full(l, 0, lmin)] = C_val_dia

        # Then, m > 0 as well as m < 0
        # C_val_antidia is understood to be Re[C_{lm,l-m}]
        for m in range(1, l + 1):
            minus_one_to_m = -1 if m % 2 == 1 else 1
            # Let R be the real result (of which we compute the diagonal), and
            # C the complex covariance matrix computed through the 3j symbols.
            # Then the diagonal of R is given by the diagonal AND the anti-diagonal
            # of C for each m.  We compute C_val_dia = C_{lm,lm} (leads to m3 = 0)
            # and C_val_antidia = C_{lm,l-m} which results
            # in m3 = m-(-m) = 2*m > 0; both are direct lookup in the alms.

            drc3jj(l, l, -m, m, &l3min_dia, &l3max_dia, <double*>three_j_symbol_dia.data, bufsize)
            drc3jj(l, l, -m, -m, &l3min_antidia, &l3max_antidia,
                   <double*>three_j_symbol_antidia.data, bufsize)
            C_val_dia = 0
            C_val_antidia = 0
            for l3 in range(l3min_dia, l3max_dia + 1):
                # Diagonal contribution for this l3
                C_val_dia += (alm_brief[lm_to_idx_brief(l3, 0, 0)].real
                              * three_j_symbol_dia[l3 - l3min_dia]
                              * three_j_symbol_zeros[l3 - l3min_zeros] * sqrt(2*l3 + 1))

                if l3 >= l3min_antidia and l3 <= l3max_antidia:
                    if indexabs(2 * m) > l3:
                        raise AssertionError("This should not happen")
                    C_val_antidia += (
                        alm_brief[lm_to_idx_brief(l3, 2*m, 0)].real
                        * three_j_symbol_antidia[l3 - l3min_antidia]
                        * three_j_symbol_zeros[l3 - l3min_zeros] * sqrt(2*l3 + 1))
            C_val_dia *= l_scale * minus_one_to_m
            C_val_antidia *= l_scale * minus_one_to_m

            out[lm_to_idx_full(l, m, lmin)] = C_val_dia + minus_one_to_m * C_val_antidia
            out[lm_to_idx_full(l, -m, lmin)] = C_val_dia - minus_one_to_m * C_val_antidia

    return oomatrix.DiagonalMatrix(out)


def Ninv_to_harmonic_complex_block(lmin, lmax, Ninv, diagonal_only=False,
                                   Npix=None):
    """
    See inverse_pixel_noise_to_harmonic; however one can pass lmin/lmax to
    include all coefficients in the range. It is also parallelized.
    
    >>> Ninv = maps.pixel_sphere_map(12, Nside=32)
    >>> D = inverse_pixel_noise_to_harmonic_block(2, 8, Ninv).to_array()
    >>> np.allclose(np.diagonal(D), Ninv.Npix / 4 / np.pi * 12, atol=1e-10)
    True
    >>> np.allclose(D, np.diagflat(np.diagonal(D)), atol=1e-3)
    True

    >>> D = inverse_pixel_noise_to_harmonic_block(0, 1, Ninv, diagonal_only=True)
    >>> D.round(0)
    3 by 3 diagonal matrix of float64:
    [ 11734.      0.      0.]
    [     0.  11734.      0.]
    [     0.      0.  11734.]
    >>> np.round(Ninv.Npix / 4 / np.pi * 12)
    11734.0
    
    """
    ls = broadcast_l_to_lm_full(np.arange(lmin, lmax + 1), lmin=lmin)
    ms = m_range_lm_full(lmin, lmax)

    Ncoef = ls.shape[0]
    out = np.empty((Ncoef, Ncoef), complex_dtype)
    inverse_pixel_noise_to_harmonic(ls, ms, Ninv, Npix, lmax, out)

# Debug code for multithreaded support:
#    terminator = threading.Event()
#    fullset = np.arange(Ncoef, dtype=index_dtype)
#    threads = []
#    numthreads = min([openmp.get_max_threads(), Ncoef, 1])
#    for i in range(numthreads):
#        subset = fullset[i::numthreads]
#        T = threading.Thread(target=inverse_pixel_noise_to_harmonic,
#                             args=(ls, ms, Ninv, Npix, lmax, out, subset, terminator))
#        threads.append(T)
#        T.start()
        
#    for T in threads:
#        T.join()
##     try:
##         while True:
##             for T in threads:
##                 print 'join'
##                 T.join(.2)
##             if not any([T.isAlive() for T in threads]):
##                 break
##             print 'before interrupt'
##             PyErr_CheckSignals()
##             print 'after'
##     except:
##         print 'setting terminator'
##         do_terminate = True
## #        terminator.set()
##         for T in threads:
##             print 'rejoin'
##             T.join(10)
##         print 'returning'
##         raise
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
def inverse_pixel_noise_to_harmonic(ls, ms, Ninv,
                                    Npix=None, index_t lmax=-1,
                                    np.ndarray[complex_t, ndim=2] out=None,
                                    np.ndarray[index_t] subset=None):
    """
    Turns a diagonal pixel covariance (e.g. noise map) into a dense
    harmonic covariance, for use in e.g. preconditioners.

    Note that this is an *approximation* which becomes more valid for high
    Nside.

    INPUT:

     - ls, ms - Integer arrays containing l and values to include in result matrix.
                The arrays should have the same length (!!)
     - lmax - Maximum value of ls (for allocating a buffer, and spherical harmonic
              transform of Ninv). Computed if not supplied.
     

    >>> Ninv = maps.pixel_sphere_map(12, Nside=32)
    >>> inverse_pixel_noise_to_harmonic([4], [5], Ninv)
    Traceback (most recent call last):
        ...
    ValueError: Illegal m-value at index 0
    >>> D = inverse_pixel_noise_to_harmonic([4, 4, 5], [4, 3, 0], Ninv)
    >>> D.round(zero_sign=True)
    3 by 3 dense matrix of complex128:
    [ 11734.176+0.j      0.000+0.j      0.000+0.j]
    [     0.000+0.j  11734.176+0.j      0.000+0.j]
    [     0.000+0.j      0.000+0.j  11734.176+0.j]

    """
    ls = np.asarray(ls, index_dtype)
    ms = np.asarray(ms, index_dtype)
    if ls.shape != ms.shape:
        raise ValueError()

    if lmax == -1:
        lmax = np.max(ls)
    
    if Npix is None:
        # Assume Ninv is given in pixel space and get Npix from there
        Npix = Ninv.Npix
    Ninv = Ninv.to_harmonic(0, 2 * lmax + 2).to_full_complex()

    cdef np.ndarray[index_t] ls_buf = ls
    cdef np.ndarray[index_t] ms_buf = ms

    cdef index_t Ncoef = ls_buf.shape[0]
    cdef index_t i, j, k, l1, m1, l2, m2, l3, m3, l3min_zeros, l3max_zeros, l3min_ms, l3max_ms, old_l1, old_l2
    cdef np.ndarray[complex_t, mode='c'] alm = Ninv
    cdef complex_t alm_val, elem_val
    cdef index_t bufsize = 2*(lmax+1)
    cdef np.ndarray[double, mode='c'] three_j_symbol_zeros = np.zeros(bufsize, dtype=np.double)
    cdef np.ndarray[double, mode='c'] three_j_symbol_ms = np.zeros(bufsize, dtype=np.double)
    cdef double Npix_by_four_pi = Npix / 4.0 / pi
    cdef double inv_sqrt_four_pi = 1.0 / sqrt(4.0 * pi)
    cdef index_t lmin_lm
#    cdef np.ndarray[double, mode='c'] l_sqrt = np.sqrt(np.arange(2*(2*lmax + 1) + 1, dtype=np.double))

    if subset is None:
        subset = np.arange(Ncoef, dtype=index_dtype)
        
    if out is None:
        out = np.zeros((subset.shape[0], Ncoef), dtype=complex_dtype)


    cdef double temp

    old_l1 = old_l2 = -1
    for i in range(0, Ncoef):
#    for k in range(0, subset.shape[0]):
#        i = subset[k]
        if i < 0 or i >= Ncoef:
            raise ValueError()
#        print 'check_signals'
#        PyErr_CheckSignals()
#        print 'checking terminator'
#        if do_terminate:
#            print 'terminating'
#            return None
#        else:
#            print 'not set'
        l1 = ls_buf[i]; m1 = ms_buf[i]
        if indexabs(m1) > l1:
            raise ValueError("Illegal m-value at index %d" % i)
        with nogil:
            for j in range(0, i + 1):
                l2 = ls_buf[j]; m2 = ms_buf[j]

                if l1 != old_l1 or l2 != old_l2:
                    # Get Wigner coefficients (l3, l1, l2; 0 0 0)
                    # and range of possible l3
                    drc3jj(l1, l2, 0, 0, &l3min_zeros, &l3max_zeros,
                           <double*>three_j_symbol_zeros.data, bufsize)

                # For each l3, the second 3j-symbol is nonzero for exactly one m3,
                # namely m3 - m1 + m2 = 0 => m3 = m1 - m2
                # Also the second 3j-symbol is always more restrictive in range, so
                # that is the range we use (this ensures |m3| <= l3).
                drc3jj(l1, l2, -m1, m2, &l3min_ms, &l3max_ms,
                       <double*>three_j_symbol_ms.data, bufsize)
                elem_val = 0
                m3 = m1 - m2
                for l3 in range(l3min_ms, l3max_ms + 1):
##                     if indexabs(m3) > l3:
##                         raise AssertionError("This should not happen")
##                     if indexabs(m2 - m1) > l3:
##                         raise AssertionError("This should not happen")
                    alm_val = alm[lm_to_idx_full(l3, m3, 0)]
                    temp = (three_j_symbol_zeros[l3 - l3min_zeros] *
                            three_j_symbol_ms[l3 - l3min_ms] * sqrt(2*l3 + 1))
                    elem_val = elem_val + (alm_val * temp)
                elem_val = (elem_val *
                            sqrt(2*l1 + 1) * sqrt(2*l2 + 1) *
                            (Npix_by_four_pi * inv_sqrt_four_pi))
                if (m1 % 2 == 1):
                    elem_val = -elem_val
                if i == j:
                    out[i, i] = elem_val.real
                else:                
                    out[i, j] = elem_val
                    out[j, i] = conj(elem_val)

    return out

def inverse_pixel_noise_to_harmonic_matmul(Py_ssize_t lmin, Py_ssize_t lmax, Ninv_map):
    cdef Py_ssize_t l, m, col_idx
    cdef Py_ssize_t Npix = Ninv_map.Npix, Nside = Ninv_map.Nside

    Ninv_map = Ninv_map.to_ring()

    Ncoef = (lmax+1)**2 - lmin**2 #l_to_lm(lmax + 1) - l_to_lm(lmin)
    out = np.zeros((Ncoef, Ncoef), dtype=real_dtype, order='F')
    cdef np.ndarray[real_t] unit_vector = maps.harmonic_sphere_map(0, lmin, lmax, is_complex=False)

    scaled_Ninv_map = Ninv_map * Npix / 4 / np.pi
    
    for l in range(lmin, lmax + 1):
        for m in range(-l, l+1):
            col_idx = lm_to_idx_full(l, m, lmin)
            unit_vector[col_idx] = 1
            unit_vector_px = unit_vector.to_complex().to_pixel(Nside)
            col = (scaled_Ninv_map * unit_vector_px).to_harmonic(
                lmin, lmax, use_weights=False).to_real()
            out[:, col_idx] = col
            unit_vector[col_idx] = 0
    return out
    
