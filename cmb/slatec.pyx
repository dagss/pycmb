from __future__ import division

##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

# Wrapper for SLATEC functions
# http://netlib.org/slatec

cimport numpy as np

nocopy = False
cdef bint must_copy(arr):
    cdef bint result
    while len(arr.shape) > 0 and arr.shape[0] == 1:
        arr = arr[0,:]
    while len(arr.shape) > 0 and arr.shape[-1] == 1:
        arr = arr[:,0]
    result = not arr.flags.f_contiguous
    if result and nocopy:
        raise ValueError(u"Copying disabled but have to copy!")
    return result

cdef extern:
    void drc3jj_(double* l2, double* l3, double* m2, double* m3,
                 double* l1min, double* l1max, double* thrcof,
                 int* ndim, int* ier) nogil

    void drc3jm_(double* l1, double* l2, double* l3,
                 double* m1, double* m2min, double* m2max,
                 double* thrcof, int* ndim, int* ier) nogil

# drc3jj

cdef int drc3jj_fast(double l2, double l3, double m2, double m3,
             double* l1min, double* l1max, double* thrcof,
             int ndim) nogil:
    cdef int ier = 0
    drc3jj_(&l2, &l3, &m2, &m3, l1min, l1max, thrcof, &ndim, &ier)
#    if ier == 5:
#        raise Exception(u"drc3jj: L1 is in (%g, %g), while buffer size is %d." % (l1min[0], l1max[0], ndim))
#    elif ier != 0:
#        raise Exception(u"Error flag set in drc3jj: %d." % ier)
    return ier

cpdef object drc3jj(double l2, double l3, double m2, double m3,
             thrcof):
    cdef:
        np.ndarray[double, ndim=1] thrcof_work
        double l1min, l1max

    if must_copy(thrcof):
        thrcof_work = thrcof.copy(u'F')
    else:
        thrcof_work = thrcof

    drc3jj_fast(l2, l3, m2, m3, &l1min, &l1max,
                <double*>thrcof_work.data,
                thrcof_work.shape[0])
    if thrcof_work is not thrcof:
        thrcof[...] = thrcof_work
    return (int(l1min), int(l1max))

# drc3jm

cdef int drc3jm_fast(double l1, double l2, double l3,
                     double m1, double* m2min, double* m2max,
                     double* thrcof, int ndim) nogil:
    cdef int ier = 0
    drc3jm_(&l1, &l2, &l3, &m1, m2min, m2max, thrcof, &ndim, &ier)
#    if ier != 0:
#        raise Exception(u"Error flag set in drc3jj: %d." % ier)
    return ier

cpdef object drc3jm(double l1, double l2, double l3, double m1, thrcof):
    cdef:
        np.ndarray[double, ndim=1] thrcof_work
        double m2min, m2max

    if must_copy(thrcof):
        thrcof_work = thrcof.copy(u'F')
    else:
        thrcof_work = thrcof

    drc3jj_fast(l1, l2, l3, m1, &m2min, &m2max,
                <double*>thrcof_work.data,
                thrcof_work.shape[0])
    if thrcof_work is not thrcof:
        thrcof[...] = thrcof_work
    return (int(m2min), int(m2max))

