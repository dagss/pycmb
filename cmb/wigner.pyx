from __future__ import division
##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

from slatec cimport drc3jj_fast
import sys
import numpy as np
cimport numpy as np
from stdlib cimport malloc, free
from python_exc cimport PyErr_NoMemory
cimport cython

# Do not change dtype without changing Fortran function called!
dtype = np.double
ctypedef double dtype_t

np.import_array()

def log(*msg):
    sys.stderr.write(repr(msg))
    sys.stderr.write(b'\n')

cdef extern from "math.h":
    double fabs(double) nogil
    double ceil(double) nogil
    bint isnan(double x) nogil
    double round(double) nogil

cdef inline double fmax(double a, double b) nogil:
    return a if b < a else b

cdef extern from "numpy/npy_math.h":
    cdef double NPY_NAN
        

DEF ERR_NO_MEM = -2

def wigner_3j(l1, l2, l3, m1, m2, m3=None, double eps=1e-80, out=None):
    """
    WARNING: This routine should be fixed for l1==l2==l3==0.
    
    wigner_3j(l1, l2, l3, m1, m2, m3=None, out=None)
    
    Computes the Wigner 3j symbol. If ``m3`` is not provided, it will
    be taken as ``-m1-m2`` (which is the only value which will not
    yield 0 as a result). The result has type np.double; passing in
    arrays of np.double for processing will be most efficient.

    The Wigner 3j symbol is 0 if:
    
     - ``l1 < abs(m1)``, ``l2 < abs(m2)`` or ``l3 < abs(m3)``,
       or any l negative
     - ``abs(l1 - l2) <= l3 <= l1 + l2`` is not satisfied
     - ``m1 + m2 + m3 != 0``

    Although conventionally the parameters satisfy certain
    restrictions, such as being integers or integers plus 1/2, the
    restrictions imposed on input to this function are somewhat
    weaker. [drc3jj]_ contains details.

    ``eps`` is used as epsilon for comparing m3 with -(m1+m2).

    Various errors concerning non-integer arguments will result in NaN
    value as well as optionally reporting an error according to the
    setting of np.seterr(). NaNs in input are silently propagated.

    The algorithm is suited to applications in which large quantum
    numbers arise, such as in molecular dynamics.

    See [drc3jj]_ for full details and references.
    
    :Examples:

    >>> wigner_3j(1, 1, 0, 0, 0) - (-np.sqrt(1/3.)) < 1e-4
    True

    >>> wigner_3j(1, 1, 0, 0, 0, 0) - (-np.sqrt(1/3.)) < 1e-4
    True

    >>> wigner_3j(1, 1, 0, 0, 0, 1)
    0.0

    >>> abs(wigner_3j(-1, 1, 0, 0, 0, 0)) < 1e-200
    True

    >>> print("%.3e" % wigner_3j(100, 40, 60, -10, 20))
    9.830e-05
    
    >>> abs(wigner_3j(100, 0, 60, 0, 0)) < 1e-200
    True

    >>> wigner_3j(1, 1, 0, 2, -2)
    0.0

    >>> wigner_3j(2, 2, 0, 0, 0, 0)
    0.44721359549995793
    >>> wigner_3j(1, 1, 0, -1, 1, 0)
    0.57735026918962584
    >>> wigner_3j(3, 2, 1, 0, 0, 0)
    -0.29277002188455992
    >>> np.round(wigner_3j(3, 2, 1, -2, 1, 1), 12)
    -0.30860669992400003

    >>> np.round(wigner_3j(l1=[1,2,3], l2=[[1],[2],[3]], l3=3,
    ...           m1=1, m2=0), 5)
    array([[ 0.     ,  0.27603, -0.10911],
           [ 0.23905, -0.16903, -0.14639],
           [-0.26726, -0.06901,  0.1543 ]])

    >>> olderr = np.seterr(invalid='raise')
    >>> wigner_3j(1.1, 1.1, 1.8, 0, 0)
    Traceback (most recent call last):
       ...
    FloatingPointError: drc3jj failed with non-integer error; NaN created

    >>> dummy=np.seterr(invalid='ignore')
    >>> wigner_3j([1.1,1], [1.1,0], [1.8,1], 0, 0)
    array([        NaN, -0.57735027])

    >>> def callback(a, b): print repr(('callback', a, b))
    >>> dummy = np.seterr(invalid='call')
    >>> oldcall = np.seterrcall(callback)
    >>> wigner_3j([1.1,1], [1.1,0], [1.8,1], 0, 0)
    ('callback', 'invalid', 8)
    array([        NaN, -0.57735027])

    >>> dummy = np.seterr(**olderr)
    >>> dummy = np.seterrcall(oldcall)

    >>> wigner_3j(np.nan, 1, 0, 2, -2)
    nan

    :Authors:
     - Gordon, R. G., Harvard University (drc3jj)
     - Schulten, K., Max Planck Institute (drc3jj)
     - Seljebotn, D. S, University of Oslo (Python wrapper)

    .. [drc3jj]_ http://netlib.org/slatec/src/drc3jj.f
    """
    cdef:
        double l1val, l2val, l3val, m1val, m2val, m3val
        double l1min, l1max, outval
        np.broadcast it
        double* thrcof = NULL
        Py_ssize_t thrcof_needed, thrcof_len, ier
        bint nan_created = False
        bint m3_provided

    l1 = np.asarray(l1, dtype)
    l2 = np.asarray(l2, dtype)
    l3 = np.asarray(l3, dtype)
    m1 = np.asarray(m1, dtype)
    m2 = np.asarray(m2, dtype)
    m3_provided = (m3 is not None)
    if m3 is None: m3 = 0 # must broadcast over something
    m3 = np.asarray(m3, dtype)

    # Figuring out the broadcast shape is not a seperate utility
    # function so need to create a dummy broadcast object.
    it = np.broadcast(l1, l2, l3, m1, m2, m3)
    shape = (<object>it).shape
    if out is None:
        out = np.empty(shape, dtype)
    else:
        if out.shape != shape:
            raise ValueError(u"out argument has incorrect shape")
        if out.dtype != dtype:
            raise ValueError(u"out argument has incorrect dtype")
    it = np.broadcast(l1, l2, l3, m1, m2, m3, out)
    with nogil:
        thrcof_len = 32
        thrcof = <double*>malloc(thrcof_len * sizeof(dtype_t))
        while np.PyArray_MultiIter_NOTDONE(it):
            l1val = (<dtype_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            l2val = (<dtype_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            l3val = (<dtype_t*>np.PyArray_MultiIter_DATA(it, 2))[0]
            m1val = (<dtype_t*>np.PyArray_MultiIter_DATA(it, 3))[0]
            m2val = (<dtype_t*>np.PyArray_MultiIter_DATA(it, 4))[0]
            if m3_provided:
                m3val = (<dtype_t*>np.PyArray_MultiIter_DATA(it, 5))[0]
            else:
                m3val = -m1val - m2val

            if (isnan(l1val) or isnan(l2val) or isnan(l3val) or
                isnan(m1val) or isnan(m2val)):
                outval = NPY_NAN
            elif m3_provided and m3val + m1val + m2val > eps:
                outval = 0
            else:
                while True:
                    ier = drc3jj_fast(l2val, l3val, m2val, m3val,
                                      &l1min, &l1max,
                                      thrcof, thrcof_len)
                    if ier == 5: # need to reallocate thrcof
                        thrcof_len *= 2
                        free(thrcof)
                        thrcof = <double*>malloc(
                            thrcof_len * sizeof(dtype_t))
                        if thrcof == NULL:
                            thrcof_len = ERR_NO_MEM
                            break # cannot raise exception within nogil
                        continue #
                    else:
                        break
                if thrcof_len == ERR_NO_MEM:
                    break # life without exceptions...

                # Now, ier != 5
                if ier == 1 or ier == 4:
                    # Parameters do not satisfy constraints; in these
                    # cases the 3j symbol is defined as 0
                    outval = 0
                elif ier == 2 or ier == 3:
                    # IER=2 Either L2+ABS(M2) or L3+ABS(M3) non-integer.
                    # IER=3 L1MAX-L1MIN not an integer.
                    #
                    # We interpret this as parameters which do not make
                    # sense => nan
                    outval = NPY_NAN
                    nan_created = True
                else:
                    outval = thrcof[<Py_ssize_t>round(l1val - l1min)]

            (<dtype_t*>np.PyArray_MultiIter_DATA(it, 6))[0] = outval
            np.PyArray_MultiIter_NEXT(it)

    if thrcof_len == ERR_NO_MEM:
        PyErr_NoMemory() # raises OutOfMemoryError
    if thrcof != NULL:
        free(thrcof)        
    if nan_created:
        handlertype = np.geterr()['invalid']
        msg = u'drc3jj failed with non-integer error; NaN created'
        if handlertype == b'ignore':
            pass # check for common easy case first
        elif handlertype == b'raise':
            raise FloatingPointError(msg)
        elif handlertype == b'warn':
            import warnings
            warnings.warn(RuntimeWarning(msg))
        elif handlertype == b'call':
            import inspect
            call = np.geterrcall()
            if inspect.isfunction(call) or inspect.isbuiltin(call):
                call(b'invalid', 8)
            else:
                call.write(msg)
    if out.shape == ():
        out = out[()]
    return out

## drc3jj_fast


__test__ = {
    "wigner_3j (line 6)" : wigner_3j.__doc__
}
