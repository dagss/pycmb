
cpdef object drc3jj(double l2, double l3, double m2, double m3, thrcof)
cdef int drc3jj_fast(double l2, double l3, double m2, double m3,
             double* l1min, double* l1max, double* thrcof,
             int ndim) nogil


cpdef object drc3jm(double l1, double l2, double l3, double m1, thrcof)
cdef int drc3jm_fast(double l1, double l2, double l3,
                     double m1, double* m2min, double* m2max,
                     double* thrcof, int ndim) nogil
