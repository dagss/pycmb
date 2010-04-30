from cmbtypes cimport complex_t, real_t, index_t

cdef inline index_t l_to_lm(index_t l) nogil:
    return (l * (l+1)) // 2

cdef inline index_t lm_to_idx_brief(index_t l, index_t m, index_t lmin) nogil:
    return (l * (l+1) - (lmin * (lmin + 1))) // 2 + m

cdef inline index_t lm_to_idx_full(index_t l, index_t m, index_t lmin) nogil:
    return l*l + l + m - lmin*lmin

cdef inline index_t lm_count_full(index_t lmin, index_t lmax) nogil:
    return (lmax + 1)**2 - lmin**2

cdef inline index_t lm_count_brief(index_t lmin, index_t lmax) nogil:
    return lm_to_idx_brief(lmax + 1, 0, lmin)

