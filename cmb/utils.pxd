from cmbtypes cimport complex_t, real_t, index_t

cdef inline index_t imax(index_t a, index_t b):
    return b if b > a else a

cdef inline complex_t alm_fixnegidx(complex_t value, index_t m):
    # Hack as we wait for efficient buffer passing in Cython...
    # Corrects lookups of alm values with negative m
    # use like this:
    #   alm = alm_fixnegidx(alms[l_to_lm(l) + abs(m)], m)
    if m >= 0:
        return value
    else:
        if m % 2 == 1:
            value = -value
        return value.conjugate()

