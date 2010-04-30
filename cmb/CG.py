from __future__ import division
import numpy as np

#from IPython.Debugger import Tracer; debug_here = Tracer()


class ConvergenceError(Exception):
    pass

def _identity(x):
    return x

def _tridiag(n, a, b, c):
    return (np.diag(np.ones(n-1)*a, -1)
            + np.diag(np.ones(n)*b, 0)
            + np.diag(np.ones(n-1)*c, 1))

def _as_float_array(x):
    if not isinstance(x, np.ndarray): # make sure to preserve ndarray subclasses
        x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        if np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError("Only real dtypes for CG")
        else:
            x = x.astype(np.float)
    return x


def CG(matrix, b, x0=None, maxit=10**3, eps=10**-8, relative_eps=True,
       precond=_identity, raise_error=True, norm_order=None,
       logger=None):
    """
    INPUT:
      matrix -- Either a 2D NumPy ndarray, or a callable which will
                be called with the vector as argument.

    
    Example from Shewchuck, converges in 2 iterations.
    >>> A = np.array([[3, 2],[2, 6]], dtype=np.float)
    >>> b = np.array([2, -8], dtype=np.float)
    >>> x0 = np.r_[0,0].astype(float)
    >>> def mymul(x): return np.dot(A, x)
    >>> x, k = CG(mymul, b, x0, 200, 10**-8)
    >>> np.dot(A, x)
    array([ 2., -8.])
    >>> k
    2
    >>> x, k = CG(mymul, b, x, maxit=3, relative_eps=False)
    >>> np.dot(A, x)
    array([ 2., -8.])
    >>> k
    1

    Try a bigger matrix; -1 on the first off-diagonal and
    increasing sequence on diagonal.
    >>> A = _tridiag(100, -1, 2, -1).astype(np.float)
    >>> A += np.diag(np.r_[0:100])
    >>> np.all(np.linalg.eig(A) > 0) # positive-definite?
    True
    >>> b = np.r_[10:110].astype(np.float)

    Normal:
    
    >>> x, k = CG(A, b, maxit=200)
    >>> np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) <= 10**-8
    True
    >>> k
    49

    Be happy with lower convergence:

    >>> x, k = CG(A, b, eps=1/10)
    >>> np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) < 1/10
    True
    >>> k
    3

    Use a diagonal preconditioner:

    >>> def precond(x): return x / (2+np.r_[0:100])
    >>> x, k = CG(A, b, precond=precond)
    >>> np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) <= 10**-8
    True
    >>> k
    10
   

    """
    if isinstance(matrix, (np.ndarray, list)):       
        assert len(args) == len(kw) == 0
        matrix = _as_float_array(matrix)
        def matmul(x):
            return np.dot(matrix, x)
    else:
        matmul = matrix

    b = _as_float_array(b)
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)
    else:
        x0 = _as_float_array(x0)
    assert b.ndim == x0.ndim == 1

    info = {}

    # Terminology/variable names follow Shewchuk, ch. B3
    #  r - residual
    #  d - preconditioned residual, "P r"
    #  
    # P = inv(M)
    r = b - matmul(x0)

    residual = np.linalg.norm(r, norm_order)
    if logger is not None:
        logger.info('Initial residual %e', residual)

    d = precond(r)
    
    delta_0 = delta_new = np.dot(r, d)

    info['residuals'] = residuals = [residual]
    info['error'] = None

    eps *= residual

    if residual < eps:
        info['iterations'] = 0
        return (x0, info)

    x = x0
    for k in xrange(maxit):
        q = matmul(d)
        dAd = np.dot(d, q)
        if not np.isfinite(dAd) or dAd == 0:
            raise AssertionError()
        alpha = delta_new / dAd
        x = x + alpha * d

        r = r - alpha * q
        if k > 0 and k % 50 == 0:
            r_est = r
            r = b - matmul(x)
            logger.info('Recomputing residual, relative error in estimate: %e',
                        np.linalg.norm(r - r_est) / np.linalg.norm(r))
            del r_est

        residual = np.linalg.norm(r, norm_order)
        if logger is not None:
            logger.info('Iteration %d: Residual %e (terminating at %e)', k, residual, eps)
        residuals.append(residual)
        
        if residual < eps:
            # Before terminating, make sure to recompute the residual
            # exactly, to avoid terminating too early due to numerical errors
            r_est = r
            r = b - matmul(x)
            logger.info('Recomputing residual, relative error in estimate: %e',
                        np.linalg.norm(r - r_est) / np.linalg.norm(r))
            residual = np.linalg.norm(r, norm_order)
            if residual < eps:
                info['iterations'] = k + 1
                return (x, info)
            else:
                logger.info('Avoided early termination due to recomputing residual')
                
        s = precond(r)
        delta_old = delta_new
        delta_new = np.dot(r, s)
        beta = delta_new / delta_old
        d = s + beta * d

    err = ConvergenceError("Did not converge in %d iterations" % maxit)
    if raise_error:
        raise err
    else:
        info['iterations'] = maxit
        info['error'] = err
        return (x, info)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
