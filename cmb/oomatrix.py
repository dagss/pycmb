"""
Object-oriented linear algebra

Simple classes to perform linear algebra on a variety of matrix types
with the same interface -- the same code can be used for, say,
diagonal or sparse Hermitian matrices.

These classes all focus on matrices as linear transforms, not as data.
There is e.g. currently no item access (!) (though that could be added).
Usually one will end up multiplying one or more of these matrices with
NumPy arrays. A NumPy array is essentially treated as a "stack of vectors"
no matter its dimensionality.

Matrices are assumed to be immutable. Decompositions etc. made are
cached.

Matrices support:
 - multiplication (within, scalars, NumPy arrays)
 - log_determinant (only Cholesky)
 - conjugate_transpose, .H
 - cholesky

BUGS:
 - solve_right must coerce result type
 - pickling a dense matrix with a cholesky factor does solve_right after unpickle


TODO:
 - addition, subtraction (only with matrices of same shape)
 - solve_left, solve_right
 - apply_ufunc
 - get rid of MatrixKind



"""

import numpy as np
import scipy.sparse as sparse
_sparse_index_dtype = np.intc # Native dtype for indices in xscipy.sparse

def debug(*args):
    import logging
    import traceback
    tb = traceback.extract_stack(limit=2)
    call_line = "%s:%d (%s)" % tb[0][0:3]
    logging.debug("%s\n%r\n" % (call_line, args))

class MatrixKind:
    """
    Represents a kind of matrix. Each instance of this class is
    considered a seperate kind.

    TODO: Is this really needed? Perhaps flags to constructor functions is all one needs.
    """
    def __init__(self, repr, default_constructor, sparse=False, hermitian=False,
                 diagonal=False, triangular=False, lower=False):
        self.repr = repr
        self.is_sparse = sparse
        self.is_hermitian = hermitian
        self.is_diagonal = diagonal
        self.is_triangular = triangular
        self.is_lower = lower
        self.default_constructor = default_constructor

    def __repr__(self):
        return self.repr

class MatrixNotPositiveDefiniteError(ValueError):
    pass

class Matrix(object):
    
    __array_priority__ = 10000000 # Very large, since we don't subclass ndarray
    
    def __init__(self, data_behaviour=None):
        self._cache = {}
        self._data_behaviour = data_behaviour

    def cache(self, key, value):
        self._cache[key] = value

    def fetch(self, key, default=None):
        return self._cache.get(key, default)

    def __repr__(self):
        shape = self.shape

        # Map platform default integer to 'int' rather than int64/int32
        # for easier doctest portability
        if self.dtype == np.dtype(np.int):
            dtype_str = 'integers'
        else:
            dtype_str = str(self.dtype)

        if max(shape) > 10:
            return "%d by %d %s matrix of %s" % (shape[0], shape[1], self.kind, dtype_str)
        else:
            # Remove the double [[, ]] from matrix repr
            data = self.full_matrix().dense_matrix()._data
            datalines = str(data).split('\n')
            datalines = [x.replace(' [', '[').replace('[[', '[').
                         replace(']]',']').replace('] ', ']')
                         for x in datalines]
            datastr = '\n'.join(datalines)
        
            return "%d by %d %s matrix of %s:\n%s" % (shape[0], shape[1],
                                                      self.kind, dtype_str,
                                                      datastr)

    def __rmul__(self, left):
        """
        EXAMPLES::

            >>> A = matrix(np.double, 2, 2, range(4))
            >>> 2 * A
            2 by 2 dense matrix of float64:
            [ 0.  2.]
            [ 4.  6.]
        
        Matrices can be multiplied with arrays and lists::

            >>> [10, 100] * matrix(np.int, 2, 2, [2, 1, 0, 1])
            [20, 110]
            >>> A = matrix(np.double, 2, 3, [2, 1, 1, 0, 1, 0])
            >>> A
            2 by 3 dense matrix of float64:
            [ 2.  1.  1.]
            [ 0.  1.  0.]
            >>> D = np.array([10, 100])
            >>> D * A
            array([  20.,  110.,   10.])
            >>> D[None, :] * A
            array([[  20.,  110.,   10.]])
            >>> D[None, None, :] * A
            array([[[  20.,  110.,   10.]]])

            >>> DDD = np.hstack((D[:,None], D[:,None], D[:,None]))
            >>> D3D = np.dstack((DDD[:,:,None], DDD[:,:,None]))
            >>> (D3D * A).shape
            (2, 3, 3)
            >>> R = D3D.T * A; R
            array([[[  20.,  110.,   10.],
                    [  20.,  110.,   10.],
                    [  20.,  110.,   10.]],
            <BLANKLINE>
                   [[  20.,  110.,   10.],
                    [  20.,  110.,   10.],
                    [  20.,  110.,   10.]]])
            >>> x = D3D.T * A.sparse_matrix()
            >>> x.shape == R.shape
            True
            >>> np.all(x == R)
            True

        But not with some other types...
        
            >>> object() * A #doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            TypeError: Matrices cannot operate with <object...>

        """
        if np.isscalar(left):
            return self._mul_scalar(left)
        else:
            return _matrix_mul(left, self)

    def __mul__(self, right):
        """
        EXAMPLES::

            >>> A = matrix(np.double, 2, 2, range(4))
            >>> B = matrix(np.int, 2, 3, range(6))
            >>> A * B
            2 by 3 dense matrix of float64:
            [  3.   4.   5.]
            [  9.  14.  19.]

            >>> A * 2
            2 by 2 dense matrix of float64:
            [ 0.  2.]
            [ 4.  6.]


        Matrices can be multiplied with arrays and lists::

            >>> matrix(np.int, 2, 2, [2, 0, 1, 1]) * [10, 100]
            [20, 110]
            >>> A = matrix(np.double, 3, 2, [2, 0, 1, 1, 1, 0])
            >>> A
            3 by 2 dense matrix of float64:
            [ 2.  0.]
            [ 1.  1.]
            [ 1.  0.]
            >>> D = np.array([10, 100])
            >>> A * D
            array([  20.,  110.,   10.])
            >>> A * D[:,None]
            array([[  20.],
                   [ 110.],
                   [  10.]])
            >>> A * D[:,None,None]
            array([[[  20.]],
            <BLANKLINE>             
                   [[ 110.]],
            <BLANKLINE>                    
                   [[  10.]]])

            >>> DDD = np.hstack((D[:,None], D[:,None], D[:,None]))
            >>> D3D = np.dstack((DDD[:,:,None], DDD[:,:,None]))
            >>> (A * D3D).shape
            (3, 3, 2)
            >>> R = A * D3D; R
            array([[[  20.,   20.],
                    [  20.,   20.],
                    [  20.,   20.]],
            <BLANKLINE>
                   [[ 110.,  110.],
                    [ 110.,  110.],
                    [ 110.,  110.]],
            <BLANKLINE>
                   [[  10.,   10.],
                    [  10.,   10.],
                    [  10.,   10.]]])
            >>> R[:,:,0]
            array([[  20.,   20.,   20.],
                   [ 110.,  110.,  110.],
                   [  10.,   10.,   10.]])
            >>> x = A.sparse_matrix() * D3D;
            >>> x.shape == R.shape
            True
            >>> np.all(x == R)
            True


        TESTS::
        
            >>> A * object() #doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            TypeError: Matrices cannot operate with <object...>

            
        """
        if np.isscalar(right):
            return self._mul_scalar(right)
        else: #if isinstance(right, (Matrix, np.ndarray)):
            # Matrix-matrix or matrix-array multiplication
            return _matrix_mul(self, right)
##         elif isinstance(right, list):
##             return (self * np.asarray(right)).tolist()
##         else:
##             raise TypeError('Cannot multiply matrix with %r' % right)
            
    def conjugate_transpose(self):
        return self._conjugate_transpose()

    def _mul_scalar(self, c):
        """
        >>> matrix(np.double, 2, 2, 1, DENSE) * 4
        2 by 2 dense matrix of float64:
        [ 4.  0.]
        [ 0.  4.]
        >>> matrix(np.double, 2, 2, 1, SPARSE) * 4
        2 by 2 sparse matrix of float64:
        [ 4.  0.]
        [ 0.  4.]
        """
        if c == 1: return self
        return type(self)(self._data * c, shape=self.shape)

    def sandwich(self, *args):
        """
        Sandwiches the matrix between the given matrix/matrices. E.g. A.sandwich(M) results in

        M^H A M

        while A.sandwich(B, C) results in

        C^H B^H A B C

        I.e. one specifies the matrices to be multiplied on the right side in the natural
        order, and the transposes are left-multiplied in the same order.

        EXAMPLES::

            >>> A = identity_matrix(2)
            >>> M = matrix(np.double, 1, 2, [2,3])
            >>> A.sandwich(M)
            1 by 1 dense matrix of float64:
            [ 13.]
            >>> A.sandwich(2*A, M)
            1 by 1 dense matrix of float64:
            [ 52.]
    
        """
        return self._sandwich(*args)

    def _sandwich(self, *args):
        result = self
        for M in args:
            result = M.H * result * M
        return result

    #
    # Type conversions
    #

    def full_matrix(self):
        """
        Returns self in a non-special matrix space::

            >>> matrix(np.complex, 2, 2, [1, 0, 1-1j, 2], SPARSE_HERMITIAN).full_matrix()
            2 by 2 sparse matrix of complex128:
            [ 1.+0.j  1.+1.j]
            [ 1.-1.j  2.+0.j]
        """
        return self._full_matrix()

    def dense_matrix(self):
        return self._dense_matrix()

    def sparse_matrix(self):
        return self._sparse_matrix()

    def numpy(self):
        return self._full_matrix()._dense_matrix()._data

    def _scipy_sparse(self):
        return self._full_matrix()._sparse_matrix()._data
    
    def scipy_sparse(self):
        return self._scipy_sparse()

    def scipy_csc(self):
        return self._scipy_sparse().tocsc()

    def scipy_coo(self):
        return self._scipy_sparse().tocoo()

    #
    # Rounding
    #
    def round(self, ndigits=3, zero_sign=False):
        return self._round(ndigits, zero_sign)

    def _round(self, ndigits, zero_sign):
        return self.apply(_round, ndigits=ndigits, zero_sign=zero_sign)

    #
    # Basic utilities
    #
    def __cmp__(self, other):
        if self is other:
            return 0
        # Since this __cmp__ was reached, the identical type case is
        # already taken care of. Try to convert both to full.
        # Using 'is' along the way avoids infinite loops -- this must
        # be recoded if we ever loose the mutability restriction.
        new_other = other.full_matrix()
        new_self = other.full_matrix()
        if new_other is not other or new_self is not self:
            return cmp(new_other, new_self)
        else:
            # Both were already full -- convert to dense which as a
            # well-defined cmp
            return cmp(self.dense_matrix(), other.dense_matrix())

    def is_complex(self):
        """
        >>> matrix(np.complex128, 1, 1, [1], DENSE).is_complex()
        True
        >>> matrix(np.float64, 1, 1, [1], DENSE).is_complex()
        False
        """
        
        return issubclass(self.dtype.type, np.complex_)

    def is_hermitian(self):
        """
        Docstring
        """
        return self._is_hermitian()

    def is_triangular(self, lower=True):
        return self._is_triangular()
 
    def _is_hermitian(self):
        return self == self.H

    def is_sparse(self):
        return self.kind.is_sparse

    def is_square(self):
        return self.shape[0] == self.shape[1]

    def diagonal(self):
        """
        NumPy array of diagonal of matrix
        """
        return self._diagonal()

    def diagonal_as_matrix(self):
        return DiagonalMatrix(self._diagonal())

    #
    # Indexing and slicing
    #
    def __getitem__(self, indices):
        # Do some sanity checking (right number of indices)
        if not isinstance(indices, tuple) or len(indices) != 2:
            raise ValueError("Only indexing/slicing with 2 indices allowed")
        i, j = indices

        if isinstance(i, slice) and isinstance(j, slice):
            if i.step not in (1, None) or j.step not in (1, None):
                raise NotImplementedError("Non-1 steps in slices not supported. "
                        "Strange inconsistensies with NumPy vs. scipy.sparse slicing, "
                        "must revisit issue.")
            if i.start == j.start and i.stop == j.stop:
                return self._get_symmetric_slice(i.start, i.stop)
        return self._getitem(i, j)

    def _getitem(self, i, j):
        return self.full_matrix()._getitem(i, j)

    def _get_symmetric_slice(self, start, stop):
        return self._getitem(slice(start, stop, None), slice(start, stop, None))
    
    def _normalize_slices(self, i, j):
        if isinstance(i, slice):
            i = _normalize_slice(i, self.shape[0])
        if isinstance(j, slice):
            j = _normalize_slice(j, self.shape[1])
        return (i, j)
    
##         i, j = indices
##         for idx in indices:
##             if not isinstance(idx, slice):
##                 raise NotImplementedError("Only slicing is currently allowed")
##             if idx.step not in (1, None):
##                 raise NotImplementedError("Striding not implemented yet")

        

    #
    # Algorithms -- decomposition and solves
    #
    def _algorithms(self):
        return (('lu', 'right'), ('cholesky', 'right'), ('triangular', 'only_invert'))

    def _select_algorithms(self, algorithm=None):
        # TODO: Fix this up -- should check which are cached, return more than
        # one, etc. etc
        assert not isinstance(algorithm, list), "TODO"
        x = self._algorithms()
        default_algorithm = x[0][0]
        algorithms = dict(x)
        if algorithm is None:
            algorithm = default_algorithm
        elif algorithm not in algorithms:
            raise ValueError('Algorithm %s not supported' % algorithm)
        return [algorithm]

    def solve(self, B, on_right, check=True, algorithm=None, **options):
        """

        EXAMPLES::

            >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], DENSE)
            >>> M.solve_right([1, 2j])
            [(4+0j), 3j]

            >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], SPARSE_HERMITIAN)
            >>> x = M.solve_right([1, 2j]); x
            [(4+0j), 3j]
            >>> M * x
            [(1+0j), 2j]

        """

        B_was_list = isinstance(B, list)
        if B_was_list:
            B = np.asarray(B)
        if isinstance(B, np.ndarray) and B.ndim > 1:
            raise NotImplementedError()
        A = self

        # TODO: Fixup algorithm selection
        algorithm = self._select_algorithms(algorithm)[0]
        algorithms = dict(self._algorithms())
        supported_sides = algorithms[algorithm]
        transposed = ((on_right and supported_sides == 'left') or
                      (not on_right and supported_sides == 'right'))
        if transposed:
            on_right = not on_right
            B = conjugate_transpose(B)
            A = A.conjugate_transpose()
            
        if algorithm == 'cholesky':
            # Support for negative-definite matrices is TODO until __getitem__ is implemented
            #negative = A[0,0] < 0
            negative = False
            if negative:
                A = -A
            X = A._solve_cholesky(B, on_right=on_right, check=check, **options)
            if negative:
                X = -X
        elif algorithm == 'triangular':
            X = A._solve_triangular(B, on_right=on_right, check=check, **options)
        elif algorithm == 'lu':
            X = A._solve_lu(B, on_right=on_right, check=check, **options)
        elif algorithm == 'trivial':
            X = A._solve_trivial(B, on_right=on_right, check=check, **options)
        else:
            raise NotImplementedError()
        if transposed:
            X = conjugate_transpose(X)
        if B_was_list:
            X = X.tolist()
        if X is B and isinstance(X, np.ndarray):
           # If solving is identity and B is an array, we want to return
           # a copy for consistency
           X = X.copy('A')
        return X

    def inverse(self, algorithm=None, check=True, **options):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Cannot invert non-square matrix")
        algorithms = self._select_algorithms(algorithm)
        for alg in algorithms:
            if alg == 'cholesky':
                return self._invert_cholesky(check=check, **options)
            elif alg == 'triangular':
                return self._invert_triangular(check=check, **options)
            elif alg == 'lu':
                return self._invert_lu(check=check, **options)                
            elif alg == 'trivial':
                return self._invert_trivial(check=check, **options)                
            else:
                raise NotImplementedError()
            assert False, "TODO"
        
    def solve_right(self, B, check=True, algorithm=None, **options):
        return self.solve(B, on_right=True, check=True, algorithm=algorithm, **options)

    def solve_left(self, B, check=True, algorithm=None, **options):
        return self.solve(B, on_right=False, check=True, algorithm=algorithm, **options)
    
    def cholesky(self, check=True):
        """
        Docstring etc.

        >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], SPARSE_HERMITIAN)
        >>> P, L = M.cholesky()
        >>> P
        2 by 2 scaled identity matrix of integers:
        [1 0]
        [0 1]
        >>> L
        2 by 2 Cholesky factor (internal CHOLMOD format) matrix of complex128:
        [ 1.+0.j  0.+0.j]
        [ 0.-1.j  1.+0.j]

        >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], SPARSE)
        >>> M.cholesky() == (P, L)
        True
        >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], SPARSE)
        >>> M.cholesky()
        Traceback (most recent call last):
          . . .
        ValueError: Matrix is not Hermitian
        >>> M.cholesky(check=False) == (P, L)
        True

        """
        return self._cholesky(check=check)

    def _cholesky(self, check):
        raise NotImplementedError()

    def log_determinant(self, check=True, algorithm='cholesky'):
        """
        >>> M = matrix(np.complex128, 2, 2, [4, 4j, -4j, 8], DENSE)
        >>> np.allclose(M.log_determinant(algorithm='cholesky'), 2*2*np.log(2))
        True
        
        >>> N = matrix(np.complex128, 2, 2, [1, 0, 1, 1], SPARSE_HERMITIAN)
        >>> N.log_determinant()
        Traceback (most recent call last):
            ...
        NotImplementedError: Determinant can currently only be found for positive definite matrices
        """
        if algorithm != 'cholesky' or not self.is_hermitian():
            raise NotImplementedError()
        
        
        try:
            return self._log_determinant_cholesky(check=check)
        except MatrixNotPositiveDefiniteError:
            raise NotImplementedError('Determinant can currently only be found for positive definite matrices')

        return self._log_determinant()

    def _log_determinant_cholesky(self, check):
        P, L = self.cholesky(check=check)
        return 2 * np.sum(np.log(L.diagonal().real))

    def _solve_cholesky(self, B, on_right, check, **options):
        """
            >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], SPARSE_HERMITIAN)
            >>> x = M.solve_right([1, 2j]); x
            [(4+0j), 3j]
            >>> M * x
            [(1+0j), 2j]
            >>> M.solve_left([1, 2j])
            [-0j, 1j]
            >>> M.solve_right(M)
            2 by 2 sparse matrix of complex128:
            [ 1.+0.j  0.+0.j]
            [ 0.+0.j  1.+0.j]
        """
        assert on_right
        P, L = self.cholesky(check=check, **options)
        PB = P * B
        Y = L.solve_right(PB)
        PX = L.H.solve_right(Y)
        X = P.H * PX
        return X

    def _invert_cholesky(self, check, **options):
        """
            >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], DENSE)
            >>> Minv = M.inverse(algorithm='cholesky'); Minv
            2 by 2 dense matrix of complex128:
            [ 2.+0.j  0.-1.j]
            [ 0.+1.j  1.+0.j]
            >>> Minv * M
            2 by 2 dense matrix of complex128:
            [ 1.+0.j  0.+0.j]
            [ 0.+0.j  1.+0.j]
        """
        P, L = self.cholesky(check=check, **options)
        Linv = L.inverse()
        return P.H * (Linv.H * Linv) * P
        

    #
    # Plotting
    #
    def plot(self, ax=None, show=None, **kw):
        from matplotlib.ticker import FuncFormatter
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            if self.is_complex():
                ax = (fig.add_subplot(1,2,1), fig.add_subplot(1,2,2))
            else:
                ax = fig.add_subplot(1,1,1)
            if show is None:
                show = True
        elif show:
            raise ValueError("show can not be set when ax is not provided")
        if self.is_complex() and not (isinstance(ax, tuple) and len(ax) == 2):
            raise ValueError('Cannot plot complex matrix on one axes, please provide 2')
        data = self.full_matrix().dense_matrix()._data

        def format_matrix_ticker(x, pos):
            return "%d" % round(float(x))

        def doit(data, ax):
            img = ax.imshow(data, interpolation='nearest', **kw)
            ax.figure.colorbar(img, ax=ax)
            ax.xaxis.set_major_formatter(FuncFormatter(format_matrix_ticker))
            ax.yaxis.set_major_formatter(FuncFormatter(format_matrix_ticker))
            return img
            
        if self.is_complex():
            ir = doit(data.real, ax[0])
            ii = doit(data.imag, ax[1])
            result = (ir, ii)
        else:
            result = doit(data, ax)
        if show:
            fig.show()
        return result
    
        
    #
    # Default implementations for cases where self._data exists and
    # is "similar enough" to a NumPy array
    #
    def _get_shape(self):
        return self._data.shape
    
    def _get_dtype(self):
        return self._data.dtype

    def _get_nnz(self):
        raise NotImplementedError()

    #
    # Properties
    #
    def get_dtype(self):
        return self._get_dtype()
    def get_shape(self):
        return self._get_shape()
    def get_kind(self):
        return self._get_kind()
    def get_nnz(self):
        return self._get_nnz()
    
    H = property(fget=conjugate_transpose)
    shape = property(fget=get_shape)
    dtype = property(fget=get_dtype)
    kind = property(fget=get_kind)
    nnz = property(fget=get_nnz)

#
# Matrix multiplication -- see matmul_impl docstring
#
_mul_implementations = {}
def _listify(x):
    return x if isinstance(x, (list, tuple)) else (x,)

class matmul_impl:
    """
    Decorator used for defining matrix multiplication implementations.

    These are looked up by double dispatch on the matrix kinds of left and
    right matrix.

    Implementations...
     - ...can assume matrices conform and are of right kind
     - Must deal with input matrices of different dtypes
     - Can either return a full Matrix instance or a tuple (data, kind).
       In the latter case, data must have a dtype attribute.
     - Only needs to define either left-muls or right-muls; transposing
       is done if necesarry by the caller

    coerces_types specifies whether the input matrices should be coerced
    to the right type prior to entry -- if
    """
    def __init__(self, left_types, right_types):
        self.left_types = _listify(left_types)
        self.right_types = _listify(right_types)

    def __call__(self, func):
        for l in self.left_types:
            for r in self.right_types:
                _mul_implementations[l, r] = func
        return func

#
# Dense matrix
#

class DenseMatrix(Matrix):
    def __init__(self, data, data_format=None, shape=None, dtype=None):
        """
        >>> matrix(np.double, 2, 2, 4, DENSE)
        2 by 2 dense matrix of float64:
        [ 4.  0.]
        [ 0.  4.]
        >>> matrix(np.double, 2, 2, range(4), DENSE)
        2 by 2 dense matrix of float64:
        [ 0.  1.]
        [ 2.  3.]
        >>> matrix(np.double, 2, 3, [[1,2,0],[3,4,5]], DENSE)
        2 by 3 dense matrix of float64:
        [ 1.  2.  0.]
        [ 3.  4.  5.]
        """
        super(DenseMatrix, self).__init__()
        if data_format == 'scalar':
            out = np.eye(shape[0], shape[1], dtype=dtype) * data
        else:
            out = np.asarray(data, dtype=dtype)
            if out.ndim == 1:
                if shape is None:
                    raise ValueError('1D data provided but not shape')
                out = out.reshape(shape) # may raise exception
            elif shape is not None and out.shape != shape:
                raise ValueError('Illegal array shape')
        self._data = out

    def _invert_lu(self, check, **options):
        """
            >>> M = matrix(np.complex128, 2, 2, [1, 1j, -4j, 2], DENSE)
            >>> Minv = M.inverse(algorithm='lu'); Minv
            2 by 2 dense matrix of complex128:
            [-1.0+0.j   0.0+0.5j]
            [ 0.0-2.j  -0.5+0.j ]
            >>> Minv * M
            2 by 2 dense matrix of complex128:
            [ 1.+0.j  0.+0.j]
            [ 0.+0.j  1.+0.j]
        """
        from scipy.linalg import inv
        return DenseMatrix(inv(self._data))

    def _cholesky(self, check, **options):
        """
        >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], DENSE)
        >>> P, L = M.cholesky()
        >>> P
        2 by 2 scaled identity matrix of integers:
        [1 0]
        [0 1]
        >>> L
        2 by 2 dense lower triangular matrix of complex128:
        [ 1.+0.j  0.+0.j]
        [ 0.-1.j  1.+0.j]

        >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], DENSE)
        >>> M.cholesky() == (P, L)
        True
        >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], DENSE)
        >>> M.cholesky()
        Traceback (most recent call last):
          . . .
        ValueError: Matrix is not Hermitian
        >>> M.cholesky(check=False) == (P, L)
        True
        >>> M.cholesky() == (P, L)
        True
        """
        x = self.fetch('cholesky')
        if x is not None: return x
        
        from scipy.linalg import cho_factor
        if check and not self.is_hermitian():
            raise ValueError('Matrix is not Hermitian')
        L_data, is_lower = cho_factor(self._data, lower=True, overwrite_a=False)
        assert is_lower
        L = DenseLowerTriangularMatrix(L_data, dtype=self.dtype, check=False)
        P = ScaledIdentityMatrix(1, dtype=np.intp, shape=self.shape)
        x = (P, L)
        self.cache('cholesky', x)
        return x

    def _solve_cholesky(self, B, on_right, check, **options):
        """
            >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], DENSE)
            >>> M.solve_right([1, 2j])
            [(4+0j), 3j]
            >>> M.solve_left([1, 2j])
            [-0j, 1j]
            >>> M.solve_right(M)
            2 by 2 dense matrix of complex128:
            [ 1.+0.j  0.+0.j]
            [ 0.+0.j  1.+0.j]
        """
        assert on_right
        P, L = self._cholesky(check=check, **options)
        from scipy.linalg import cho_solve
        was_matrix = isinstance(B, Matrix)
        if was_matrix:
            B = B.full_matrix().dense_matrix()._data
        if B.ndim > 2:
            raise NotImplementedError
        X = cho_solve((L._data, True), B)
        if was_matrix:
           X = DenseMatrix(X)
        return X

    def _solve_lu(self, B, on_right, check, **options):
        """
            >>> M = matrix(np.complex128, 2, 2, [1, 1j, -4j, 2], DENSE)
            >>> M.solve_right([1, 2j])
            [(-2+0j), -3j]
            >>> M.solve_left([1, 2j])
            [(3-0j), -0.5j]
            >>> M.solve_right(M)
            2 by 2 dense matrix of complex128:
            [ 1.+0.j  0.+0.j]
            [ 0.+0.j  1.+0.j]
        """
        assert on_right
        from scipy.linalg import solve
        if isinstance(B, np.ndarray):
            return solve(self._data, B)
        else:
            return as_matrix(solve(self._data, B.full_matrix().dense_matrix()._data))

    def _get_kind(self):
        return DENSE
    
    def _full_matrix(self):
        return self

    def _dense_matrix(self):
        return self

    def _conjugate_transpose(self):
        M = self.fetch('conjugate_transpose')
        if M is not None:
            return M
        M = DenseMatrix(self._data.T.conjugate())
        self.cache('conjugate_transpose', M)
        return M

    def _sparse_matrix(self):
        return SparseMatrix(self._data)

    def __cmp__(self, other):
        """
        >>> A = matrix(np.double, 1, 2, [2, 2])
        >>> B = matrix(np.double, 1, 2, [2, 1])
        >>> cmp(A, B)
        1
        >>> cmp(A, A)
        0
        >>> cmp(A, A.sparse_matrix().dense_matrix())
        0
        >>> cmp(A, A.sparse_matrix())
        0
        >>> cmp(A, B.sparse_matrix())
        1
        >>> cmp(A.sparse_matrix(), B)
        1
        
        """
        if self is other:
            return 0
        elif type(other) is DenseMatrix:
            return _cmp_ndarray(self._data, other._data)
        else:
            return cmp(self, other.full_matrix().dense_matrix())

    def _diagonal(self):
        return np.diagonal(self._data)

    def apply(self, ufunc, *args, **kw):
        return DenseMatrix(ufunc(self._data, *args, **kw))

    def _is_triangular(self, lower):
        if lower:
            return np.all(self._data == np.tril(self._data))
        else:
            return np.all(self._data == np.triu(self._data))

    def _invert_triangular(self, check, lower=None, **options):
        """
        >>> M = DenseMatrix([[1, 0, 0], [2, 4, 0], [9, 16, 25]])
        >>> Minv = M.inverse(algorithm='triangular'); Minv
        3 by 3 dense lower triangular matrix of float64:
        [ 1.    0.    0.  ]
        [-0.5   0.25  0.  ]
        [-0.04 -0.16  0.04]
        >>> Minv * M
        3 by 3 dense matrix of float64:
        [ 1.  0.  0.]
        [ 0.  1.  0.]
        [ 0.  0.  1.]

        >>> M.inverse(algorithm='triangular', lower=False)
        Traceback (most recent call last):
          . . .
        ValueError: Matrix is not upper triangular
        >>> DenseMatrix([[1,2],[3,5]]).inverse(algorithm='triangular', lower=True)
        Traceback (most recent call last):
          . . .
        ValueError: Matrix is not lower triangular
        >>> A = DenseMatrix([[1,2],[3,5]])
        >>> Ainv = A.inverse(algorithm='triangular', check=False, lower=True); Ainv
        2 by 2 dense lower triangular matrix of float64:
        [ 1.   0. ]
        [-0.6  0.2]
        >>> Ainv * DenseMatrix([[1,0], [3,5]])
        2 by 2 dense matrix of float64:
        [ 1.  0.]
        [ 0.  1.]

        >>> DenseMatrix([[1, 0], [2, 0]]).inverse(algorithm='triangular', lower=True)
        Traceback (most recent call last):
          . . .
        ValueError: Element at [1,1] is zero, matrix is not invertible
        """
        if lower is None:
            lower = self._is_triangular(lower=True)
            if lower:
                check = False
        if check and not self._is_triangular(lower=lower):
            raise ValueError("Matrix is not %s triangular" % ('lower' if lower else 'upper'))
        from scipy.linalg.lapack import get_lapack_funcs
        (trtri,) = get_lapack_funcs(('trtri',), (self._data,))
        out, info = trtri(self._data, lower=lower, unitdiag=False, overwrite_c=False)
        if info != 0:
            if info > 0:
                raise ValueError("Element at [%d,%d] is zero, matrix is not invertible" % (info - 1, info - 1))
            else:
                raise RuntimeError("Bug in call of LAPACK TRTRI function")
        if lower:
            return DenseLowerTriangularMatrix(out, check=False)
        else:
            raise NotImplementedError()

    def _getitem(self, i, j):
        """
        >>> DenseMatrix([[1,2,3],[4,5,6]])[0:1, 1:3]
        1 by 2 dense matrix of integers:
        [2 3]
        """
        return DenseMatrix(self._data[i, j])

    def _get_nnz(self):
        return np.prod(self.shape)    
        
class _MatrixOpsClass(type):
    """
    Metaclass which simply registers the operations in the framework
    """
    def __init__(cls, name, bases, dct):
        pass

class _MatrixOps(object):
    __metaclass__ = _MatrixOpsClass
    want_out = False

class _DenseArrayOps(_MatrixOps):
    left = DenseMatrix
    right = np.ndarray
#    def mul(left, array, behaviour, 
    

@matmul_impl(DenseMatrix, np.ndarray)
def _mul_dense_array(left, array, out=None):
    if array.ndim <= 2:
        tmp = np.dot(left._data, array)
        if out is not None:
            out[...] = tmp
            return out
        else:
            return tmp
    else:
        # np.dot generalizes to letting the "data vectors" be along
        # second-to-last dimension, while we generalize to first dimension
        # Therefore, manual iterate and call np.dot for 2D x 2D instances.
        if out is None:
            out = np.empty((left.shape[0],) + array.shape[1:],
                           dtype=_pushout_dtype(left._data.dtype, array.dtype))
        for last_idx in range(array.shape[-1]):
            _mul_dense_array(left, array[...,last_idx], out=out[...,last_idx])
        return out

@matmul_impl(np.ndarray, DenseMatrix)
def _mul_array_dense(array, right):
    # Much simpler than _mul_dense_array, since we agree with
    # np.dot here -- right is always 2D
    return np.dot(array, right._data)

@matmul_impl(DenseMatrix, DenseMatrix)
def _mul_dense(left, right):
    """
    TESTS::
    
        >>> A = matrix(np.double, 2, 2, range(4), DENSE)
        >>> B = matrix(np.int, 2, 3, range(6), DENSE)
        >>> A * B
        2 by 3 dense matrix of float64:
        [  3.   4.   5.]
        [  9.  14.  19.]
    """
    return DenseMatrix(np.dot(left._data, right._data))

class DenseLowerTriangularMatrix(DenseMatrix):
    """
    Uses the full sized dense buffer, but allows random data to exist in the upper
    half without ever being referred to. That is, the upper half is "lazily" zeroed;
    calling full_matrix() will zero it.

    Using this class efficiently requires wrapping parts of BLAS which
    NumPy/SciPy doesn't.

    ::

        >>> matrix(np.double, 2, 2, 4, DENSE_LOWER)
        2 by 2 dense lower triangular matrix of float64:
        [ 4.  0.]
        [ 0.  4.]
        >>> M1 = matrix(np.float, 2, 2, [1, 0, 3, 4], DENSE_LOWER); M1
        2 by 2 dense lower triangular matrix of float64:
        [ 1.  0.]
        [ 3.  4.]
        >>> matrix(np.double, 2, 2, [1, 2, 3, 4], DENSE_LOWER)
        Traceback (most recent call last):
            . . .
        ValueError: Non-zero entries provided outside of lower triangle

    Passing data with non-zero upper half is actually fine; it is lazily zeroed::

        >>> M = DenseLowerTriangularMatrix([1, 2, 3, 4], check=False, shape=(2,2))
        >>> M._data[0, 1]
        2
        >>> M == M1
        True
        >>> M
        2 by 2 dense lower triangular matrix of integers:
        [1 0]
        [3 4]
        >>> M._data[0, 1]
        0
    """
    def __init__(self, data, data_format=None, shape=None, dtype=None, check=True):
        super(DenseLowerTriangularMatrix, self).__init__(data, data_format, shape, dtype)
        if check and not np.all(np.triu(self._data, 1) == 0):
            raise ValueError("Non-zero entries provided outside of lower triangle")

    def _get_kind(self):
        return DENSE_LOWER

    def _full_matrix(self):
        x = self.fetch('full')
        if x is not None: return x
        
        # Zero our upper half, and return DenseMatrix instance sharing the data
        self._data = np.tril(self._data)
        x = DenseMatrix(self._data)
        self.cache('full', x)
        return x

    def __cmp__(self, other):
        if self is other:
            return 0
        return cmp(self.full_matrix(), other)

    def _algorithms(self):
        return (('triangular', 'invert_only'),) # TODO

    def _is_triangular(self, lower):
        return lower

    def _conjugate_transpose(self):
        """

        >>> M = DenseLowerTriangularMatrix([1, 2, 3, 4], check=False, shape=(2,2))
        >>> M.H
        2 by 2 dense matrix of integers:
        [1 3]
        [0 4]
        
        """
        # TODO: Upper triangular
        return self._full_matrix()._conjugate_transpose()

#
# Sparse matrix
#

class SparseMatrix(Matrix):
    def __init__(self, data, data_format=None, shape=None, dtype=None):
        """
        >>> matrix(np.double, 2, 2, 4, SPARSE)
        2 by 2 sparse matrix of float64:
        [ 4.  0.]
        [ 0.  4.]
        >>> matrix(np.double, 2, 2, range(4), SPARSE)
        2 by 2 sparse matrix of float64:
        [ 0.  1.]
        [ 2.  3.]
        >>> matrix(np.double, 2, 3, [[1,2,0],[3,4,5]], SPARSE)
        2 by 3 sparse matrix of float64:
        [ 1.  2.  0.]
        [ 3.  4.  5.]
        >>> matrix(np.double, 2, 2,
        ...   coo=([0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]), kind=SPARSE)
        2 by 2 sparse matrix of float64:
        [ 1.  0.]
        [ 0.  1.]

        >>> matrix(np.double, 2, 2, sparse.eye(2, 2), SPARSE)
        2 by 2 sparse matrix of float64:
        [ 1.  0.]
        [ 0.  1.]
        """
        super(SparseMatrix, self).__init__()
        if data_format == 'scalar':
            out = sparse.eye(shape[0], shape[1], dtype=dtype) * data
        elif data_format == 'coo':
            i, j, values = data
            out = sparse.csc_matrix((values, (i, j)), shape, dtype=dtype)
        elif isinstance(data, sparse.spmatrix):
            out = data.tocsc()
        else:
            dense = DenseMatrix(data, data_format, shape, dtype)
            out = sparse.csc_matrix(dense._data)
        self._data = out

    def _get_kind(self):
        return SPARSE
    
    def _conjugate_transpose(self):
        M = self.fetch('conjugate_transpose')
        if M is not None:
            return M
        M = SparseMatrix(self._data.transpose(copy=False).conj())
        self.cache('conjugate_transpose', M)
        return M

    def _full_matrix(self):
        return self

    def _sparse_matrix(self):
        return self

    def _dense_matrix(self):
        return DenseMatrix(self._data.toarray())

    def __cmp__(self, other):
        return cmp(self.dense_matrix(), other.dense_matrix())

    def _diagonal(self):
        """
        >>> matrix(np.double, 2, 2, [1,2,3,4], SPARSE).diagonal()
        array([ 1.,  4.])
        """
        return self._data.diagonal()

    def _cholesky(self, check=True):
        """
        >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], SPARSE)
        >>> P, L = M.cholesky()
        >>> P
        2 by 2 scaled identity matrix of integers:
        [1 0]
        [0 1]
        >>> L
        2 by 2 Cholesky factor (internal CHOLMOD format) matrix of complex128:
        [ 1.+0.j  0.+0.j]
        [ 0.-1.j  1.+0.j]

        >>> M = matrix(np.complex128, 2, 2, [1, 1j, -1j, 2], SPARSE)
        >>> M.cholesky() == (P, L)
        True
        >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], SPARSE)
        >>> M.cholesky()
        Traceback (most recent call last):
          . . .
        ValueError: Matrix is not Hermitian
        >>> M.cholesky(check=False) == (P, L)
        True
        """
        x = self.fetch('cholesky')
        if x is not None: return x

        mode = "supernodal" # force mode for now:
        # Issue #1: If mode is 'auto', CHOLMOD does not check for positive definite!
        # Issue #2: Method of extracting log-determinant/diagonal etc. differs
        # between the two. Perhaps LDL^T should be a seperate call

        if check:
            # CHOLMOD will only access lower triangle at any rate, so must
            # check for Hermitianness. The Hermitian subclass overrides
            # is_hermitian with "return True".
            if not self.is_hermitian():
                raise ValueError("Matrix is not Hermitian")
        
        import scikits.sparse.cholmod as cholmod
        n = self.shape[0]
        try:
            F = cholmod.cholesky(self._data, beta=0, mode=mode)
        except cholmod.CholmodError as e:
            if 'positive definite' in e.args[0]:
                raise MatrixNotPositiveDefiniteError()
            raise
        P = F.P()
        if np.all(P == np.arange(n)):
            MP = ScaledIdentityMatrix(1, dtype=np.int, shape=(n, n))
        else:
            MP = PermutationMatrix(P)
        MF = CholmodFactorMatrix(F, shape=self.shape, dtype=self._data.dtype)
        x = MP, MF
        self.cache('cholesky', x)
        return x

    def _log_determinant_cholesky(self, check):
        """
        >>> M = matrix(np.complex128, 2, 2, [4, 0, -4j, 8], SPARSE_HERMITIAN)
        >>> P, L = M.cholesky(); L       
        2 by 2 Cholesky factor (internal CHOLMOD format) matrix of complex128:
        [ 2.+0.j  0.+0.j]
        [ 0.-2.j  2.+0.j]
        >>> np.allclose(M.log_determinant(), 2*2*np.log(2))
        True
        """
        # Override since CHOLMOD prefers extracting the D of LDL^T
        # also for Cholesky factorization. L will always be CholmodFactorMatrix
        # which provides _diagonal_squared for this purpose.
        P, L = self.cholesky(check=check)
        return np.sum(np.log(L._diagonal_squared()))

    def apply(self, ufunc, *args, **kw):
        return SparseMatrix(self._data._with_data(ufunc(self._data.data, *args, **kw)))

    def _getitem(self, i, j):
        """
        >>> SparseMatrix([[1,2,3],[4,5,6]])[0:1, 1:3]
        1 by 2 sparse matrix of integers:
        [2 3]
        """
        return SparseMatrix(self._data[i, j])

    def _get_nnz(self):
        return self._data.nnz
    

@matmul_impl(SparseMatrix, (SparseMatrix, DenseMatrix))
def _mul_lsparse(left, right):
    """
    TESTS::
    
        >>> A = matrix(np.double, 2, 2, range(4), SPARSE)
        >>> B = matrix(np.int, 2, 3, range(6), DENSE)
        >>> A * B
        2 by 3 dense matrix of float64:
        [  3.   4.   5.]
        [  9.  14.  19.]
        >>> A = matrix(np.double, 2, 2, range(4), SPARSE)
        >>> B = matrix(np.int, 2, 3, range(6), SPARSE)
        >>> A * B
        2 by 3 sparse matrix of float64:
        [  3.   4.   5.]
        [  9.  14.  19.]

        >>> B.H * A.H
        3 by 2 sparse matrix of float64:
        [  3.   9.]
        [  4.  14.]
        [  5.  19.]
    """
    return type(right)(left._data * right._data)

@matmul_impl(SparseMatrix, np.ndarray)
def _mul_sparse_array(left, array, out=None):
    # scipy.sparse only deal with rmul, and only for ndim <= 2
    if array.ndim <= 2:
        tmp = left._data * array
        if out is not None:
            out[...] = tmp
            return out
        else:
            return tmp
    else:
        if out is None:
            out = np.empty((left.shape[0],) + array.shape[1:],
                           dtype=_pushout_dtype(left._data.dtype, array.dtype))
        for last_idx in range(array.shape[-1]):
            _mul_sparse_array(left, array[...,last_idx], out=out[...,last_idx])
        return out
        
#
# Sparse hermitian
#
class SparseHermitianMatrix(SparseMatrix):
    """
        >>> A = matrix(np.complex, 2, 2, [[1,0],[1+1j,2]], SPARSE_HERMITIAN)
        >>> Asq = A * A; Asq
        2 by 2 sparse matrix of complex128:
        [ 3.+0.j  3.-3.j]
        [ 3.+3.j  6.+0.j]

        >>> A[1:2,1:2]
        1 by 1 sparse Hermitian matrix of complex128:
        [ 2.+0.j]

        >>> A[1:2,0:2]
        1 by 2 sparse matrix of complex128:
        [ 1.+1.j  2.+0.j]

    """
    def __init__(self, data, data_format=None, shape=None, dtype=None):
        """

        >>> matrix(np.complex, 2, 2, [[1,0],[1+1j,2]], SPARSE_HERMITIAN)
        2 by 2 sparse Hermitian matrix of complex128:
        [ 1.+0.j  1.-1.j]
        [ 1.+1.j  2.+0.j]
        >>> matrix(np.complex, 2, 2, [1, 1-1j, 1+1j, 2], SPARSE_HERMITIAN)
        Traceback (most recent call last):
            ...
        ValueError: Hermitian matrix must be all symbolic 0 above diagonal
        >>> matrix(np.complex, 2, 2, [1, 0, 0, 1+1e-10j], SPARSE_HERMITIAN)
        Traceback (most recent call last):
            ...
        ValueError: Hermitian matrix must have real entries on diagonal
        """
        # Use original constructor, but do some validation of data
        super(SparseHermitianMatrix, self).__init__(data, data_format, shape, dtype)
        coo_data = self._data.tocoo()
        i, j, values = coo_data.row, coo_data.col, coo_data.data
        if np.any(i < j):
            raise ValueError('Hermitian matrix must be all symbolic 0 above diagonal')
        if np.any(values[i == j].imag):
            raise ValueError('Hermitian matrix must have real entries on diagonal')

    def _get_kind(self):
        return SPARSE_HERMITIAN
    
    def _full_matrix(self):
        """
        Returns self in a non-special matrix space.

        >>> matrix(np.complex, 2, 2, [1, 0, 1-1j, 2], SPARSE_HERMITIAN).full_matrix()
        2 by 2 sparse matrix of complex128:
        [ 1.+0.j  1.+1.j]
        [ 1.-1.j  2.+0.j]
        """
        coo_data = self._data.tocoo()
        i, j, values = coo_data.row, coo_data.col, coo_data.data
        # Construct the upper half from the lower and add it
        lower_set = (i > j)
        uppervals = values[lower_set].conjugate()
        values = np.concatenate((values, values[lower_set].conjugate()))
        i = np.concatenate((i, j[lower_set]))
        j = np.concatenate((j, i[lower_set]))
        full_data = sparse.csc_matrix((values, (i, j)), self.shape,
                                      dtype=self._data.dtype)
        return SparseMatrix(full_data)

    def _conjugate_transpose(self):
        return self

    def _dense_matrix(self):
        raise NotImplementedError()

    def _sparse_matrix(self):
        return self

    def _is_hermitian(self):
        return True

    def _get_symmetric_slice(self, start, stop):
        return SparseHermitianMatrix(self._data[start:stop, start:stop])

        

class CholmodFactorMatrix(Matrix):
    """
    EXAMPLES::
    
        >>> M = matrix(np.complex128, 2, 2, [4, 0, -4j, 8], SPARSE_HERMITIAN)
        >>> P, L = M.cholesky(); L
        2 by 2 Cholesky factor (internal CHOLMOD format) matrix of complex128:
        [ 2.+0.j  0.+0.j]
        [ 0.-2.j  2.+0.j]
        >>> type(L) #doctest: +ELLIPSIS
        <class '...CholmodFactorMatrix'>

        >>> L * [3, 3j]
        [(6+0j), 0j]
        >>> L.H * [3,3j]
        [0j, 6j]
        >>> [3,3j] * L
        [(12-0j), 6j]
        >>> [3,3j] * L.H
        [(6-0j), 12j]

        >>> L.solve_right([2,0])
        [(1+0j), 1j]

        >>> L.H.solve_right([2,0])
        [(1+0j), 0j]
    """
    
    def __init__(self, cholmod_factor, dtype, shape, nontransposed=None):
        super(CholmodFactorMatrix, self).__init__()
        self._factor = cholmod_factor
        self._original = nontransposed
        self._shape = shape
        self._dtype = dtype
        self._is_transposed = nontransposed is not None            

    def _get_shape(self):
        return self._shape

    def _get_kind(self):
        return CHOLMOD_FACTOR

    def _get_dtype(self):
        return self._dtype

    def _get_nnz(self):
        return self._full_matrix()._get_nnz()

    def _full_matrix(self):
        M = self.fetch('full')
        if M is not None:
            return M
        if self._is_transposed:
            # Go to SparseMatrix before conjugation to allow sharing cached
            # transposes
            return self._original.full_matrix().conjugate_transpose()
        else:
            M = SparseMatrix(self._factor.L())
        self.cache('full', M)
        return M

    def _conjugate_transpose(self):
        if self._is_transposed:
            return self._original
        return CholmodFactorMatrix(self._factor, dtype=self._dtype,
                                   shape=self._shape, nontransposed=self)

    def _algorithms(self):
        return (('triangular', 'right'),)

    def _solve_triangular(self, B, on_right, check):
        """
        Do a full Cholesky solve manually, testing functionality along the way::

            >>> M = matrix(np.complex128, 2, 2, [1, 0, -1j, 2], SPARSE_HERMITIAN)
            >>> P, L = M.cholesky(); L
            2 by 2 Cholesky factor (internal CHOLMOD format) matrix of complex128:
            [ 1.+0.j  0.+0.j]
            [ 0.-1.j  1.+0.j]
            >>> x = L.solve_right([1, 2j]); x
            [(1+0j), 3j]
            >>> L * x
            [(1+0j), 2j]
            >>> x = L.solve_left([1,2j]); x
            [(-1+0j), 2j]
            >>> x * L
            [(1-0j), 2j]
        
        """
        if not on_right:
            raise AssertionError()
        if isinstance(B, np.ndarray):
            array = B
        else:
            B = B.full_matrix()
            array = B._data # does not matter whether sparse or dense

        CHOLMOD_L = 4
        CHOLMOD_Lt = 5

        self._factor._ensure_L_or_LD_inplace(True)
        if not self._is_transposed:
            X = self._factor._solve(array, CHOLMOD_L)
        else:
            X = self._factor._solve(array, CHOLMOD_Lt)

        if array.ndim == 1 and X.ndim == 2:
            X = X[:,0]

        if isinstance(B, np.ndarray):
            return X
        elif B.is_sparse():
            return SparseMatrix(X)
        else:
            return DenseMatrix(X)

    def _diagonal(self):
        return np.sqrt(self._diagonal_squared())

    def _diagonal_squared(self):
        return self._factor.D()

class ScaledIdentityMatrix(Matrix):
    """
    >>> I = identity_matrix(3, dtype=np.float64); I
    3 by 3 scaled identity matrix of float64:
    [ 1.  0.  0.]
    [ 0.  1.  0.]
    [ 0.  0.  1.]
    
    >>> I * 3
    3 by 3 scaled identity matrix of float64:
    [ 3.  0.  0.]
    [ 0.  3.  0.]
    [ 0.  0.  3.]

    >>> (2+3j) * I
    3 by 3 scaled identity matrix of complex128:
    [ 2.+3.j  0.+0.j  0.+0.j]
    [ 0.+0.j  2.+3.j  0.+0.j]
    [ 0.+0.j  0.+0.j  2.+3.j]

    >>> I2 = identity_matrix(2)
    >>> D = matrix(np.float64, 2, 2, [1,2,3,4])
    >>> (2 * I2) * D
    2 by 2 dense matrix of float64:
    [ 2.  4.]
    [ 6.  8.]
    >>> D * (I2 * 2)
    2 by 2 dense matrix of float64:
    [ 2.  4.]
    [ 6.  8.]

    >>> S = D.sparse_matrix()
    >>> (2 * I2) * D == (2 * I2) * S
    True
    >>> D * (I2 * 2) == S * (I2 * 2)
    True
    """
    
    def __init__(self, value, shape, dtype=None):
        super(ScaledIdentityMatrix, self).__init__()
        # Ensure that value is a NumPy dtype scalar
        if dtype is None:
            dtype = value.dtype
        else:
            dtype = np.dtype(dtype)
            value = dtype.type(value)
        self._dtype = dtype
        self._data = value
        self._shape = shape
        if self._shape[0] != self._shape[1]:
            raise NotImplementedError()

    def _algorithms(self):
        return (('special', 'both'),)

    def _get_kind(self):
        return SCALED_IDENTITY

    def _get_shape(self):
        return self._shape

    def _full_matrix(self):
        return self.sparse_matrix()

    def _sparse_matrix(self):
        return SparseMatrix(sparse.eye(*self._shape, dtype=self._dtype) * self._data)

    def _dense_matrix(self):
        return DenseMatrix(self._sparse_matrix()._data.toarray())

    def _diagonal(self):
        return np.repeat(self._data, self.shape[0])

    def _conjugate_transpose(self):
        return self

    def _solve_trivial(self, B, on_right, check, **options):
        if self._data == 1:
            return B
        else:
            return B * (1/self._data)

    def _get_symmetric_slice(self, start, stop):
        """
        >>> D = identity_matrix(4)
        >>> D[0:3,0:3]
        3 by 3 scaled identity matrix of integers:
        [1 0 0]
        [0 1 0]
        [0 0 1]
        >>> D[0:3,2:4]
        3 by 2 sparse matrix of integers:
        [0 0]
        [0 0]
        [1 0]
        """
        n = stop - start
        return ScaledIdentityMatrix(self._data, (n, n), self._dtype)


@matmul_impl(ScaledIdentityMatrix,
             (DenseMatrix, SparseMatrix, SparseHermitianMatrix, np.ndarray))
def _mul_ident_dense(left, right):
    return right * left._data

@matmul_impl((DenseMatrix, SparseMatrix, SparseHermitianMatrix, np.ndarray,
              ScaledIdentityMatrix),
             ScaledIdentityMatrix)
def _mul_dense_ident(left, right):
    return left * right._data

#
# Diagonal matrix
#
class DiagonalMatrix(Matrix):
    """
    >>> D = diagonal_matrix([10,10,100,100]); D
    4 by 4 diagonal matrix of integers:
    [ 10   0   0   0]
    [  0  10   0   0]
    [  0   0 100   0]
    [  0   0   0 100]
    >>> A = matrix(np.float, 4, 4, range(16)); A
    4 by 4 dense matrix of float64:
    [  0.   1.   2.   3.]
    [  4.   5.   6.   7.]
    [  8.   9.  10.  11.]
    [ 12.  13.  14.  15.]
    >>> D * A
    4 by 4 dense matrix of float64:
    [    0.    10.    20.    30.]
    [   40.    50.    60.    70.]
    [  800.   900.  1000.  1100.]
    [ 1200.  1300.  1400.  1500.]
    >>> A * D
    4 by 4 dense matrix of float64:
    [    0.    10.   200.   300.]
    [   40.    50.   600.   700.]
    [   80.    90.  1000.  1100.]
    [  120.   130.  1400.  1500.]

    """
    def __init__(self, entries, nrows=None, ncols=None, dtype=None):
        super(DiagonalMatrix, self).__init__()
        self._data = np.asarray(entries, dtype=dtype)
        if self._data.ndim != 1:
            raise ValueError('Must provide 1D diagonal')
        if nrows is None:
            nrows = self._data.shape[0]
        if ncols is None:
            ncols = nrows
        
        if not (self._data.shape[0] == ncols == nrows):
            # TODO, but this makes for easier multiplication code
            raise NotImplementedError()
        self._nrows = nrows
        self._ncols = ncols
            
    def _get_shape(self):
        return (self._nrows, self._ncols)

    def _full_matrix(self):
        M = sparse.dia_matrix((self._data[None, :], [0]),
                               shape=(self._nrows, self._ncols),
                               dtype=self._data.dtype)
        return SparseMatrix(M)

    def _get_kind(self):
        return DIAGONAL

    def apply(self, ufunc, *args, **kw):
        return DiagonalMatrix(ufunc(self._data, *args, **kw))

    def _cholesky(self, check):
        assert self._nrows == self._ncols
        P = identity_matrix(self._nrows, self._data.dtype)
        L = self.apply(np.sqrt)
        return (P, L)

    def _diagonal(self):
        return self._data

    def _algorithms(self):
        return (('trivial', 'right'),)

    def _solve_trivial(self, B, on_right, check, **options):
        """
        >>> DiagonalMatrix([1,2,4]).solve_right([1,50,100])
        [1.0, 25.0, 25.0]
        >>> DiagonalMatrix([1,2,4]).solve_left([1,50,100])
        [1.0, 25.0, 25.0]
        """
        return self._invert_trivial() * B

    def _conjugate_transpose(self):
        if self.is_complex():
            return self.apply(np.conjugate)
        else:
            return self

    def _invert_trivial(self, **options):
        """
        >>> DiagonalMatrix([1,2,4])._inverse()
        3 by 3 diagonal matrix of float64:
        [ 1.    0.    0.  ]
        [ 0.    0.5   0.  ]
        [ 0.    0.    0.25]
        """
        x = self.fetch('inverse')
        if x is not None: return x

        x = DiagonalMatrix(1.0 / self._data)
        self.cache('inverse', x)
        return x

    def _mul_scalar(self, c):
        return DiagonalMatrix(self._data * c)

    def _get_symmetric_slice(self, start, stop):
        """
        >>> D = DiagonalMatrix([1,2,3,4])
        >>> D[1:3, 1:3]
        2 by 2 diagonal matrix of integers:
        [2 0]
        [0 3]
        >>> D[0:3,2:4]
        3 by 2 sparse matrix of integers:
        [0 0]
        [0 0]
        [3 0]
        """
        return DiagonalMatrix(self._data[start:stop])

@matmul_impl(DiagonalMatrix, DenseMatrix)
def _mul_dia_dense(dia, dense):
    """
    >>> diagonal_matrix([10, 100]) * matrix(np.float, 2, 3, range(6))
    2 by 3 dense matrix of float64:
    [   0.   10.   20.]
    [ 300.  400.  500.]
    """
    return DenseMatrix(dia._data[:, None] * dense._data)

@matmul_impl(DenseMatrix, DiagonalMatrix)
def _mul_dense_dia(dense, dia):
    """
    >>> matrix(np.float, 2, 3, range(6)) * diagonal_matrix([10, 100, 1000])
    2 by 3 dense matrix of float64:
    [    0.   100.  2000.]
    [   30.   400.  5000.]
    """
    return DenseMatrix(dia._data[None, :] * dense._data)

@matmul_impl(DiagonalMatrix, np.ndarray)
def _mul_dia_array(dia, array):
    """
    >>> diagonal_matrix([10, 100, 1000]) * [1, 2, 3]
    [10, 200, 3000]
    >>> x = diagonal_matrix([10, 100, 1000]) * np.arange(24).reshape(3, 2, 4)
    >>> x.shape
    (3, 2, 4)
    >>> x[:,0,0]
    array([    0,   800, 16000])
    >>> x[2,:,:]
    array([[16000, 17000, 18000, 19000],
           [20000, 21000, 22000, 23000]])
    """
    return dia._data[(slice(None),) + (None,) * (array.ndim-1)] * array

@matmul_impl(np.ndarray, DiagonalMatrix)
def _mul_array_dia(array, dia):
    """
    >>> [1, 2, 3] * diagonal_matrix([10, 100, 1000])
    [10, 200, 3000]
    >>> x = np.arange(24).reshape(4, 2, 3) * diagonal_matrix([10, 100, 1000])
    >>> x.shape
    (4, 2, 3)
    >>> x[0,0,:]
    array([   0,  100, 2000])
    >>> x[:,:,2]
    array([[ 2000,  5000],
           [ 8000, 11000],
           [14000, 17000],
           [20000, 23000]])
    """
    return dia._data[(None,) * (array.ndim-1) + (slice(None),)] * array

@matmul_impl(DiagonalMatrix, DiagonalMatrix)
def _mul_array_dia(a, b):
    """
    >>> D = diagonal_matrix([1, 2, 3])
    >>> D * D
    3 by 3 diagonal matrix of integers:
    [1 0 0]
    [0 4 0]
    [0 0 9]
    """
    return DiagonalMatrix(a._data * b._data)

# TODO: Efficient mul with sparse matrices

#
# Permutation matrix
#
class PermutationMatrix(Matrix):
    """
    >>> 
    """
    def __init__(self, permutation, value=np.int_(1), dtype=None, _transposed=None):
        super(PermutationMatrix, self).__init__()
        # Ensure that value is a NumPy dtype scalar
        if value != 1:
            # Didn't bother to implement efficient solve with scalars yet,
            # + perhaps the scalar multiple should be generalized into a superclass
            raise NotImplementedError()
        if dtype is None:
            dtype = value.dtype
        else:
            dtype = np.dtype(dtype)
            value = dtype.type(value)
        self._dtype = dtype
        self._value = value
        # Accept any integer NumPy array; use intp if one has to be constructed
        if (isinstance(permutation, np.ndarray) and
            issubclass(permutation.dtype.type, np.int_)):
            self._perm = permutation
        else:
            self._perm = np.asarray(permutation, dtype=np.intp)
        self._n = self._perm.shape[0]
        # Compute the inverse/transpose right away
        if _transposed is not None:
            self._transposed = _transposed
        else:
            self._transposed = PermutationMatrix(
                _inverse_permutation(self._perm), self._value, self._dtype, self)

    def _algorithms(self):
        return (('special', 'both'),)

    def _get_kind(self):
        return PERMUTATION

    def _get_dtype(self):
        return self._dtype

    def _get_shape(self):
        return (self._n, self._n)

    def _full_matrix(self):
        return self.sparse_matrix()

    def _sparse_matrix(self):
        """
        >>> PermutationMatrix([1, 0, 2, 3]).sparse_matrix()
        4 by 4 sparse matrix of integers:
        [0 1 0 0]
        [1 0 0 0]
        [0 0 1 0]
        [0 0 0 1]

        Repr calls full_matrix which calls sparse_matrix:
        >>> PermutationMatrix([1, 0, 2, 3], dtype=np.float)
        4 by 4 permutation matrix of float64:
        [ 0.  1.  0.  0.]
        [ 1.  0.  0.  0.]
        [ 0.  0.  1.  0.]
        [ 0.  0.  0.  1.]
        """
        rows = np.arange(self._n, dtype=_sparse_index_dtype)
        cols = self._perm
        values = np.ones(self._n, dtype=self.dtype)
        out = sparse.coo_matrix((values, (rows, cols)), shape=(self._n, self._n))
        return SparseMatrix(out)

    def _dense_matrix(self):
        return DenseMatrix(self._sparse_matrix()._data.toarray())

    def _diagonal(self):
        """
        >>> PermutationMatrix([1, 0, 2, 3]).diagonal()
        array([0, 0, 1, 1])
        >>> PermutationMatrix([1, 0, 2, 3], dtype=np.float).diagonal()
        array([ 0.,  0.,  1.,  1.])
        """
        out = np.zeros(self._n, dtype=self._dtype)
        out[self._perm == np.arange(self._n)] = 1
        return out

    def _conjugate_transpose(self):
        return self._transposed

    def _solve_trivial(self, B, on_right, check, **options):
        assert self._value == 1
        return self._transposed * P

def _inverse_permutation(p):
    """
    >>> _inverse_permutation(np.array([2, 0, 1]))
    array([1, 2, 0])
    >>> _inverse_permutation(np.array([1, 2, 0]))
    array([2, 0, 1])
    >>> _inverse_permutation(np.array([0, 1, 2]))
    array([0, 1, 2])
    """
    assert isinstance(p, np.ndarray)
    invp = np.empty_like(p)
    invp[p] = np.arange(p.shape[0], dtype=p.dtype)
    return invp

# Only define right-muls
# Permutations are given by row; column permutation (left-muls) require
# the transpose/inverse anyhow

@matmul_impl(PermutationMatrix, (DenseMatrix, SparseMatrix))
def _mul_perm_matrix(perm, M):
    """
    Right-mul:
    >>> P = PermutationMatrix([2, 0, 1]); P
    3 by 3 permutation matrix of integers:
    [0 0 1]
    [1 0 0]
    [0 1 0]
    >>> M = DenseMatrix(np.arange(12).reshape(3, 4)); M
    3 by 4 dense matrix of integers:
    [ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]
    >>> P * M
    3 by 4 dense matrix of integers:
    [ 8  9 10 11]
    [ 0  1  2  3]
    [ 4  5  6  7]
    >>> P * M.sparse_matrix()
    3 by 4 sparse matrix of integers:
    [ 8  9 10 11]
    [ 0  1  2  3]
    [ 4  5  6  7]

    Left-mul:
    >>> N = M.conjugate_transpose()
    >>> N * P
    4 by 3 dense matrix of integers:
    [ 4  8  0]
    [ 5  9  1]
    [ 6 10  2]
    [ 7 11  3]
    >>> N.sparse_matrix() * P
    4 by 3 sparse matrix of integers:
    [ 4  8  0]
    [ 5  9  1]
    [ 6 10  2]
    [ 7 11  3]
    """
    return type(M)(M._data[perm._perm,:])
    

@matmul_impl(PermutationMatrix, np.ndarray)
def _mul_perm_array(perm, array):
    """

    Test right-mul:
    >>> P = PermutationMatrix([2, 0, 1])
    >>> x = np.arange(24).reshape(3, 2, 4)
    >>> P * [10, 100, 1000]
    [1000, 10, 100]
    >>> np.all(PermutationMatrix([0, 1, 2]) * x == x)
    True
    >>> z = P * x
    >>> np.all(z[0,:,:] == x[2,:,:])
    True
    >>> np.all(z[1,:,:] == x[0,:,:])
    True

    Test left-mul:

    >>> P = PermutationMatrix([2, 0, 1])
    >>> x = np.arange(24).reshape(4, 2, 3)
    >>> [10, 100, 1000] * P
    [100, 1000, 10]
    >>> np.all(x * PermutationMatrix([0, 1, 2]) == x)
    True
    >>> y = x * P
    >>> x.shape
    (4, 2, 3)
    >>> y[0,0,:]
    array([1, 2, 0])
    >>> np.all(x[:,:,1] == y[:,:,0])
    True
    >>> np.all(x[:,:,0] == y[:,:,2])
    True
    """
    return array[perm._perm, ...]

#
# Matrix product
#
class MatrixProduct(Matrix):
    def __init__(self, matrices):
        if len(matrices) == 0:
            raise ValueError()
        M = matrices[0]
        nrows = M.shape[0]
        ncols = M.shape[1]
        for M in matrices[1:]:
            if ncols != M.shape[0]:
                raise ValueError('Shapes do not conform')
            ncols = M.shape[1]
        self._shape = (nrows, ncols)
        self._matrices = matrices

    def _is_hermitian(self):
        raise NotImplementedError()

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        raise NotImplementedError()

@matmul_impl(MatrixProduct, np.ndarray)
def _mul_matprod_array(matprod, array):
    for M in matprod._matrices:
        array = M * array
    return array

class MatrixWrapProduct(Matrix):
    """
    Symbolic product ... B2^H B1^H A B1 B2 ...,
    where A is supplied as the kernel argument and
    wraps is a list of the Bs. 
    """
    def __init__(self, kernel, wraps):
        raise NotImplementedError()
        if len(matrices) == 0:
            raise ValueError()
        M = matrices[0]
        nrows = M.shape[0]
        ncols = M.shape[1]
        for M in matrices[1:]:
            if ncols != M.shape[0]:
                raise ValueError('Shapes do not conform')
            ncols = M.shape[1]
        self._shape = (nrows, ncols)
        self._matrices = matrices

    def _is_hermitian(self):
        raise NotImplementedError()

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        raise NotImplementedError()

@matmul_impl(MatrixProduct, np.ndarray)
def _mul_matprod_array(matprod, array):
    for M in matprod._matrices:
        array = M * array
    return array

# The rest is TODO...

#
# Matrix kind enum values
#
DENSE = MatrixKind('dense', DenseMatrix)
SPARSE = MatrixKind('sparse', SparseMatrix, sparse=True)

## HERMITIAN_DENSE = MatrixKind('full Hermitian', hermitian=True)
SPARSE_HERMITIAN = MatrixKind('sparse Hermitian', SparseHermitianMatrix,
                              hermitian=True, sparse=True)
DIAGONAL = MatrixKind('diagonal', DiagonalMatrix, diagonal=True)
## LOWER_TRIANGULAR_SPARSE = MatrixKind('sparse lower triangular',
##                                          sparse=True, triangular=True, lower=True)
PERMUTATION = MatrixKind('permutation', PermutationMatrix)
SCALED_IDENTITY = MatrixKind('scaled identity', ScaledIdentityMatrix)
CHOLMOD_FACTOR = MatrixKind('Cholesky factor (internal CHOLMOD format)', None)

DENSE_LOWER = MatrixKind('dense lower triangular', DenseLowerTriangularMatrix,
                         sparse=False, triangular=True, lower=True)


PRODUCT = MatrixKind('product', MatrixProduct,
                     sparse=False, triangular=False, lower=False)

def conjugate_transpose(x):
    if isinstance(x, np.ndarray):
        return x.T.conjugate()
    else:
        return x.conjugate_transpose()

def _matrix_mul(left, right):
    """
    Multiplies two matrices or one array with a matrix. This primarily
    happens through a double dispatch to specific implementations defined
    below.
    
    TESTS::

    See Matrix.__mul__ and Matrix.__rmul__.
    """
    left, right, inshape, outshape, data, data_behaviour = get_data_behaviour(left, right)
    assert data is left or data is right or data is None

    # First validate shape
    if left.shape[-1] != right.shape[0]:
        raise ValueError('Matrix shapes do not conform')

    # Look up mul routine through double dispatch on matrix kind
    left, right, impl, transposed = _lookup_mul_implementation(left, right)
    if transposed:
        left, right = conjugate_transpose(right), conjugate_transpose(left)

    out = impl(left, right)
    if transposed:
        out = conjugate_transpose(out)
    if data_behaviour is not None:
        out = data_behaviour.wrap_out(None, inshape, outshape, data, out)
    return out

def _lookup_mul_implementation(l, r):
    transposed = False
    ltype = orig_ltype = type(l)
    rtype = orig_rtype = type(r)
    if issubclass(ltype, np.ndarray):
        ltype = np.ndarray
    elif issubclass(rtype, np.ndarray):
        rtype = np.ndarray
    impl = _mul_implementations.get((ltype, rtype), None)
    if impl is None:
        # Try to transpose
        impl = _mul_implementations.get((rtype, ltype), None)
        if impl is not None:
            transposed = True
    if impl is None:
        # Try to convert to full matrices
        retry = False
        if isinstance(l, Matrix):
            l = l.full_matrix()
            retry = True
        if isinstance(r, Matrix):
            r = r.full_matrix()
            retry = True
        if retry:
            l, r, impl, transposed = _lookup_mul_implementation(l, r)
    if impl is None:
        raise NotImplementedError(("Cannot multiply '%s' "
                                       "with '%s'") % (ltype.__name__, rtype.__name__))
    return l, r, impl, transposed



#
# Utils
#
def _cmp_ndarray(a, b):
    r = cmp(a.shape, b.shape)
    if r != 0: return r
    for aval, bval in np.broadcast(a, b):
        r = cmp(aval, bval)
        if r != 0: return r
    return 0

def _round(data, ndigits, zero_sign):
    out = np.round(data, ndigits)
    if zero_sign:
        if issubclass(data.dtype.type, np.complex_):
            out.real[out.real == 0] = 0
            out.imag[out.imag == 0] = 0
        else:
            out[out == 0] = 0
    return out

def _normalize_slice(s, n):
    """
    Utility function for use by _getitem implementations

    >>> _normalize_slice(slice(None), 10)
    slice(0, 10, 1)
    >>> _normalize_slice(slice(-2, -1, None), 10)
    slice(8, 10, 1)
    >>> _normalize_slice(slice(0, 10, None), 10)
    slice(0, 10, 1)
    >>> _normalize_slice(slice(0, 11, None), 10)
    Traceback (most recent call last):
      . . .
    ValueError: Index out of bounds
    >>> _normalize_slice(slice(10, 10, None), 10)
    Traceback (most recent call last):
      . . .
    ValueError: Index out of bounds
    """

    def do(x, end, default):
        if x is None:
            return default
        elif x < 0:
            x += n + (1 if end else 0)
        if x < 0 or x >= n + (1 if end else 0):
            raise ValueError("Index out of bounds")
        return x

    step = 1 if s.step is None else s.step
    start = do(s.start, False, 0)
    stop = do(s.stop, True, n)

    return slice(start, stop, step)


#
# Dtype conversions
#
def _pushout_dtype(left_dtype, right_dtype):
    return (np.array(1, left_dtype) * np.array(1, right_dtype)).dtype

#
# Construction
#
def as_matrix(x, copy=True, hermitian=False):
    if isinstance(x, Matrix):
        return x
    elif isinstance(x, np.ndarray):
        if copy:
            x = x.copy()
        if x.ndim == 1:
            return DiagonalMatrix(x)
        elif x.ndim == 2:
            return DenseMatrix(x)
        else:
            raise ValueError("Only 1D and 2D arrays supported")
    elif isinstance(x, sparse.spmatrix):
        if copy:
            x = x.copy()
        if hermitian:
            return SparseHermitianMatrix(x)
        else:
            return SparseMatrix(x)        
    else:
        raise TypeError("Unsupported type")

def matrix(dtype, nrows, ncols, entries=1, kind=DENSE, coo=None):
    ctor = kind.default_constructor
    data_format = None
    if coo is not None:
        data_format = 'coo'
        if entries != 1:
            raise ValueError('Cannot provide both coo and entries')
        entries = coo
    elif np.isscalar(entries):
        data_format = 'scalar'        
    return ctor(entries, data_format, shape=(nrows, ncols), dtype=dtype)


def identity_matrix(n, dtype=np.int, value=1):
    return ScaledIdentityMatrix(value, shape=(n, n), dtype=dtype)

def diagonal_matrix(entries, nrows=None, ncols=None, dtype=None):
    return DiagonalMatrix(entries, nrows, ncols, dtype)

def is_matrix(x):
    return isinstance(x, Matrix)

def hermitian_matrix(entries, nrows, ncols, sparse=False, dtype=None):
    kind = SPARSE_HERMITIAN if sparse else DENSE_HERMITIAN
    if (isinstance(entries, tuple) and len(entries) == 2
        and isinstance(entries[1], tuple) and len(entries[1]) == 2):
        # coo format
        values, (rows, cols) = entries
        return matrix(dtype, nrows, ncols, kind=kind, coo=(rows, cols, values))
    else:
        return matrix(dtype, nrows, ncols, entries, kind)

    
#
# Behaviour registration
#
class ToArrayBehaviour(object):
    """
    Default behaviour -- convert all input/output which are not matrices
    to np.ndarray.
    """
    def allocate_out(self, op, inshape, outshape, input, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def wrap_out(self, op, inshape, outshape, input, out):
        return out.view(np.ndarray)

    def as_array(self, op, inshape, outshape, input, dtype):
        arr = np.asarray(input)
        if arr.ndim == 0:
            # Resulted in a 0-dimensional scalar
            raise TypeError("Matrices cannot operate with %r" % input)
        return arr

class ListConversionBehaviour(ToArrayBehaviour):
    # inherit allocate_out and as_array
    def wrap_out(self, op, inshape, outshape, input, out):
        return out.tolist()


array_behaviour = ToArrayBehaviour()

_behaviours = {}
def add_default_data_behaviour(type, behaviour):
    if isinstance(type, list):
        for t in type:
            add_default_data_behaviour(t, behaviour)
        return
    # Only the exact type for now, not subclasses
    _behaviours[type] = behaviour

def get_default_data_behaviour(type):
    b = _behaviours.get(type, array_behaviour)
    return b

def get_data_behaviour(left, right):
    if isinstance(right, Matrix):
        if isinstance(left, Matrix):
            return left, right, None, None, None, None
        data, matrix = left, right
        left_is_data = True
        inshape = matrix.shape[0]
        outshape = matrix.shape[1]
    else:
        matrix, data = left, right
        left_is_data = False
        inshape = matrix.shape[1]
        outshape = matrix.shape[0]
    if matrix._data_behaviour is not None:
        b = matrix._data_behaviour
    else:
        b = get_default_data_behaviour(type(data))
    data = b.as_array(None, inshape, outshape, data, matrix.dtype)                      
    if left_is_data:
        return data, matrix, inshape, outshape, data, b
    else:
        return matrix, data, inshape, outshape, data, b
        

add_default_data_behaviour(list, ListConversionBehaviour())
