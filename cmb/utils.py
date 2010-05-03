from time import clock
import logging
import os
import numpy as np
import contextlib

__all__ = ['get_times', 'log_times', 'working_directory', 'verbosity_to_logger',
           'as_random_state', 'check_l_increases',
           'locked_pytable', 'locked_h5file',
           'pycmb_debug', 'timestats']

# Use MPI for wall time if present, otherwise try for OpenMP wall time,
# otherwise local clock
try:
    from mpi4py.MPI import Wtime as wall_time
except ImportError:
    try:
        from healpix.openmp import get_wtime as wall_time
    except ImportError:
        from time import time as wall_time

def get_times():
    return (clock(), wall_time())

def log_times(logger, t0, msg='Done (time taken: %s)'):
    logger.info(msg % timestats(t0))

def timestats(t0):
    c0, w0 = t0
    dc, dw = clock() - c0, wall_time() - w0
    if dc > 0.01 and dw > 0.01:
        timestr = '%.2f wall, %.2f CPU' % (dw, dc)
    else:
        timestr = '%.2e wall, %.2e CPU' % (dw, dc)
    return timestr

_verbosity_to_loglevel = [logging.WARNING, logging.INFO, logging.DEBUG]


def as_random_state(seed):
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise TypeError('Provide int or RandomState for random seed')

def verbosity_to_logger(verbosity=None, logger=None):
    """
    Converts an optional verbosity level and an optional logger
    into a logger instance with the right filter level set.
    """
    if verbosity is not None:
        if verbosity < len(_verbosity_to_loglevel):
            verbosity = _verbosity_to_loglevel[verbosity]
    if logger is None:
        logging.basicConfig(level=verbosity)
        logger = logging.getLogger()
    else:
        logger.setLevel(verbosity)
    return logger

class working_directory(object):
    """
    Context manager to temporarily change current working directory

    Example:
    
    with working_directory('/tmp'):
        assertEqual(os.path.basename(os.getcwd()), 'tmp')
    """

    def __init__(self, dirname, create=False):
        self.dirname = os.path.realpath(
            os.path.expandvars(os.path.expanduser(dirname)))
        if not os.path.isdir(self.dirname):
            if create:
                os.makedirs(self.dirname)
            else:
                raise ValueError("Is not a directory: %s" % self.dirname)
        self._oldcwd = os.getcwd()

    def __enter__(self):
        os.chdir(self.dirname)
        return self.dirname

    def __exit__(self, exc, value, tb):
        if os.getcwd() != self.dirname:
            import warnings
            warnings.warn('Want to exit "%s" by changing to "%s" as we exit the context manager, '
                          'but the directory was changed to "%s" in the meantime. '
                          'Proceeding anyway.' % (
                self.dirname, self._oldcwd, os.getcwd())
                )
        os.chdir(self._oldcwd)

    def __call__(self, path):
        if os.path.isabs(path):
            return os.path.realpath(path)
        else:
            return os.path.realpath(os.path.join(self.dirname, path))

@contextlib.contextmanager
def locked_h5file(filename, mode, *args, **kw):
    import h5py
    from fcntl import lockf, LOCK_EX, LOCK_SH, LOCK_UN
    filename = os.path.realpath(filename)
    if mode == 'r':
        lockmode = LOCK_SH
        flags = os.O_RDONLY
    elif mode == 'w' or mode == 'a':
        lockmode = LOCK_EX
        flags = os.O_RDWR | os.O_CREAT
    else:
        raise ValueError('Invalid mode')
    h5file = None
    fhandle = os.open(filename, flags)
    try:
        lockf(fhandle, lockmode)
        try:
            h5file = h5py.File(filename, mode, *args, **kw)
            with h5file:
                yield h5file
        finally:
            # at this point the lock is lost anyway on most systems as the
            # file was closed, but it doesn't hurt
            lockf(fhandle, LOCK_UN)
    finally:
        os.close(fhandle)

@contextlib.contextmanager
def locked_pytable(filename, mode='r', *args, **kw):
    assert False, "PyTables must be fixed"
    import tables
    with locked_file(filename):
        f = tables.openFile(filename, mode, *args, **kw)
        try:
            yield f
        finally:
            f.close()

## @contextlib.contextmanager
## def locked_h5file(filename, mode, *args, **kw):
##     import h5py
##     from fcntl lockf import lockf, F_LOCK, F_TLOCK, F_ULOCK, F_TEST
##     filename = os.path.realpath(filename)
##     if mode == 'r':
##         raise NotImplementedError('Not supported currently')
##     elif mode == 'w' or mode == 'a':
##         # Do NOT specify append, because we want to be at seek 0 for
##         # the lockf call
##         flags = os.O_RDWR | os.O_CREAT
##     else:
##         raise ValueError('Invalid mode')
##     h5file = None
##     fhandle = os.open(filename, flags)
##     try:
##         lockf(fhandle, F_LOCK, 0)
##         try:
##             h5file = h5py.File(filename, mode, *args, **kw)
##             yield h5file
##         finally:
##             # At this point, check that we still have a lock,
##             # just to be sure -- lockf semantics is horrible.
##             try:
##                 lockf(fhandle, F_TEST, 0)
##                 raise RuntimeError("Lock on file disappeared -- check that the file is never closed!")
##             except IOError, e:
##                 print e
##     finally:
##         try:
##             if h5file is not None:
##                 h5file.close()
##         finally:
##             try:
##                 lockf(fhandle, F_ULOCK, 0)
##             finally:
##                 os.close(fhandle)


## @contextlib.contextmanager
## def locked_h5file(filename, mode='a', *args, **kw):
##     import h5py
##     with locked_file(filename, mode):
##         f = h5py.openFile(filename, mode, *args, **kw)
##         try:
##             yield f
##         finally:
##             f.close()

def check_l_increases(*args):
    prev = 0
    for i in args:
        if not isinstance(i, int):
            raise TypeError('Please provide an integer')
        if i < prev:
            raise ValueError('Invalid l value (out of bounds, perhaps set by other l values)')
        prev = i

pycmb_debug = 'PYCMB_DEBUG' in os.environ and os.environ['PYCMB_DEBUG'] == '1'
