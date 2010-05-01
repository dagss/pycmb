from __future__ import division
import numpy as np
from cmb.maps import (random_real_harmonic_sphere_map, simulate_pixel_sphere_maps,
                      harmonic_sphere_map)
from cmb.CG import CG
from cmb.observation import CmbObservation
from healpix import nside2npix
from cmb.mapdatautils import py_lm_to_idx_full as lm_to_idx_full
from oomatrix import *
import noise
from time import clock
import logging
from utils import *
from observation import load_temperature_pixel_window_matrix


__all__ = ['ConstrainedSignalSampler']

class ConstrainedSignalSampler(object):
    def __init__(self, observations, model, lmin=None, lmax=None,
                 lprecond=50, verbosity=None, seed=None, logger=None,
                 max_iterations=10000, eps=1e-6, norm_order=None,
                 preconditioner=None, cache=None):
        # Check inputs, defaults
        if len(observations) != 1:
            raise NotImplementedError('Realization with more than one data band is '
                                      'not yet implemented')
        for o in observations:
            if not isinstance(o, CmbObservation):
                raise TypeError('observations: Please provide a list of CmbObservation instances')
        Nside = min([o.Nside for o in observations])
        if any([o.Nside != Nside for o in observations]):
            raise NotImplementedError('Currently assumes a single Nside (for beams), trivial '
                                      'to fix though')
            
        logger = verbosity_to_logger(verbosity, logger)
        self.random_state = as_random_state(seed)

        if lprecond is not None and preconditioner is not None:
            raise ValueError('Cannot provide both lprecond and preconditioner')

        if preconditioner is None:
            preconditioner = default_preconditioner(lprecond)
        if cache is None:
            cache = {}
        if lmin is None:
            lmin = model.default_lmin(Nside)
        if lmax is None:
            lmax = model.default_lmax(Nside)

        check_l_increases(lmin, lprecond, lmax)

        # Set values
        obs = observations[0]
        N_inv_map = obs.properties.load_Ninv_map_mutable('ring')
        beam_and_window = obs.properties.load_beam_transfer_matrix(lmin, lmax)
        pixwin = load_temperature_pixel_window_matrix(Nside, lmin, lmax)
        beam_and_window = pixwin * beam_and_window
        del pixwin

        self.observations = observations
        self.lmin = lmin
        self.lmax = lmax
        self.Npix = N_inv_map.Npix
        self.N_inv_map = N_inv_map
        self.Nside = N_inv_map.Nside
        self.uniform_noise = np.all(N_inv_map == N_inv_map[0])
        self.logger = logger
        self.preconditioner = preconditioner
        # Multiply together beam and window
        self.beam_and_window = beam_and_window

        # Remove monopole and dipole from data, and
        # compute N^{-1}d in harmonic space right away
        self.scaled_Ninv_map = N_inv_map * (self.Npix / 4 / np.pi)
        d = obs.load_temperature_mutable('ring')
        d.remove_multipoles_inplace(2, obs.properties.load_mask('ring'))
        scaled_Ninv_d = self.scaled_Ninv_map * d
        self.Ninv_d = scaled_Ninv_d.to_harmonic(lmin, lmax, use_weights=False).to_real()
        


        preconditioner.set_logger(make_sublogger(logger, 'precond'))
        preconditioner.set_l_range(lmin, lmax)
        preconditioner.set_cache(cache)
        preconditioner.set_experiment_properties([o.properties for o in observations])

        if model is not None:
            self.set_model(model)

    def set_model(self, model):
        S = model.load_covariance(self.lmin, self.lmax)
        self.S = S
        self.P_L = S.cholesky()
        self.preconditioner.set_model(model)
        

    def sample_rhs(self, find_mean=False):
        """
        Computes the sample

        R^H N^(-1) d + eta0 + R^H N^(-1/2) eta1

        where R is the Cholesky-decomposed root of the covariance such that RR^H = C.

        INPUT:

         - lmin, lmax
         - N_inv - Inverse noise map in pixel space
         - N_inv_sqrt - Square root of inverse noise map in pixel space
         - S - a covariance matrix in harmonic space
         - d - observed data
        """
        Npix = self.Npix
        P, L = self.P_L

        obs = self.observations[0]
        N_inv_map = obs.properties.load_Ninv_map_mutable('ring')
#        N_inv_map.map2gif('Ninv_map_cg.gif', title='Ninv_map_cg')
#        N_inv_map.map2gif('A1.gif', title='A1')
        N_inv_map *= Npix / 4 / np.pi
#        N_inv_map.map2gif('A2.gif', title='A2')

        d = obs.load_temperature_mutable('ring')
        
        # Remove multipoles
        d.remove_multipoles_inplace(2, obs.properties.load_mask('ring'))
        
#        d = d.to_harmonic(self.lmin, self.lmax).to_pixel(self.Nside)
        
        N_inv_map *= d
#        N_inv_map.map2gif('Ninv_d_map_cg.gif', title='Ninv_d_map_cg')
        data_part = N_inv_map.to_harmonic(self.lmin, self.lmax, use_weights=False).to_real()
#        data_part.to_pixel(self.Nside).map2gif('Ninv_d_cg.gif', title='Ninv_d_cg')
        del N_inv_map
#        data_part.to_pixel(self.Nside).map2gif('C.gif', title='C')
        data_part = self.beam_and_window * data_part
#        data_part.to_pixel(self.Nside).map2gif('A_Ninv_d_cg.gif', title='A_Ninv_d_cg')
        data_part = P.H * (L.H * (P * data_part))
#        data_part.to_pixel(self.Nside).map2gif('E.gif', title='E')
        

        rhs = data_part
        del data_part
        if not find_mean:
            # Simulate white noise maps
            eta0 = random_real_harmonic_sphere_map(self.lmin, self.lmax, state=self.random_state)
            rhs += eta0

            eta1_part = simulate_pixel_sphere_maps(self.Nside, state=self.random_state)
            eta1_part *= np.sqrt(self.scaled_Ninv_map)
            eta1_part = eta1_part.to_harmonic(self.lmin, self.lmax, weights_transform=np.sqrt).to_real()
            eta1_part = P.H * (L.H * (P * (self.beam_and_window * eta1_part)))

            rhs += eta1_part

#            eta1_part.to_pixel(self.Nside).map2gif('eta1.gif', title='eta1')
#            eta0.to_pixel(self.Nside).map2gif('eta0.gif', title='eta0')

#        rhs.to_pixel(self.Nside).map2gif('rhs_cg.gif', title='rhs_cg')
#        data_part.to_pixel(self.Nside).map2gif('data.gif', title='data')

        
        return rhs
                     
    def sample_signal_details(self, rhs=None,
                              norm_order=None, max_iterations=10000,
                              eps=1e-8, raise_error=True,
                              find_mean=False):
        P, L = self.P_L

        if rhs is None:
            rhs = self.sample_rhs(find_mean=find_mean)

        cg_logger = make_sublogger(self.logger, 'cg')
        x0 = harmonic_sphere_map(0, self.lmin, self.lmax, is_complex=False)
        x, info = CG(self.cg_mul_lhs, rhs, x0=x0, precond=self.preconditioner,
                     norm_order=norm_order, maxit=max_iterations, eps=eps,
                     raise_error=raise_error, logger=cg_logger)

        # Got x; need to scale back -- x = L^-1 sdraw
#        x.to_pixel(64).map2gif('x.gif', title='x')
        sdraw = harmonic_sphere_map(P.H * (L * (P * x)), self.lmin, self.lmax, is_complex=False)
        return sdraw, info, x

    def sample_signal(self, **kw):
        signal, info, x = self.sample_signal_details(**kw)
        return signal

    def find_mean(self, **kw):
        mean, info, x = self.sample_signal_details(find_mean=True, **kw)
        return mean

    def sample_unmasked(self, order='nested', **kw):
        signal = self.sample_signal(**kw)
        unmasked = []
        for obs in self.observations:
            unmasked.append(
                obs.unmask_temperature(signal, order=order,
                                       seed=self.random_state)
            )
        return signal, unmasked

    def cg_mul_lhs(self, x):
        return mul_lhs(self.lmin, self.lmax, self.S,
                       self.scaled_Ninv_map, self.beam_and_window, x)

class BasePreconditioner(object):
    """
    The methods should be called in the order below. Calling
    e.g. set_S can happen multiple times for each set_N_inv_map,
    but after calling set_experiment_properties, set_signal_covariance must be called (and
    so on for the rest of the hierarchy).

     1. set_l_range, set_cache
     2. set_instrument_and_noise
     3. set_signal_covariance

    NOTE that the cache provided may be a shelf -- it is not required
    to commit data except for in explicit assignment. When changing a value (like
    a list), please re-put it.
    """
    def __init__(self, name):
        self.name = name
        
    def set_logger(self, logger):
        self.logger = make_sublogger(logger, self.name)
    
    def set_l_range(self, lmin, lmax):
        self.lmin = lmin
        self.lmax = lmax

    def set_cache(self, cache):
        self.cache = cache

    def set_experiment_properties(self, instruments_and_masks):
        raise NotImplementedError()
#        self.instruments_and_masks = bands

    def set_model(self, S, covar_cache):
        raise NotImplementedError()
#        self.S = S

    def clear(self):
        pass
        # Let go of all memory
#        self.S = self.Ninv_map = self.beam = None

def lookup_lrange_cache(cache, lmin, lmax):
    """
    Looks up a (possible slice of) a cached matrix in a list of the form
    [ (lmin, lmax, M), (lmin, lmax, M), ... ]. M is assumed to be stored
    as a covar-like matrix in real form.

    Returns None if not found.
    """
    for it_lmin, it_lmax, M in cache:
        if it_lmin <= lmin and it_lmax >= lmax:
            start = lm_to_idx_full(lmin, -lmin, it_lmin)
            stop = lm_to_idx_full(lmax, lmax, it_lmin) + 1
            return M[start:stop, start:stop]
    return None
        
class FullPreconditioner(BasePreconditioner):
    def clear(self):
        self.A = self.beam_Ninv_beam = None

    def set_experiment_properties(self, instruments_and_masks):
        lmin, lmax = self.lmin, self.lmax
        assert len(instruments_and_masks) == 1
        instrument, = instruments_and_masks

        # Search cache for existing dense preconditioner over given l-range
        subcache = self.cache.get((instrument, 'harmonic_Ninv_full_with_beam'), None)
        if subcache is not None:
            # We store the cache as a list of (lmin, lmax, matrix)
            # Search the cache for something covering the range which we can
            # slice.
            beam_Ninv_beam = lookup_lrange_cache(subcache, lmin, lmax)
        else:
            subcache = []
            beam_Ninv_beam = None
            
        # Compute if it is not already computed
        if beam_Ninv_beam is None:
            self.logger.info('Computing Ninv_full for l=%d..%d'
                             % (lmin, lmax))
            t0 = get_times()
            import noise
            Ninv_map = instrument.load_Ninv_map('ring')
            Ninv_full = as_matrix(
                noise.Ninv_to_harmonic_real_block(lmin, lmax, Ninv_map),
                copy=False)
            beam = instrument.load_beam_transfer_matrix(self.lmin, self.lmax)
            beam_Ninv_beam = Ninv_full.sandwich(beam)
            subcache.append((lmin, lmax, Ninv_full))
            self.cache[instrument, 'harmonic_Ninv_full_with_beam'] = subcache
            log_times(self.logger, t0)
        else:
            self.logger.info('Ninv_full for l=%d..%d found in cache'
                             % (lmin, lmax))
            
        # Multiply with beam
        self.beam_Ninv_beam = beam_Ninv_beam

    def set_model(self, model):
        S = model.load_covariance(self.lmin, self.lmax)
        # We now form the full matrix and Cholesky decompose it
        P, L = S.cholesky()
        A = self.cache.get((model, 'A_full'), None)
        if A is None:
            self.logger.info('Dense Cholesky decomposition started (l=%d..%d)'
                             % (self.lmin, self.lmax))
            t0 = get_times()
            A = self.beam_Ninv_beam.sandwich(P.H, L, P)
            A = DenseMatrix(A._data + np.eye(A.shape[0]))
            A.cholesky(check=False) # for profiling purposes, put this here
            if 1 == 0:
                # DO NOT cache, because there's a bug in oomatrix.py regarding this
                self.cache[model, 'A_full'] = A
            log_times(self.logger, t0)
        self.A = A
                                                                        
    def __call__(self, B):
        return harmonic_sphere_map(
            self.A.solve_right(B, algorithm='cholesky', check=False),
            self.lmin, self.lmax, is_complex=False)

class DiagonalPreconditioner(BasePreconditioner):
    def clear(self):
        self.Ninv_diagonal = self.Ainv = self.beam_diagonal_square = None

    def set_experiment_properties(self, instruments_and_masks):
        lmin, lmax = self.lmin, self.lmax
        assert len(instruments_and_masks) == 1
        instrument, = instruments_and_masks

        # Search cache for existing diagonal preconditioner over given l-range
        subcache = self.cache.get((instrument, 'Ninv_diagonal_with_beam'), None)
        if subcache is not None:
            # We store the cache as a list of (lmin, lmax, matrix)
            # Search the cache for something covering the range which we can
            # slice.
            beam_Ninv_beam = lookup_lrange_cache(subcache, lmin, lmax)
        else:
            subcache = []
            beam_Ninv_beam = None
            
        # Compute if it is not already computed
        if beam_Ninv_beam is None:
            self.logger.info('Need to compute Ninv_diagonal for l=%d..%d'
                             % (lmin, lmax))
            t0 = get_times()
            Ninv_map = instrument.load_Ninv_map('ring')
            Ninv_diagonal = as_matrix(
                noise.Ninv_to_harmonic_real_diagonal(lmin, lmax, Ninv_map),
                copy=False)
            beam = instrument.load_beam_transfer_matrix(self.lmin, self.lmax)
            beam_Ninv_beam = Ninv_diagonal.sandwich(beam)            
            subcache.append((lmin, lmax, beam_Ninv_beam))
            log_times(self.logger, t0)
            self.cache[instrument, 'Ninv_diagonal_with_beam'] = subcache
        else:
            self.logger.info('Found Ninv_diagonal for l=%d..%d in cache' %
                             (lmin, lmax))
        self.beam_Ninv_beam = beam_Ninv_beam

    def set_model(self, model):
        # All matrices should be diagonal, so order is unimportant.
        # Also, don't bother caching.
        S = model.load_covariance(self.lmin, self.lmax)
        D = self.beam_Ninv_beam * S.diagonal_as_matrix()
        assert isinstance(D, DiagonalMatrix)
        self.Ainv = diagonal_matrix(1 / (D._data + 1))
                                                                        
    def __call__(self, B):
        return B
        return self.Ainv * B

class SplitPreconditioner(BasePreconditioner):
    """
    A block-diagonal preconditioner with splits for ls.
    
    E.g.::
    
        SplitPreconditioner([
            (None, FullPreconditioner()),
            (60, DiagonalPreconditioner())])
    
    """
    def __init__(self, children, name=None):
        # Input is rather complicated, so do some simple input validation
        if len(children) < 2:
            raise ValueError()
        for item in children:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError()
        if children[0][0] is not None:
            raise ValueError()
        for l, p in children[1:]: int(l)
        # Assign.
        # self.lstart and self.split_indices will contain one more item than children
        # containing appropriate end values
        self.lstarts = [x[0] for x in children] + [None]
        self.children = [x[1] for x in children]
        self.name = name

    def set_logger(self, logger):
        self.logger = make_sublogger(logger, self.name)
        for child in self.children:
            child.set_logger(logger)

    def set_l_range(self, lmin, lmax):
        self.lmin = lmin
        self.lmax = lmax
        # Update lmin, lmax
        self.lstarts[0] = lmin
        self.lstarts[-1] = lmax + 1
        
        # Now that lmin is given, it is possible to convert
        # the lstarts given in the constructors to splitting indices.
        # Each index indicates the start point of that preconditioner.
        self.split_indices = [lm_to_idx_full(l, -l, lmin)
                              for l in self.lstarts]
        self.child_start_stop = []
        # Make sure to drop any unused children
        idx_stop = lm_to_idx_full(lmax, lmax, lmin) + 1
        for idx, child in enumerate(self.children):
            if self.lstarts[idx] > lmax:
                break
            lstop = min(self.lstarts[idx + 1] - 1, lmax)
            child.set_l_range(self.lstarts[idx], lstop)
            self.child_start_stop.append(
                (child, self.split_indices[idx],
                 min(self.split_indices[idx + 1], idx_stop)))

    def set_cache(self, cache):
        for child, start, stop in self.child_start_stop:
            child.set_cache(cache)

    def set_experiment_properties(self, arg):
        for child, start, stop in self.child_start_stop:
            child.set_experiment_properties(arg)

    def set_model(self, model):
        for child, start, stop in self.child_start_stop:
            child.set_model(model)

    def __call__(self, B):
        X_list = []
        for child, start, stop in self.child_start_stop:
            sub_X = child(B[start:stop])
            X_list.append(sub_X)
        X = harmonic_sphere_map(np.hstack(X_list), self.lmin, self.lmax,
                                is_complex=False)
        return X

    def clear(self):
        for child in self.children:
            child.clear()

class RecursiveCGPreconditioner:
    """
    As a convenience, if lsplit is given, the child preconditioner
    given will be used as the dense block in a block preconditioner
    with a diagonal preconditioner above it.
    """
    def __init__(self, name, Nside_pre, eps, child, max_iterations=100000, lsplit=None):
        self.name = name
        self.Nside_pre = Nside_pre
        self.eps = eps
        self.max_iterations = max_iterations
        if lsplit is not None:
            child = SplitPreconditioner([
                (None, child),
                (lsplit, DiagonalPreconditioner('%s.diagonal' % name))])
        self.child = child

    def set_logger(self, logger):
        self.logger = logger
        self.cg_logger = make_sublogger(logger, '%s.cg' % self.name)
        self.child.set_logger(logger)

    def set_l_range(self, lmin, lmax):
        self.lmin = lmin
        self.lmax = lmax
        self.child.set_l_range(lmin, lmax)

    def set_experiment_properties(self, Ninv_map, beam, cache):
        self.beam_and_window = beam
        # Store a scaled map for multiplication
        self.scaled_Ninv_map = Ninv_map * (Ninv_map.Npix / 4. / np.pi)
        # Downgrade sub-preconditioner map if necesarry
        if Ninv_map.Nside > self.Nside_pre:
            Ninv_map = Ninv_map.change_resolution(self.Nside_pre)
        # Feed (possibly downgraded) map to child preconditioner
        self.child.set_experiment_properties(Ninv_map, beam, cache)

    def set_signal_covariance(self, S, cache):
        self.S = S
        S.cholesky(check=False)
        self.child.set_signal_covariance(S, cache)

    def cg_mul_lhs(self, x):
        return mul_lhs(self.lmin, self.lmax, self.S,
                       self.scaled_Ninv_map, self.beam_and_window, x)


    def __call__(self, B):
        x0 = harmonic_sphere_map(0, self.lmin, self.lmax, is_complex=False)
        x, info = CG(self.cg_mul_lhs, B, x0=x0, precond=self.child,
                     norm_order=None, maxit=self.max_iterations,
                     eps=self.eps,
                     raise_error=True,
                     logger=self.cg_logger)
        return x

    def clear(self):
        self.S = self.beam_and_window = self.scaled_Ninv_map = None
        self.child.clear()


class Deprecated_SignalToNoisePreconditioner:

    def __init__(self, lmin, lmax, inv_rms_map, S, beam, cutoff=5):
        raise NotImplementedError("Assumes S is diagonal")
        cutoff_low, cutoff_high = 200, 300
#        beam, S = S, beam
        # Create RMS-map, but avoid creating infinity
        inv_rms_map = inv_rms_map.copy()
        min_positive = np.min(inv_rms_map[inv_rms_map != 0])
        inv_rms_map[inv_rms_map == 0] = 2# * min_positive
        rms_map = 1 / inv_rms_map

        N_inv = inv_rms_map ** 2

        import noise
        N_inv_dia = noise.inverse_pixel_noise_to_harmonic_real_diagonal(
            lmin, lmax, N_inv)
        self.signal_to_noise = (N_inv_dia * S.diagonal_as_matrix() * beam * beam).diagonal()
        print 'b'
        self.beam_inv = beam.inverse()
        self.diagonal_preconditioner = diagonal_matrix(1 / (self.signal_to_noise + 1))
        print 'c'

        factor = (self.signal_to_noise - cutoff_low) / (cutoff_high - cutoff_low)
        factor[factor > 1] = 1
        factor[factor < 0] = 0
        self.dense_contribution_factor = factor
        
#        self.low_signal_to_noise_subset = self.signal_to_noise < cutoff
#        self.high_signal_to_noise_subset = ~self.low_signal_to_noise_subset
        print 'd'

        self.Nside = inv_rms_map.Nside
        self.Npix = inv_rms_map.Npix
        self.S = S
        self.beam = beam
        self.P_L = S.cholesky()
        self.lmin = lmin
        self.lmax = lmax
        self.scaled_N = rms_map**2 * 4 * np.pi / self.Npix

    def __call__(self, x):
        P, L = self.P_L

        # Diagonal preconditioner
        x_low = x
        x_low = self.diagonal_preconditioner * x_low

        # High signal-to-noise approximation preconditioner
        x_high = x
        x_high = self.beam_inv * (P.H * L.H.solve_right(P * x_high))
        x_high = x_high.to_complex().to_pixel(self.Nside)
        x_high *= self.scaled_N
        x_high = x_high.to_harmonic(self.lmin, self.lmax).to_real()
        x_high = P.H * L.solve_right(P * (self.beam_inv * x_high))

        # Merge the two
        x_tot = (x_low * (1 - self.dense_contribution_factor) +
                 x_high * self.dense_contribution_factor)
        
        return x_tot

def debug(*args):
    import logging
    import traceback
    tb = traceback.extract_stack(limit=2)
    call_line = "%s:%d (%s)" % tb[0][0:3]
    logging.debug("%s\n%r\n" % (call_line, args))

def make_sublogger(logger, name):
    if logger is logging.getLogger():
        return logging.getLogger(name)
    else:
        r = logging.getLogger('%s.%s' % (logger.name, name))
        return r

def mul_lhs(lmin, lmax, S, scaled_Ninv_map, beam_and_window, x):
    P, L = S.cholesky(); PH = P.H
    x_orig = x.copy()
    x = beam_and_window * (PH * (L * (P * x)))
    xpix = x.to_complex().to_pixel(scaled_Ninv_map.Nside)
    xpix *= scaled_Ninv_map
    x = xpix.to_harmonic(lmin, lmax, use_weights=False).to_real()
    x = PH * (L.H * (P * (beam_and_window * x)))
    x += x_orig
    return x

def default_preconditioner(lprecond):
    return SplitPreconditioner([
        (None, FullPreconditioner('dense_part')),
        (lprecond + 1, DiagonalPreconditioner('diagonal_part'))])
    
