from __future__ import division

import numpy as np
from isotropic import ClArray
from maps import sphere_maps_from_fits, _PixelSphereMap, pixel_sphere_map
import os
from weakref import WeakValueDictionary
from utils import *

__all__ = ['CmbObservation', 'CmbObservationProperties', 'map_from_fits',
           'load_temperature_pixel_window_matrix',
           'make_cosine_beam', 'beam_to_fits']

class FormatError(Exception):
    pass
 
#l_Cl_dtype = [('l', np.int), ('Cl', np.double)]

def fits_resolve_field(extension_obj, field, desc=None):
    """
    Converts the given field into a numerical value (0-based)
    if it is a string. Raises ValueError if it is out of range.

    extension_obj should be the pyfits extension object.
    """
    if isinstance(field, int):
        if field >= 0 and field < len(extension_obj.columns):
            return field
        # fall through to exception
    else:
        for idx, col in enumerate(extension_obj.columns):
            if field == col.name:
                return idx
    if desc is None:
        desc = field
    raise ValueError('Cannot find FITS field: %s' % repr(desc))

def probe_fits(desc, wanted_units, default_ext=None, default_field=None):
    """
    Check the existance of the FITS field *without* loading the data.
    Returns Nside parameter and desc (which may have defaults inserted)
    """
    if isinstance(desc, _PixelSphereMap):
        return desc.Nside, desc
    
    filename, extno, field = desc
    import pyfits
    hdulist = pyfits.open(filename)

    if extno is None:
        if len(hdulist) == 2:
            extno = 1
        elif default_ext is not None:
            # The context determines an obvious extension to use
            extno = default_ext
            desc = (filename, extno, field)
        else:
            raise ValueError('Must provide FITS extension number: %s' % repr(desc))
    ext = hdulist[extno]

    # Probe for presence of field and resolve it to an integer
    nfields = ext.header['TFIELDS']
    if field is None:
        if nfields == 1:
            field = 1
        elif default_field is not None:
            field = default_field
            desc = (filename, extno, field)
        else:
            raise ValueError('Must provide FITS field: %s' % repr(desc))
    if isinstance(field, int):
        if field > nfields:
            raise ValueError('Cannot find FITS field: %s' % repr(desc))
        if field < 0:
            raise ValueError('Field number must be positive: %s' % repr(desc))
    else:
        found = False
        for i in range(1, nfields+1):
            if ext.header['TTYPE%d' % i] == field:
                field = i - 1
                found = True
                break
        if not found:
            raise ValueError('Cannot find FITS field: %s' % repr(desc))

    units = ext.header['TUNIT%d' % (field + 1)].split(',')[0]
    if wanted_units == 'temperature':
        if units != 'mK':
            raise ValueError('Temperature units expected but %s found: %s' % (units, repr(desc)))
    elif wanted_units == 'counts':
        if units != 'counts':
            raise ValueError('Count units expected but %s found: %s' % (units, repr(desc)))

    return ext.header['NSIDE'], desc
 

def beam_from_fits(f, dtype=np.double):
     msg = 'Unexpected FITS data for beam'
     import pyfits
     if isinstance(f, str):
         f = pyfits.open(f)
     if len(f) != 2:
         raise ValueError(msg)
     data = f[1].data.field(0).astype(dtype)
     return ClArray(data, 0, data.shape[0]-1)

def as_beam(beam):
    if isinstance(beam, ClArray):
        return beam
    else:
        try:
            # Attempt load from file
            return beam_from_fits(beam)
        except IOError:
            pass # fall through to text file attempt
        
        # Load from text file, (l, beam)
        data = np.loadtxt(beam, dtype=np.dtype([('l', np.int), ('beam', np.double)]))
        ls = data['l']
        lmin, lmax = ls[0], ls[-1]
        if np.any(ls != np.arange(lmin, lmax + 1)):
            raise NotImplementedError('l\'s not stored consecutively in beam file')
        beam = data['beam']
        return ClArray(beam.copy(), lmin, lmax)

def beam_to_fits(filename, beam):
    import pyfits
    beamdata = beam.by_l[0:beam.lmax+1].view(np.ndarray)
    cols = pyfits.ColDefs([
        pyfits.Column(name='beam', format='E', array=beamdata)])
    tab = pyfits.new_table(cols)
    hdulist = pyfits.HDUList([pyfits.PrimaryHDU(), tab])
    hdulist.writeto(filename)

def as_kelvin(sigma):
    msg = 'Must be float (kelvin) or a string on the form "2.3 mK"'
    if isinstance(sigma, float):
        return sigma
    elif isinstance(sigma, str):
        # Parse units -- only mK and K currently
        x = sigma.split()
        if len(x) != 2:
            raise ValueError(msg)
        try:
            value = float(x[0])
        except ValueError:
            raise ValueError(msg)
        if x[1] == 'mK':
            return value * 1e-3
        elif x[1] == 'K':
            return value
        else:
            raise ValueError(msg)
    else:
        raise TypeError(msg)


def map_from_fits(desc, type, dtype=np.double):
    filename, extno, field = desc
    if extno is None or field is None:
        raise ValueError("extno and field must be provided")
    if isinstance(field, int) and field < 0:
        raise ValueError("field must be >= 0")
    import pyfits
    hdulist = pyfits.open(filename)
    try:
        ext = hdulist[extno]
        field = fits_resolve_field(ext, field, desc)
        mapdata = ext.data.field(field).ravel().astype(dtype)
        units = ext.columns[field].unit.split(',')[0]

        if type == 'temperature':
            if units == 'mK':
                mapdata *= 1e-3 # convert to Kelvin
            elif units == 'uK':
                mapdata *= 1e-6
            else:
                raise FormatError('Do not recognize temperature units: %s' % (units))
        elif type == 'count':
            if units != 'counts':
                raise FormatError('Do not recognize count units: %s' % (units))
        elif type in ('mask', 'raw'):
            pass
        else:
            raise ValueError('Wrong map type: %s' % type)
        
        map = pixel_sphere_map(mapdata,
                               pixel_order=ext.header['ORDERING'].lower() # nested or ring
                               )
    finally:
        hdulist.close()
    return map    

def as_single_map(desc, dtype=np.double,
                  type=None, copy=True):
    assert type is not None
    if isinstance(desc, _PixelSphereMap):
        if desc.ndim != 1:
            raise ValueError('Too many maps provided')
        if copy:
            desc = desc.copy()
        return desc

    return map_from_fits(desc, type=type)

def ensure_realpath(*args):
    for arg in args:
        if arg is None:
            yield arg
        elif isinstance(arg, str):
            yield os.path.realpath(arg)
        elif isinstance(arg, tuple):
            yield (os.path.realpath(arg[0]),) + arg[1:]
        else:
            yield arg

def expand_fits_desc(desc):
    # Plain maps are passed through. Otherwise,
    # turn a parameter into a 3-tuple FITS descriptor,
    # with filler None values. Also expand the file name to the real, full path.
    msg = 'Resource pointer can not be parsed: %s' % repr(desc)
    if isinstance(desc, _PixelSphereMap):
        return desc
    if isinstance(desc, str):
        desc = (desc,)
    if isinstance(desc, tuple):
        extno = field = None
        if len(desc) == 1:
            filename, = desc
        elif len(desc) == 2:
            filename, extno = desc
        elif len(desc) == 3:
            filename, extno, field = desc
        else:
            raise ValueError(msg)

        filename = os.path.realpath(filename)
        return (filename, extno, field)
    elif desc is None:
        return None
    else:
        raise ValueError(msg)

def load_temperature_pixel_window_matrix(Nside, lmin, lmax,
                                         healpix_resource=None):
    if healpix_resource is None:
        import healpix
        healpix_resource = healpix.resources.get_default()
    pixwin_temp, pixwin_pol = healpix_resource.pixel_window(Nside)
    x = ClArray(pixwin_temp, 0, pixwin_temp.shape[0]-1)
    x = x.by_l[lmin:lmax+1].as_matrix()
    return x

class CmbObservationProperties(object):
    """

     - Ninv_map - Uncorrelated
    
    """
    
    def __init__(self, data=None,
                 beam=None, mask=None, sigma0=None, n_obs=None, rms=None,
                 Nside=None, mask_downgrade_treshold=.5,
                 add_noise=None, seed=None,
                 uniform_rms=None):
        
        #
        # Ensure that all resources are either maps, 3-tuple FITS descs or
        # None. Also check that mandatory resources are provided.
        #
        if data is not None:
            if not isinstance(data, str):
                raise ValueError('"data" argument must be a str')
            data = os.path.realpath(data)
        
        mask, n_obs, rms = (expand_fits_desc(x) for x in
                                         (mask, n_obs, rms))

        num_noise_maps = (n_obs is not None) + (rms is not None) + (uniform_rms is not None)
        if  num_noise_maps > 1:
            raise ValueError('Must provide only one of n_obs, rms, uniform_rms')

        if num_noise_maps == 0:
            # No explicit noise maps, try N_OBS in the data resource
            if data is None:
                raise ValueError('No noise information provided')
            n_obs = (data, 1, 'N_OBS')

        if uniform_rms is not None:
            if add_noise is not None:
                raise ValueError('Uniform noise (constant rms) and add_noise cannot '
                                 'be specified together')


        #
        # Check that reality matches the specs -- check FITS files.
        # Also probe/validate Nside
        #
        Nsides = []

        if rms is not None:
            Nsides.append(probe_fits(rms, 'temperature')[0])
        
        if n_obs is not None:
            Nside_tmp, n_obs = probe_fits(n_obs, 'counts', 1, 'N_OBS')
            Nsides.append(Nside_tmp)
        
        if mask is not None:
            Nsides.append(probe_fits(mask, 'mask')[0])

        # No rescaling specified, probe
        if len(Nsides) == 0:
            if Nside is None:
                raise ValueError()
            Nsides = [Nside]
        for Nside_maps in Nsides[1:]:
            if Nside_maps != Nsides[0]:
                raise ValueError('Resolution of maps does not agree')
        if Nside is None:
            Nside = Nsides[0]
        Nside_maps = Nsides[0]
        self._rms_regrade_factor = np.sqrt(Nside / Nside_maps)
        
        #
        # More trivial resources...
        #
        if n_obs is not None:
            if sigma0 is None:
                raise ValueError('Must provide sigma0, when n_obs is provided')
            self._sigma0 = as_kelvin(sigma0)
        else:
            # silently ignore provided sigma0
            self._sigma0 = None
            
        if add_noise is not None:
            add_noise = as_kelvin(add_noise)

        if uniform_rms is not None:
            uniform_rms = as_kelvin(uniform_rms)

        if beam is not None:
            beam = as_beam(beam)

        self._cache = cache = WeakValueDictionary()
        self._mask_downgrade_treshold = mask_downgrade_treshold
        self.Nside = Nside
        self._n_obs = n_obs
        self._rms = rms
        self._uniform_rms = uniform_rms
        self._beam = beam
        self._mask = mask
        self._add_noise = add_noise
        self._random_state = as_random_state(seed)

        # Populate cache with any maps provided by value so that copies aren't made
        if rms is not None and isinstance(rms, _PixelSphereMap):
            cache['rms_without_added_noise', rms.pixel_order, rms.dtype] = rms
        if mask is not None and isinstance(mask, _PixelSphereMap):
            cache['mask', mask.pixel_order, mask.dtype] = mask
            

    # Mutable resource loaders
    def load_rms_mutable(self, order='nested', dtype=np.double, include_added_noise=True):
        if self._uniform_rms is not None:
            rms = pixel_sphere_map(self._uniform_rms, Nside=self.Nside, pixel_order=order)
            return rms # We're done -- add_noise disallowed for uniform noise
        elif self._n_obs is not None:
            n_obs = as_single_map(self._n_obs, dtype=dtype, type='count', copy=False)
            rms = self._sigma0  * self._rms_regrade_factor / np.sqrt(n_obs)
        elif self._rms is not None:
            rms = as_single_map(self._rms * self._rms_regrade_factor,
                                dtype=dtype, type='temperature', copy=True)
        else:
            assert False


        rms = rms.reformat_and_discard(order, self.Nside, _strict=True)
        if self._add_noise is not None and include_added_noise:
            rms += self._add_noise
        return rms

    def load_Ninv_map_mutable(self, order='nested', dtype=np.double):
        # Attempt to only do one order change rather than two,
        # and use cached mask if available
        x = self.load_rms_mutable(order, dtype)
        mask = self.load_mask(order)
        x **= -2
        x *= mask
        return x

    def load_mask_mutable(self, order, dtype=np.double):
        assert dtype == np.double
        if self._mask is not None:
            x = as_single_map(self._mask, dtype=dtype, type='mask')
            downgrading = self.Nside < x.Nside
            x = x.reformat_and_discard(order, self.Nside, _strict=True)
            if downgrading:
                x = pixel_sphere_map(
                    (x > self._mask_downgrade_treshold).view(np.ndarray),
                    pixel_order=order)
        else:
            x = pixel_sphere_map(1, Nside=self.Nside, pixel_order=order)
        return x

    # Immutable resource loaders. These simply share a single
    # mutable resource through a WeakValueDictionary
    def _load(self, name, loader, order, dtype, *args):
        key = (name, order, dtype)
        x = self._cache.get(key, None)
        if x is None:
            map = loader(order, dtype, *args)
            x = readonly_array(map)
            x = pixel_sphere_map(x, pixel_order=map.pixel_order, Nside=map.Nside)
            self._cache[key] = x
        return x
    
    def load_temperature(self, order='nested', dtype=np.double):
        return self._load('temperature', self.load_temperature_mutable, order, dtype)

    def load_rms(self, order, dtype=np.double, include_added_noise=True):
        if include_added_noise and self._add_noise:
            return self._load('rms_with_added_noise', self.load_rms_mutable, order, dtype, True)
        else:
            return self._load('rms_without_added_noise',
                              self.load_rms_mutable, order, dtype, False)

    def load_Ninv_map(self, order='nested', dtype=np.double):
        return self._load('Ninv_map', self.load_Ninv_map_mutable, order, dtype)

    def load_mask(self, order='nested', dtype=np.double):
        return self._load('mask', self.load_mask_mutable, order, dtype)

    def load_beam_transfer(self, lmin=None, lmax=None):
        if self._beam is None:
            raise ValueError('This resource does not contain a beam')
        if lmin is None and lmax is None:
            return self._beam
        else:
            return self._beam.by_l[lmin:lmax + 1]

    def load_beam_transfer_matrix(self, lmin, lmax):
        return self.load_beam_transfer(lmin, lmax).as_matrix()

        
class CmbObservation(object):
    def __init__(self,
                 properties=None,
                 data=None,
                 temperature=None,
                 Nside=None,
                 **properties_args):
        if properties is not None:
            if len(properties_args) > 0:
                raise TypeError('Unexpected arguments: %s' % repr(properties_args.keys()))
        else:
            properties = CmbObservationProperties(data=data, Nside=Nside,
                                                  **properties_args)
        
        if temperature is None and data is not None:
            # Try data resource.
            if not isinstance(data, str):
                raise ValueError('"data" argument must be a str')
            data = os.path.realpath(data)
            temperature = (data, 1, 'TEMPERATURE')

        if temperature is None:
            raise ValueError('temperature data must be provided')

        temperature = expand_fits_desc(temperature)
        Nside_temp, temperature = probe_fits(temperature, 'temperature', 1, 'TEMPERATURE')
        if Nside is None and Nside_temp != properties.Nside:
            # no explicit Nside, must be obvious an choice
            raise ValueError('Resolutions of maps does not agree, and Nside not provided')
        
        self.Nside = properties.Nside
        self._temperature = temperature
        self.properties = properties
        self._cache = cache = WeakValueDictionary()

        if isinstance(temperature, _PixelSphereMap):
            cache['temperature', temperature.pixel_order, temperature.dtype] = temperature

    def persist(self, filename, order='nested', format='no_noise'):
        if format != 'no_noise':
            raise NotImplementedError()
        self.load_temperature(order).to_fits(filename)

    #
    # Utilities
    #
    def unmask_temperature(self, signal, order='nested', seed=None):
        """
        Given the harmonic sphere map ``signal`` as the underlying signal,
        provide a map where the mask has been removed and replaced with the
        contents of signal. Noise consistent with the noise properties
        of the observation (without the mask) will be added.
        """
        Nside, lmin, lmax = self.Nside, signal.lmin, signal.lmax
        
        random_state = as_random_state(seed)
        temperature = self.load_temperature_mutable(order)
        inverse_mask = (self.properties.load_mask_mutable(order) == 1).view(np.ndarray)
        np.negative(inverse_mask, inverse_mask) # invert the mask in-place

        # First, smooth the signal with the beam and pixel window
        smoothed_signal = self.properties.load_beam_transfer_matrix(lmin, lmax) * signal
        pixwin = load_temperature_pixel_window_matrix(Nside, lmin, lmax)
        smoothed_signal = pixwin * smoothed_signal

        # Create map from signal, and replace unmasked values in temperature map
        signal_map = smoothed_signal.to_pixel(self.Nside)
        signal_map.change_order_inplace(order)
        temperature[inverse_mask] = signal_map[inverse_mask]

        # Finally, add RMS to unmasked area
        rms_in_mask = self.properties.load_rms(order)[inverse_mask]
        temperature[inverse_mask] += random_state.normal(scale=rms_in_mask)
        return temperature
    
    def load_temperature_mutable(self, order='nested', dtype=np.double):
        x = as_single_map(self._temperature, dtype=dtype, type='temperature', copy=True)
        x = x.reformat_and_discard(order, self.Nside, _strict=True)
        if self.properties._add_noise is not None:
            x += self._random_state.normal(scale=self.properties._add_noise, size=x.shape)
        x *= self.properties.load_mask(order)
        return x

    def load_temperature(self, order='nested', dtype=np.double):
        key = ('temperature', order, dtype)
        x = self._cache.get(key, None)
        if x is None:
            map = self.load_temperature_mutable(order, dtype)
            x = readonly_array(map)
            x = pixel_sphere_map(x, pixel_order=map.pixel_order, Nside=map.Nside)
            self._cache[key] = x
        return x


class _ReadonlyArray(object):
    def __init__(self, arr):
        self.__array_interface__ = dict(
            shape=arr.shape,
            typestr=arr.dtype.str,
            data=(arr.ctypes.data, True), # read-only flag on
            strides=arr.strides,
            version=3
            )

def readonly_array(arr):
    return np.asarray(_ReadonlyArray(arr))


def make_cosine_beam(lmin, lbeamstart, lbeamstop, lmax=None):
    if lmax is None:
        lmax = lbeamstop
    beam = np.ones(lmax + 1 - lmin, dtype=np.double)
    l = np.arange(lbeamstart, lbeamstop + 1)
    beam[l - lmin] = (1 + np.cos(np.pi * (l - lbeamstart) / (lbeamstop - lbeamstart))) / 2
    beam[lbeamstop:lmax+1] = 0
    return ClArray(beam, lmin, lmax)
