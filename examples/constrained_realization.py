from __future__ import division
from cmb import *


# working_directory is used for specifying what relative paths inside
# the section is relative to. $VAR (environment variable) and ~ (home
# directory) is allowed.
with working_directory('$WMAP_PATH'):

    # Use the standard isotropic model, i.e. diagonal signal covariance matrix
    model = IsotropicCmbModel(
        # The power spectrum data file
        # (Need to support more formats here? FITS? Other text files?)
        power_spectrum='wmap_lcdm_sz_lens_wmap5_cl_v3.dat',
    )

    d = CmbObservation(
        #
        # Data options
        #
        
        # Map giving both temperature and n_obs information
        data='wmap_da_forered_iqumap_r9_7yr_V1_v4.fits',
        
        # If n_obs is used, provide the sigma0 used to compute the RMS
        # (either as a float in Kelvin, or as a string with units,
        # e.g. '3.3 mK')
        sigma0=3.319e-3,

        # Beam transfer function
        # (more data formats needed?)
        beam='wmap_V1_ampl_bl_7yr_v4.txt',

        # Mask
        mask=('wmap_temperature_analysis_mask_r9_7yr_v4.fits', 1, 'TEMPERATURE'),

        # Provide an RMS map directly.
        # This overrides any n_obs information in data (and cannot be
        # given together with n_obs)
        rms=(rms_file, 1, 'TEMPERATURE'),

        # Use uniform RMS instead of a map
        ##uniform_rms=5e-5,

        # Add uniform RMS in addition to map
        ##add_noise=5e-5,

        # Explicit n_obs map
        ##n_obs=('my_nobs_map.fits', 1, 'TEMPERATURE'),

        # Explicit temperature map
        ##temperature=('my_temp.fits', 1, 'TEMPERATURE'),

        #
        # Data downgrade information -- set this to automatically downgrade
        # the data.
        #

        #Nside=32,

        # When downgrading the mask, pixel values below this value
        # will be set to 0, the ones above to 1:
        #mask_downgrade_treshold=.5

        # A random seed can be provided for the use of uniform_rms/add_noise:
        ##seed=34,
    )

sampler = ConstrainedSignalSampler(
    # The ...CmbModel instance modelling the signal
    # (provides the power spectrum)
    model=model,
        
    # A list of CmbObservation instances providing the
    # noise and data information
    observations=[d],

    # lmax for the sample; the power spectrum is taken as 0 above
    # this point
    lmax=40,
        
    # How many ls to include in the dense part of the preconditioner?
    lprecond=30,

    # How much logging is wanted to standard output? Set
    # to 0 for no output.
    verbosity=3,

    # Random seed (optional, taken from the system if not provided)
    seed=32,

    # Rather than setting a verbosity, a Python logger can
    # be provided for logging to file, over a network etc. See the logging
    # module.
    ##logger=None

    # lmin can also be provided if one has special needs
    ##lmin=2
)


signal, [map1] = sampler.sample_unmasked()
map1.to_fits('output.fits')
