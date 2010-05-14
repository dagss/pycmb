#!/usr/bin/env python
from __future__ import division

##############################################################################
#    Copyright (C) 2010 Dag Sverre Seljebotn <dagss@student.matnat.uio.no>
#  Distributed under the terms of the GNU General Public License (GPL),
#  either version 2 of the License, or (at your option) any later version.
#  The full text of the GPL is available at:
#                  http://www.gnu.org/licenses/
##############################################################################

__doc__ = """\
cmbcr - Constrained Realization of Cosmic Microwave Background maps

  Makes a constrained simulated CMB map given an input map, a noise
  map, an assumed power spectrum, a beam, and a mask. The effect is to
  effectively "unmask" the input map -- provide a sample of what
  is under the mask, given the power spectrum and noise properties.

  The signal sample used to replace the information in the masked area
  is taken from the posterior distribution of the signal conditional
  on the input listed above, assuming normality, and using a uniform
  prior. Only temperature maps with uncorrelated noise is supported.

  TODO: Provide paper reference.

author and license:

  Copyright (C) 2010 Dag Sverre Seljebotn

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version. See <http://www.gnu.org/licenses/>.

examples:

  Make a single sample, using lmax=Nside^3-1:

    cmbcr --rms=rms.fits beam.dat mask.fits indata.fits

  The result is saved to crsample-0-0.fits (no overwriting is done
  by default).

  TODO: Write about file formats and other noise formats,
  give a more complicated example.

acknowledgement:

  This software would have been impossible without the help of Hans
  Kristian Eriksen and the example of his CMB Gibbs sampler software
  Commander.
"""
  
## FILE FORMATS:

## The input and output files are expected as FITS files, in the form
## "filename:extno:field" (0-based extension number, 1 is the
## default). If no extension number is supplied, 1 is assumed. The field
## can be provided either by index (0-based!) or by name. If no field
## number is specified, the map is assumed to contain either a single
## map, or contain both a 'TEMPERATURE' and 'N_OBS' field.

##"""

import argparse
import numpy as np
import sys
import os
import logging

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

# Optional args + noise choices
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed (integer value). It will be combined with the '
                    'process ID so that parallel jobs don''t produce the same samples.')
parser.add_argument('-l', '--lmax', type=int, default=None,
                    help='Maximum l value considered when sampling the underlying signal.')
parser.add_argument('--rms', type=str, default=None,
                    help='Filename (or FITS-pointer, see above) of a map specifying the noise '
                    'as RMS.')
parser.add_argument('--nobs', type=str, default=None,
                    help='Filename (or FITS-pointer, see above) of a map specifying the noise '
                    'as number of observations. When specifying this, --sigma must also be '
                    'given.')
parser.add_argument('--sigma', type=str, default=None,
                    help='sigma0-value to use with --nobs to produce RMS. Valid forms: '
                    '0.002, 2e-3, 0.002K, 2mK, 2000uK (default unit is Kelvin).')
parser.add_argument('-n', '--repeat', type=int, default=1,
                    help='Number of samples to produce (per process!).')
parser.add_argument('--lprecond', type=int, default=None,
                    help='Tune the size of the preconditioner used in the CG iterations. '
                    'Larger value increase memory use and startup overhead but *may* '
                    'decrease number of iterations needed for convergence. '
                    'Default: max(min(lmax/2, 60), 30).')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='Acceptable relative error of solution (default: 1e-6).')
parser.add_argument('-f', '--force', type=bool, default=False,
                    help='Overwrite existing output files rather than skipping.')
parser.add_argument('--quiet', action='store_true', default=False,
                    help='Only output error messages (same as --verbose=0)')
parser.add_argument('-v', '--verbose', metavar='VERBOSITY', default=1, type=int,
                    help='Set verbosity level (0-4, default 1)')
parser.add_argument('--procid', type=str, default=None,
                    help='ID of process for log message and ouput file (default: '
                    'Try MPI process number, then $SLURM_PROCID, then use 0). '
                    'Environment variables accepted, see -o.')
parser.add_argument('-o', '--output', type=str, default='crsample-{p}-{i}.fits',
                    help='Output filename pattern. {p} is replaced with process ID and '
                    '{i} with sample index. Environment variables $SOMEVAR are expanded '
                    'per process (but remember to protect the string with ''-quotes so '
                    'that the shell doesn''t expand them first)')
parser.add_argument('--render', action='store_true', default=False,
                    help='Make a GIF image (by invoking HEALPix map2gif) for each sample. '
                    'Useful during debugging.')
parser.add_argument('--render-span', metavar='SPAN', type=float, default=0.4,
                    help='If --render is given, use +- this value as the range')
parser.add_argument('--output-dir', type=str, default='.',
                    help='Output directory (default: current location)')

## parser.add_argument('--uniform-noise',
##                     help='Use uniform noise of the given value per pixel in the  This is '
##                     'used aft
##                          (see --sigma for possible values).'

# Mandatory arguments
parser.add_argument('powerspectrum', type=str,
                    help='The assumed power spectrum.')
parser.add_argument('beam', type=str,
                    help='The beam.')
parser.add_argument('mask', type=str,
                    help='The mask.')
parser.add_argument('observed', type=str,
                    help='The observed map.')


def main():
    # Parse args
    args = parser.parse_args()

    # Fill in some defaults

    def find_procid():
        try:
            from mpi4py.MPI import COMM_WORLD as comm
            if comm.Get_size() > 0:
                return comm.Get_rank()
        except ImportError:
            pass

        if 'SLURM_PROCID' in os.environ:
            return int(os.environ['SLURM_PROCID'])

        return 0

    args.output = os.path.expandvars(args.output)
    if args.procid is None:
        args.procid = find_procid()
    else:
        args.procid = os.path.expandvars(args.procid)
        try:
            # Canonicalize to integer if possible (more predictable
            # random seed behaviour)
            args.procid = int(args.procid)
        except ValueError:
            pass

    def fitsdesc(x, defext=1, deffield=0):
        if x is None:
            return None
        else:
            if ':' in x:
                return x.split(':')
            else:
                return (x, defext, deffield)

    # Combine random seed with process ID to create unique seed
    # for each process
    if args.seed is None:
        args.seed = np.random.randint(sys.maxint)
    elif args.seed > 0x7FFF:
         # If we proceed we'll loose bits and two different random seeds can produce
         # identical results. Of course, we could use less than 16 bits
         # for the process ID if this becomes a problem.
        raise ValueError('Random seed too large, must be less than %d' % 0x7FFF)
    args.seed = int(((args.seed << 16) ^ hash(args.procid)) & 0x7FFFFFFF)

    try:
        args.sigma = float(args.sigma)
    except ValueError:
        pass

    #
    # Configuration of loggers/verbosity levels. Basically map verbosity
    # integers to different levels for different logs
    #
    log_configs = [
         # default level, [overrides]
    # Level 0
        (logging.WARNING, []),
    # Level 1
        (logging.INFO, [
          ('cr', logging.WARNING),
          ('cr.precond', logging.WARNING),
          ('cr.cg', logging.WARNING),
        ]),
    # Level 2
        (logging.INFO, [
          ('cr', logging.INFO),
          ('cr.precond', logging.INFO),
          ('cr.cg', logging.WARNING),
        ]),
    # Level 3
        (logging.INFO, {}),
    # Level 4
        (logging.DEBUG, {}),
    ]

    logging.basicConfig(format='%s %%(levelname)s:%%(name)s:%%(message)s' % args.procid,
                        stream=sys.stdout)
    if args.verbose >= len(log_configs):
        args.verbose = len(log_configs) - 1
    lc = log_configs[args.verbose]

    logger = logging.getLogger()
    logger.setLevel(lc[0])
    for name, lvl in lc[1]:
        logging.getLogger(name).setLevel(lvl)



    #
    # Set up the sampler
    #

    import cmb # time-consuming, so putting after argparsing


    model = cmb.IsotropicCmbModel(
        # The power spectrum data file
        power_spectrum=args.powerspectrum
    )

    observation = cmb.CmbObservation(
        sigma0=args.sigma,
        beam=args.beam,
        mask=fitsdesc(args.mask),
        rms=fitsdesc(args.rms),
        n_obs=fitsdesc(args.nobs, 1, 'N_OBS'),
        temperature=fitsdesc(args.observed)
    )

    if args.lmax is None:
        args.lmax = 3*observation.Nside - 1
    if args.lprecond is None:
        args.lprecond = max([min([args.lmax // 2, 60]), 30])

    t0 = cmb.get_times()
    logger.info('Process-specific random seed: %d; lmax=%d; lprecond=%d' % (args.seed, args.lmax,
                                                                            args.lprecond))
    logger.info('Initializing')
    sampler_logger = logging.getLogger("cr")
    sampler = cmb.ConstrainedSignalSampler(
        model=model,
        observations=[observation],
        lmax=args.lmax,
        lprecond=min([args.lprecond, args.lmax]),
        logger=sampler_logger,
        seed=args.seed,
        eps=args.eps
    )
    cmb.log_times(logger, t0, 'Done initializing (%s)')

    with cmb.working_directory(args.output_dir):
        for i in range(args.repeat):
            outfile = args.output.format(i=i, p=args.procid)
            if os.path.exists(outfile) and not args.force:
                logger.warning('File exists, skipping sample: %s' % outfile)
            else:
                t0 = cmb.get_times()
                signal, [map] = sampler.sample_unmasked()
                map.to_fits(outfile)
                cmb.log_times(logger, t0, 'Sample %d of %d (%%s): %s' % (i+1, args.repeat, outfile))
                if args.render:
                    giffile = '%s.gif' % outfile
                    if os.path.exists(giffile):
                        logger.warning('Not creating GIF file, it already exists (and will '
                                       'be outdated): %s' % giffile)
                    map.map2gif(giffile, title=outfile, bar=True, max=args.render_span,
                                min=-args.render_span)
    logger.info('All samples taken, terminating')

if __name__ == '__main__':
    main()
