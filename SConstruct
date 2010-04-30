import os
import numpy
import distutils.sysconfig

env = Environment(
    FORTRAN="ifort",
    F90="ifort",
    FORTRANFLAGS=["-g", "-vec_report0"],
    PYEXT_USE_DISTUTILS=True)


env.Tool("pyext")
env.Tool("cython")

env.Append(PYEXTINCPATH=[numpy.get_include()])
env.Replace(PYEXTCFLAGS=['-fno-strict-aliasing', '-DNDEBUG', '-Wall',
                         '-fwrapv', '-g', '-Wstrict-prototypes'],#, '-DCYTHON_REFNANNY'],
            CYTHON="python /uio/arkimedes/s07/dagss/cython/stable/cython.py",
            CYTHONFLAGS=['-a', '-I/uio/arkimedes/s07/dagss/cmb/cmblib',
                         '-I/uio/arkimedes/s07/dagss/cmb/slatec',
                         '-I/uio/arkimedes/s07/dagss/cmb/healpix4py'])
env['ENV']['PATH'] = os.environ['PATH']

Export('env')

SConscript(['cmb/SConscript'])

