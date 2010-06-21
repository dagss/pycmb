import os
import numpy
import distutils.sysconfig

v = Variables('config')
v.Add('HEALPIX4PY_INCLUDE')
v.Add('FORTRAN')
v.Add('FORTRANFLAGS')
v.Add('FORTRAN_FPU_DETECT_FLAGS', 'Flags to use for d1mach.f', ['-O0', '-ffloat-store'])
v.Add('PYEXTCFLAGS')
v.Add('PYEXTLINKFLAGS')
v.Add('FORTRANPYEXTLINK')
env = Environment(variables=v,
                  PYEXT_USE_DISTUTILS=True)

env.Tool("pyext")
env.Tool("cython")

env.Append(PYEXTINCPATH=[numpy.get_include()],
           CYTHONFLAGS='-I$HEALPIX4PY_INCLUDE')

env['ENV']['PATH'] = os.environ['PATH']

Export('env')

SConscript(['cmb/SConscript'])

