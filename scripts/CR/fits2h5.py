import sys
import os
import glob
import h5py
from cmb import *

sourceglob = sys.argv[1]
destfile = sys.argv[2]

fitsfiles = glob.glob(sourceglob)
print fitsfiles[0]
Nalm = harmonic_sphere_map_from_fits(fitsfiles[0]).to_real().shape[0]

f = h5py.File(destfile, 'w')
grp = f.create_dataset('samples', (len(fitsfiles), Nalm), np.double, maxshape=(None, Nalm))
for idx, fitsname in enumerate(fitsfiles):
    print idx, fitsname
    grp[idx,:] = harmonic_sphere_map_from_fits(fitsname).to_real()
f.close()
