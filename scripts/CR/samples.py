from __future__ import division
from common import *
from mpi4py import MPI
import tables as tab
from cmb.mpiutils import *

comm = MPI.COMM_WORLD

sampler = ConstrainedSignalSampler(
    model=model,
    observations=[obs],
    verbosity=0,
    lprecond=lprecond,
    seed=83 + comm.Get_rank()*10000,
    lmax=lmax)            

J = 1000
Nalm = (lmax + 1)**2 - lmin**2

time_lock = time_io = 0

def persist_sample(signal):
    global time_lock, time_io
    t0 = MPI.Wtime()
    print '.   Locking datafile', comm.Get_rank(), MPI.Wtime()
    with locked_h5file(samples_filename, 'a') as f:
        time_lock += MPI.Wtime() - t0
        t0 = MPI.Wtime()
        print ' .  Opened datafile', comm.Get_rank(), MPI.Wtime()
        if 'samples' not in f:
            samples = f.create_dataset('samples', (1, Nalm), np.double, maxshape=(None, Nalm))
        else:
            samples = f['samples']
            samples.resize((samples.shape[0] + 1, samples.shape[1]))
        samples[-1,:] = signal
    time_io += MPI.Wtime() - t0
    t = MPI.Wtime() - t0_global
    print '  . Closed datafile', comm.Get_rank(), MPI.Wtime()
    print 'Stats: Time io rate: %f, time lockwait rate: %f' % (time_io / t, time_lock / t)

def get_sample_count():
    with locked_h5file(samples_filename, 'a') as f:
        print 'Opened datafile'
        if 'samples' not in f:
            return 0
        else:
            return f['samples'].shape[0]

@parallel()
def draw_signal(indices, comm):
    for j in indices:
        
        print 'Sample #%d of %d @ process %d' % (j+1, J, comm.Get_rank())
        signal = sampler.sample_signal().to_real()
        persist_sample(signal)
        del signal

t0_global = MPI.Wtime()
if comm.Get_rank() == 0:
    draw_signal(range(get_sample_count(), J), comm=comm)

parallel_barrier(comm)
