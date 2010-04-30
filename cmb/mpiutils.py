from mpi4py import MPI
import threading
import functools
import contextlib
from time import sleep
import os

__all__ = ['parallel_barrier',
           'parallel',
           'parallel_release_lock',
           'parallel_acquire_lock',
           'parallel_locked_filename',
           'parallel_locked_file',
           'parallel_locked_h5file'
           ]

def parallel_barrier(comm):
    if comm.Get_rank() == 0:
        # Signal to workers that we are done. At this point,
        # they should all be listening for new commands (parallel
        # functions called by root will wait until all workers are
        # in wait-mode before returning)
        for client in range(0, comm.Get_size()):
            comm.send('done', dest=client, tag=COMMAND_TAG)
    else:
        # Become a worker
        work_loop(comm)

class parallel(object):
    """
    Decorator which executes functions in parallel through task
    dispatching. By default this will run two threads in the root node
    (one simple task dispatcher), which will not work well unless one
    is sure to release the GIL during computations (so that the
    dispatcher threads get to dispatch tasks).
    """
    def __init__(self, argcount=1, include_root=True):
        self.broadcast_arg_count = argcount

    def __call__(self, func):
        @functools.wraps(func)
        def call_executor(*args, **kw):
            if '_through_pickle' in kw:
                del kw['_through_pickle']
                func(*args, **kw)
            elif 'comm' not in kw:
                # Bypass MPI altogether
                return func(*args, **kw)                
            else:
                # The below is a bit inobvious -- we need to pickle the returned
                # decorated function over the network (as it can be looked up
                # by name), and then we reenter over
                comm = kw['comm']
                if comm.Get_rank() != 0:
                    raise Exception("Cannot call %s.%s from non-root MPI process" %
                                    (func.__module__, func.__name__))
                del kw['comm']
                sched = Scheduler(call_executor, comm,
                                  self.broadcast_arg_count,
                                  *args, **kw)
                sched.run()
                
        return call_executor

class RootWorker(threading.Thread):
    # Just to get the return value
    def __init__(self, target, args, kwargs):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        work_loop()
#        try:
#            self.return_value = self.target(*self.args, **self.kwargs)
#        except Exception, e:
#            import traceback
#            traceback.print_exc()
#            self.return_value = 'exception' # TODO

class Scheduler(object):
    def __init__(self, func, comm, arg_count, *args, **kw):
        self.func = func
        self.comm = comm
        self.arg_count = arg_count
        self.args = args
        self.kw = kw
        self._locks = {}
        self._lock_queue = {}
        
    def run(self):
        # Execute tasks, both on other nodes through MPI, and on this
        # node through another thread (which must release the
        # GIL often enough).
        # TODO: Reimplement this in GIL-less Cython.

        # We are communicating with the worker thread in our own process
        # through MPI as well. The mesages are seperated by the tags.
        # However, this means that we must use an Iprobe/sleep combination
        # so that messages with different tags can get through the queue.
        # (I think...)
        
        comm = self.comm
        broadcast_args = self.args[:self.arg_count]
        args = self.args[self.arg_count:]
        numprocs = comm.Get_size()
        status = MPI.Status()
        root_worker = threading.Thread(target=work_loop, args=(comm,))
        root_worker.start()
        # Dispatch loop
        for curargs in zip(*broadcast_args):
            args_as_lists = [[x] for x in curargs]
            while True:
                if comm.Iprobe(source=MPI.ANY_SOURCE, tag=REQUEST_TAG, status=status):
                    # Can dispatch the task on another node
                    client = status.Get_source()
                    ready = self.process_request(client)
                    if ready:
                        comm.send((self.func, args_as_lists, self.kw), dest=client,
                                  tag=COMMAND_TAG)
                        break # go to next task in outer loop
                else:
                    # Sleep and try again
                    sleep(0.05)

        # Make sure all workers are done before returning (TODO: get final return values)
        clients_to_terminate = {}
        for client in range(0, numprocs):
            clients_to_terminate[client] = True
        while len(clients_to_terminate) > 0:
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=REQUEST_TAG, status=status):
                client = status.Get_source()
                ready = self.process_request(client)
                if ready:
                    del clients_to_terminate[client]
            else:
                sleep(0.05)
                
    def process_request(self, client):
        retval = self.comm.recv(source=client, tag=REQUEST_TAG)
        if retval is not None:
            cmd, arg = retval
            if cmd == 'acquire_lock':
                self.acquire_lock(arg, client)
                return False
            elif cmd == 'release_lock':
                self.release_lock(arg, client)
                return False
            else:
                # return value handling not yet supported
                raise RuntimeError('protocol error')
        else:
            return True # ready to work

    def acquire_lock(self, lockname, client):
        if lockname not in self._locks:
            self._locks[lockname] = client
            self.comm.send('got_lock', dest=client, tag=COMMAND_TAG)
        else:
            # Queue client for notification when the file is unlocked
            q = self._lock_queue.get(lockname, [])
            q.append(client)
            self._lock_queue[lockname] = q

    def release_lock(self, lockname, client):
        if lockname not in self._locks:
            raise RuntimeError('Lock "%s" released by client %d was never acquired!' %
                               (lockname, client))
        x = self._locks[lockname]
        if x != client:
            raise RuntimeError('Lock "%s" acquired by client %d was released by client %d!' %
                               (lockname, x, client))
        del self._locks[lockname]
        # No response to the client is required, but we should process
        # queued requests for the same lock
        q = self._lock_queue.get(lockname, None)
        if q is not None:
            client = q.pop(0)
            if len(q) == 0:
                del self._lock_queue[lockname]
            self.acquire_lock(lockname, client)

COMMAND_TAG = 1
REQUEST_TAG = 2

def work_loop(comm):
    while True:
        comm.send(None, dest=0, tag=REQUEST_TAG)
        obj = comm.recv(source=0, tag=COMMAND_TAG)
        if obj == 'wait':
            pass # root just fetched result, start to wait again
        elif obj == 'done':
            return
        else:
            func, args, kw = obj
            kw['_through_pickle'] = True
            func(*args, comm=comm, **kw)

def parallel_acquire_lock(comm, lockname):
    comm.send(('acquire_lock', lockname), dest=0, tag=REQUEST_TAG)
    # The following recv blocks until we have the lock
    response = comm.recv(source=0, tag=COMMAND_TAG)
    if response != 'got_lock':
        raise RuntimeError('Invalid response from root: %s' % (response,))

def parallel_release_lock(comm, lockname):
    comm.send(('release_lock', lockname), dest=0, tag=REQUEST_TAG)    





@contextlib.contextmanager
def parallel_locked_filename(comm, filename):
    lockname = 'file:/%s' % os.path.realpath(filename)
    parallel_acquire_lock(comm, lockname)
    try:
        yield
    finally:
        parallel_release_lock(comm, lockname)
    
@contextlib.contextmanager
def parallel_locked_file(comm, filename, mode):
    with parallel_locked_filename(comm, filename):
        with file(filename, mode) as f:
            yield f

@contextlib.contextmanager
def parallel_locked_h5file(comm, filename, *args, **kw):
    import h5py
    with parallel_locked_filename(comm, filename):
        f = h5py.File(filename, *args, **kw)
        try:
            yield f
        finally:
            print 'closing f'
            f.close()
            print 'closed'
