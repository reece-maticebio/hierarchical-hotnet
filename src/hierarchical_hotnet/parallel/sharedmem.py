"""Read-only sharing of a large NumPy array across pool workers.

``ProcessPoolExecutor`` pickles every ``initargs`` entry once *per worker*, so
handing a dense N x N similarity matrix to a W-worker pool materializes W
independent copies. For Hierarchical HotNet that array is the dominant memory
cost and it is identical, read-only data for every worker -- W-1 of those
copies are pure waste, and they are what pushes large runs into OOM.

:func:`shared_ndarray` instead copies the array once into a POSIX shared-memory
segment and hands out a small picklable :class:`SharedArraySpec` (a name plus
shape/dtype). Workers call :func:`attach_shared_ndarray` to map the *same*
physical pages -- one copy total, regardless of worker count.

Ownership model
---------------
The parent process is the sole owner of the segment: :func:`shared_ndarray`
creates it and unlinks it on context exit. Workers only *attach* (and close
their handle when they exit); they never create or unlink. Callers must nest
the worker pool *inside* the :func:`shared_ndarray` block::

    with shared_ndarray(matrix) as spec:
        with maybe_pool(n_jobs, initializer=init, initargs=(spec,)) as run:
            ...

so the pool is fully shut down -- every worker detached -- before the parent
unlinks. That ordering means there is never a live mapping at unlink time and
no race to clean up after a crashed child.
"""

import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np


@dataclass(frozen=True)
class SharedArraySpec:
    """Picklable handle to an ndarray living in shared memory.

    Cheap to pass through a pool's ``initargs`` (a string and two tuples),
    unlike the array itself. Resolve it inside a worker with
    :func:`attach_shared_ndarray`.
    """

    name: str
    shape: tuple
    dtype: str


_SHM_PATH = "/dev/shm"

_SHM_REMEDIES = {
    "container": (
        "Restart the container with a larger shared-memory size:\n"
        "  docker run --shm-size=4g ...        (compose:  shm_size: 4gb)"
    ),
    "kubernetes": (
        "Mount a memory-backed emptyDir at /dev/shm in the pod spec:\n"
        "  volumes:      [{name: dshm, emptyDir: {medium: Memory, sizeLimit: 4Gi}}]\n"
        "  volumeMounts: [{name: dshm, mountPath: /dev/shm}]"
    ),
    "host": (
        "/dev/shm is normally a tmpfs sized at ~50% of RAM; here it looks\n"
        "restricted. Remount it larger (needs root):\n"
        "  sudo mount -o remount,size=4g /dev/shm\n"
        "If /dev/shm is already ~half your RAM, this network's matrix is\n"
        "simply too large for this machine."
    ),
}


def _runtime_environment():
    """Best-effort detection of the runtime, for a tailored /dev/shm hint.

    Returns ``"kubernetes"``, ``"container"`` (Docker/Podman), or ``"host"``.
    Kubernetes is checked first because a pod is also a container.
    """
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return "container"
    return "host"


def _dev_shm_guard(nbytes):
    """Fail early, with an actionable message, if /dev/shm is too small.

    POSIX shared memory on Linux is backed by /dev/shm, a tmpfs. tmpfs
    allocates lazily, so an oversized segment is created and mapped without
    error, then dies with an opaque ``SIGBUS`` ("Bus error", no traceback)
    the instant the array is written into it. This turns that into a clear,
    environment-specific error before any allocation happens.

    Best-effort only: it is a no-op off Linux (macOS POSIX shm is not
    /dev/shm-backed) or when /dev/shm cannot be inspected, and the
    free-space check is inherently racy against other processes.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        free = shutil.disk_usage(_SHM_PATH).free
    except OSError:
        return  # No /dev/shm to inspect -- let SharedMemory try regardless.
    if free >= nbytes:
        return

    mb = 1024 * 1024
    raise RuntimeError(
        f"Cannot create a {nbytes / mb:.0f} MB shared-memory segment: "
        f"{_SHM_PATH} has only {free / mb:.0f} MB free.\n"
        "The similarity matrix is shared across worker processes via POSIX "
        "shared memory (/dev/shm).\n"
        f"{_SHM_REMEDIES[_runtime_environment()]}\n"
        "Or run with n_jobs=1, which uses no shared memory."
    )


@contextmanager
def shared_ndarray(array):
    """Copy ``array`` into shared memory; yield a :class:`SharedArraySpec`.

    Use as a context manager wrapping the lifetime of the pool that consumes
    the spec (see the module docstring for the required nesting). The shared
    segment is unlinked when the block exits, so it must outlive the pool.

    The input may be any array-like; it is copied as a C-contiguous block, so
    non-contiguous views (transposes, strided slices) are reproduced
    faithfully by their logical contents.
    """
    src = np.ascontiguousarray(array)
    # Fail early with a clear message if /dev/shm cannot hold the segment,
    # rather than later with an opaque SIGBUS when it is written into.
    _dev_shm_guard(src.nbytes)
    # SharedMemory rejects size 0; clamp so a (degenerate) empty array still
    # round-trips -- a 0-element ndarray needs 0 bytes and fits any buffer.
    shm = SharedMemory(create=True, size=max(src.nbytes, 1))
    try:
        # Populate the segment through a temporary writable view, then drop
        # the view before yielding: SharedMemory.close() raises BufferError
        # if a NumPy array still exports a pointer into shm.buf.
        view = np.ndarray(src.shape, dtype=src.dtype, buffer=shm.buf)
        view[:] = src
        del view
        yield SharedArraySpec(shm.name, src.shape, str(src.dtype))
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            # Already unlinked (e.g. a duplicate cleanup pass). Unlink is
            # idempotent from this owner's point of view.
            pass


def attach_shared_ndarray(spec: SharedArraySpec):
    """Attach (inside a worker) to the segment described by ``spec``.

    Returns ``(array, shm)``. ``array`` is a **read-only** view onto the
    shared pages. The caller must keep ``shm`` alive for as long as ``array``
    is used: dropping the last reference to ``shm`` unmaps the buffer and
    invalidates ``array``.
    """
    shm = SharedMemory(name=spec.name)
    array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    # Workers only ever read the matrix (every consumer fancy-indexes a copy
    # out of it). Marking the view read-only turns an accidental in-place
    # write into a loud error instead of silent cross-worker corruption.
    array.flags.writeable = False
    return array, shm
