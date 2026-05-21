"""Internal helpers for running step batches serially or on a process pool.

Each step that supports batch processing uses :func:`maybe_pool` to either run
serially (``n_jobs=1``, no pool overhead) or on a ``ProcessPoolExecutor``.
Worker functions read shared per-batch data from a module-level dict populated
via the executor's ``initializer``.

Small per-batch data (score maps, flags) is passed straight through
``initargs`` -- pickled once per worker. A large read-only array shared by
every worker (the similarity matrix) should instead be placed in shared
memory via :func:`shared_ndarray` so the pool maps one physical copy rather
than pickling W of them; see :mod:`hierarchical_hotnet.parallel.sharedmem`.
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager

from hierarchical_hotnet.parallel.oom import OomProbe, worker_failure_error
from hierarchical_hotnet.parallel.sharedmem import (
    SharedArraySpec,
    attach_shared_ndarray,
    shared_ndarray,
)

__all__ = [
    "maybe_pool",
    "shared_ndarray",
    "attach_shared_ndarray",
    "SharedArraySpec",
]


@contextmanager
def maybe_pool(n_jobs, initializer=None, initargs=(), *, stage="batch"):
    """Yield a ``map``-like callable.

    Parameters
    ----------
    n_jobs : int
        ``1`` runs serially (no pool created). ``-1`` lets the pool pick a
        worker count. Any other positive integer uses that many workers.
    initializer, initargs :
        Passed through to :class:`concurrent.futures.ProcessPoolExecutor`. The
        initializer is invoked once per worker process; use it to populate a
        module-level state dict with data shared across all tasks in the batch.
    stage : str
        Human-readable name of the batch step, used only to label the error
        raised if a worker dies (see below).

    Notes
    -----
    If a worker is killed mid-batch -- overwhelmingly because the run ran out
    of memory, since peak memory scales with ``n_jobs`` -- the executor
    raises a bare ``BrokenProcessPool``. This wrapper rewrites that (and a
    propagated ``MemoryError``) into a message that names ``n_jobs`` and the
    fixes; see :mod:`hierarchical_hotnet.parallel.oom`.
    """
    if n_jobs == 1:
        yield map
        return
    workers = None if n_jobs == -1 else n_jobs
    # Snapshot the cgroup OOM counter before the pool starts so a failure can
    # be classified as a definite out-of-memory kill where the host allows it.
    probe = OomProbe()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        try:
            yield executor.map
        except (BrokenProcessPool, MemoryError) as exc:
            raise worker_failure_error(n_jobs, stage, exc, probe) from exc
