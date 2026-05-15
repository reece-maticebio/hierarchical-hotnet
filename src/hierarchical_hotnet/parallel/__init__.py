"""Internal helpers for running step batches serially or on a process pool.

Each step that supports batch processing uses :func:`maybe_pool` to either run
serially (``n_jobs=1``, no pool overhead) or on a ``ProcessPoolExecutor``.
Worker functions read shared per-batch data from a module-level dict populated
via the executor's ``initializer`` (so heavy objects like similarity matrices
are pickled once per worker rather than once per task).
"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager


@contextmanager
def maybe_pool(n_jobs, initializer=None, initargs=()):
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
    """
    if n_jobs == 1:
        yield map
        return
    workers = None if n_jobs == -1 else n_jobs
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        yield executor.map
