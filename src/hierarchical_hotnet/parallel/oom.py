"""Turn opaque pool-worker deaths into actionable memory-overflow errors.

When a process pool runs out of memory, the OS OOM killer SIGKILLs a worker
and :class:`concurrent.futures.process.BrokenProcessPool` surfaces as a bare
"a process ... was terminated abruptly" -- with no mention of memory or of
the worker count that caused it. :func:`worker_failure_error` rewrites that
into a message that names ``n_jobs`` and the fixes (fewer workers, or
disk-backed storage).

On cgroup v2 hosts (containers, and most SLURM setups) :class:`OomProbe`
reads the kernel's per-cgroup ``oom_kill`` counter, so the message can state
*definitively* whether the death was an out-of-memory kill rather than, say,
a crash in the native clustering backend.
"""


def _read_cgroup_oom_kills():
    """Cumulative ``oom_kill`` count for this process's cgroup (v2).

    Returns ``None`` when the host is not cgroup v2, the files are missing,
    or anything is unreadable -- callers treat ``None`` as "cannot tell".
    The counter covers every OOM kill of a process in the cgroup, whether
    triggered by the cgroup's own limit or by the global OOM killer.
    """
    try:
        rel = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.rstrip("\n").split(":", 2)
                # cgroup v2 has exactly one entry, with an empty controller
                # field: "0::/the/path".
                if parts[0] == "0":
                    rel = parts[2]
                    break
        if rel is None:
            return None
        path = "/sys/fs/cgroup" + rel.rstrip("/") + "/memory.events"
        with open(path) as f:
            for line in f:
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
    except (OSError, ValueError, IndexError):
        return None
    return None


class OomProbe:
    """Snapshot the cgroup OOM-kill counter to classify a later failure.

    Construct one before launching a pool; after a worker dies, ask
    :meth:`oom_killed_since`.
    """

    def __init__(self):
        self.before = _read_cgroup_oom_kills()

    def oom_killed_since(self):
        """Was a process in this cgroup OOM-killed since construction?

        ``True``/``False`` when cgroup v2 gives a definitive answer,
        ``None`` when it cannot be determined (e.g. macOS, cgroup v1).
        """
        after = _read_cgroup_oom_kills()
        if self.before is None or after is None:
            return None
        return after > self.before


def worker_failure_error(n_jobs, stage, exc, probe=None):
    """Build a descriptive exception for an abrupt pool-worker failure.

    Parameters
    ----------
    n_jobs : int
        Worker count the pool was launched with.
    stage : str
        Human-readable name of the batch step (e.g. ``"construct_hierarchies"``).
    exc : BaseException
        The original failure (``BrokenProcessPool`` or ``MemoryError``).
    probe : OomProbe, optional
        If given, used to state definitively whether this was an OOM kill.

    Returns
    -------
    RuntimeError or MemoryError
        Same family as ``exc`` (``MemoryError`` stays a ``MemoryError`` so
        callers keying on it still work); raise it with ``from exc``.
    """
    oom = probe.oom_killed_since() if probe is not None else None

    lines = [f"A worker process failed during '{stage}' (n_jobs={n_jobs})."]
    if oom is True:
        lines.append(
            "Confirmed out of memory: the cgroup OOM killer terminated a "
            "worker. Every worker runs concurrently, so peak memory scales "
            "with n_jobs."
        )
    else:
        lines.append(
            "This almost always means the run ran out of memory: all "
            f"{n_jobs} workers run at once, each holding the similarity "
            "matrix plus a multi-hundred-MB working set, so peak memory "
            "scales with n_jobs."
        )
    lines += [
        "Fixes:",
        "  - reduce n_jobs (try n_jobs=4, then n_jobs=2)",
        "  - pass workdir=<path> so per-permutation artifacts stream to disk",
    ]
    if oom is not True:
        lines.append(
            "(A worker can also die from a crash in the native clustering "
            "backend; if memory is plentiful, suspect that instead.)"
        )

    message = "\n".join(lines)
    if isinstance(exc, MemoryError):
        return MemoryError(message)
    return RuntimeError(message)
