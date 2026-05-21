"""Descriptive errors when a pool worker dies (parallel.oom + maybe_pool).

A worker killed by the OS OOM killer otherwise surfaces as a bare
``BrokenProcessPool``. These tests check that maybe_pool rewrites it into a
message naming n_jobs and the fixes, and that the OOM probe never guesses.
"""

import os
from concurrent.futures.process import BrokenProcessPool

import pytest

from hierarchical_hotnet.parallel import maybe_pool
from hierarchical_hotnet.parallel.oom import OomProbe, worker_failure_error


def _suicidal_worker(_):
    # Exit hard, without raising, so the executor reports a BrokenProcessPool.
    os._exit(1)


def _double(x):
    return x * 2


class _FakeProbe:
    def __init__(self, verdict):
        self._verdict = verdict

    def oom_killed_since(self):
        return self._verdict


def test_maybe_pool_translates_broken_pool_to_descriptive_error():
    with pytest.raises(RuntimeError) as excinfo:
        with maybe_pool(2, stage="unit-stage") as run:
            list(run(_suicidal_worker, list(range(8))))
    msg = str(excinfo.value)
    assert "n_jobs=2" in msg
    assert "unit-stage" in msg
    assert "reduce n_jobs" in msg
    assert "workdir=" in msg
    # The original failure is preserved as the exception cause.
    assert isinstance(excinfo.value.__cause__, BrokenProcessPool)


def test_maybe_pool_serial_path_is_unaffected():
    # n_jobs=1 takes the no-pool path: a plain map, nothing wrapped.
    with maybe_pool(1, stage="serial") as run:
        assert list(run(_double, [1, 2, 3])) == [2, 4, 6]


def test_worker_failure_error_keeps_memory_error_type():
    # A propagated MemoryError stays a MemoryError so callers keying on the
    # type still work -- only the message is enriched.
    err = worker_failure_error(4, "some-stage", MemoryError("boom"))
    assert isinstance(err, MemoryError)
    assert "n_jobs=4" in str(err)
    assert "some-stage" in str(err)


def test_worker_failure_error_is_runtimeerror_for_broken_pool():
    err = worker_failure_error(8, "s", BrokenProcessPool("died"))
    assert isinstance(err, RuntimeError)
    assert not isinstance(err, MemoryError)


def test_worker_failure_error_states_confirmed_oom_when_probe_says_so():
    err = worker_failure_error(8, "s", BrokenProcessPool("x"), _FakeProbe(True))
    assert "Confirmed out of memory" in str(err)


def test_worker_failure_error_hedges_when_oom_unconfirmed():
    err = worker_failure_error(8, "s", BrokenProcessPool("x"), _FakeProbe(None))
    text = str(err)
    assert "Confirmed out of memory" not in text
    assert "native clustering backend" in text  # the alternative cause


def test_oom_probe_never_falsely_reports_an_oom():
    # No OOM happens in this test. The probe must return False (cgroup v2) or
    # None (no cgroup v2, e.g. macOS) -- never True, never an exception.
    probe = OomProbe()
    assert probe.oom_killed_since() in (None, False)
