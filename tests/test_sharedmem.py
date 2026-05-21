"""Tests for parallel.sharedmem: one shared, read-only copy of a large array.

The end-to-end behavior (construct_hierarchies with n_jobs>1 now routes the
similarity matrix through shared memory) is already exercised by
test_batch_functions.py and test_batch_store.py. These tests cover the helper
itself: faithful round-trip, read-only enforcement, and cleanup.
"""

import numpy as np
import pytest

from hierarchical_hotnet.parallel import (
    SharedArraySpec,
    attach_shared_ndarray,
    shared_ndarray,
)
from hierarchical_hotnet.parallel import sharedmem as _sm


def test_round_trips_array_contents():
    rng = np.random.default_rng(0)
    arr = rng.random((40, 40))
    with shared_ndarray(arr) as spec:
        assert isinstance(spec, SharedArraySpec)
        view, shm = attach_shared_ndarray(spec)
        try:
            np.testing.assert_array_equal(view, arr)
            assert view.dtype == arr.dtype
            assert view.shape == arr.shape
        finally:
            shm.close()


def test_spec_is_small_and_picklable():
    # The whole point: the spec, not the array, crosses the pool boundary.
    import pickle

    arr = np.zeros((500, 500))
    with shared_ndarray(arr) as spec:
        blob = pickle.dumps(spec)
        # Far smaller than the 2 MB array -- a name string plus shape/dtype.
        assert len(blob) < 1024
        assert pickle.loads(blob) == spec


def test_attached_view_is_read_only():
    arr = np.ones((8, 8))
    with shared_ndarray(arr) as spec:
        view, shm = attach_shared_ndarray(spec)
        try:
            with pytest.raises(ValueError):
                view[0, 0] = 2.0
        finally:
            shm.close()


def test_segment_unlinked_after_context_exit():
    arr = np.zeros((4, 4))
    with shared_ndarray(arr) as spec:
        pass
    # The parent unlinked the segment on exit; attaching by name must fail.
    with pytest.raises(FileNotFoundError):
        attach_shared_ndarray(spec)


def test_non_contiguous_input_is_copied_faithfully():
    # A transposed, strided view is not C-contiguous; shared_ndarray must
    # still reproduce its logical contents.
    base = np.arange(100, dtype=np.float64).reshape(10, 10)
    arr = base.T[::2]
    assert not arr.flags["C_CONTIGUOUS"]
    with shared_ndarray(arr) as spec:
        view, shm = attach_shared_ndarray(spec)
        try:
            np.testing.assert_array_equal(view, arr)
        finally:
            shm.close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_preserves_dtype(dtype):
    arr = np.arange(16, dtype=dtype).reshape(4, 4)
    with shared_ndarray(arr) as spec:
        view, shm = attach_shared_ndarray(spec)
        try:
            assert view.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(view, arr)
        finally:
            shm.close()


# --- /dev/shm preflight guard (Linux) ----------------------------------------


def _patch_shm(monkeypatch, free_bytes, *, platform="linux"):
    """Make _dev_shm_guard see a /dev/shm with ``free_bytes`` free."""
    import collections

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(_sm.sys, "platform", platform)
    monkeypatch.setattr(
        _sm.shutil, "disk_usage",
        lambda path: usage(free_bytes * 2, free_bytes, free_bytes),
    )


def test_dev_shm_guard_passes_when_space_is_sufficient(monkeypatch):
    _patch_shm(monkeypatch, 4 * 1024**3)
    _sm._dev_shm_guard(100 * 1024**2)  # 100 MB needed, 4 GB free -- no raise


def test_dev_shm_guard_raises_when_too_small(monkeypatch):
    _patch_shm(monkeypatch, 64 * 1024**2)
    with pytest.raises(RuntimeError) as excinfo:
        _sm._dev_shm_guard(2600 * 1024**2)
    msg = str(excinfo.value)
    assert "/dev/shm" in msg
    assert "2600 MB" in msg and "64 MB" in msg
    assert "n_jobs=1" in msg


def test_dev_shm_guard_is_noop_off_linux(monkeypatch):
    # Tiny /dev/shm, huge need -- but not Linux, so no check applies.
    _patch_shm(monkeypatch, 1, platform="darwin")
    _sm._dev_shm_guard(10**10)


def test_dev_shm_guard_hint_is_environment_specific(monkeypatch):
    _patch_shm(monkeypatch, 1024)
    real = _sm.os.path.exists

    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr(_sm.os.path, "exists",
                        lambda p: p == "/.dockerenv" or real(p))
    with pytest.raises(RuntimeError, match="shm-size"):
        _sm._dev_shm_guard(10**9)

    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    with pytest.raises(RuntimeError, match="emptyDir"):
        _sm._dev_shm_guard(10**9)


def test_runtime_environment_detection(monkeypatch):
    real = _sm.os.path.exists
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    monkeypatch.setattr(
        _sm.os.path, "exists",
        lambda p: real(p) and p not in ("/.dockerenv", "/run/.containerenv"),
    )
    assert _sm._runtime_environment() == "host"

    monkeypatch.setattr(_sm.os.path, "exists",
                        lambda p: p == "/.dockerenv" or real(p))
    assert _sm._runtime_environment() == "container"

    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert _sm._runtime_environment() == "kubernetes"
