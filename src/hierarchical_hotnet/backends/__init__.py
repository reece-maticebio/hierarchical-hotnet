"""Backend dispatch for the native-accelerated operations.

The clustering and statistics code in :mod:`hierarchical_hotnet.core` calls
six operations that have a Fortran fast path and a pure-Python fallback.
The choice happens once at import time:

  * ``HHNET_BACKEND`` env var unset or ``auto``: try Fortran, fall back to
    pure Python if the extension was not built.
  * ``HHNET_BACKEND=fortran``: require the Fortran extension (raises if it
    is missing — useful for CI to guard against silent perf regressions).
  * ``HHNET_BACKEND=python``: force the pure-Python path (for testing the
    fallback, profiling, or environments where the extension is broken).

The selected backend's name is exposed via :data:`BACKEND` for assertions
and logging.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_REQUESTED = os.environ.get("HHNET_BACKEND", "auto").lower()

if _REQUESTED not in ("auto", "fortran", "python"):
    raise ValueError(
        f"HHNET_BACKEND={_REQUESTED!r} is not recognized; "
        "expected one of: auto, fortran, python"
    )


def _load_backend():
    if _REQUESTED == "python":
        from hierarchical_hotnet.backends import _python as impl
        logger.info("hierarchical_hotnet: HHNET_BACKEND=python; using pure-Python backend")
        return impl, "python"

    if _REQUESTED == "fortran":
        from hierarchical_hotnet.backends import _fortran as impl  # may raise ImportError
        logger.info("hierarchical_hotnet: HHNET_BACKEND=fortran; using Fortran backend")
        return impl, "fortran"

    # auto: prefer Fortran, fall back loudly to Python.
    try:
        from hierarchical_hotnet.backends import _fortran as impl
        logger.info("hierarchical_hotnet: using Fortran backend")
        return impl, "fortran"
    except ImportError:
        from hierarchical_hotnet.backends import _python as impl
        logger.warning(
            "hierarchical_hotnet: Fortran backend unavailable; falling back to "
            "pure-Python (slow). Rebuild the package with a working Fortran "
            "toolchain to enable the fast path, or set HHNET_BACKEND=python "
            "explicitly to silence this warning."
        )
        return impl, "python"


_impl, BACKEND = _load_backend()

condense_adjacency_matrix = _impl.condense_adjacency_matrix
find_distinct_weights = _impl.find_distinct_weights
slice_array = _impl.slice_array
strongly_connected_components_labels = _impl.strongly_connected_components_labels
threshold_edges = _impl.threshold_edges
summarize_sizes = _impl.summarize_sizes

__all__ = [
    "BACKEND",
    "condense_adjacency_matrix",
    "find_distinct_weights",
    "slice_array",
    "strongly_connected_components_labels",
    "threshold_edges",
    "summarize_sizes",
]
