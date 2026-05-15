"""Backward-compatibility shim for the old ``hhio`` module name.

The IO helpers moved to :mod:`hierarchical_hotnet.file_io`. Importing this
module emits a :class:`DeprecationWarning`. Remove on the next minor release.
"""

import warnings

warnings.warn(
    "hierarchical_hotnet.hhio has been renamed; import from "
    "hierarchical_hotnet.file_io instead",
    DeprecationWarning,
    stacklevel=2,
)

from hierarchical_hotnet.file_io import (  # noqa: E402, F401
    load_edge_list,
    load_gene_score,
    load_index_gene,
    load_matrix,
    progress,
    save_edge_list,
    save_gene_score,
    save_index_gene,
    save_matrix,
)
