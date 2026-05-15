"""Pure-Python compute primitives for Hierarchical HotNet.

Modules here contain the algorithm: similarity-matrix construction, score and
network permutation, hierarchical decomposition, statistics, consensus.
They do not import argparse, do not perform file IO except via
:mod:`hierarchical_hotnet.file_io`, and do not own parallel-pool lifecycles
except via :mod:`hierarchical_hotnet.parallel`.

Most names users care about are re-exported from the top-level
:mod:`hierarchical_hotnet` package, so end-user code should reach for
``from hierarchical_hotnet import construct_hierarchy`` rather than
importing the submodules here directly.
"""
