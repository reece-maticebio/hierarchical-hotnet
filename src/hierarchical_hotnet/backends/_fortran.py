"""Fortran-accelerated implementations of the backend-dispatched operations.

This module imports the compiled Fortran extension at module load. Importing
it fails if the extension was not built — :mod:`hierarchical_hotnet.backends.__init__`
catches that and falls back to :mod:`._python`.

The public functions here match :mod:`._python` exactly. Each one wraps the
low-level Fortran routine and performs the 0-vs-1 indexing translation and
any padding into the rectangular arrays the Fortran code expects.
"""

from __future__ import annotations

import numpy as np

from hierarchical_hotnet import fortran_module  # noqa: F401 — ImportError surfaces


# --- clustering primitives ----------------------------------------------------


def condense_adjacency_matrix(A, components):
    n = len(components)
    nodes = np.array([i for component in components for i in component], dtype=np.int64)
    sizes = np.array([len(component) for component in components], dtype=np.int64)
    indices = np.array([np.sum(sizes[:i]) for i in range(n + 1)], dtype=np.int64)
    return fortran_module.condense_adjacency_matrix(A, nodes + 1, indices + 1)


def find_distinct_weights(A):
    B, l = fortran_module.unique_entries(A)
    return B[:l]


def slice_array(A, rows, columns):
    return fortran_module.slice_array(
        A,
        np.array(columns, dtype=np.int64) + 1,
        np.array(rows, dtype=np.int64) + 1,
    )


def strongly_connected_components_labels(A):
    return fortran_module.strongly_connected_components(A)


def threshold_edges(A, weight):
    return fortran_module.threshold_matrix(A, weight)


# --- statistics aggregation ---------------------------------------------------


def summarize_sizes(distinct_heights, permuted_heights, permuted_sizes, max_indices):
    num_permutations = len(permuted_heights)
    max_index = int(np.max(max_indices))

    # Fortran wants rectangular arrays; pad each variable-length permutation.
    heights = np.zeros((num_permutations, max_index))
    sizes = np.zeros((num_permutations, max_index))
    for i in range(num_permutations):
        heights[i, : max_indices[i]] = permuted_heights[i]
        sizes[i, : max_indices[i]] = permuted_sizes[i]

    summary = fortran_module.summarize_sizes(distinct_heights, heights, sizes, max_indices)
    return summary[:, 0], summary[:, 1], summary[:, 2]
