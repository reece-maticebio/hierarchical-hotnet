"""Pure-Python reference implementations of the backend-dispatched operations.

These exist so the package works without the Fortran extension built; they
are also the easier-to-read implementations and the oracle the Fortran path
must agree with. New backends (Rust, Numba, ...) should be added as sibling
modules and selected by :mod:`hierarchical_hotnet.backends.__init__`.
"""

from __future__ import annotations

import numpy as np


# --- clustering primitives ----------------------------------------------------


def condense_adjacency_matrix(A, components):
    """Condense an adjacency matrix by collapsing SCCs into single nodes.

    The (i, j) entry of the result is the minimum nonzero edge weight from
    any node in ``components[j]`` to any node in ``components[i]``.
    """
    n = len(components)
    B = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            if i != j:
                C = A[np.ix_(components[j], components[i])]
                nonzero_indices = np.nonzero(C)
                if np.size(nonzero_indices) > 0:
                    B[i, j] = np.min(C[nonzero_indices])
    return B


def find_distinct_weights(A):
    """Sorted unique nonzero entries of ``A`` (in fact: all unique entries)."""
    return np.unique(A)


def slice_array(A, rows, columns):
    """Equivalent to ``A[np.ix_(rows, columns)]``."""
    return A[np.ix_(rows, columns)]


def strongly_connected_components_labels(A):
    """Tarjan SCC. Returns a length-N vector ``labels`` where ``labels[i]``
    is the component label for node ``i`` (1-indexed labels)."""
    m, n = np.shape(A)
    nodes = range(n)

    index = -np.ones(n, dtype=np.int64)
    lowlink = -np.ones(n, dtype=np.int64)
    found = np.zeros(n, dtype=bool)
    queue = np.zeros(n, dtype=np.int64)
    subqueue = np.zeros(n, dtype=np.int64)
    component = np.zeros(n, dtype=np.int64)

    neighbors = np.zeros((n, n), dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    for v in nodes:
        neighbors_v = np.where(A[v] > 0)[0]
        degree_v = np.size(neighbors_v)
        neighbors[v, 0:degree_v] = neighbors_v
        degree[v] = degree_v

    i = 0
    j = 0
    k = 0
    l = 0

    for u in nodes:
        if not found[u]:
            queue[k] = u
            k += 1

            while k >= 1:
                v = queue[k - 1]
                if index[v] == -1:
                    i += 1
                    index[v] = i

                updated_queue = False
                for w in neighbors[v, 0:degree[v]]:
                    if index[w] == -1:
                        queue[k] = w
                        k += 1
                        updated_queue = True
                        break

                if not updated_queue:
                    lowlink[v] = index[v]
                    for w in neighbors[v, 0:degree[v]]:
                        if not found[w]:
                            if index[w] > index[v]:
                                lowlink[v] = min(lowlink[v], lowlink[w])
                            else:
                                lowlink[v] = min(lowlink[v], index[w])
                    k -= 1

                    if lowlink[v] == index[v]:
                        found[v] = True
                        j += 1
                        component[v] = j
                        while l >= 1 and index[subqueue[l - 1]] > index[v]:
                            w = subqueue[l - 1]
                            l -= 1
                            found[w] = True
                            component[w] = j
                    else:
                        subqueue[l] = v
                        l += 1

    return component


def threshold_edges(A, weight):
    """Zero out entries of ``A`` strictly greater than ``weight``.

    (Used by tarjan_HD when walking edges in order — entries below the
    threshold survive; entries above are 'not yet added' to the graph.)
    """
    B = A.copy()
    B[B > weight] = 0
    return B


# --- statistics aggregation ---------------------------------------------------


def summarize_sizes(distinct_heights, permuted_heights, permuted_sizes, max_indices):
    """Aggregate min/mean/max cluster size across permutations at each height.

    Inputs are *lists* of per-permutation arrays (variable lengths between
    permutations). Returns three length-``len(distinct_heights)`` arrays
    (min, expected, max).
    """
    num_permutations = len(permuted_heights)
    num_distinct = len(distinct_heights)

    cur_indices = np.zeros(num_permutations, dtype=np.int64)
    cur_sizes = np.zeros(num_permutations)
    min_sizes = np.zeros(num_distinct)
    expected_sizes = np.zeros(num_distinct)
    max_sizes = np.zeros(num_distinct)

    for k in range(num_distinct):
        distinct_height = distinct_heights[k]
        for i in range(num_permutations):
            while (
                cur_indices[i] < max_indices[i] - 1
                and permuted_heights[i][cur_indices[i] + 1] >= distinct_height
            ):
                cur_indices[i] += 1
            cur_sizes[i] = permuted_sizes[i][cur_indices[i]]
        min_sizes[k] = np.min(cur_sizes)
        expected_sizes[k] = np.mean(cur_sizes)
        max_sizes[k] = np.max(cur_sizes)

    return min_sizes, expected_sizes, max_sizes
