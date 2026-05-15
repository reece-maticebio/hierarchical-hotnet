"""Construct the Hierarchical HotNet similarity matrix."""

import numpy as np
import scipy as sp
from hierarchical_hotnet.core.common import hh_similarity_matrix


def _difference(A, beta, threshold):
    P = hh_similarity_matrix(A, beta)
    np.fill_diagonal(P, 0)
    r = np.sum(P[np.where(A >= threshold)])
    s = np.sum(P[np.where(A < threshold)])
    return r - s


def balanced_beta(A, threshold=1.0, num_digits=2):
    """Find the restart probability beta balancing within/between-edge mass."""
    try:
        return sp.optimize.ridder(lambda beta: _difference(A, beta, threshold), a=0.1 ** num_digits, b=1.0 - 0.1 ** num_digits, xtol=0.1 ** (num_digits + 1))
    except Exception:
        return 0.5


def edges_to_adjacency(edges, directed=False):
    """Build a 1-indexed adjacency matrix from a weighted edge list."""
    k = min((min(edge[:2]) for edge in edges))
    l = max((max(edge[:2]) for edge in edges))
    A = np.zeros((l - k + 1, l - k + 1), dtype=np.float64)
    if directed:
        for i, j, weight in edges:
            A[j - k, i - k] = weight
    else:
        for i, j, weight in edges:
            A[i - k, j - k] = A[j - k, i - k] = weight
    return A


def compute_similarity_matrix(edges, *, directed=False, beta=None, threshold=1.0, num_digits=2):
    """Compute the Hierarchical HotNet similarity matrix.

    Parameters
    ----------
    edges : iterable of (i, j, w)
        Weighted edge list (1-indexed) as returned by :func:`load_edge_list`.
    directed : bool
        Treat the graph as directed.
    beta : float or None
        Restart probability in (0, 1). If ``None``, chosen automatically to
        balance edge-weight mass at ``threshold``.
    threshold : float
        Threshold for edge weights when choosing beta.
    num_digits : int
        Precision (digits) used in the beta search.

    Returns
    -------
    P : np.ndarray
        Similarity matrix.
    beta : float
        Restart probability actually used.
    """
    edges = list(edges)
    A = edges_to_adjacency(edges, directed=directed)
    if beta is None:
        beta = balanced_beta(A, threshold=threshold, num_digits=num_digits)
    elif not 0 < beta < 1:
        raise ValueError(f'{beta} invalid; beta must satisfy 0 < beta < 1.')
    P = hh_similarity_matrix(A, beta)
    return (P, beta)
