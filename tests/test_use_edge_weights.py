"""Tests for the use_edge_weights toggle on compute_similarity_matrix.

The default (``use_edge_weights=False``) ignores the third component of
each edge tuple and treats the network as unweighted, matching the
canonical Hierarchical HotNet methodology. Opting in propagates the
input weights into the adjacency matrix.
"""

import numpy as np
import pytest

from hierarchical_hotnet.core.similarity import compute_similarity_matrix


# Same triangle topology, three different weight configurations.
TOPOLOGY = [(1, 2), (2, 3), (3, 1)]
EDGES_UNIT = [(i, j, 1.0) for i, j in TOPOLOGY]
EDGES_WEIGHTED = [(1, 2, 5.0), (2, 3, 0.1), (3, 1, 2.7)]


class TestDefaultIgnoresWeights:
    def test_default_matches_unit_weights(self):
        """use_edge_weights=False makes weighted edges equivalent to unit edges."""
        P_unit, beta_unit = compute_similarity_matrix(EDGES_UNIT, beta=0.4)
        P_weighted, beta_weighted = compute_similarity_matrix(EDGES_WEIGHTED, beta=0.4)
        np.testing.assert_allclose(P_unit, P_weighted)
        assert beta_unit == beta_weighted

    def test_use_edge_weights_false_is_explicit_default(self):
        P_implicit, _ = compute_similarity_matrix(EDGES_WEIGHTED, beta=0.4)
        P_explicit, _ = compute_similarity_matrix(
            EDGES_WEIGHTED, beta=0.4, use_edge_weights=False,
        )
        np.testing.assert_array_equal(P_implicit, P_explicit)


class TestOptInUsesWeights:
    def test_opt_in_diverges_from_unit(self):
        """use_edge_weights=True makes weighted edges produce a different P."""
        P_unit, _ = compute_similarity_matrix(EDGES_UNIT, beta=0.4)
        P_weighted, _ = compute_similarity_matrix(
            EDGES_WEIGHTED, beta=0.4, use_edge_weights=True,
        )
        assert not np.allclose(P_unit, P_weighted), (
            "use_edge_weights=True should yield a different similarity matrix "
            "from the unit-weight case when input weights are non-uniform."
        )

    def test_opt_in_with_unit_input_matches_default(self):
        """Edge case: unit-weight input gives the same P regardless of the flag."""
        P_default, _ = compute_similarity_matrix(EDGES_UNIT, beta=0.4)
        P_opt_in, _ = compute_similarity_matrix(
            EDGES_UNIT, beta=0.4, use_edge_weights=True,
        )
        np.testing.assert_allclose(P_default, P_opt_in)


class TestEdgeListIsNotMutated:
    def test_caller_edge_list_unchanged(self):
        """The function rebuilds the list internally; the caller's stays intact."""
        edges = list(EDGES_WEIGHTED)
        snapshot = list(edges)
        compute_similarity_matrix(edges, beta=0.4)  # use_edge_weights=False default
        assert edges == snapshot
