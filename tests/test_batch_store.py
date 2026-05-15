"""Tests for the new Store paths on permute_scores_many and construct_hierarchies.

The historical list-returning behavior is exercised by test_batch_functions.py.
These tests cover the parallel branch (out=Store): same results, written into
the store keyed by seed (for permute_scores) or input index (for hierarchies).
"""

import pytest

from hierarchical_hotnet.construct_hierarchy import construct_hierarchies
from hierarchical_hotnet.permute_scores import permute_scores_many
from hierarchical_hotnet.storage import MemoryStore


# --- shared fixtures: a tiny adjacency-derived similarity matrix --------------


@pytest.fixture
def tiny_inputs():
    """Minimal inputs for construct_hierarchies and permute_scores_many."""
    import numpy as np

    from hierarchical_hotnet.construct_similarity_matrix import compute_similarity_matrix

    edges = [(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0), (3, 4, 1.0), (4, 3, 1.0)]
    P, _ = compute_similarity_matrix(edges, beta=0.4)
    index_to_gene = {1: "A", 2: "B", 3: "C", 4: "D"}
    gene_to_score = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    bins = [["A", "B", "C", "D"]]
    return np.asarray(P), index_to_gene, gene_to_score, bins


# --- permute_scores_many ------------------------------------------------------


class TestPermuteScoresManyStore:
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_store_path_writes_one_entry_per_seed(self, tiny_inputs, n_jobs):
        _, _, gene_to_score, bins = tiny_inputs
        store: MemoryStore = MemoryStore()
        seeds = [1, 2, 3]
        returned = permute_scores_many(gene_to_score, bins, seeds=seeds, n_jobs=n_jobs, out=store)
        assert returned is store
        assert set(store.keys()) == {"1", "2", "3"}

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_store_path_matches_list_path(self, tiny_inputs, n_jobs):
        _, _, gene_to_score, bins = tiny_inputs
        seeds = [1, 2, 3]

        list_result = permute_scores_many(gene_to_score, bins, seeds=seeds, n_jobs=n_jobs)

        store: MemoryStore = MemoryStore()
        permute_scores_many(gene_to_score, bins, seeds=seeds, n_jobs=n_jobs, out=store)

        assert [store.get(str(s)) for s in seeds] == list_result


# --- construct_hierarchies ----------------------------------------------------


class TestConstructHierarchiesStore:
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_store_path_writes_one_entry_per_input(self, tiny_inputs, n_jobs):
        P, idx_to_gene, gs, _ = tiny_inputs
        store: MemoryStore = MemoryStore()
        returned = construct_hierarchies(P, idx_to_gene, [gs, gs, gs], n_jobs=n_jobs, out=store)
        assert returned is store
        assert set(store.keys()) == {"0", "1", "2"}

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_store_path_matches_list_path(self, tiny_inputs, n_jobs):
        P, idx_to_gene, gs, _ = tiny_inputs

        list_result = construct_hierarchies(P, idx_to_gene, [gs, gs], n_jobs=n_jobs)

        store: MemoryStore = MemoryStore()
        construct_hierarchies(P, idx_to_gene, [gs, gs], n_jobs=n_jobs, out=store)

        for i in range(2):
            T_list, idx_list = list_result[i]
            T_store, idx_store = store.get(str(i))
            assert T_list == T_store
            assert idx_list == idx_store
