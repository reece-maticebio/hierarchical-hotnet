"""Smoke tests for the pure-function API of each step."""

import numpy as np
import pytest

import hierarchical_hotnet as hhn


def _load_inputs(data_dir):
    index_to_gene, gene_to_index = hhn.load_index_gene(data_dir / "network_1_index_gene.tsv")
    edges = hhn.load_edge_list(data_dir / "network_1_edge_list.tsv")
    scores_1 = hhn.load_gene_score(data_dir / "scores_1.tsv")
    scores_2 = hhn.load_gene_score(data_dir / "scores_2.tsv")
    return index_to_gene, gene_to_index, edges, scores_1, scores_2


def test_compute_similarity_matrix(data_dir):
    _, _, edges, _, _ = _load_inputs(data_dir)
    P, beta = hhn.compute_similarity_matrix(edges)
    n = max(max(e[0], e[1]) for e in edges)
    assert P.shape == (n, n)
    assert 0 < beta < 1


def test_permute_scores_is_deterministic(data_dir):
    _, _, _, scores_1, _ = _load_inputs(data_dir)
    p1 = hhn.permute_scores(scores_1, seed=42)
    p2 = hhn.permute_scores(scores_1, seed=42)
    assert p1 == p2
    p3 = hhn.permute_scores(scores_1, seed=43)
    assert p3 != p1  # different seed → different output
    # same gene set, just reshuffled values
    assert sorted(p1.values()) == sorted(scores_1.values())


def test_permute_network_preserves_edge_count(data_dir):
    _, _, edges, _, _ = _load_inputs(data_dir)
    unweighted = [(e[0], e[1]) for e in edges]
    permuted = hhn.permute_network(unweighted, seed=1, preserve_connectivity=True, Q=10)
    assert len(permuted) == len(unweighted)


def test_compute_permutation_bins(data_dir):
    index_to_gene, _, edges, scores_1, _ = _load_inputs(data_dir)
    gene_edges = [(index_to_gene[i], index_to_gene[j]) for i, j, _ in edges]
    bins = hhn.compute_permutation_bins(gene_edges, scores_1, min_size=1000)
    # All binned genes are scored.
    binned = {gene for b in bins for gene in b}
    assert binned.issubset(scores_1.keys())


def test_construct_hierarchy(data_dir):
    index_to_gene, _, edges, scores_1, _ = _load_inputs(data_dir)
    P, _ = hhn.compute_similarity_matrix(edges)
    T, common_index_to_gene = hhn.construct_hierarchy(P, index_to_gene, gene_to_score=scores_1)
    assert len(T) >= len(common_index_to_gene) - 1
    heights = [h for _, _, h in T]
    assert all(np.isfinite(h) for h in heights)
