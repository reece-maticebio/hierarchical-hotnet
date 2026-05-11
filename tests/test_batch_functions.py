"""Each step's batch function should equal serial application + work in parallel."""

import numpy as np
import pytest

import hierarchical_hotnet as hhn


def _load(data_dir):
    index_to_gene, _ = hhn.load_index_gene(data_dir / "network_1_index_gene.tsv")
    edges = hhn.load_edge_list(data_dir / "network_1_edge_list.tsv")
    scores = hhn.load_gene_score(data_dir / "scores_1.tsv")
    return index_to_gene, edges, scores


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_permute_scores_many_matches_serial(data_dir, n_jobs):
    _, _, scores = _load(data_dir)
    seeds = [1, 2, 3, 4, 5]

    expected = [hhn.permute_scores(scores, seed=s) for s in seeds]
    got = hhn.permute_scores_many(scores, seeds=seeds, n_jobs=n_jobs)

    assert len(got) == len(seeds)
    for e, g in zip(expected, got):
        assert e == g


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_permute_network_many_matches_serial(data_dir, n_jobs):
    _, edges, _ = _load(data_dir)
    unweighted = [(e[0], e[1]) for e in edges]
    seeds = [1, 2, 3]

    expected = [hhn.permute_network(unweighted, seed=s, Q=10) for s in seeds]
    got = hhn.permute_network_many(unweighted, seeds=seeds, n_jobs=n_jobs, Q=10)

    assert len(got) == len(seeds)
    for e, g in zip(expected, got):
        assert sorted(map(tuple, map(sorted, e))) == sorted(map(tuple, map(sorted, g)))


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_construct_hierarchies_matches_serial(data_dir, n_jobs):
    index_to_gene, edges, scores = _load(data_dir)
    P, _ = hhn.compute_similarity_matrix(edges)

    # Two distinct score sets: original + a permutation.
    permuted = hhn.permute_scores(scores, seed=42)
    score_sets = [scores, permuted]

    expected = [
        hhn.construct_hierarchy(P, index_to_gene, gene_to_score=s)
        for s in score_sets
    ]
    got = hhn.construct_hierarchies(P, index_to_gene, score_sets, n_jobs=n_jobs)

    assert len(got) == 2
    for (eT, eIdx), (gT, gIdx) in zip(expected, got):
        assert eIdx == gIdx
        assert len(eT) == len(gT)
        # Heights are stable; topology should match exactly.
        assert sorted((s, t, float(h)) for s, t, h in eT) == sorted((s, t, float(h)) for s, t, h in gT)
