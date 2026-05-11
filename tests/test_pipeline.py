"""End-to-end pipeline test: matches the canonical example_commands.sh outputs."""

import pytest

import hierarchical_hotnet as hhn


def _load_inputs(data_dir):
    index_to_gene, _ = hhn.load_index_gene(data_dir / "network_1_index_gene.tsv")
    edges = hhn.load_edge_list(data_dir / "network_1_edge_list.tsv")
    scores_1 = hhn.load_gene_score(data_dir / "scores_1.tsv")
    scores_2 = hhn.load_gene_score(data_dir / "scores_2.tsv")
    return index_to_gene, edges, scores_1, scores_2


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_pipeline_matches_canonical_consensus(data_dir, reference_dir, n_jobs):
    index_to_gene, edges, scores_1, scores_2 = _load_inputs(data_dir)

    result = hhn.run_pipeline(
        edges,
        index_to_gene,
        {"scores_1": scores_1, "scores_2": scores_2},
        num_permutations=100,
        n_jobs=n_jobs,
        min_bin_size=1000,
        lower_size_bound=1,           # toy example
        consensus_threshold=2,
    )

    expected_nodes = (reference_dir / "example_consensus_nodes.tsv").read_text().strip()
    expected_edges = (reference_dir / "example_consensus_edges.tsv").read_text().strip()

    actual_nodes = "\n".join("\t".join(group) for group in result.consensus.nodes)
    actual_edges = "\n".join("\t".join(edge) for edge in result.consensus.edges)

    assert actual_nodes == expected_nodes
    assert actual_edges == expected_edges


def test_pipeline_result_shape(data_dir):
    index_to_gene, edges, scores_1, scores_2 = _load_inputs(data_dir)
    result = hhn.run_pipeline(
        edges,
        index_to_gene,
        {"scores_1": scores_1, "scores_2": scores_2},
        num_permutations=10,
        n_jobs=1,
        min_bin_size=1000,
        lower_size_bound=1,
    )

    assert set(result.score_results.keys()) == {"scores_1", "scores_2"}
    for label, res in result.score_results.items():
        assert 0.0 <= res.p_value <= 1.0
        assert res.observed_cut_size >= 1
        assert isinstance(res.observed_clusters, set)
    assert result.consensus is not None
    assert isinstance(result.consensus.nodes, list)
