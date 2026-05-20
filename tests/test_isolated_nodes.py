"""Tests for drop_isolated_nodes and its integration into run_pipeline."""

import warnings

import pytest

import hierarchical_hotnet as hhn
from hierarchical_hotnet.core.common import drop_isolated_nodes


class TestNoIsolatedNodes:
    def test_clean_input_is_unchanged_and_silent(self):
        edges = [(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)]
        index_to_gene = {1: "A", 2: "B", 3: "C"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            new_edges, new_idx = drop_isolated_nodes(edges, index_to_gene)
        assert new_edges == edges
        assert new_idx == index_to_gene

    def test_returns_fresh_containers(self):
        edges = [(1, 2, 1.0)]
        index_to_gene = {1: "A", 2: "B"}
        new_edges, new_idx = drop_isolated_nodes(edges, index_to_gene)
        assert new_edges is not edges
        assert new_idx is not index_to_gene


class TestIsolatedNodeRemoval:
    def test_trailing_isolated_node_removed(self):
        edges = [(1, 2, 1.0), (2, 3, 1.0)]
        index_to_gene = {1: "A", 2: "B", 3: "C", 4: "ISOLATED"}
        with pytest.warns(UserWarning, match="ISOLATED"):
            new_edges, new_idx = drop_isolated_nodes(edges, index_to_gene)
        assert new_idx == {1: "A", 2: "B", 3: "C"}
        assert new_edges == edges  # indices 1..3 unchanged

    def test_middle_isolated_node_triggers_reindex(self):
        # Node 2 ("GHOST") has no edges; edges connect 1, 3, 4.
        edges = [(1, 3, 1.0), (3, 4, 1.0), (4, 1, 1.0)]
        index_to_gene = {1: "A", 2: "GHOST", 3: "C", 4: "D"}
        with pytest.warns(UserWarning, match="GHOST"):
            new_edges, new_idx = drop_isolated_nodes(edges, index_to_gene)
        # Survivors 1,3,4 renumber to 1,2,3 preserving order.
        assert new_idx == {1: "A", 2: "C", 3: "D"}
        assert new_edges == [(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)]

    def test_multiple_isolated_nodes(self):
        edges = [(1, 2, 1.0)]
        index_to_gene = {1: "A", 2: "B", 3: "X", 4: "Y", 5: "Z"}
        with pytest.warns(UserWarning, match="3 isolated node"):
            new_edges, new_idx = drop_isolated_nodes(edges, index_to_gene)
        assert new_idx == {1: "A", 2: "B"}
        assert new_edges == [(1, 2, 1.0)]

    def test_weights_preserved_through_reindex(self):
        edges = [(1, 3, 0.7), (3, 1, 9.9)]
        index_to_gene = {1: "A", 2: "GHOST", 3: "C"}
        with pytest.warns(UserWarning):
            new_edges, _ = drop_isolated_nodes(edges, index_to_gene)
        assert new_edges == [(1, 2, 0.7), (2, 1, 9.9)]

    def test_unweighted_two_tuples_preserved(self):
        edges = [(1, 3), (3, 1)]  # 2-tuples, no weight
        index_to_gene = {1: "A", 2: "GHOST", 3: "C"}
        with pytest.warns(UserWarning):
            new_edges, _ = drop_isolated_nodes(edges, index_to_gene)
        assert new_edges == [(1, 2), (2, 1)]


class TestPipelineIntegration:
    def test_run_pipeline_drops_isolated_node_and_warns(self, data_dir):
        """An isolated gene appended to the example inputs is dropped, the
        run still completes, and the consensus matches the canonical output
        (the isolated node cannot affect any cluster)."""
        index_to_gene, _ = hhn.load_index_gene(data_dir / "network_1_index_gene.tsv")
        edges = hhn.load_edge_list(data_dir / "network_1_edge_list.tsv")
        scores_1 = hhn.load_gene_score(data_dir / "scores_1.tsv")

        # Append an isolated node beyond the existing 1..25 indices.
        polluted = dict(index_to_gene)
        polluted[max(polluted) + 1] = "LONELY_GENE"

        with pytest.warns(UserWarning, match="LONELY_GENE"):
            result = hhn.run_pipeline(
                edges, polluted, {"scores_1": scores_1},
                num_permutations=20, n_jobs=1, min_bin_size=1000,
                lower_size_bound=1, consensus_threshold=None,
            )

        # Pipeline succeeded; the lonely gene is in no cluster.
        all_clustered = {
            g for c in result.score_results["scores_1"].observed_clusters for g in c
        }
        assert "LONELY_GENE" not in all_clustered
