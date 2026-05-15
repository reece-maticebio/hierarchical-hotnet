"""Tests for DiskStore + the ScoreMap/Hierarchy codecs."""

from pathlib import Path

import pytest

from hierarchical_hotnet.storage import (
    DiskStore,
    HierarchyCodec,
    ScoreMapCodec,
    Store,
)


@pytest.fixture
def score_store(tmp_path):
    return DiskStore[dict](tmp_path / "scores", ScoreMapCodec())


@pytest.fixture
def hierarchy_store(tmp_path):
    return DiskStore[tuple](tmp_path / "hierarchies", HierarchyCodec())


class TestDiskStoreScoreMap:
    def test_put_then_get_round_trips(self, score_store):
        scores = {"GENE_A": 1.5, "GENE_B": -0.3, "GENE_C": 2.0}
        score_store.put("0", scores)
        assert score_store.get("0") == scores

    def test_writes_one_file_per_key(self, score_store, tmp_path):
        score_store.put("0", {"GENE_A": 1.0})
        score_store.put("1", {"GENE_A": 2.0})
        files = sorted((tmp_path / "scores").glob("*.tsv"))
        assert [f.name for f in files] == ["0.tsv", "1.tsv"]

    def test_overwrite_replaces_file_contents(self, score_store):
        score_store.put("0", {"GENE_A": 1.0})
        score_store.put("0", {"GENE_A": 2.0})
        assert score_store.get("0") == {"GENE_A": 2.0}

    def test_get_missing_key_raises(self, score_store):
        with pytest.raises(KeyError):
            score_store.get("missing")

    def test_contains_reflects_filesystem(self, score_store, tmp_path):
        assert "0" not in score_store
        score_store.put("0", {"GENE_A": 1.0})
        assert "0" in score_store

    def test_existing_files_are_visible_after_reopen(self, tmp_path):
        first = DiskStore[dict](tmp_path / "scores", ScoreMapCodec())
        first.put("0", {"GENE_A": 1.0})
        first.put("1", {"GENE_A": 2.0})

        second = DiskStore[dict](tmp_path / "scores", ScoreMapCodec())
        assert len(second) == 2
        assert "0" in second
        assert second.get("1") == {"GENE_A": 2.0}


class TestDiskStoreIteration:
    def test_keys_sorted_lexicographically(self, score_store):
        score_store.put("10", {"GENE_A": 10.0})
        score_store.put("2", {"GENE_A": 2.0})
        score_store.put("1", {"GENE_A": 1.0})
        # Lexicographic, not numeric. Callers that want numeric order must pad
        # their keys (e.g. "001", "002", "010") or sort themselves.
        assert list(score_store.keys()) == ["1", "10", "2"]

    def test_iter_yields_key_value_pairs(self, score_store):
        a = {"GENE_A": 1.0}
        b = {"GENE_A": 2.0}
        score_store.put("a", a)
        score_store.put("b", b)
        assert dict(score_store) == {"a": a, "b": b}


class TestHierarchyCodec:
    def test_round_trip_through_disk_store(self, hierarchy_store):
        T = [(1, 2, 0.5), (2, 3, 0.7), (3, 4, 1.0)]
        index_to_gene = {1: "A", 2: "B", 3: "C", 4: "D"}
        hierarchy_store.put("0", (T, index_to_gene))
        got_T, got_idx = hierarchy_store.get("0")
        assert got_T == T
        assert got_idx == index_to_gene

    def test_writes_two_files_per_hierarchy(self, hierarchy_store, tmp_path):
        T = [(1, 2, 0.5)]
        index_to_gene = {1: "A", 2: "B"}
        hierarchy_store.put("0", (T, index_to_gene))
        names = sorted(p.name for p in (tmp_path / "hierarchies").iterdir())
        assert names == ["0.edges.tsv", "0.genes.tsv"]

    def test_keys_uses_edges_file_as_canonical(self, hierarchy_store, tmp_path):
        """If only one of the pair exists, the store still reports the key
        based on the edges file. Pulling a partial hierarchy will then fail
        loudly inside the codec rather than silently returning garbage."""
        T = [(1, 2, 0.5)]
        index_to_gene = {1: "A", 2: "B"}
        hierarchy_store.put("0", (T, index_to_gene))
        (tmp_path / "hierarchies" / "0.genes.tsv").unlink()
        assert "0" in hierarchy_store
        with pytest.raises(FileNotFoundError):
            hierarchy_store.get("0")


class TestProtocolConformance:
    def test_disk_store_satisfies_store_protocol(self, tmp_path):
        store = DiskStore[dict](tmp_path / "s", ScoreMapCodec())
        assert isinstance(store, Store)
