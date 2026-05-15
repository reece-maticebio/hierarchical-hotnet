"""Tests for the storage abstraction (Store protocol + MemoryStore)."""

import pytest

from hierarchical_hotnet.storage import MemoryStore, Store


class TestMemoryStoreBasics:
    def test_put_then_get_round_trips(self):
        store = MemoryStore[int]()
        store.put("a", 1)
        assert store.get("a") == 1

    def test_get_missing_key_raises_key_error(self):
        store = MemoryStore[int]()
        with pytest.raises(KeyError):
            store.get("missing")

    def test_len_reflects_put_count(self):
        store = MemoryStore[int]()
        assert len(store) == 0
        store.put("a", 1)
        store.put("b", 2)
        assert len(store) == 2

    def test_overwrite_replaces_value_without_growing(self):
        store = MemoryStore[int]()
        store.put("a", 1)
        store.put("a", 2)
        assert store.get("a") == 2
        assert len(store) == 1

    def test_contains(self):
        store = MemoryStore[int]()
        store.put("a", 1)
        assert "a" in store
        assert "b" not in store

    def test_close_is_no_op(self):
        store = MemoryStore[int]()
        store.put("a", 1)
        store.close()
        assert store.get("a") == 1


class TestMemoryStoreIteration:
    def test_iter_yields_insertion_order(self):
        store = MemoryStore[int]()
        store.put("c", 3)
        store.put("a", 1)
        store.put("b", 2)
        assert list(store) == [("c", 3), ("a", 1), ("b", 2)]

    def test_keys_returns_inserted_keys(self):
        store = MemoryStore[int]()
        store.put("a", 1)
        store.put("b", 2)
        assert set(store.keys()) == {"a", "b"}

    def test_iteration_after_overwrite_keeps_original_position(self):
        store = MemoryStore[int]()
        store.put("a", 1)
        store.put("b", 2)
        store.put("a", 99)
        assert list(store) == [("a", 99), ("b", 2)]


class TestMemoryStoreValues:
    def test_holds_arbitrary_objects(self):
        store = MemoryStore[dict]()
        score_map = {"GENE_A": 1.5, "GENE_B": 2.3}
        store.put("perm_0", score_map)
        assert store.get("perm_0") is score_map

    def test_values_are_not_copied(self):
        store = MemoryStore[list]()
        values = [1, 2, 3]
        store.put("k", values)
        values.append(4)
        assert store.get("k") == [1, 2, 3, 4]


class TestProtocolConformance:
    def test_memory_store_satisfies_store_protocol(self):
        store: Store[int] = MemoryStore[int]()
        assert isinstance(store, Store)

    def test_plain_dict_does_not_satisfy_store_protocol(self):
        """Guard against false-positive structural matches with builtin dict."""
        assert not isinstance({}, Store)
