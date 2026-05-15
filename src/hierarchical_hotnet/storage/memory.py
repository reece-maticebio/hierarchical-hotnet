"""In-memory implementation of the Store protocol."""

from __future__ import annotations

from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class MemoryStore(Generic[T]):
    """Dict-backed :class:`Store` for tests and small runs.

    Values are held by reference (no copies). Iteration follows insertion
    order. ``close`` is a no-op; the store remains usable after it is called.
    """

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def put(self, key: str, value: T) -> None:
        self._items[key] = value

    def get(self, key: str) -> T:
        return self._items[key]

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(self._items.items())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def close(self) -> None:
        pass
