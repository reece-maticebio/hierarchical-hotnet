"""Protocols for the storage abstraction.

A :class:`Store` is a keyed container of artifacts produced by a fan-out
pipeline stage. Producers write each artifact once via ``put``; consumers
iterate or retrieve by key. Implementations may keep values in memory or
spill them to disk; iteration is required to be lazy so consumers can
process N artifacts in bounded memory.

A :class:`Codec` describes how to serialize and deserialize one artifact
type to and from a path. Disk-backed stores pair with a Codec; in-memory
stores do not need one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, Iterable, Iterator, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Store(Protocol, Generic[T]):
    """Keyed container of artifacts, written once and read on demand.

    Producers call :meth:`put` for each artifact. Consumers iterate the store
    (yielding ``(key, value)`` pairs in deterministic order) or call
    :meth:`get` for random access. Iteration must be lazy: a disk-backed
    store must not materialise all values up front.

    Keys are strings so they map cleanly to filenames in disk-backed
    implementations. ``put`` may overwrite an existing key.
    """

    def put(self, key: str, value: T) -> None: ...

    def get(self, key: str) -> T: ...

    def keys(self) -> Iterable[str]: ...

    def __iter__(self) -> Iterator[tuple[str, T]]: ...

    def __len__(self) -> int: ...

    def __contains__(self, key: str) -> bool: ...

    def close(self) -> None:
        """Release any underlying resources. Idempotent."""
        ...


@runtime_checkable
class Codec(Protocol, Generic[T]):
    """Serialize and deserialize one artifact type to and from a file path.

    ``extension`` includes the leading dot (e.g. ``".tsv"``) and is used by
    disk-backed stores to name the file for a given key.
    """

    extension: str

    def write(self, value: T, path: Path) -> None: ...

    def read(self, path: Path) -> T: ...
