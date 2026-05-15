"""Disk-backed implementation of the Store protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Generic, Iterable, Iterator, TypeVar

from hierarchical_hotnet.storage.base import Codec

T = TypeVar("T")


class DiskStore(Generic[T]):
    """Filesystem-backed :class:`Store`.

    Each ``put(key, value)`` serializes ``value`` via the codec to
    ``<path>/<key><codec.extension>``. ``get`` deserializes on demand.
    Iteration is lazy: keys are listed from the filesystem and each value
    is decoded only when the consumer pulls it from the iterator.

    The directory is created on construction if it does not already exist.
    Existing files in the directory are visible immediately, which is the
    mechanism the pipeline's ``reuse=True`` mode relies on.
    """

    def __init__(self, path: Path | str, codec: Codec[T]) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.codec = codec

    def _file_for(self, key: str) -> Path:
        return self.path / f"{key}{self.codec.extension}"

    def put(self, key: str, value: T) -> None:
        self.codec.write(value, self._file_for(key))

    def get(self, key: str) -> T:
        path = self._file_for(key)
        if not path.exists():
            raise KeyError(key)
        return self.codec.read(path)

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def keys(self) -> Iterable[str]:
        ext = self.codec.extension
        for p in sorted(self.path.glob(f"*{ext}")):
            yield p.name[: -len(ext)]

    def __iter__(self) -> Iterator[tuple[str, T]]:
        for key in self.keys():
            yield key, self.codec.read(self._file_for(key))

    def __len__(self) -> int:
        return sum(1 for _ in self.path.glob(f"*{self.codec.extension}"))

    def __contains__(self, key: str) -> bool:
        return self._file_for(key).exists()

    def close(self) -> None:
        pass
