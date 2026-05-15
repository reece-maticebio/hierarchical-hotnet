"""Storage abstractions for fan-out pipeline artifacts.

Stages that produce N artifacts (one per permutation) write through a
:class:`Store`. The same stage works against an in-memory store (tests, small
runs) or a disk-backed store (large runs that would otherwise overflow RAM)
without changing the stage's code.

Consumers iterate the store lazily, so the pipeline never needs to hold all
N artifacts in memory at once.
"""

from hierarchical_hotnet.storage.base import Codec, Store
from hierarchical_hotnet.storage.memory import MemoryStore

__all__ = ["Codec", "MemoryStore", "Store"]
