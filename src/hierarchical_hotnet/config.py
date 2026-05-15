"""Typed configuration objects for the Hierarchical HotNet pipeline.

These dataclasses group the flat parameters currently accepted by
:func:`hierarchical_hotnet.pipeline.run_pipeline` into the conceptual blocks
they belong to. They are additive: ``run_pipeline`` still takes the flat
kwargs, and nothing in the existing pipeline consumes these types yet. They
exist so that later refactor waves (Store integration, CLI extraction) have
a stable shape to accept and pass around instead of growing the flat
parameter list further.

Defaults match the current :func:`run_pipeline` defaults exactly. Changing
a default here is a behavior change and must be matched in ``run_pipeline``
until the flat kwargs are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimilarityConfig:
    """Parameters controlling the diffusion similarity matrix construction.

    ``beta`` is the random-walk restart probability. ``None`` means
    auto-select via the procedure in :mod:`construct_similarity_matrix`.
    """

    beta: Optional[float] = None
    similarity_threshold: float = 1.0
    num_digits: int = 2
    directed: bool = False


@dataclass
class ScoringConfig:
    """Per-score-set preprocessing applied before hierarchy construction."""

    log_transform: bool = False
    score_threshold: float = float("nan")


@dataclass
class PermutationConfig:
    """How permuted score sets are generated for the null distribution.

    ``seed_offset`` shifts the seed sequence; permutation ``i`` uses seed
    ``seed_offset + i + 1``. ``min_bin_size`` is the lower bound on the
    bin sizes used for degree-preserving score permutation.
    """

    num_permutations: int = 100
    seed_offset: int = 0
    min_bin_size: float = 1000


@dataclass
class HierarchyConfig:
    """Cluster-size bounds used when selecting the cut of each hierarchy."""

    lower_size_bound: float = 10.0
    upper_size_bound: float = float("inf")


@dataclass
class ConsensusConfig:
    """Cross-score-set consensus parameters.

    ``threshold`` is the minimum number of score sets in which a node or
    edge must appear to be retained in the consensus. ``None`` disables
    the consensus stage entirely.
    """

    threshold: Optional[int] = 2


@dataclass
class RuntimeConfig:
    """Execution-time knobs (parallelism, logging)."""

    n_jobs: int = 1
    verbose: bool = False
