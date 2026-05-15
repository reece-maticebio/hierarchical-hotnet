"""Tests for the config dataclasses.

Defaults are pinned because they must stay in lockstep with
:func:`hierarchical_hotnet.pipeline.run_pipeline` until the flat kwargs are
removed. If a default changes here without changing ``run_pipeline`` (or
vice versa), behavior will silently diverge.
"""

import math

from hierarchical_hotnet import (
    ConsensusConfig,
    HierarchyConfig,
    PermutationConfig,
    RuntimeConfig,
    ScoringConfig,
    SimilarityConfig,
)


class TestSimilarityConfigDefaults:
    def test_defaults_match_run_pipeline(self):
        cfg = SimilarityConfig()
        assert cfg.beta is None
        assert cfg.similarity_threshold == 1.0
        assert cfg.num_digits == 2
        assert cfg.directed is False


class TestScoringConfigDefaults:
    def test_defaults_match_run_pipeline(self):
        cfg = ScoringConfig()
        assert cfg.log_transform is False
        assert math.isnan(cfg.score_threshold)


class TestPermutationConfigDefaults:
    def test_defaults_match_run_pipeline(self):
        cfg = PermutationConfig()
        assert cfg.num_permutations == 100
        assert cfg.seed_offset == 0
        assert cfg.min_bin_size == 1000


class TestHierarchyConfigDefaults:
    def test_defaults_match_run_pipeline(self):
        cfg = HierarchyConfig()
        assert cfg.lower_size_bound == 10.0
        assert math.isinf(cfg.upper_size_bound)


class TestConsensusConfigDefaults:
    def test_defaults_match_run_pipeline(self):
        cfg = ConsensusConfig()
        assert cfg.threshold == 2


class TestRuntimeConfigDefaults:
    def test_defaults_match_run_pipeline(self):
        cfg = RuntimeConfig()
        assert cfg.n_jobs == 1
        assert cfg.verbose is False


class TestConfigsAreOverridable:
    def test_similarity_overrides(self):
        cfg = SimilarityConfig(beta=0.5, directed=True)
        assert cfg.beta == 0.5
        assert cfg.directed is True
        assert cfg.num_digits == 2  # unchanged default

    def test_consensus_threshold_can_disable(self):
        cfg = ConsensusConfig(threshold=None)
        assert cfg.threshold is None
