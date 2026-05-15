"""Hierarchical HotNet: identifying hierarchies of altered subnetworks."""

__version__ = "0.2.0"

from hierarchical_hotnet.core.similarity import compute_similarity_matrix
from hierarchical_hotnet.core.bins import compute_permutation_bins
from hierarchical_hotnet.core.permute_scores import permute_scores, permute_scores_many
from hierarchical_hotnet.core.permute_network import permute_network, permute_network_many
from hierarchical_hotnet.core.hierarchy import construct_hierarchy, construct_hierarchies
from hierarchical_hotnet.core.process_hierarchies import (
    process_hierarchies,
    ProcessHierarchiesResult,
)
from hierarchical_hotnet.core.consensus import (
    perform_consensus,
    ConsensusInput,
    ConsensusResult,
)
from hierarchical_hotnet.pipeline import run_pipeline, PipelineResult
from hierarchical_hotnet.config import (
    ConsensusConfig,
    HierarchyConfig,
    PermutationConfig,
    RuntimeConfig,
    ScoringConfig,
    SimilarityConfig,
)
from hierarchical_hotnet.file_io import (
    load_edge_list,
    load_gene_score,
    load_index_gene,
    load_matrix,
    save_edge_list,
    save_gene_score,
    save_index_gene,
    save_matrix,
)

__all__ = [
    "__version__",
    # building blocks
    "compute_similarity_matrix",
    "compute_permutation_bins",
    "permute_scores",
    "permute_scores_many",
    "permute_network",
    "permute_network_many",
    "construct_hierarchy",
    "construct_hierarchies",
    "process_hierarchies",
    "perform_consensus",
    # high-level
    "run_pipeline",
    # result types
    "PipelineResult",
    "ProcessHierarchiesResult",
    "ConsensusInput",
    "ConsensusResult",
    # configs
    "SimilarityConfig",
    "ScoringConfig",
    "PermutationConfig",
    "HierarchyConfig",
    "ConsensusConfig",
    "RuntimeConfig",
    # IO helpers
    "load_edge_list",
    "load_gene_score",
    "load_index_gene",
    "load_matrix",
    "save_edge_list",
    "save_gene_score",
    "save_index_gene",
    "save_matrix",
]
