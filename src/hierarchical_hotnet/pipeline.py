"""High-level Hierarchical HotNet pipeline.

Each step in :func:`run_pipeline` owns its own parallelism (via the
``*_many`` / ``construct_hierarchies`` batch functions). The pipeline itself
is sequential composition: parallelism is applied within a single score set's
permutations, not across score sets.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hierarchical_hotnet.construct_hierarchy import construct_hierarchies
from hierarchical_hotnet.construct_similarity_matrix import compute_similarity_matrix
from hierarchical_hotnet.find_permutation_bins import compute_permutation_bins
from hierarchical_hotnet.perform_consensus import (
    ConsensusInput,
    ConsensusResult,
    perform_consensus,
)
from hierarchical_hotnet.permute_scores import permute_scores_many
from hierarchical_hotnet.process_hierarchies import (
    ProcessHierarchiesResult,
    process_hierarchies,
)

logger = logging.getLogger(__name__)


def _ensure_visible_logging() -> None:
    """Attach a default handler so ``verbose=True`` prints progress.

    Idempotent: if the package logger already has handlers (because the caller
    configured logging explicitly), do nothing.
    """
    pkg_logger = logging.getLogger("hierarchical_hotnet")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    if pkg_logger.level == logging.NOTSET or pkg_logger.level > logging.INFO:
        pkg_logger.setLevel(logging.INFO)


@dataclass
class PipelineResult:
    similarity_matrix: np.ndarray = field(repr=False)
    beta: float
    score_results: dict            # score_label -> ProcessHierarchiesResult
    consensus: Optional[ConsensusResult] = None


def _gene_labeled_edges(edges, index_to_gene):
    """Translate integer-indexed edges to an undirected gene-labeled pair set."""
    out = set()
    for edge in edges:
        u, v = edge[0], edge[1]
        out.add(frozenset((index_to_gene[u], index_to_gene[v])))
    return out


def run_pipeline(
    edges,
    index_to_gene,
    score_sets,
    *,
    num_permutations: int = 100,
    n_jobs: int = 1,
    seed_offset: int = 0,
    min_bin_size: float = 1000,
    beta: Optional[float] = None,
    similarity_threshold: float = 1.0,
    num_digits: int = 2,
    directed: bool = False,
    log_transform: bool = False,
    score_threshold: float = float('nan'),
    lower_size_bound: float = 10.0,
    upper_size_bound: float = float('inf'),
    consensus_threshold: Optional[int] = 2,
    verbose: bool = False,
):
    """Run the full Hierarchical HotNet pipeline for one network and N score sets.

    Each step parallelizes its own work via ``n_jobs``:

      1. Build the similarity matrix (single-shot).
      2. For each score set:
         a. Compute permutation bins.
         b. Build ``num_permutations`` permuted score sets
            (:func:`permute_scores_many`, parallel).
         c. Build the observed + permuted hierarchies
            (:func:`construct_hierarchies`, parallel).
         d. Process them into a cut + clusters (:func:`process_hierarchies`,
            internally parallel over permutations).
      3. Consensus across score sets (single-shot).

    The pipeline does not parallelize *across* score sets -- each label is
    handled sequentially. This matches typical real-world usage where one
    score set saturates available cores.

    Parameters
    ----------
    edges : iterable of (i, j, w)
        Weighted, 1-indexed edge list.
    index_to_gene : Mapping[int, str]
    score_sets : Mapping[str, Mapping[str, float]]
        Named gene-score maps.
    num_permutations, n_jobs, seed_offset, min_bin_size : see step docs.
    beta : float or None
        Pinned restart probability, or ``None`` to auto-pick.
    consensus_threshold : int or None
        ``None`` skips the consensus step.

    Returns
    -------
    PipelineResult
    """
    if not isinstance(score_sets, Mapping):
        raise TypeError('score_sets must be a mapping {label: gene_to_score}')

    if verbose:
        _ensure_visible_logging()

    edges = list(edges)

    logger.info('Building similarity matrix...')
    similarity_matrix, beta_used = compute_similarity_matrix(
        edges,
        directed=directed,
        beta=beta,
        threshold=similarity_threshold,
        num_digits=num_digits,
    )

    gene_edges = _gene_labeled_edges(edges, index_to_gene)
    unweighted_gene_edges = [tuple(sorted(e)) for e in gene_edges]
    seeds = list(range(seed_offset + 1, seed_offset + num_permutations + 1))

    score_results: dict = {}
    consensus_payload: dict = {}

    for label, gene_to_score in score_sets.items():
        gene_to_score = dict(gene_to_score)

        logger.info('[%s] Computing permutation bins...', label)
        bins = compute_permutation_bins(unweighted_gene_edges, gene_to_score, min_size=min_bin_size)

        logger.info('[%s] Permuting scores (%d permutations, n_jobs=%d)...', label, num_permutations, n_jobs)
        permuted_scores_list = permute_scores_many(
            gene_to_score, bins, seeds=seeds, n_jobs=n_jobs,
        )

        logger.info('[%s] Building %d hierarchies (n_jobs=%d)...', label, num_permutations + 1, n_jobs)
        hierarchies = construct_hierarchies(
            similarity_matrix,
            index_to_gene,
            [gene_to_score, *permuted_scores_list],
            n_jobs=n_jobs,
            log_transform=log_transform,
            score_threshold=score_threshold,
        )
        observed_T, observed_idx2gene = hierarchies[0]
        permuted_Ts = [T for T, _ in hierarchies[1:]]
        permuted_idx2genes = [g for _, g in hierarchies[1:]]

        logger.info('[%s] Processing hierarchies (n_jobs=%d)...', label, n_jobs)
        result = process_hierarchies(
            observed_T, observed_idx2gene, permuted_Ts, permuted_idx2genes,
            lower_size_bound=lower_size_bound,
            upper_size_bound=upper_size_bound,
            n_jobs=n_jobs,
        )
        score_results[label] = result

        # Consensus uses only the largest observed cluster per (network, score).
        sorted_clusters = sorted(
            sorted(map(sorted, result.observed_clusters)), key=len, reverse=True,
        )
        consensus_payload[label] = sorted_clusters[:1]

    consensus = None
    if consensus_threshold is not None and score_sets:
        logger.info('Performing consensus (threshold=%s)...', consensus_threshold)
        inputs = [
            ConsensusInput(edges=gene_edges, components=consensus_payload[label])
            for label in score_sets
        ]
        consensus = perform_consensus(inputs, threshold=consensus_threshold)

    return PipelineResult(
        similarity_matrix=similarity_matrix,
        beta=beta_used,
        score_results=score_results,
        consensus=consensus,
    )
