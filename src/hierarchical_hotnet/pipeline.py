"""High-level Hierarchical HotNet pipeline.

Each step in :func:`run_pipeline` owns its own parallelism (via the
``*_many`` / ``construct_hierarchies`` batch functions). The pipeline itself
is sequential composition: parallelism is applied within a single score set's
permutations, not across score sets.

Disk-backed runs
----------------
When ``workdir`` is set, every artifact is written to disk under a fixed
layout::

    <workdir>/similarity_matrix.h5
    <workdir>/beta.txt
    <workdir>/bins/<label>.tsv
    <workdir>/permuted_scores/<label>/<seed>.tsv
    <workdir>/hierarchies/<label>/<i>.edges.tsv  (+ .genes.tsv)

``reuse=True`` checks each artifact path individually: existing files are
read back instead of recomputed, missing ones are computed normally. This
makes interrupted runs resumable without external bookkeeping.
"""

import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from hierarchical_hotnet.construct_hierarchy import construct_hierarchies
from hierarchical_hotnet.construct_similarity_matrix import compute_similarity_matrix
from hierarchical_hotnet.file_io import load_matrix, save_matrix
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
from hierarchical_hotnet.storage import (
    DiskStore,
    HierarchyCodec,
    MemoryStore,
    ScoreMapCodec,
    Store,
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


def _save_bins(path: Path, bins: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(b) for b in bins) + "\n")


def _load_bins(path: Path) -> list:
    bins = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            bins.append(line.split("\t"))
    return bins


def _scores_store(workdir: Optional[Path], label: str) -> Store:
    """Return a Store backing the permuted-scores fan-out for one label."""
    if workdir is None:
        return MemoryStore()
    return DiskStore[dict](workdir / "permuted_scores" / label, ScoreMapCodec())


def _hierarchies_store(workdir: Optional[Path], label: str) -> Store:
    """Return a Store backing the hierarchies fan-out for one label."""
    if workdir is None:
        return MemoryStore()
    return DiskStore(workdir / "hierarchies" / label, HierarchyCodec())


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
    workdir: Optional[Path] = None,
    reuse: bool = False,
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
    workdir : Path or None
        If set, every artifact is written under this directory (see module
        docstring for the layout). The fan-out artifacts (permuted scores and
        hierarchies) are streamed through :class:`DiskStore`, so peak memory
        is bounded even with large permutation counts. ``None`` keeps every
        artifact in memory.
    reuse : bool
        Only meaningful with ``workdir``. When ``True``, each artifact path is
        checked individually; existing files are read back instead of being
        recomputed. Lets interrupted runs resume without external state.

    Returns
    -------
    PipelineResult
    """
    if not isinstance(score_sets, Mapping):
        raise TypeError('score_sets must be a mapping {label: gene_to_score}')
    if reuse and workdir is None:
        raise ValueError("reuse=True requires workdir= to be set")

    if verbose:
        _ensure_visible_logging()

    workdir = Path(workdir) if workdir is not None else None
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)

    edges = list(edges)

    # 1. Similarity matrix --------------------------------------------------
    sim_path = workdir / "similarity_matrix.h5" if workdir else None
    beta_path = workdir / "beta.txt" if workdir else None

    if reuse and sim_path is not None and sim_path.exists() and beta_path.exists():
        logger.info("Loading similarity matrix from %s", sim_path)
        similarity_matrix = load_matrix(str(sim_path))
        beta_used = float(beta_path.read_text().strip())
    else:
        logger.info('Building similarity matrix...')
        similarity_matrix, beta_used = compute_similarity_matrix(
            edges,
            directed=directed,
            beta=beta,
            threshold=similarity_threshold,
            num_digits=num_digits,
        )
        if workdir is not None:
            save_matrix(str(sim_path), similarity_matrix)
            # repr() round-trips a float losslessly so reuse=True returns the
            # exact same PipelineResult.beta as the first run. CLI-style
            # rounded output (using num_digits) is a separate concern handled
            # by hhnet-construct-similarity-matrix.
            beta_path.write_text(repr(beta_used) + "\n")

    gene_edges = _gene_labeled_edges(edges, index_to_gene)
    unweighted_gene_edges = [tuple(sorted(e)) for e in gene_edges]
    seeds = list(range(seed_offset + 1, seed_offset + num_permutations + 1))

    score_results: dict = {}
    consensus_payload: dict = {}

    # 2. Per-score-set processing ------------------------------------------
    for label, gene_to_score in score_sets.items():
        gene_to_score = dict(gene_to_score)

        # 2a. Bins
        bins_path = workdir / "bins" / f"{label}.tsv" if workdir else None
        if reuse and bins_path is not None and bins_path.exists():
            logger.info('[%s] Loading permutation bins from %s', label, bins_path)
            bins = _load_bins(bins_path)
        else:
            logger.info('[%s] Computing permutation bins...', label)
            bins = compute_permutation_bins(unweighted_gene_edges, gene_to_score, min_size=min_bin_size)
            if bins_path is not None:
                _save_bins(bins_path, bins)

        # 2b. Permuted scores: skip seeds already present when reuse=True
        scores_store = _scores_store(workdir, label)
        missing_seeds = [s for s in seeds if not (reuse and str(s) in scores_store)]
        if missing_seeds:
            logger.info(
                '[%s] Permuting scores (%d/%d permutations, n_jobs=%d)...',
                label, len(missing_seeds), num_permutations, n_jobs,
            )
            permute_scores_many(
                gene_to_score, bins, seeds=missing_seeds, n_jobs=n_jobs, out=scores_store,
            )
        else:
            logger.info('[%s] All %d permuted scores found on disk', label, num_permutations)

        # 2c. Hierarchies: observed at key "0", permutations at "1".."N".
        # Only compute the keys that aren't already in the store.
        hier_store = _hierarchies_store(workdir, label)
        all_keys = [str(i) for i in range(num_permutations + 1)]
        to_compute_keys: list = []
        to_compute_sets: list = []
        for i, key in enumerate(all_keys):
            if reuse and key in hier_store:
                continue
            if i == 0:
                to_compute_sets.append(gene_to_score)
            else:
                to_compute_sets.append(scores_store[str(seeds[i - 1])])
            to_compute_keys.append(key)

        if to_compute_keys:
            logger.info(
                '[%s] Building %d/%d hierarchies (n_jobs=%d)...',
                label, len(to_compute_keys), num_permutations + 1, n_jobs,
            )
            construct_hierarchies(
                similarity_matrix,
                index_to_gene,
                to_compute_sets,
                keys=to_compute_keys,
                n_jobs=n_jobs,
                log_transform=log_transform,
                score_threshold=score_threshold,
                out=hier_store,
            )
        else:
            logger.info('[%s] All %d hierarchies found on disk', label, num_permutations + 1)

        # 2d. Process: pull observed once; iterate permutations lazily so we
        # don't materialize all N+1 hierarchies at once (DiskStore mode).
        logger.info('[%s] Processing hierarchies (n_jobs=%d)...', label, n_jobs)
        observed_T, observed_idx2gene = hier_store["0"]
        permuted_pairs = (hier_store[str(i)] for i in range(1, num_permutations + 1))
        permuted_Ts_src, permuted_idx_src = itertools.tee(permuted_pairs, 2)
        permuted_Ts = (p[0] for p in permuted_Ts_src)
        permuted_idx_list = (p[1] for p in permuted_idx_src)

        result = process_hierarchies(
            observed_T,
            observed_idx2gene,
            permuted_Ts,
            permuted_idx_list,
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

    # 3. Consensus ---------------------------------------------------------
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
