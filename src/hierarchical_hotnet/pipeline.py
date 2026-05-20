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

from hierarchical_hotnet.core.hierarchy import construct_hierarchies
from hierarchical_hotnet.core.similarity import compute_similarity_matrix
from hierarchical_hotnet.core.common import drop_isolated_nodes
from hierarchical_hotnet.file_io import load_matrix, save_matrix
from hierarchical_hotnet.core.bins import compute_permutation_bins
from hierarchical_hotnet.core.consensus import (
    ConsensusInput,
    ConsensusResult,
    perform_consensus,
)
from hierarchical_hotnet.core.permute_scores import permute_scores_many
from hierarchical_hotnet.core.process_hierarchies import (
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


def save_pipeline_result(
    result: "PipelineResult",
    workdir: Path,
    *,
    plot: bool = False,
) -> None:
    """Save the final outputs from a :class:`PipelineResult` under ``<workdir>/results/``.

    The on-disk format mirrors what ``hhnet process-hierarchies`` and
    ``hhnet perform-consensus`` produce, so a workdir from
    :func:`run_pipeline` is interoperable with anything downstream that
    consumes the CLI output.

    Per score-set:

      * ``<label>_clusters.tsv`` — header comments (cut height, p-value,
        ratios) followed by one cluster per line, tab-separated genes,
        sorted by cluster size descending.
      * ``<label>_observed_sizes.tsv`` — ``height\\tlargest_cluster_size``
        for the observed hierarchy.
      * ``<label>_permuted_sizes.tsv`` —
        ``height\\tmin\\tmean(expected)\\tmax`` aggregated across permutations.
      * ``<label>_sizes.pdf`` — observed vs expected size plot (only when
        ``plot=True``; requires matplotlib).

    Consensus (when present): ``consensus_nodes.tsv``, ``consensus_edges.tsv``.
    """
    workdir = Path(workdir)
    results_dir = workdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for label, r in result.score_results.items():
        sorted_clusters = sorted(
            sorted(map(sorted, r.observed_clusters)), key=len, reverse=True,
        )
        cluster_lines = "\n".join("\t".join(c) for c in sorted_clusters)
        header = (
            f"# Observed cut height: {r.observed_cut_height}\n"
            f"# Observed size of largest cluster at observed cut height: {r.observed_cut_size}\n"
            f"# Expected size of largest cluster at observed cut height: {r.expected_cut_size}\n"
            f"# Observed maximum ratio statistic: {r.observed_cut_ratio:.3f}\n"
            f"# Expected maximum ratio statistic: {r.expected_cut_ratio:.3f}\n"
            f"# p-value: {r.p_value}\n"
            f"# Clusters:\n"
        )
        # No trailing newline -- matches what hhnet process-hierarchies writes,
        # so this file diffs cleanly against the canonical CLI output.
        (results_dir / f"{label}_clusters.tsv").write_text(header + cluster_lines)

        (results_dir / f"{label}_observed_sizes.tsv").write_text(
            "\n".join(f"{h}\t{s}" for h, s in zip(r.observed_heights, r.observed_sizes))
        )
        (results_dir / f"{label}_permuted_sizes.tsv").write_text(
            "# height\tmin\texpected\tmax\n"
            + "\n".join(
                f"{h}\t{mn}\t{ex}\t{mx}"
                for h, mn, ex, mx in zip(
                    r.distinct_heights, r.min_sizes, r.expected_sizes, r.max_sizes
                )
            )
        )

        if plot:
            # Imported lazily so the optional matplotlib dep is only required
            # when the caller actually asks for a plot.
            from hierarchical_hotnet.core.process_hierarchies import plot_cluster_sizes
            plot_cluster_sizes(
                r.observed_heights, r.observed_sizes,
                r.distinct_heights, r.min_sizes,
                r.distinct_heights, r.expected_sizes,
                r.distinct_heights, r.max_sizes,
                r.permuted_heights_collection,
                r.permuted_sizes_collection,
                r.observed_cut_height,
                label,
                str(results_dir / f"{label}_sizes.pdf"),
            )

    if result.consensus is not None:
        (results_dir / "consensus_nodes.tsv").write_text(
            "\n".join("\t".join(g) for g in result.consensus.nodes)
        )
        (results_dir / "consensus_edges.tsv").write_text(
            "\n".join("\t".join(e) for e in result.consensus.edges)
        )


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
    use_edge_weights: bool = False,
    log_transform: bool = False,
    score_threshold: float = float('nan'),
    lower_size_bound: float = 10.0,
    upper_size_bound: float = float('inf'),
    consensus_threshold: Optional[int] = 2,
    verbose: bool = False,
    workdir: Optional[Path] = None,
    reuse: bool = False,
    plot: bool = False,
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
    use_edge_weights : bool
        When ``False`` (default), input edge weights are ignored and every
        edge contributes weight ``1.0`` to the adjacency matrix. Set to
        ``True`` to use the input weights (e.g. STRING confidence scores)
        in the diffusion. See :func:`compute_similarity_matrix` for details.
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
    plot : bool
        Only meaningful with ``workdir``. When ``True``, also write a
        ``<workdir>/results/<label>_sizes.pdf`` for each score set. Requires
        the optional ``matplotlib`` dependency (``pip install '.[plot]'``).

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

    # Drop nodes with no edges: they cannot belong to any cluster and would
    # produce all-zero rows in the similarity matrix. Re-indexes if needed.
    edges, index_to_gene = drop_isolated_nodes(edges, index_to_gene)

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
            use_edge_weights=use_edge_weights,
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

    pipeline_result = PipelineResult(
        similarity_matrix=similarity_matrix,
        beta=beta_used,
        score_results=score_results,
        consensus=consensus,
    )

    if workdir is not None:
        save_pipeline_result(pipeline_result, workdir, plot=plot)

    return pipeline_result
