"""Tests for run_pipeline(workdir=..., reuse=...) — disk-backed pipeline runs.

The canonical-correctness tests in test_pipeline.py already cover the
in-memory path. These tests focus on the workdir layout, the reuse=True
resume semantics, and the parity between workdir and in-memory results.
"""

import time
from pathlib import Path

import pytest

import hierarchical_hotnet as hhn


PIPELINE_KWARGS = dict(
    num_permutations=20,
    n_jobs=1,
    min_bin_size=1000,
    lower_size_bound=1,
    consensus_threshold=2,
)


def _load_inputs(data_dir):
    index_to_gene, _ = hhn.load_index_gene(data_dir / "network_1_index_gene.tsv")
    edges = hhn.load_edge_list(data_dir / "network_1_edge_list.tsv")
    scores_1 = hhn.load_gene_score(data_dir / "scores_1.tsv")
    scores_2 = hhn.load_gene_score(data_dir / "scores_2.tsv")
    return index_to_gene, edges, scores_1, scores_2


def _cluster_signature(clusters):
    return sorted(sorted(c) for c in clusters)


def _result_signature(result):
    """Reduce a PipelineResult to a comparable structure for parity checks."""
    return {
        "beta": float(result.beta),
        "scores": {
            label: {
                "p_value": float(r.p_value),
                "observed_cut_size": int(r.observed_cut_size),
                "observed_clusters": _cluster_signature(r.observed_clusters),
            }
            for label, r in result.score_results.items()
        },
        "consensus_nodes": [sorted(g) for g in (result.consensus.nodes or [])],
    }


def test_reuse_without_workdir_raises(data_dir):
    index_to_gene, edges, scores_1, _ = _load_inputs(data_dir)
    with pytest.raises(ValueError, match="reuse=True requires workdir"):
        hhn.run_pipeline(
            edges, index_to_gene, {"scores_1": scores_1},
            **PIPELINE_KWARGS, reuse=True,
        )


def test_workdir_creates_expected_layout(tmp_path, data_dir):
    index_to_gene, edges, scores_1, scores_2 = _load_inputs(data_dir)
    workdir = tmp_path / "run"

    hhn.run_pipeline(
        edges, index_to_gene,
        {"scores_1": scores_1, "scores_2": scores_2},
        **PIPELINE_KWARGS, workdir=workdir,
    )

    # Single artifacts
    assert (workdir / "similarity_matrix.h5").is_file()
    assert (workdir / "beta.txt").is_file()
    assert (workdir / "bins" / "scores_1.tsv").is_file()
    assert (workdir / "bins" / "scores_2.tsv").is_file()

    # Fan-out artifacts: one permuted-score file per seed, one hierarchy
    # (edges + genes pair) per observed+permutation.
    n_perm = PIPELINE_KWARGS["num_permutations"]
    for label in ("scores_1", "scores_2"):
        scores_dir = workdir / "permuted_scores" / label
        hier_dir = workdir / "hierarchies" / label
        assert len(list(scores_dir.glob("*.tsv"))) == n_perm
        assert len(list(hier_dir.glob("*.edges.tsv"))) == n_perm + 1
        assert len(list(hier_dir.glob("*.genes.tsv"))) == n_perm + 1


def test_workdir_result_matches_in_memory(tmp_path, data_dir):
    """A workdir run produces the same PipelineResult as an in-memory run."""
    index_to_gene, edges, scores_1, scores_2 = _load_inputs(data_dir)

    in_memory = hhn.run_pipeline(
        edges, index_to_gene,
        {"scores_1": scores_1, "scores_2": scores_2},
        **PIPELINE_KWARGS,
    )
    on_disk = hhn.run_pipeline(
        edges, index_to_gene,
        {"scores_1": scores_1, "scores_2": scores_2},
        **PIPELINE_KWARGS, workdir=tmp_path / "run",
    )
    assert _result_signature(in_memory) == _result_signature(on_disk)


def test_reuse_skips_recomputation(tmp_path, data_dir):
    """Second run with reuse=True doesn't touch the artifacts on disk."""
    index_to_gene, edges, scores_1 = _load_inputs(data_dir)[:3]
    workdir = tmp_path / "run"

    first = hhn.run_pipeline(
        edges, index_to_gene, {"scores_1": scores_1},
        **PIPELINE_KWARGS, workdir=workdir,
    )

    sentinel_paths = [
        workdir / "similarity_matrix.h5",
        workdir / "beta.txt",
        workdir / "bins" / "scores_1.tsv",
        workdir / "permuted_scores" / "scores_1" / "1.tsv",
        workdir / "hierarchies" / "scores_1" / "0.edges.tsv",
        workdir / "hierarchies" / "scores_1" / "5.edges.tsv",
    ]
    mtimes_before = {p: p.stat().st_mtime_ns for p in sentinel_paths}

    # Sleep a hair so a write would visibly bump mtime on coarse filesystems.
    time.sleep(0.05)

    second = hhn.run_pipeline(
        edges, index_to_gene, {"scores_1": scores_1},
        **PIPELINE_KWARGS, workdir=workdir, reuse=True,
    )

    for p in sentinel_paths:
        assert p.stat().st_mtime_ns == mtimes_before[p], f"reuse rewrote {p}"

    assert _result_signature(first) == _result_signature(second)


def test_reuse_fills_in_missing_hierarchy(tmp_path, data_dir):
    """If some hierarchies are missing on disk, reuse computes only those."""
    index_to_gene, edges, scores_1 = _load_inputs(data_dir)[:3]
    workdir = tmp_path / "run"

    first = hhn.run_pipeline(
        edges, index_to_gene, {"scores_1": scores_1},
        **PIPELINE_KWARGS, workdir=workdir,
    )

    # Delete one permuted hierarchy on disk (both files of the pair).
    missing_key = "7"
    hier_dir = workdir / "hierarchies" / "scores_1"
    edges_file = hier_dir / f"{missing_key}.edges.tsv"
    genes_file = hier_dir / f"{missing_key}.genes.tsv"
    edges_file.unlink()
    genes_file.unlink()

    # Track which other files survive untouched.
    survivor = hier_dir / "0.edges.tsv"
    survivor_mtime = survivor.stat().st_mtime_ns

    time.sleep(0.05)

    second = hhn.run_pipeline(
        edges, index_to_gene, {"scores_1": scores_1},
        **PIPELINE_KWARGS, workdir=workdir, reuse=True,
    )

    # The missing pair is back, the survivor is untouched, results match.
    assert edges_file.is_file()
    assert genes_file.is_file()
    assert survivor.stat().st_mtime_ns == survivor_mtime
    assert _result_signature(first) == _result_signature(second)


def test_reuse_false_overwrites(tmp_path, data_dir):
    """Without reuse, a second run recomputes (and rewrites) every artifact."""
    index_to_gene, edges, scores_1 = _load_inputs(data_dir)[:3]
    workdir = tmp_path / "run"

    hhn.run_pipeline(
        edges, index_to_gene, {"scores_1": scores_1},
        **PIPELINE_KWARGS, workdir=workdir,
    )

    survivor = workdir / "hierarchies" / "scores_1" / "0.edges.tsv"
    mtime_before = survivor.stat().st_mtime_ns

    time.sleep(0.05)

    hhn.run_pipeline(
        edges, index_to_gene, {"scores_1": scores_1},
        **PIPELINE_KWARGS, workdir=workdir,  # reuse=False (default)
    )
    assert survivor.stat().st_mtime_ns > mtime_before
