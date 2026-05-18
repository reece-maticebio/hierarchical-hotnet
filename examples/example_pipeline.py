"""Run the Hierarchical HotNet pipeline on the example toy dataset.

Companion to ``examples/example_commands.sh`` — same inputs, same parameters,
same canonical consensus output, but driven through the Python
``hhn.run_pipeline()`` API instead of the CLI commands.

Run from the repository root::

    python examples/example_pipeline.py

The script:

  1. Loads the toy dataset (25-node network, two score sets).
  2. Calls ``run_pipeline`` in memory mode and prints per-score statistics.
  3. Verifies the consensus matches the committed canonical fixtures
     (``example_consensus_nodes.tsv``, ``example_consensus_edges.tsv``).
  4. Re-runs the pipeline with ``workdir=<tmp>``, prints the on-disk layout,
     and confirms it produces the same results as the in-memory run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import hierarchical_hotnet as hhn
from hierarchical_hotnet import backends


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
CANONICAL = REPO / "examples"

# These match examples/example_commands.sh exactly. ``lower_size_bound=1`` is
# specific to the toy example (25 nodes); for real networks, use the default
# of 10 (or larger) to filter out trivial-size cuts.
PIPELINE_KWARGS = dict(
    num_permutations=100,
    n_jobs=1,
    min_bin_size=1000,
    lower_size_bound=1,
    consensus_threshold=2,
)


def load_inputs():
    """Load the toy dataset into the Python objects ``run_pipeline`` expects."""
    index_to_gene, _ = hhn.load_index_gene(DATA / "network_1_index_gene.tsv")
    edges = hhn.load_edge_list(DATA / "network_1_edge_list.tsv")
    scores_1 = hhn.load_gene_score(DATA / "scores_1.tsv")
    scores_2 = hhn.load_gene_score(DATA / "scores_2.tsv")
    return index_to_gene, edges, {"scores_1": scores_1, "scores_2": scores_2}


def summarize(result) -> None:
    print(f"  beta (auto-selected):       {result.beta:.6f}")
    print(f"  similarity matrix shape:    {result.similarity_matrix.shape}")
    for label, r in result.score_results.items():
        print(f"  [{label}]")
        print(f"    p_value:                {r.p_value:.4f}")
        print(f"    observed cut height:    {r.observed_cut_height:.6f}")
        print(f"    largest observed cluster: {r.observed_cut_size} genes")
        print(f"    observed cut ratio:     {r.observed_cut_ratio:.4f}")
        print(f"    expected cut ratio:     {r.expected_cut_ratio:.4f}")
        print(f"    n clusters at cut:      {len(r.observed_clusters)}")
    print(f"  consensus clusters:         {len(result.consensus.nodes)}")
    print(f"  consensus edges:            {len(result.consensus.edges)}")


def matches_canonical(result) -> bool:
    expected_nodes = (CANONICAL / "example_consensus_nodes.tsv").read_text().strip()
    expected_edges = (CANONICAL / "example_consensus_edges.tsv").read_text().strip()
    actual_nodes = "\n".join("\t".join(g) for g in result.consensus.nodes)
    actual_edges = "\n".join("\t".join(e) for e in result.consensus.edges)
    return actual_nodes == expected_nodes and actual_edges == expected_edges


def list_artifacts(workdir: Path) -> None:
    print("  artifacts:")
    for p in sorted(workdir.iterdir()):
        if p.is_file():
            print(f"    {p.name}")
        else:
            count = sum(1 for child in p.rglob("*") if child.is_file())
            print(f"    {p.name}/  ({count} files)")


def main() -> None:
    print(f"Clustering backend: {backends.BACKEND}\n")
    index_to_gene, edges, score_sets = load_inputs()

    print("=" * 60)
    print("In-memory run (workdir=None)")
    print("=" * 60)
    in_mem = hhn.run_pipeline(edges, index_to_gene, score_sets, **PIPELINE_KWARGS)
    summarize(in_mem)
    print(f"  matches canonical:          {matches_canonical(in_mem)}")

    with tempfile.TemporaryDirectory(prefix="hhnet_example_") as tmp:
        workdir = Path(tmp)
        print()
        print("=" * 60)
        print(f"Disk-backed run (workdir={workdir})")
        print("=" * 60)
        on_disk = hhn.run_pipeline(
            edges, index_to_gene, score_sets, workdir=workdir, **PIPELINE_KWARGS,
        )
        summarize(on_disk)
        print(f"  matches canonical:          {matches_canonical(on_disk)}")
        list_artifacts(workdir)

    same = (
        in_mem.beta == on_disk.beta
        and all(
            in_mem.score_results[k].p_value == on_disk.score_results[k].p_value
            for k in in_mem.score_results
        )
    )
    print()
    print(f"in-memory and disk-backed runs identical: {same}")


if __name__ == "__main__":
    main()
