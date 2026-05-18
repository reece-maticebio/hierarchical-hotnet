"""Run the Hierarchical HotNet pipeline on the example toy dataset.

Companion to ``examples/example_commands.sh`` — same inputs, same parameters,
same canonical consensus output, but driven through the Python
``hhn.run_pipeline()`` API instead of the CLI commands.

Run from the repository root::

    python examples/example_pipeline.py
    python examples/example_pipeline.py --workdir my_output
    python examples/example_pipeline.py --reuse           # resume a previous run

The pipeline writes every artifact (similarity matrix, beta, bins, permuted
scores, hierarchies) under ``--workdir`` in the layout described in the
README. The script then prints per-score statistics and verifies that the
consensus matches the committed canonical fixtures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hierarchical_hotnet as hhn
from hierarchical_hotnet import backends


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
CANONICAL = REPO / "examples"
DEFAULT_WORKDIR = REPO / "examples" / "pipeline_output"

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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir", type=Path, default=DEFAULT_WORKDIR,
        help=f"Directory to save run artifacts (default: {DEFAULT_WORKDIR.relative_to(REPO)})",
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="Reuse artifacts already present in workdir (skip recomputation).",
    )
    args = parser.parse_args()

    print(f"Clustering backend: {backends.BACKEND}")
    print(f"Workdir:            {args.workdir}")
    print(f"Reuse:              {args.reuse}")
    print()

    index_to_gene, edges, score_sets = load_inputs()
    result = hhn.run_pipeline(
        edges, index_to_gene, score_sets,
        workdir=args.workdir, reuse=args.reuse,
        **PIPELINE_KWARGS,
    )

    summarize(result)
    print(f"  matches canonical:          {matches_canonical(result)}")
    list_artifacts(args.workdir)


if __name__ == "__main__":
    main()
