"""Unified ``hhnet`` command-line entry point.

Replaces the eight separate ``hhnet-*`` console scripts with a single
``hhnet`` entry and subcommands::

    hhnet --help
    hhnet construct-similarity-matrix -i edges.tsv -o sim.h5
    hhnet construct-hierarchy ...

Each subcommand still owns its own argparse parser in
``hierarchical_hotnet.cli.<name>``. This module is a thin dispatcher: it
finds the right module by subcommand name, fixes up ``sys.argv`` so the
subcommand's own ``--help`` reports a sensible program name, and calls
its :func:`main`.
"""

from __future__ import annotations

import importlib
import sys
from typing import Optional, Sequence

# (subcommand-name, module-name, short description)
_SUBCOMMANDS: list[tuple[str, str, str]] = [
    ("construct-similarity-matrix", "construct_similarity_matrix",
     "Build the diffusion similarity matrix from a weighted edge list."),
    ("construct-hierarchy", "construct_hierarchy",
     "Build the hierarchical decomposition of the SCCs of the similarity matrix."),
    ("find-permutation-bins", "find_permutation_bins",
     "Compute degree-preserving permutation bins for a network + score set."),
    ("permute-scores", "permute_scores",
     "Generate a permuted gene-score file within precomputed bins."),
    ("permute-network", "permute_network",
     "Generate a permuted edge list, optionally preserving connectivity."),
    ("process-hierarchies", "process_hierarchies",
     "Process observed + permuted hierarchies into clusters and a p-value."),
    ("perform-consensus", "perform_consensus",
     "Combine cluster sets across networks/scores into consensus nodes/edges."),
    ("generate-example-graph", "generate_example_graph",
     "Generate the example toy graph used by the docs."),
]

_NAME_TO_MODULE: dict[str, str] = {name: mod for name, mod, _ in _SUBCOMMANDS}


def _print_help(stream=sys.stdout) -> None:
    name_width = max(len(name) for name, _, _ in _SUBCOMMANDS)
    print("usage: hhnet SUBCOMMAND [OPTIONS]", file=stream)
    print(file=stream)
    print("Hierarchical HotNet command-line interface.", file=stream)
    print(file=stream)
    print("Subcommands:", file=stream)
    for name, _, desc in _SUBCOMMANDS:
        print(f"  {name:<{name_width}}  {desc}", file=stream)
    print(file=stream)
    print("Run 'hhnet SUBCOMMAND --help' for command-specific options.", file=stream)


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_help(sys.stdout if argv else sys.stderr)
        sys.exit(0 if argv else 2)

    subcommand = argv[0]
    if subcommand not in _NAME_TO_MODULE:
        print(f"hhnet: unknown subcommand: {subcommand!r}", file=sys.stderr)
        print("Run 'hhnet --help' for the list of subcommands.", file=sys.stderr)
        sys.exit(2)

    mod = importlib.import_module(f"hierarchical_hotnet.cli.{_NAME_TO_MODULE[subcommand]}")

    # Make the subcommand's own --help report a sensible program name.
    saved_argv = sys.argv
    sys.argv = [f"hhnet {subcommand}"] + argv[1:]
    try:
        mod.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
