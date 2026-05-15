"""Consensus summarization across multiple hierarchical clusterings."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import networkx as nx

from hierarchical_hotnet.file_io import load_edge_list, load_index_gene, progress


@dataclass
class ConsensusInput:
    """One (network, scores) cluster set with its underlying network."""
    edges: set            # set[frozenset[str]] — gene-labeled edges
    components: Sequence  # list[list[str]] — top-level clusters from process_hierarchies


@dataclass
class ConsensusResult:
    """Result bundle from :func:`perform_consensus`."""
    nodes: list   # list[list[str]] — consensus connected components, sorted desc
    edges: list   # list[list[str]] — consensus edges (each as [u, v])


def load_components(component_file):
    """Read the first cluster (the largest) from a ``process_hierarchies`` output file."""
    components = []
    with open(component_file, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                components.append(sorted(l.rstrip('\n').split('\t')))
                break
    return components


def perform_consensus(consensus_inputs: Iterable[ConsensusInput], threshold: int) -> ConsensusResult:
    """Aggregate clusterings into a consensus network.

    For each input, every pair of genes that co-occur in a cluster *and* are
    connected by an edge in that input's network contributes one vote. Edges
    receiving at least ``threshold`` votes form the consensus network; its
    connected components are the consensus node groups.

    Parameters
    ----------
    consensus_inputs : iterable of ConsensusInput
    threshold : int
        Minimum number of votes for an edge to enter the consensus.
    """
    edge_to_tally = defaultdict(int)

    for inp in consensus_inputs:
        for component in inp.components:
            for u, v in combinations(component, 2):
                edge = frozenset((u, v))
                if edge in inp.edges:
                    edge_to_tally[edge] += 1

    thresholded_edges = {edge for edge, tally in edge_to_tally.items() if tally >= threshold}

    G = nx.Graph()
    G.add_edges_from(thresholded_edges)
    nodes = sorted(
        sorted([sorted(c) for c in nx.connected_components(G)]),
        key=len,
        reverse=True,
    )
    edges = sorted(map(sorted, thresholded_edges))

    return ConsensusResult(nodes=nodes, edges=edges)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf',  '--component_files',    type=str, required=True, nargs='*')
    parser.add_argument('-igf', '--index_gene_files',   type=str, required=True, nargs='*')
    parser.add_argument('-elf', '--edge_list_files',    type=str, required=True, nargs='*')
    parser.add_argument('-n',   '--networks',           type=str, required=True, nargs='*')
    parser.add_argument('-s',   '--scores',             type=str, required=True, nargs='*')
    parser.add_argument('-t',   '--threshold',          type=int, required=True)
    parser.add_argument('-cnf', '--consensus_node_file', type=str, required=False)
    parser.add_argument('-cef', '--consensus_edge_file', type=str, required=False)
    parser.add_argument('-v',   '--verbose',            action='store_true')
    return parser


def run(args):
    if args.verbose:
        progress('Loading data...')

    n = len(args.component_files)
    if not (n == len(args.index_gene_files) == len(args.edge_list_files)
            == len(args.networks) == len(args.scores)):
        raise ValueError('--component_files, --index_gene_files, --edge_list_files, --networks, --scores must be the same length')

    inputs = []
    for index_gene_file, edge_list_file, component_file in zip(
        args.index_gene_files, args.edge_list_files, args.component_files
    ):
        index_to_gene, _ = load_index_gene(index_gene_file)
        edges = {frozenset(edge) for edge in load_edge_list(edge_list_file, index_to_gene, unweighted=True)}
        components = load_components(component_file)
        inputs.append(ConsensusInput(edges=edges, components=components))

    if args.verbose:
        progress('Processing data...')

    result = perform_consensus(inputs, args.threshold)

    if args.verbose:
        progress('Saving data...')

    if args.consensus_node_file is not None:
        with open(args.consensus_node_file, 'w') as f:
            f.write('\n'.join('\t'.join(group) for group in result.nodes))

    if args.consensus_edge_file is not None:
        with open(args.consensus_edge_file, 'w') as f:
            f.write('\n'.join('\t'.join(edge) for edge in result.edges))

    if args.verbose:
        progress()


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
