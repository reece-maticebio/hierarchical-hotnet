"""Consensus summarization across multiple hierarchical clusterings."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence
import networkx as nx


@dataclass
class ConsensusInput:
    """One (network, scores) cluster set with its underlying network."""
    edges: set
    components: Sequence


@dataclass
class ConsensusResult:
    """Result bundle from :func:`perform_consensus`."""
    nodes: list
    edges: list


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
    nodes = sorted(sorted([sorted(c) for c in nx.connected_components(G)]), key=len, reverse=True)
    edges = sorted(map(sorted, thresholded_edges))
    return ConsensusResult(nodes=nodes, edges=edges)
