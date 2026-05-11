"""Find permutation bins for score-permutation tests."""

import argparse
from collections import defaultdict

import networkx as nx

from hierarchical_hotnet.hhio import load_edge_list, load_gene_score, load_index_gene


def compute_permutation_bins(edges, gene_to_score, *, min_size=float('inf')):
    """Bin scored genes by network degree for degree-preserving permutation.

    Parameters
    ----------
    edges : iterable of (gene_a, gene_b)
        Gene-labeled, unweighted edge list (e.g. via
        ``load_edge_list(edge_list_file, index_to_gene, unweighted=True)``).
    gene_to_score : Mapping[str, float]
        Gene-score map; only these genes are binned.
    min_size : float
        Minimum size of each bin (the highest-degree bin may exceed it).

    Returns
    -------
    list[list[str]]
        Bins of gene names. Genes within a bin are interchangeable under
        permutation.
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    G = G.subgraph(gene_to_score)
    if len(G) == 0:
        return []
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    common_genes = set(G.nodes())

    degree_to_nodes = defaultdict(set)
    for node in common_genes:
        degree_to_nodes[G.degree(node)].add(node)

    bins = []
    current_bin = []
    for degree in sorted(degree_to_nodes, reverse=True):
        current_bin += sorted(degree_to_nodes[degree])
        if len(current_bin) >= min_size:
            bins.append(current_bin)
            current_bin = []
    if bins:
        bins[-1] += current_bin
    elif current_bin:
        bins.append(current_bin)
    return bins


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-igf', '--index_gene_file', type=str,   required=True,  help='Index-gene filename')
    parser.add_argument('-elf', '--edge_list_file',  type=str,   required=True,  help='Edge list filename')
    parser.add_argument('-gsf', '--gene_score_file', type=str,   required=True,  help='Gene score filename')
    parser.add_argument('-ms',  '--min_size',        type=float, required=False, default=float('inf'), help='Minimum permutation bin size')
    parser.add_argument('-o',   '--output_file',     type=str,   required=True,  help='Output filename')
    return parser


def run(args):
    index_to_gene, _ = load_index_gene(args.index_gene_file)
    edges = load_edge_list(args.edge_list_file, index_to_gene, unweighted=True)
    gene_to_score = load_gene_score(args.gene_score_file)

    bins = compute_permutation_bins(edges, gene_to_score, min_size=args.min_size)

    with open(args.output_file, 'w') as f:
        f.write('\n'.join('\t'.join(current_bin) for current_bin in bins))


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
