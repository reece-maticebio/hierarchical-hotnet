"""Command-line interface for ``hierarchical_hotnet.find_permutation_bins``."""

import argparse
from hierarchical_hotnet.file_io import load_edge_list, load_gene_score, load_index_gene
from hierarchical_hotnet.core.bins import compute_permutation_bins


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-igf', '--index_gene_file', type=str, required=True, help='Index-gene filename')
    parser.add_argument('-elf', '--edge_list_file', type=str, required=True, help='Edge list filename')
    parser.add_argument('-gsf', '--gene_score_file', type=str, required=True, help='Gene score filename')
    parser.add_argument('-ms', '--min_size', type=float, required=False, default=float('inf'), help='Minimum permutation bin size')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output filename')
    return parser


def run(args):
    index_to_gene, _ = load_index_gene(args.index_gene_file)
    edges = load_edge_list(args.edge_list_file, index_to_gene, unweighted=True)
    gene_to_score = load_gene_score(args.gene_score_file)
    bins = compute_permutation_bins(edges, gene_to_score, min_size=args.min_size)
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(('\t'.join(current_bin) for current_bin in bins)))


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
