"""Command-line interface for ``hierarchical_hotnet.construct_hierarchy``."""

import argparse
from hierarchical_hotnet.file_io import load_gene_score, load_index_gene, load_matrix, save_edge_list, save_index_gene
from hierarchical_hotnet.core.hierarchy import construct_hierarchy


def get_parser():
    parser = argparse.ArgumentParser(description='Construct the hierarchical decomposition of the SCCs of the Hierarchical HotNet similarity matrix.')
    parser.add_argument('-smf', '--similarity_matrix_file', type=str, required=True, help='HH similarity matrix filename')
    parser.add_argument('-smn', '--similarity_matrix_name', type=str, required=False, default='PPR', help='HH similarity matrix name')
    parser.add_argument('-igf', '--index_gene_file', type=str, required=True, help='Index-gene filename')
    parser.add_argument('-gsf', '--gene_score_file', type=str, required=False, help='Gene-score filename')
    parser.add_argument('-lt', '--log_transform', action='store_true', help='Log transform scores')
    parser.add_argument('-st', '--score_threshold', type=float, required=False, default=float('nan'), help='Score threshold')
    parser.add_argument('-helf', '--hierarchy_edge_list_file', type=str, required=True, help='Hierarchy edge list filename')
    parser.add_argument('-higf', '--hierarchy_index_gene_file', type=str, required=True, help='Hierarchy index-gene filename')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    return parser


def run(args):
    if args.verbose:
        print('Loading data...')
    index_to_gene, _ = load_index_gene(args.index_gene_file)
    P = load_matrix(args.similarity_matrix_file, args.similarity_matrix_name)
    gene_to_score = load_gene_score(args.gene_score_file) if args.gene_score_file else None
    T, common_index_to_gene = construct_hierarchy(P, index_to_gene, gene_to_score=gene_to_score, log_transform=args.log_transform, score_threshold=args.score_threshold, verbose=args.verbose)
    if args.verbose:
        print('Saving results...')
    save_edge_list(args.hierarchy_edge_list_file, T)
    save_index_gene(args.hierarchy_index_gene_file, common_index_to_gene)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
