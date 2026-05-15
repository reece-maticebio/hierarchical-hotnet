"""Command-line interface for ``hierarchical_hotnet.permute_scores``."""

import argparse
from hierarchical_hotnet.file_io import load_gene_score, save_gene_score
from hierarchical_hotnet.core.permute_scores import load_bins, permute_scores


def get_parser():
    parser = argparse.ArgumentParser(description='Permute gene scores.')
    parser.add_argument('-i', '--gene_score_file', type=str, required=True, help='Input filename')
    parser.add_argument('-bf', '--bin_file', type=str, required=False, help='Bin filename')
    parser.add_argument('-s', '--seed', type=int, required=False, help='Random seed')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output filename')
    return parser


def run(args):
    gene_to_score = load_gene_score(args.gene_score_file)
    bins = load_bins(args.bin_file) if args.bin_file else None
    permuted = permute_scores(gene_to_score, bins, seed=args.seed)
    save_gene_score(args.output_file, permuted)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
