"""Command-line interface for ``hierarchical_hotnet.construct_similarity_matrix``."""

import argparse
from hierarchical_hotnet.file_io import load_edge_list, save_matrix
from hierarchical_hotnet.construct_similarity_matrix import compute_similarity_matrix


def get_parser():
    parser = argparse.ArgumentParser(description='Construct the Hierarchical HotNet similarity matrix and/or the restart probability beta.')
    parser.add_argument('-i', '--edge_list_file', type=str, required=True, help='Edge list filename')
    parser.add_argument('-d', '--directed', action='store_true', help='Is directed')
    parser.add_argument('-b', '--beta', type=float, required=False, help='Restart probability beta')
    parser.add_argument('-nd', '--num_digits', type=int, required=False, default=2, help='Number of digits in beta')
    parser.add_argument('-t', '--threshold', type=float, required=False, default=1.0, help='Threshold for edge weights')
    parser.add_argument('-n', '--name', type=str, required=False, default='PPR', help='Similarity matrix name')
    parser.add_argument('-o', '--output_file', type=str, required=False, help='Similarity matrix output filename')
    parser.add_argument('-bof', '--beta_output_file', type=str, required=False, help='Beta output filename')
    return parser


def run(args):
    edges = load_edge_list(args.edge_list_file)
    P, beta = compute_similarity_matrix(edges, directed=args.directed, beta=args.beta, threshold=args.threshold, num_digits=args.num_digits)
    if args.output_file is not None:
        save_matrix(args.output_file, P, args.name)
    if args.beta_output_file is not None:
        fmt = '{:.' + str(args.num_digits) + 'f}'
        with open(args.beta_output_file, 'w') as f:
            f.write(fmt.format(beta))


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
