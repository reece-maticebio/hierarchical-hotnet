"""Command-line interface for ``hierarchical_hotnet.permute_network``."""

import argparse
from hierarchical_hotnet.file_io import load_edge_list, save_edge_list
from hierarchical_hotnet.permute_network import permute_network


def get_parser():
    parser = argparse.ArgumentParser(description='Permute networks with the double edge swap algorithm.')
    parser.add_argument('-i', '--edge_list_file', type=str, required=True, help='Edge list filename')
    parser.add_argument('-c', '--connected', action='store_true', help='Preserve connectivity of graph')
    parser.add_argument('-s', '--seed', type=int, required=False, help='Random seed')
    parser.add_argument('-q', '--Q', type=float, required=False, default=100, help='Minimum of Q*|E| edge swaps')
    parser.add_argument('-o', '--permuted_edge_list_file', type=str, required=True, help='Permuted edge list filename')
    return parser


def run(args):
    edges = load_edge_list(args.edge_list_file, unweighted=True)
    permuted = permute_network(edges, seed=args.seed, preserve_connectivity=args.connected, Q=args.Q)
    save_edge_list(args.permuted_edge_list_file, permuted)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
