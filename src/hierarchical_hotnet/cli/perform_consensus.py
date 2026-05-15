"""Command-line interface for ``hierarchical_hotnet.perform_consensus``."""

import argparse
from hierarchical_hotnet.file_io import load_edge_list, load_index_gene, progress
from hierarchical_hotnet.perform_consensus import ConsensusInput, load_components, perform_consensus


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--component_files', type=str, required=True, nargs='*')
    parser.add_argument('-igf', '--index_gene_files', type=str, required=True, nargs='*')
    parser.add_argument('-elf', '--edge_list_files', type=str, required=True, nargs='*')
    parser.add_argument('-n', '--networks', type=str, required=True, nargs='*')
    parser.add_argument('-s', '--scores', type=str, required=True, nargs='*')
    parser.add_argument('-t', '--threshold', type=int, required=True)
    parser.add_argument('-cnf', '--consensus_node_file', type=str, required=False)
    parser.add_argument('-cef', '--consensus_edge_file', type=str, required=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def run(args):
    if args.verbose:
        progress('Loading data...')
    n = len(args.component_files)
    if not n == len(args.index_gene_files) == len(args.edge_list_files) == len(args.networks) == len(args.scores):
        raise ValueError('--component_files, --index_gene_files, --edge_list_files, --networks, --scores must be the same length')
    inputs = []
    for index_gene_file, edge_list_file, component_file in zip(args.index_gene_files, args.edge_list_files, args.component_files):
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
            f.write('\n'.join(('\t'.join(group) for group in result.nodes)))
    if args.consensus_edge_file is not None:
        with open(args.consensus_edge_file, 'w') as f:
            f.write('\n'.join(('\t'.join(edge) for edge in result.edges)))
    if args.verbose:
        progress()


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
