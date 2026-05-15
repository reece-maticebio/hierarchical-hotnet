"""Command-line interface for ``hierarchical_hotnet.process_hierarchies``."""

import argparse
from hierarchical_hotnet.file_io import load_edge_list, load_index_gene, progress
from hierarchical_hotnet.core.process_hierarchies import plot_cluster_sizes, process_hierarchies


def get_parser():
    parser = argparse.ArgumentParser(description='Process observed and permuted hierarchies.')
    parser.add_argument('-oelf', '--observed_edge_list_file', type=str, required=True, help='Observed edge list filename')
    parser.add_argument('-oigf', '--observed_index_gene_file', type=str, required=True, help='Observed index-gene filename')
    parser.add_argument('-pelf', '--permuted_edge_list_files', type=str, required=True, nargs='*', help='Permuted edge list filenames')
    parser.add_argument('-pigf', '--permuted_index_gene_files', type=str, required=True, nargs='*', help='Permuted index-gene filenames')
    parser.add_argument('-lsb', '--lower_size_bound', type=float, required=False, default=10.0, help='Lower bound for cut size')
    parser.add_argument('-usb', '--upper_size_bound', type=float, required=False, default=float('inf'), help='Upper bound for cut size')
    parser.add_argument('-nc', '--num_cores', type=int, required=False, default=1, help='Number of cores')
    parser.add_argument('-cf', '--cluster_file', type=str, required=False, help='Cluster filename')
    parser.add_argument('-pf', '--plot_file', type=str, required=False, help='Plot filename')
    parser.add_argument('-pl', '--plot_label', type=str, required=False, nargs='*', help='Plot label')
    parser.add_argument('-osf', '--observed_size_file', type=str, required=False, help='Observed cluster size filename')
    parser.add_argument('-esf', '--expected_size_file', type=str, required=False, help='Expected cluster size filename')
    parser.add_argument('-minsf', '--min_size_file', type=str, required=False, help='Minimum cluster size filename')
    parser.add_argument('-maxsf', '--max_size_file', type=str, required=False, help='Maximum cluster size filename')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    return parser


def run(args):
    if args.verbose:
        progress('Loading hierarchy data...')
    observed_hierarchy = load_edge_list(args.observed_edge_list_file)
    observed_index_to_gene, _ = load_index_gene(args.observed_index_gene_file)
    permuted_hierarchies = [load_edge_list(f) for f in args.permuted_edge_list_files]
    permuted_index_to_gene_list = [load_index_gene(f)[0] for f in args.permuted_index_gene_files]
    if args.verbose:
        progress('Processing hierarchies...')
    result = process_hierarchies(observed_hierarchy, observed_index_to_gene, permuted_hierarchies, permuted_index_to_gene_list, lower_size_bound=args.lower_size_bound, upper_size_bound=args.upper_size_bound, n_jobs=args.num_cores)
    if args.observed_size_file is not None:
        with open(args.observed_size_file, 'w') as f:
            f.write('\n'.join((f'{h}\t{s}' for h, s in zip(result.observed_heights, result.observed_sizes))))
    if args.expected_size_file is not None:
        with open(args.expected_size_file, 'w') as f:
            f.write('\n'.join((f'{h}\t{s}' for h, s in zip(result.distinct_heights, result.expected_sizes))))
    if args.min_size_file is not None:
        with open(args.min_size_file, 'w') as f:
            f.write('\n'.join((f'{h}\t{s}' for h, s in zip(result.distinct_heights, result.min_sizes))))
    if args.max_size_file is not None:
        with open(args.max_size_file, 'w') as f:
            f.write('\n'.join((f'{h}\t{s}' for h, s in zip(result.distinct_heights, result.max_sizes))))
    if args.cluster_file is not None:
        sorted_clusters = sorted(sorted(map(sorted, result.observed_clusters)), key=len, reverse=True)
        cluster_lines = '\n'.join(('\t'.join(c) for c in sorted_clusters))
        header = f'# Observed cut height: {result.observed_cut_height}\n# Observed size of largest cluster at observed cut height: {result.observed_cut_size}\n# Expected size of largest cluster at observed cut height: {result.expected_cut_size}\n# Observed maximum ratio statistic: {result.observed_cut_ratio:.3f}\n# Expected maximum ratio statistic: {result.expected_cut_ratio:.3f}\n# p-value: {result.p_value}\n# Clusters:\n'
        with open(args.cluster_file, 'w') as f:
            f.write(header + cluster_lines)
    if args.plot_file is not None:
        if args.verbose:
            progress('Plotting cluster sizes...')
        plot_cluster_sizes(result.observed_heights, result.observed_sizes, result.distinct_heights, result.min_sizes, result.distinct_heights, result.expected_sizes, result.distinct_heights, result.max_sizes, result.permuted_heights_collection, result.permuted_sizes_collection, result.observed_cut_height, ' '.join(args.plot_label) if args.plot_label else '', args.plot_file)
    if args.verbose:
        progress()


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
