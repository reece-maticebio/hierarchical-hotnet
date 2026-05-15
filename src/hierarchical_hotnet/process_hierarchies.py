"""Process observed and permuted hierarchies."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    from hierarchical_hotnet import fortran_module
    imported_fortran_module = True
except ImportError:
    imported_fortran_module = False

from hierarchical_hotnet.file_io import load_edge_list, load_index_gene, progress


def compute_statistic(sizes):
    return max(sizes)


def compute_statistics(T, index_to_gene, reverse=True):
    """Cluster-size statistic at each height of a dendrogram."""
    heights = []
    statistics = []

    index_to_node = defaultdict(set)
    for index, gene in index_to_gene.items():
        index_to_node[index] = {gene}

    height = float('inf') if reverse else 0.0
    statistics.append(compute_statistic(len(node) for node in index_to_node.values()))
    heights.append(height)

    T = sorted(T, key=lambda x: x[2], reverse=reverse)
    m = len(T)

    for i, edge in enumerate(T):
        source, target, height = edge
        a = index_to_node[source]
        b = index_to_node[target]
        c = set(a) | set(b)
        del index_to_node[source]
        index_to_node[target] = c

        if i == m - 1 or height != T[i + 1][2]:
            statistics.append(compute_statistic(len(node) for node in index_to_node.values()))
            heights.append(height)

    height = 0.0 if reverse else float('inf')
    statistics.append(compute_statistic(len(node) for node in index_to_node.values()))
    heights.append(height)

    return np.array(heights), np.array(statistics)


def find_statistics(edge_list_file, index_gene_file, reverse=True):
    T = load_edge_list(edge_list_file)
    index_to_gene, _ = load_index_gene(index_gene_file)
    return compute_statistics(T, index_to_gene, reverse)


def _statistics_worker(args):
    T, index_to_gene = args
    return compute_statistics(T, index_to_gene, reverse=True)


def _statistics_worker_from_files(args):
    edge_list_file, index_gene_file = args
    return find_statistics(edge_list_file, index_gene_file, reverse=True)


def summarize_cluster_sizes(permuted_heights_collection, permuted_sizes_collection):
    """Aggregate min/mean/max cluster size across permutations at each height."""
    num_permutations = len(permuted_heights_collection)
    distinct_heights = np.unique(np.concatenate(permuted_heights_collection))[::-1]
    num_distinct_heights = len(distinct_heights)

    max_indices = np.array([len(h) for h in permuted_heights_collection], dtype=np.int64)

    if imported_fortran_module:
        max_index = int(np.max(max_indices))
        heights = np.zeros((num_permutations, max_index))
        sizes = np.zeros((num_permutations, max_index))
        for i in range(num_permutations):
            heights[i, : max_indices[i]] = permuted_heights_collection[i]
            sizes[i, : max_indices[i]] = permuted_sizes_collection[i]

        summary_sizes = fortran_module.summarize_sizes(distinct_heights, heights, sizes, max_indices)
        min_sizes = summary_sizes[:, 0]
        expected_sizes = summary_sizes[:, 1]
        max_sizes = summary_sizes[:, 2]
    else:
        cur_indices = np.zeros(num_permutations, dtype=np.int64)
        cur_sizes = np.zeros(num_permutations)
        min_sizes = np.zeros(num_distinct_heights)
        expected_sizes = np.zeros(num_distinct_heights)
        max_sizes = np.zeros(num_distinct_heights)

        for k in range(num_distinct_heights):
            distinct_height = distinct_heights[k]
            for i in range(num_permutations):
                while (cur_indices[i] < max_indices[i] - 1
                       and permuted_heights_collection[i][cur_indices[i] + 1] >= distinct_height):
                    cur_indices[i] += 1
                cur_sizes[i] = permuted_sizes_collection[i][cur_indices[i]]
            min_sizes[k] = np.min(cur_sizes)
            expected_sizes[k] = np.mean(cur_sizes)
            max_sizes[k] = np.max(cur_sizes)

    return distinct_heights, min_sizes, expected_sizes, max_sizes


def find_cut(observed_heights, observed_sizes, expected_heights, expected_sizes,
             lower_size_bound, upper_size_bound):
    """Cut height with the largest observed/expected size ratio."""
    num_observed_heights = len(observed_heights)
    num_expected_heights = len(expected_heights)
    ratios = np.zeros(num_observed_heights)

    j = 0
    for i in range(num_observed_heights):
        if lower_size_bound <= observed_sizes[i] <= upper_size_bound:
            while j < num_expected_heights - 1 and expected_heights[j + 1] >= observed_heights[i]:
                j += 1
            ratios[i] = float(observed_sizes[i]) / float(expected_sizes[j])

    max_index = int(np.argmax(ratios))
    return observed_heights[max_index], ratios[max_index]


def _find_cut_worker(args):
    return find_cut(*args)


def find_clusters(preordered_T, index_to_gene, threshold, reverse=True):
    """Cut a dendrogram at ``threshold`` and return the resulting clusters."""
    index_to_node = defaultdict(set)
    for index, gene in index_to_gene.items():
        index_to_node[index] = frozenset([gene])

    clusters = set(index_to_node.values())

    T = sorted(preordered_T, key=lambda x: x[2], reverse=reverse)
    for edge in T:
        source, target, height = edge
        if (not reverse and height > threshold) or (reverse and height < threshold):
            break
        a = index_to_node[source]
        b = index_to_node[target]
        c = frozenset(set(a) | set(b))
        clusters.discard(a)
        clusters.discard(b)
        clusters.add(c)
        del index_to_node[source]
        index_to_node[target] = c

    return clusters


def cut_hierarchy(edge_list_file, index_gene_file, cut_height):
    T = load_edge_list(edge_list_file)
    index_to_gene, _ = load_index_gene(index_gene_file)
    return find_clusters(T, index_to_gene, cut_height)


@dataclass
class ProcessHierarchiesResult:
    """Result bundle from :func:`process_hierarchies`."""
    observed_clusters: set
    observed_cut_height: float
    observed_cut_ratio: float
    expected_cut_ratio: float
    p_value: float
    observed_cut_size: float
    expected_cut_size: float
    observed_heights: np.ndarray
    observed_sizes: np.ndarray
    distinct_heights: np.ndarray
    min_sizes: np.ndarray
    expected_sizes: np.ndarray
    max_sizes: np.ndarray
    permuted_heights_collection: tuple = field(repr=False)
    permuted_sizes_collection: tuple = field(repr=False)


def _resolve_map(num_workers):
    """Return ``(map_fn, cleanup_fn)`` for a desired parallelism level."""
    if num_workers == 1:
        return map, lambda: None
    workers = None if num_workers in (-1, None) else num_workers
    executor = ProcessPoolExecutor(max_workers=workers)
    return executor.map, executor.shutdown


def process_hierarchies(
    observed_hierarchy,
    observed_index_to_gene,
    permuted_hierarchies,
    permuted_index_to_gene_list,
    *,
    lower_size_bound=10.0,
    upper_size_bound=float('inf'),
    n_jobs=1,
):
    """Process an observed hierarchy and its permutations.

    Parameters
    ----------
    observed_hierarchy : list[tuple[int, int, float]]
        Dendrogram edges as returned by :func:`construct_hierarchy`.
    observed_index_to_gene : Mapping[int, str]
    permuted_hierarchies : Sequence[list[tuple[int, int, float]]]
    permuted_index_to_gene_list : Sequence[Mapping[int, str]]
    lower_size_bound, upper_size_bound : float
        Bounds on cluster size considered when choosing the cut.
    n_jobs : int
        Worker processes for the (per-permutation) maps. ``1`` is serial;
        ``-1`` lets ``ProcessPoolExecutor`` choose.

    Returns
    -------
    ProcessHierarchiesResult
    """
    if len(permuted_hierarchies) != len(permuted_index_to_gene_list):
        raise ValueError('permuted_hierarchies and permuted_index_to_gene_list lengths differ')

    map_fn, cleanup = _resolve_map(n_jobs)
    try:
        stats_input = [(observed_hierarchy, observed_index_to_gene)]
        stats_input += list(zip(permuted_hierarchies, permuted_index_to_gene_list))
        stats_output = list(map_fn(_statistics_worker, stats_input))
    finally:
        cleanup()

    observed_heights, observed_sizes = stats_output[0]
    permuted_heights_collection, permuted_sizes_collection = zip(*stats_output[1:])

    distinct_heights, min_sizes, expected_sizes, max_sizes = summarize_cluster_sizes(
        permuted_heights_collection, permuted_sizes_collection
    )

    map_fn, cleanup = _resolve_map(n_jobs)
    try:
        cut_input = [(observed_heights, observed_sizes, distinct_heights, expected_sizes,
                      lower_size_bound, upper_size_bound)]
        cut_input += [(permuted_heights, permuted_sizes, distinct_heights, expected_sizes,
                       lower_size_bound, upper_size_bound)
                      for permuted_heights, permuted_sizes
                      in zip(permuted_heights_collection, permuted_sizes_collection)]
        cut_output = list(map_fn(_find_cut_worker, cut_input))
    finally:
        cleanup()

    observed_cut_height, observed_cut_ratio = cut_output[0]
    _, permuted_cut_ratios = zip(*cut_output[1:])
    expected_cut_ratio = float(np.mean(permuted_cut_ratios))
    p_value = sum(r >= observed_cut_ratio for r in permuted_cut_ratios) / float(len(permuted_cut_ratios))

    observed_clusters = find_clusters(observed_hierarchy, observed_index_to_gene, observed_cut_height)
    observed_cut_size = compute_statistic(len(c) for c in observed_clusters)
    expected_cut_size = observed_cut_size / observed_cut_ratio if observed_cut_ratio else float('nan')

    return ProcessHierarchiesResult(
        observed_clusters=observed_clusters,
        observed_cut_height=observed_cut_height,
        observed_cut_ratio=observed_cut_ratio,
        expected_cut_ratio=expected_cut_ratio,
        p_value=p_value,
        observed_cut_size=observed_cut_size,
        expected_cut_size=expected_cut_size,
        observed_heights=observed_heights,
        observed_sizes=observed_sizes,
        distinct_heights=distinct_heights,
        min_sizes=min_sizes,
        expected_sizes=expected_sizes,
        max_sizes=max_sizes,
        permuted_heights_collection=permuted_heights_collection,
        permuted_sizes_collection=permuted_sizes_collection,
    )


def plot_cluster_sizes(observed_heights, observed_sizes, min_heights, min_sizes,
                       expected_heights, expected_sizes, max_heights, max_sizes,
                       permuted_heights_collection, permuted_sizes_collection,
                       cut_height, plot_label, plot_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    observed_heights = np.clip(observed_heights, 1e-15, float('inf'))
    min_heights = np.clip(min_heights, 1e-15, float('inf'))
    expected_heights = np.clip(expected_heights, 1e-15, float('inf'))
    max_heights = np.clip(max_heights, 1e-15, float('inf'))
    permuted_heights_collection = [np.clip(h, 1e-15, float('inf'))
                                    for h in permuted_heights_collection]

    observed_color = (0.8, 0.0, 0.0)
    permuted_color = (0.0, 0.0, 0.8)
    background_permuted_color = (0.5, 0.5, 0.8)
    alpha = 0.2

    plt.figure(figsize=(5, 5))

    plt.step(1.0 / observed_heights, observed_sizes, where='post', c=observed_color, linewidth=2, zorder=5, label='Observed sizes')
    plt.step(1.0 / expected_heights, expected_sizes, where='post', c=permuted_color, linewidth=2, zorder=4, label='Expected sizes')

    if cut_height:
        i = max(k for k, height in enumerate(observed_heights) if height >= cut_height)
        j = max(k for k, height in enumerate(expected_heights) if height >= cut_height)
        plt.plot((1.0 / cut_height, 1.0 / cut_height), (observed_sizes[i], expected_sizes[j]),
                 c='k', linewidth=2, alpha=0.75, zorder=6, label='Chosen cut')

    plt.step(1.0 / min_heights, min_sizes, where='post', c=background_permuted_color, linewidth=1, linestyle='dotted', zorder=3, label='Permuted sizes (minimum)')

    plt.step([float('nan')], [float('nan')], where='post', c=background_permuted_color, linewidth=1, zorder=1, label='Permuted sizes (all)')
    for permuted_heights, permuted_sizes in zip(permuted_heights_collection, permuted_sizes_collection):
        plt.step(1.0 / permuted_heights, permuted_sizes, where='post', c=background_permuted_color, linewidth=0.5, alpha=alpha, zorder=1)

    plt.step(1.0 / max_heights, max_sizes, where='post', c=background_permuted_color, linewidth=1, linestyle='dashed', zorder=3, label='Permuted sizes (maximum)')

    plt.xlim(0.8 * np.min(1.0 / observed_heights[1:-1]), 1.2 * np.max(1.0 / observed_heights[1:-1]))
    plt.ylim(0.8, 1.2 * np.max(observed_sizes))
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'Cut height $\delta$ ($1/\delta$)')
    plt.ylabel(r'Largest cluster size at $\delta$')
    if plot_label:
        plt.title(r'Cluster sizes across hierarchy cuts for' + '\n' + r'{}'.format(plot_label))

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.setp(ax.spines.values(), color='#555555')
    plt.grid(color='#555555', linestyle='dotted', alpha=0.25)

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    legend.get_frame().set_alpha(0.0)
    plt.tight_layout()

    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()


def get_parser():
    parser = argparse.ArgumentParser(description='Process observed and permuted hierarchies.')
    parser.add_argument('-oelf',  '--observed_edge_list_file',   type=str,   required=True,  help='Observed edge list filename')
    parser.add_argument('-oigf',  '--observed_index_gene_file',  type=str,   required=True,  help='Observed index-gene filename')
    parser.add_argument('-pelf',  '--permuted_edge_list_files',  type=str,   required=True,  nargs='*', help='Permuted edge list filenames')
    parser.add_argument('-pigf',  '--permuted_index_gene_files', type=str,   required=True,  nargs='*', help='Permuted index-gene filenames')
    parser.add_argument('-lsb',   '--lower_size_bound',          type=float, required=False, default=10.0, help='Lower bound for cut size')
    parser.add_argument('-usb',   '--upper_size_bound',          type=float, required=False, default=float('inf'), help='Upper bound for cut size')
    parser.add_argument('-nc',    '--num_cores',                 type=int,   required=False, default=1, help='Number of cores')
    parser.add_argument('-cf',    '--cluster_file',              type=str,   required=False, help='Cluster filename')
    parser.add_argument('-pf',    '--plot_file',                 type=str,   required=False, help='Plot filename')
    parser.add_argument('-pl',    '--plot_label',                type=str,   required=False, nargs='*', help='Plot label')
    parser.add_argument('-osf',   '--observed_size_file',        type=str,   required=False, help='Observed cluster size filename')
    parser.add_argument('-esf',   '--expected_size_file',        type=str,   required=False, help='Expected cluster size filename')
    parser.add_argument('-minsf', '--min_size_file',             type=str,   required=False, help='Minimum cluster size filename')
    parser.add_argument('-maxsf', '--max_size_file',             type=str,   required=False, help='Maximum cluster size filename')
    parser.add_argument('-v',     '--verbose',                   action='store_true',         help='Verbose')
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

    result = process_hierarchies(
        observed_hierarchy,
        observed_index_to_gene,
        permuted_hierarchies,
        permuted_index_to_gene_list,
        lower_size_bound=args.lower_size_bound,
        upper_size_bound=args.upper_size_bound,
        n_jobs=args.num_cores,
    )

    if args.observed_size_file is not None:
        with open(args.observed_size_file, 'w') as f:
            f.write('\n'.join(f'{h}\t{s}' for h, s in zip(result.observed_heights, result.observed_sizes)))
    if args.expected_size_file is not None:
        with open(args.expected_size_file, 'w') as f:
            f.write('\n'.join(f'{h}\t{s}' for h, s in zip(result.distinct_heights, result.expected_sizes)))
    if args.min_size_file is not None:
        with open(args.min_size_file, 'w') as f:
            f.write('\n'.join(f'{h}\t{s}' for h, s in zip(result.distinct_heights, result.min_sizes)))
    if args.max_size_file is not None:
        with open(args.max_size_file, 'w') as f:
            f.write('\n'.join(f'{h}\t{s}' for h, s in zip(result.distinct_heights, result.max_sizes)))

    if args.cluster_file is not None:
        sorted_clusters = sorted(sorted(map(sorted, result.observed_clusters)), key=len, reverse=True)
        cluster_lines = '\n'.join('\t'.join(c) for c in sorted_clusters)
        header = (
            f'# Observed cut height: {result.observed_cut_height}\n'
            f'# Observed size of largest cluster at observed cut height: {result.observed_cut_size}\n'
            f'# Expected size of largest cluster at observed cut height: {result.expected_cut_size}\n'
            f'# Observed maximum ratio statistic: {result.observed_cut_ratio:.3f}\n'
            f'# Expected maximum ratio statistic: {result.expected_cut_ratio:.3f}\n'
            f'# p-value: {result.p_value}\n'
            f'# Clusters:\n'
        )
        with open(args.cluster_file, 'w') as f:
            f.write(header + cluster_lines)

    if args.plot_file is not None:
        if args.verbose:
            progress('Plotting cluster sizes...')
        plot_cluster_sizes(
            result.observed_heights, result.observed_sizes,
            result.distinct_heights, result.min_sizes,
            result.distinct_heights, result.expected_sizes,
            result.distinct_heights, result.max_sizes,
            result.permuted_heights_collection, result.permuted_sizes_collection,
            result.observed_cut_height,
            ' '.join(args.plot_label) if args.plot_label else '',
            args.plot_file,
        )

    if args.verbose:
        progress()


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
