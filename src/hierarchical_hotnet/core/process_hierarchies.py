"""Process observed and permuted hierarchies."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import numpy as np
from hierarchical_hotnet import backends
from hierarchical_hotnet.file_io import load_edge_list, load_index_gene


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
    statistics.append(compute_statistic((len(node) for node in index_to_node.values())))
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
            statistics.append(compute_statistic((len(node) for node in index_to_node.values())))
            heights.append(height)
    height = 0.0 if reverse else float('inf')
    statistics.append(compute_statistic((len(node) for node in index_to_node.values())))
    heights.append(height)
    return (np.array(heights), np.array(statistics))


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
    distinct_heights = np.unique(np.concatenate(permuted_heights_collection))[::-1]
    max_indices = np.array([len(h) for h in permuted_heights_collection], dtype=np.int64)
    min_sizes, expected_sizes, max_sizes = backends.summarize_sizes(
        distinct_heights, permuted_heights_collection, permuted_sizes_collection, max_indices,
    )
    return (distinct_heights, min_sizes, expected_sizes, max_sizes)


def find_cut(observed_heights, observed_sizes, expected_heights, expected_sizes, lower_size_bound, upper_size_bound):
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
    return (observed_heights[max_index], ratios[max_index])


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
        if not reverse and height > threshold or (reverse and height < threshold):
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
        return (map, lambda: None)
    workers = None if num_workers in (-1, None) else num_workers
    executor = ProcessPoolExecutor(max_workers=workers)
    return (executor.map, executor.shutdown)


def process_hierarchies(observed_hierarchy, observed_index_to_gene, permuted_hierarchies, permuted_index_to_gene_list, *, lower_size_bound=10.0, upper_size_bound=float('inf'), n_jobs=1):
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
    # When both inputs are sized (the historical case — both are lists), we can
    # still cheaply validate their lengths match. Iterables without __len__
    # (e.g. generators that pull hierarchies from disk one at a time) skip the
    # check; zip will simply stop at the shorter side if they diverge.
    if hasattr(permuted_hierarchies, '__len__') and hasattr(permuted_index_to_gene_list, '__len__'):
        if len(permuted_hierarchies) != len(permuted_index_to_gene_list):
            raise ValueError('permuted_hierarchies and permuted_index_to_gene_list lengths differ')

    def _stats_input():
        # Generator — keeps peak memory at one hierarchy at a time when the
        # caller passes lazy iterables (DiskStore iteration in the pipeline).
        yield (observed_hierarchy, observed_index_to_gene)
        yield from zip(permuted_hierarchies, permuted_index_to_gene_list)

    map_fn, cleanup = _resolve_map(n_jobs)
    try:
        stats_output = list(map_fn(_statistics_worker, _stats_input()))
    finally:
        cleanup()
    observed_heights, observed_sizes = stats_output[0]
    permuted_heights_collection, permuted_sizes_collection = zip(*stats_output[1:])
    distinct_heights, min_sizes, expected_sizes, max_sizes = summarize_cluster_sizes(permuted_heights_collection, permuted_sizes_collection)
    map_fn, cleanup = _resolve_map(n_jobs)
    try:
        cut_input = [(observed_heights, observed_sizes, distinct_heights, expected_sizes, lower_size_bound, upper_size_bound)]
        cut_input += [(permuted_heights, permuted_sizes, distinct_heights, expected_sizes, lower_size_bound, upper_size_bound) for permuted_heights, permuted_sizes in zip(permuted_heights_collection, permuted_sizes_collection)]
        cut_output = list(map_fn(_find_cut_worker, cut_input))
    finally:
        cleanup()
    observed_cut_height, observed_cut_ratio = cut_output[0]
    _, permuted_cut_ratios = zip(*cut_output[1:])
    expected_cut_ratio = float(np.mean(permuted_cut_ratios))
    p_value = sum((r >= observed_cut_ratio for r in permuted_cut_ratios)) / float(len(permuted_cut_ratios))
    observed_clusters = find_clusters(observed_hierarchy, observed_index_to_gene, observed_cut_height)
    observed_cut_size = compute_statistic((len(c) for c in observed_clusters))
    expected_cut_size = observed_cut_size / observed_cut_ratio if observed_cut_ratio else float('nan')
    return ProcessHierarchiesResult(observed_clusters=observed_clusters, observed_cut_height=observed_cut_height, observed_cut_ratio=observed_cut_ratio, expected_cut_ratio=expected_cut_ratio, p_value=p_value, observed_cut_size=observed_cut_size, expected_cut_size=expected_cut_size, observed_heights=observed_heights, observed_sizes=observed_sizes, distinct_heights=distinct_heights, min_sizes=min_sizes, expected_sizes=expected_sizes, max_sizes=max_sizes, permuted_heights_collection=permuted_heights_collection, permuted_sizes_collection=permuted_sizes_collection)


def plot_cluster_sizes(observed_heights, observed_sizes, min_heights, min_sizes, expected_heights, expected_sizes, max_heights, max_sizes, permuted_heights_collection, permuted_sizes_collection, cut_height, plot_label, plot_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    observed_heights = np.clip(observed_heights, 1e-15, float('inf'))
    min_heights = np.clip(min_heights, 1e-15, float('inf'))
    expected_heights = np.clip(expected_heights, 1e-15, float('inf'))
    max_heights = np.clip(max_heights, 1e-15, float('inf'))
    permuted_heights_collection = [np.clip(h, 1e-15, float('inf')) for h in permuted_heights_collection]
    observed_color = (0.8, 0.0, 0.0)
    permuted_color = (0.0, 0.0, 0.8)
    background_permuted_color = (0.5, 0.5, 0.8)
    alpha = 0.2
    plt.figure(figsize=(5, 5))
    plt.step(1.0 / observed_heights, observed_sizes, where='post', c=observed_color, linewidth=2, zorder=5, label='Observed sizes')
    plt.step(1.0 / expected_heights, expected_sizes, where='post', c=permuted_color, linewidth=2, zorder=4, label='Expected sizes')
    if cut_height:
        i = max((k for k, height in enumerate(observed_heights) if height >= cut_height))
        j = max((k for k, height in enumerate(expected_heights) if height >= cut_height))
        plt.plot((1.0 / cut_height, 1.0 / cut_height), (observed_sizes[i], expected_sizes[j]), c='k', linewidth=2, alpha=0.75, zorder=6, label='Chosen cut')
    plt.step(1.0 / min_heights, min_sizes, where='post', c=background_permuted_color, linewidth=1, linestyle='dotted', zorder=3, label='Permuted sizes (minimum)')
    plt.step([float('nan')], [float('nan')], where='post', c=background_permuted_color, linewidth=1, zorder=1, label='Permuted sizes (all)')
    for permuted_heights, permuted_sizes in zip(permuted_heights_collection, permuted_sizes_collection):
        plt.step(1.0 / permuted_heights, permuted_sizes, where='post', c=background_permuted_color, linewidth=0.5, alpha=alpha, zorder=1)
    plt.step(1.0 / max_heights, max_sizes, where='post', c=background_permuted_color, linewidth=1, linestyle='dashed', zorder=3, label='Permuted sizes (maximum)')
    plt.xlim(0.8 * np.min(1.0 / observed_heights[1:-1]), 1.2 * np.max(1.0 / observed_heights[1:-1]))
    plt.ylim(0.8, 1.2 * np.max(observed_sizes))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cut height $\\delta$ ($1/\\delta$)')
    plt.ylabel('Largest cluster size at $\\delta$')
    if plot_label:
        plt.title('Cluster sizes across hierarchy cuts for' + '\n' + '{}'.format(plot_label))
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.setp(ax.spines.values(), color='#555555')
    plt.grid(color='#555555', linestyle='dotted', alpha=0.25)
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    legend.get_frame().set_alpha(0.0)
    plt.tight_layout()
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
