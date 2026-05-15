"""Find permutation bins for score-permutation tests."""

from collections import defaultdict
import networkx as nx


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
