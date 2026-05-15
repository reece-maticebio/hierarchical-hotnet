"""Verify tarjan_HD against an independent naive reference implementation.

Ported from the legacy ``hierarchical_clustering_tests.py`` script that used
to live inside ``src/``. The naive_HD algorithm here is O(n^3 log n) — it
walks edges in weight order and recomputes SCCs at each weight change. It
exists purely as a slow-but-obvious oracle for the fast tarjan_HD
implementation.

The Tarjan-1983 example (graph from Figure 1 of Tarjan 1983) is the canonical
correctness check for tarjan_HD. The random-graph parametrization is a small
fuzz layer; capped at n=13 to keep total test time under a second.
"""

import networkx as nx
import numpy as np
import pytest

from hierarchical_hotnet.core.clustering import (
    find_height_to_clusters,
    tarjan_HD,
)


def _edges_to_adjacency_matrix(edges):
    nodes = sorted(set(e[0] for e in edges) | set(e[1] for e in edges))
    node_to_index = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for u, v, w in edges:
        A[node_to_index[u], node_to_index[v]] = w
    return nodes, A


def _tarjan_1983_example():
    """The graph from Figure 1 of Tarjan (1983)."""
    edges = [
        ('a', 'b', 10), ('b', 'a', 12), ('b', 'c', 30), ('d', 'c', 6),
        ('d', 'e', 16), ('e', 'd', 13), ('e', 'f', 8), ('f', 'a', 26),
        ('a', 'g', 15), ('g', 'b', 35), ('c', 'g', 45), ('g', 'c', 22),
        ('d', 'g', 14), ('g', 'e', 50), ('f', 'g', 20),
    ]
    nodes, A = _edges_to_adjacency_matrix(edges)
    return nodes, A


def _naive_HD(A, reverse=False):
    """Reference hierarchical decomposition via repeated SCC computation.

    Walk edges in weight order; whenever the SCC partition changes, emit
    merge events into the hierarchy. Quadratic-ish per weight class; only
    suitable as a correctness oracle on small graphs.
    """
    num_nodes = A.shape[0]
    nodes = list(range(num_nodes))
    edges = sorted(
        ((i, j, A[i, j]) for i in nodes for j in nodes if i != j and A[i, j] != 0),
        reverse=reverse,
        key=lambda edge: edge[2],
    )

    num_components = num_nodes
    next_id = num_nodes
    ancestors = list(nodes)
    T = []

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for k, edge in enumerate(edges):
        G.add_edge(edge[0], edge[1], weight=edge[2])
        is_last_of_weight_class = k == len(edges) - 1 or edges[k + 1][2] != edge[2]
        if not is_last_of_weight_class:
            continue
        components = list(nx.strongly_connected_components(G))
        if len(components) != num_components:
            num_components = len(components)
            for component in components:
                component_ancestors = set(ancestors[i] for i in component)
                if len(component_ancestors) > 1:
                    for i in component:
                        ancestors[i] = next_id
                    for i in component_ancestors:
                        T.append((i, next_id, edge[2]))
                    next_id += 1
        if len(components) == 1:
            break

    return [(u + 1, v + 1, w) for u, v, w in T]


def _trees_are_equivalent(S, T, index_to_gene, reverse=False):
    return (
        find_height_to_clusters(S, index_to_gene, reverse=reverse)
        == find_height_to_clusters(T, index_to_gene, reverse=reverse)
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_tarjan_HD_matches_naive_on_tarjan_1983(reverse):
    nodes, A = _tarjan_1983_example()
    index_to_gene = {i + 1: v for i, v in enumerate(nodes)}

    naive = _naive_HD(A, reverse=reverse)
    fast = tarjan_HD(A, reverse=reverse)

    assert _trees_are_equivalent(naive, fast, index_to_gene, reverse=reverse)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n", [3, 5, 8, 13])
def test_tarjan_HD_matches_naive_on_random_graphs(n, seed):
    """Small fuzz against the naive reference. Capped at n=13 for speed."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    np.fill_diagonal(A, 0.0)

    # Bridge components if not already strongly connected, so naive_HD has
    # something to merge all the way up.
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                G.add_edge(i, j, weight=A[i, j])
    components = list(nx.strongly_connected_components(G))
    for k in range(len(components) - 1):
        a = next(iter(components[k]))
        b = next(iter(components[k + 1]))
        A[a, b] = float(rng.random())
        A[b, a] = float(rng.random())

    index_to_gene = {i + 1: i for i in range(n)}
    naive = _naive_HD(A, reverse=False)
    fast = tarjan_HD(A, reverse=False)
    assert _trees_are_equivalent(naive, fast, index_to_gene, reverse=False)
