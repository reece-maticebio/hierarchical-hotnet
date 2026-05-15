"""Permute networks with the double-edge-swap algorithm."""

import argparse
import math
import random

import networkx as nx

from hierarchical_hotnet._parallel import maybe_pool
from hierarchical_hotnet.file_io import load_edge_list, save_edge_list


def permute_network(edges, *, seed=None, preserve_connectivity=False, Q=100):
    """Return a permuted edge list using double-edge-swap.

    Parameters
    ----------
    edges : iterable of (u, v)
        Undirected, unweighted edge list (integers).
    seed : int or None
        Seed for :mod:`random`. ``None`` leaves the global state unchanged.
    preserve_connectivity : bool
        If True, run the connected double-edge-swap on the largest connected
        component, preserving connectivity.
    Q : float
        Each call requests at least ``Q * |E|`` successful edge swaps.

    Returns
    -------
    list[tuple[int, int]]
        Permuted edge list.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    if seed is not None:
        random.seed(seed)

    minimum_swaps = int(math.ceil(Q * G.number_of_edges()))

    if not preserve_connectivity:
        G = nx.double_edge_swap(G, minimum_swaps, 2**30)
    else:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        # The connected double-edge-swap algorithm does not guarantee a minimum
        # number of successful swaps, so we iterate until reached.
        current_swaps = 0
        while current_swaps < minimum_swaps:
            remaining_swaps = max(minimum_swaps - current_swaps, 100)
            current_swaps += nx.connected_double_edge_swap(G, remaining_swaps)

    return list(G.edges())


# --- batch / parallel API -----------------------------------------------------

_state: dict = {}


def _init_worker(edges, preserve_connectivity, Q):
    _state['edges'] = edges
    _state['preserve_connectivity'] = preserve_connectivity
    _state['Q'] = Q


def _worker(seed):
    return permute_network(
        _state['edges'],
        seed=seed,
        preserve_connectivity=_state['preserve_connectivity'],
        Q=_state['Q'],
    )


def permute_network_many(edges, *, seeds, n_jobs=1, preserve_connectivity=False, Q=100):
    """Generate a batch of permuted networks.

    Parameters
    ----------
    edges, preserve_connectivity, Q :
        See :func:`permute_network`.
    seeds : iterable of int
    n_jobs : int
        ``1`` runs serially. ``-1`` lets the pool pick worker count.
    """
    seeds = list(seeds)
    if n_jobs == 1:
        return [permute_network(edges, seed=s, preserve_connectivity=preserve_connectivity, Q=Q) for s in seeds]
    with maybe_pool(n_jobs, initializer=_init_worker, initargs=(edges, preserve_connectivity, Q)) as map_fn:
        return list(map_fn(_worker, seeds))


# --- CLI ----------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(description='Permute networks with the double edge swap algorithm.')
    parser.add_argument('-i', '--edge_list_file',          type=str,   required=True,  help='Edge list filename')
    parser.add_argument('-c', '--connected',               action='store_true',         help='Preserve connectivity of graph')
    parser.add_argument('-s', '--seed',                    type=int,   required=False, help='Random seed')
    parser.add_argument('-q', '--Q',                       type=float, required=False, default=100, help='Minimum of Q*|E| edge swaps')
    parser.add_argument('-o', '--permuted_edge_list_file', type=str,   required=True,  help='Permuted edge list filename')
    return parser


def run(args):
    edges = load_edge_list(args.edge_list_file, unweighted=True)
    permuted = permute_network(edges, seed=args.seed, preserve_connectivity=args.connected, Q=args.Q)
    save_edge_list(args.permuted_edge_list_file, permuted)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
