#!/usr/bin/python

import math, numpy as np
from collections import defaultdict

from hierarchical_hotnet import backends

####################################################################################################################################
#
# Hierarchical clustering generation functions
#
####################################################################################################################################

def tarjan_HD(A, reverse=True, verbose=False):
    '''
    Compute the hierarchical decomposition of a graph into strongly connected components using Tarjan's algorithm (1983).
    This function is the driver function for the implementation of this algorithm.
    '''
    if verbose:
        print(f'- Clustering backend: {backends.BACKEND}')

    # Initialize variables.
    n = np.shape(A)[0]
    V = list(range(n))
    T = list()
    i = 0
    root = n

    # Compute hierarchical decomposition.
    # We reduce memory consumption by temporarily reducing the precision of the weights in the adjacency matrix, which is usually
    # fine because we do not perform arithmetic on the weights; another strategy would replace the weights by their ranks.
    if not reverse:
        weights = np.unique(A)
        m = len(weights)
        B = np.asarray(A, dtype=np.float32)

        T, root = tarjan_HD_recursive(V, B, T, i, root)

        S = list()
        for u, v, a in T:
            j = min(max(np.searchsorted(weights, a), 1), m-1)
            x = weights[j-1]
            y = weights[j]
            if abs(x-a)<abs(y-a):
                w = x
            else:
                w = y
            S.append((u, v, w))
        T = S

    else:
        # When adding edges in reverse, we "reverse" the weights in the adjacency matrix and then "reverse" the corresponding heights.
        weights = np.unique(A)
        m = len(weights)
        max_weight = weights[m-1]
        B = np.asarray(2*max_weight-A, dtype=np.float32)
        np.fill_diagonal(B, 0)

        T, root = tarjan_HD_recursive(V, B, T, i, root)

        S = list()
        for u, v, a in T:
            b = 2*max_weight-a
            j = min(max(np.searchsorted(weights, b), 1), m-1)
            x = weights[j-1]
            y = weights[j]
            if abs(x-b)<abs(y-b):
                w = x
            else:
                w = y
            S.append((u, v, w))
        T = S

    T = [(u+1, v+1, w) for u, v, w in T]

    return T

def tarjan_HD_recursive(V, A, T, i, root):
    '''
    Compute the hierarchical decomposition of a graph into strongly connected components using Tarjan's algorithm (1983).
    This function is not the driver function for the implementation of this algorithm; call the tarjan_HD function directly instead
    of this function.
    '''
    weights = find_distinct_weights(A)
    m = len(weights)-1
    r = m-i

    if r==1:
        # Case 1
        weight_m = weights[m]
        del weights

        for v in V:
            T.append((v, root, weight_m))
        return T, root

    else:
        # Case 2
        j = int(math.ceil(0.5*float(i+m)))
        weight_i = weights[i]
        weight_j = weights[j]
        del weights

        A_j = threshold_edges(A, weight_j)
        components = strongly_connected_components(A_j)

        if len(components)==1:
            # Case 2a
            return tarjan_HD_recursive(V, A_j, T, i, root)
        else:
            # Case 2b
            Y = list()
            for component in components:
                if len(component)>1:
                    X = index_vertices(V, component)
                    B = slice_array(A_j, component, component)
                    k = subproblem_index(B, weight_i)
                    subtree, root = tarjan_HD_recursive(X, B, list(), k, root)
                    T.extend(subtree)
                    Y.append(root)
                    root += 1
                else:
                    Y.extend(index_vertices(V, component))

            B = condense_graph(A, components)
            k = subproblem_index(B, weight_j)
            return tarjan_HD_recursive(Y, B, T, k, root)

def condense_graph(A, components):
    return backends.condense_adjacency_matrix(A, components)

def find_distinct_weights(A):
    return backends.find_distinct_weights(A)

def index_vertices(vertices, indices):
    return [vertices[index] for index in indices]

def slice_array(A, rows, columns):
    return backends.slice_array(A, rows, columns)

def strongly_connected_components(A):
    """Return the SCCs of ``A`` as a list of component lists (0-indexed nodes)."""
    indices = backends.strongly_connected_components_labels(A)
    index_to_component = defaultdict(list)
    for i, j in enumerate(indices):
        index_to_component[j].append(i)
    return list(index_to_component.values())

def subproblem_index(A, weight):
    B = find_distinct_weights(A)[1:]
    i = np.searchsorted(B, weight, side='right')
    return i

def threshold_edges(A, weight):
    return backends.threshold_edges(A, weight)

####################################################################################################################################
#
# Hierarchical clustering processing functions
#
####################################################################################################################################

def find_height_to_clusters(preordered_T, index_to_gene, reverse=True):
    '''
    Find clusters for every distinct height of the dendrogram.
    '''
    # Associate nodes of dendrogram with indices.
    index_to_node = defaultdict(set)
    for index, gene in index_to_gene.items():
        index_to_node[index] = frozenset([gene])

    # Initialize clusters with leaf nodes.
    height_to_clusters = dict()
    height = float('inf')
    clusters = set(index_to_node.values())
    height_to_clusters[height] = clusters.copy()

    # Update clusters while ascending dendrogram.
    T = sorted(preordered_T, key=lambda x: x[2], reverse=reverse)
    m = len(T)

    for i, edge in enumerate(T):
        source, target, height = edge

        a = index_to_node[source]
        b = index_to_node[target]
        c = frozenset(set(a) | set(b))

        clusters.discard(a)
        clusters.discard(b)
        clusters.add(c)

        del index_to_node[source]
        index_to_node[target] = c

        if i==m-1 or height!=T[i+1][2]:
            height_to_clusters[height] = clusters.copy()

    # Add cluster for root node.
    height = 0.0
    clusters = set(frozenset(index_to_node.values()))
    height_to_clusters[height] = clusters.copy()

    return height_to_clusters

def find_height_to_sizes(T, index_to_gene, reverse=True):
    '''
    Find composition of clusters for every distinct height of the dendrogram.
    '''
    index_to_index = dict((index, index) for index, gene in index_to_gene.items())
    height_to_clusters = find_height_to_clusters(T, index_to_index, reverse)
    height_to_sizes = dict((height, map(len, clusters)) for height, clusters in height_to_clusters.items())
    return height_to_sizes

def find_cut(preordered_T, index_to_gene, threshold, reverse=True):
    '''
    Find clusters for specific height in dendrogram.
    '''
    # Associate nodes of dendrogram with indices.
    index_to_node = defaultdict(set)
    for index, gene in index_to_gene.items():
        index_to_node[index] = frozenset([gene])

    # Initialize clusters with leaf nodes.
    clusters = set(index_to_node.values())

    # Update clusters while ascending dendrogram.
    T = sorted(preordered_T, key=lambda x: x[2], reverse=reverse)
    m = len(T)

    for k, edge in enumerate(T):
        source, target, height = edge
        if (not reverse and height>threshold) or (reverse and height<threshold):
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
