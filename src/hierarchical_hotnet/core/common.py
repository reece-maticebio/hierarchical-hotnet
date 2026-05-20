#!/usr/bin/python

import math, warnings
import numpy as np

####################################################################################################################################
#
# Method functions
#
####################################################################################################################################

def degree_sequence(A):
    '''
    Find the degree sequence for an adjacency matrix.
    '''
    return np.sum(A, axis=0)

def laplacian_matrix(A):
    '''
    Find the Laplacian matrix.
    '''
    return np.diag(degree_sequence(A))-A

def walk_matrix(A):
    '''
    Find the walk matrix for the random walk.
    '''
    d = degree_sequence(A)
    d[np.where(d<=0)] = 1
    return np.asarray(A, dtype=np.float64)/np.asarray(d, dtype=np.float64)

def hotnet_similarity_matrix(A, t):
    '''
    Perform the diffusion process in HotNet.
    '''
    from scipy.linalg import eigh
    D, V = eigh(-t*laplacian_matrix(A))
    return np.dot(np.exp(D)*V, sp.linalg.inv(V))

def hotnet2_similarity_matrix(A, beta):
    '''
    Perform the random walk with restart process in HotNet2.
    '''
    from scipy.linalg import inv
    return beta*inv(np.eye(*np.shape(A))-(1-beta)*walk_matrix(A))

def hotnet2_similarity_matrix_iterative(A, beta, steps=1):
    '''
    Perform the random walk with restart process in HotNet2 iteratively.
    '''
    W = walk_matrix(A)
    P = np.eye(*np.shape(A))
    F = np.eye(*np.shape(A))
    for _ in range(steps):
        P[:, :] = (1-beta)*np.dot(W, P)+beta*F
    return P

def hh_similarity_matrix(A, beta):
    '''
    Construct the Hierarchical HotNet similarity matrix.
    '''
    return hotnet2_similarity_matrix(A, beta)

def drop_isolated_nodes(edges, index_to_gene):
    """Remove nodes that appear in no edge, re-indexing to stay consecutive.

    Hierarchical HotNet cannot handle isolated nodes: a node with no edges
    produces an all-zero row/column in the similarity matrix, which the
    downstream strongly-connected-component code chokes on. The integer
    indices must also be consecutive ``1..N``. This function enforces both —
    any index in ``index_to_gene`` absent from every edge is dropped, and if
    anything was dropped the survivors are renumbered ``1..M`` so the indices
    stay gap-free.

    Parameters
    ----------
    edges : iterable of (i, j) or (i, j, w)
        Edge list with integer endpoints. Tuple arity is preserved.
    index_to_gene : Mapping[int, str]

    Returns
    -------
    (edges, index_to_gene) : tuple
        The cleaned edge list and a dict copy of ``index_to_gene``. When
        there are no isolated nodes, both are returned unchanged (the edge
        list as a fresh list, the mapping as a fresh dict).

    Warns
    -----
    UserWarning
        Names how many isolated nodes were removed (and which genes), if any.
    """
    edges = list(edges)

    connected = set()
    for edge in edges:
        connected.add(edge[0])
        connected.add(edge[1])

    isolated = [idx for idx in index_to_gene if idx not in connected]
    if not isolated:
        return edges, dict(index_to_gene)

    isolated_genes = sorted(str(index_to_gene[idx]) for idx in isolated)
    shown = ", ".join(isolated_genes[:10])
    if len(isolated_genes) > 10:
        shown += f", ... (+{len(isolated_genes) - 10} more)"
    warnings.warn(
        f"Removed {len(isolated)} isolated node(s) with no edges: {shown}. "
        "Isolated nodes cannot belong to any cluster and would produce "
        "all-zero rows in the similarity matrix.",
        stacklevel=2,
    )

    surviving = sorted(idx for idx in index_to_gene if idx in connected)
    old_to_new = {old: new for new, old in enumerate(surviving, start=1)}

    new_index_to_gene = {new: index_to_gene[old] for old, new in old_to_new.items()}
    new_edges = [(old_to_new[e[0]], old_to_new[e[1]], *e[2:]) for e in edges]
    return new_edges, new_index_to_gene


def combined_similarity_matrix(P, gene_to_index, gene_to_score):
    '''
    Find the combined similarity submatrix from the topological similarity matrix and the gene
    scores.
    '''
    m, n = np.shape(P)
    min_index = min(gene_to_index.values())
    max_index = max(gene_to_index.values())

    # We do not currently support nodes without edges, which seems to be the most likely way for
    # someone to encounter this error.
    if m!=n:
        raise IndexError('The similarity matrix is not square.')
    if max_index-min_index+1!=n:
        raise IndexError('The similarity matrix and the index-gene associations have different dimensions.')

    topology_genes = set(gene_to_index)
    score_genes = set(gene_to_score)
    common_genes = sorted(topology_genes & score_genes)
    common_indices = [gene_to_index[gene]-min_index for gene in common_genes]

    if not common_genes:
        raise Exception('No network genes with gene scores.')

    f = np.array([gene_to_score[gene] for gene in common_genes])
    S = P[np.ix_(common_indices, common_indices)]*f
    common_index_to_gene = dict((i+1, gene) for i, gene in enumerate(common_genes))
    common_gene_to_index = dict((gene, i) for i, gene in common_index_to_gene.items())

    return S, common_index_to_gene, common_gene_to_index
