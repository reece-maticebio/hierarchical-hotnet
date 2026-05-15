"""Construct the hierarchical decomposition of the SCCs of the similarity matrix."""

import argparse
import math

import numpy as np

from hierarchical_hotnet._parallel import maybe_pool
from hierarchical_hotnet.common import combined_similarity_matrix
from hierarchical_hotnet.file_io import (
    load_gene_score,
    load_index_gene,
    load_matrix,
    save_edge_list,
    save_index_gene,
)
from hierarchical_hotnet.hierarchical_clustering import (
    strongly_connected_components,
    tarjan_HD,
)


def _apply_score_transform(gene_to_score, log_transform, score_threshold):
    if not math.isnan(score_threshold):
        if log_transform:
            gene_to_score = {g: s for g, s in gene_to_score.items() if s <= score_threshold}
        else:
            gene_to_score = {g: s for g, s in gene_to_score.items() if s >= score_threshold}
    if log_transform:
        gene_to_score = {
            g: (-math.log10(s) if s != 1.0 else 0.0) for g, s in gene_to_score.items()
        }
    return gene_to_score


def construct_hierarchy(
    similarity_matrix,
    index_to_gene,
    gene_to_score=None,
    *,
    log_transform=False,
    score_threshold=float('nan'),
    verbose=False,
):
    """Construct the hierarchical decomposition of the SCCs of the similarity matrix.

    Parameters
    ----------
    similarity_matrix : np.ndarray
    index_to_gene : Mapping[int, str]
        1-indexed mapping aligning with rows/columns of ``similarity_matrix``.
    gene_to_score : Mapping[str, float] or None
        Scores per gene. If ``None``, all genes get score 1.0.
    log_transform : bool
        Apply ``-log10`` to scores (with 1.0 mapped to 0.0).
    score_threshold : float
        Threshold; with ``log_transform``, keeps scores ``<= threshold``;
        otherwise keeps ``>= threshold``. ``NaN`` disables filtering.

    Returns
    -------
    T : list[tuple[int, int, float]]
        Hierarchy edges (source, target, height) of the dendrogram.
    common_index_to_gene : dict[int, str]
        Index-gene map restricted to the largest strongly connected component.
    """
    gene_to_index = {gene: idx for idx, gene in index_to_gene.items()}

    if gene_to_score is None:
        gene_to_score = {gene: 1.0 for gene in gene_to_index}
    else:
        gene_to_score = _apply_score_transform(gene_to_score, log_transform, score_threshold)

    if verbose:
        print('Processing data...')

    S, common_index_to_gene, common_gene_to_index = combined_similarity_matrix(
        similarity_matrix, gene_to_index, gene_to_score
    )

    # Restrict to a largest strongly connected component if the graph isn't connected.
    components = strongly_connected_components(S)
    if len(components) > 1:
        component = sorted(max(components, key=len))
        S = S[np.ix_(component, component)]
        common_index_to_gene = {i + 1: common_index_to_gene[j + 1] for i, j in enumerate(component)}

    if verbose:
        print('Constructing hierarchical decomposition...')

    T = tarjan_HD(np.asarray(S, dtype=np.float32), reverse=True, verbose=verbose)

    return T, common_index_to_gene


# --- batch / parallel API -----------------------------------------------------

_state: dict = {}


def _init_worker(similarity_matrix, index_to_gene, log_transform, score_threshold):
    _state['similarity_matrix'] = similarity_matrix
    _state['index_to_gene'] = index_to_gene
    _state['log_transform'] = log_transform
    _state['score_threshold'] = score_threshold


def _worker(gene_to_score):
    return construct_hierarchy(
        _state['similarity_matrix'],
        _state['index_to_gene'],
        gene_to_score=gene_to_score,
        log_transform=_state['log_transform'],
        score_threshold=_state['score_threshold'],
    )


def construct_hierarchies(
    similarity_matrix,
    index_to_gene,
    gene_to_score_sets,
    *,
    n_jobs=1,
    log_transform=False,
    score_threshold=float('nan'),
):
    """Build a hierarchy per ``gene_to_score`` set, sharing ``similarity_matrix``.

    Parameters
    ----------
    similarity_matrix, index_to_gene, log_transform, score_threshold :
        See :func:`construct_hierarchy` (shared across all sets in the batch).
    gene_to_score_sets : iterable of Mapping[str, float] or None
        One hierarchy is built per element; ``None`` elements use unit scores.
    n_jobs : int
        ``1`` runs serially. ``-1`` lets the pool pick worker count. The pool
        pickles ``similarity_matrix`` once per worker (via ``initializer``)
        rather than once per task.

    Returns
    -------
    list[tuple[T, common_index_to_gene]]
        One per input score set, in input order.
    """
    sets = list(gene_to_score_sets)
    if n_jobs == 1:
        return [
            construct_hierarchy(
                similarity_matrix, index_to_gene, gene_to_score=gs,
                log_transform=log_transform, score_threshold=score_threshold,
            )
            for gs in sets
        ]
    with maybe_pool(
        n_jobs,
        initializer=_init_worker,
        initargs=(similarity_matrix, index_to_gene, log_transform, score_threshold),
    ) as map_fn:
        return list(map_fn(_worker, sets))


# --- CLI ----------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(
        description='Construct the hierarchical decomposition of the SCCs of the Hierarchical HotNet similarity matrix.'
    )
    parser.add_argument('-smf',  '--similarity_matrix_file',   type=str,   required=True,  help='HH similarity matrix filename')
    parser.add_argument('-smn',  '--similarity_matrix_name',   type=str,   required=False, default='PPR', help='HH similarity matrix name')
    parser.add_argument('-igf',  '--index_gene_file',          type=str,   required=True,  help='Index-gene filename')
    parser.add_argument('-gsf',  '--gene_score_file',          type=str,   required=False, help='Gene-score filename')
    parser.add_argument('-lt',   '--log_transform',            action='store_true',         help='Log transform scores')
    parser.add_argument('-st',   '--score_threshold',          type=float, required=False, default=float('nan'), help='Score threshold')
    parser.add_argument('-helf', '--hierarchy_edge_list_file', type=str,   required=True,  help='Hierarchy edge list filename')
    parser.add_argument('-higf', '--hierarchy_index_gene_file', type=str,  required=True,  help='Hierarchy index-gene filename')
    parser.add_argument('-v',    '--verbose',                  action='store_true',         help='Verbose')
    return parser


def run(args):
    if args.verbose:
        print('Loading data...')

    index_to_gene, _ = load_index_gene(args.index_gene_file)
    P = load_matrix(args.similarity_matrix_file, args.similarity_matrix_name)
    gene_to_score = load_gene_score(args.gene_score_file) if args.gene_score_file else None

    T, common_index_to_gene = construct_hierarchy(
        P,
        index_to_gene,
        gene_to_score=gene_to_score,
        log_transform=args.log_transform,
        score_threshold=args.score_threshold,
        verbose=args.verbose,
    )

    if args.verbose:
        print('Saving results...')

    save_edge_list(args.hierarchy_edge_list_file, T)
    save_index_gene(args.hierarchy_index_gene_file, common_index_to_gene)


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
