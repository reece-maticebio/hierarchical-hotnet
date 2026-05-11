"""Permute gene scores within degree-preserving bins."""

import argparse

import numpy as np

from hierarchical_hotnet._parallel import maybe_pool
from hierarchical_hotnet.hhio import load_gene_score, save_gene_score


def load_bins(filename):
    bins = []
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                bins.append(set(l.rstrip('\n').split('\t')))
    return bins


def permute_scores(gene_to_score, bins=None, *, seed=None):
    """Return a permuted ``gene_to_score`` dict.

    Scores are permuted independently within each bin. If ``bins`` is ``None``,
    all genes are permuted in a single bin.

    Parameters
    ----------
    gene_to_score : Mapping[str, float]
    bins : iterable of iterable of str, optional
        Each bin lists the genes that are interchangeable under permutation.
    seed : int or None
        Seed for ``numpy.random``. ``None`` leaves the global state unchanged.
    """
    if bins is None:
        bins = [sorted(gene_to_score)]

    genes = sorted(gene_to_score)
    scores = np.array([gene_to_score[gene] for gene in genes])

    if seed is not None:
        np.random.seed(seed)

    for permute_genes in bins:
        permute_genes = set(permute_genes)
        permute_indices = [i for i, gene in enumerate(genes) if gene in permute_genes]
        scores[permute_indices] = np.random.permutation(scores[permute_indices])

    return dict(zip(genes, scores))


# --- batch / parallel API -----------------------------------------------------

_state: dict = {}


def _init_worker(gene_to_score, bins):
    _state['gene_to_score'] = gene_to_score
    _state['bins'] = bins


def _worker(seed):
    return permute_scores(_state['gene_to_score'], _state['bins'], seed=seed)


def permute_scores_many(gene_to_score, bins=None, *, seeds, n_jobs=1):
    """Generate a batch of permuted gene-score dicts.

    Parameters
    ----------
    gene_to_score, bins :
        See :func:`permute_scores`.
    seeds : iterable of int
        One permutation produced per seed; results returned in the same order.
    n_jobs : int
        ``1`` runs serially. ``-1`` lets the pool pick worker count. The pool
        pickles ``gene_to_score`` and ``bins`` once per worker, not per task.
    """
    seeds = list(seeds)
    if n_jobs == 1:
        return [permute_scores(gene_to_score, bins, seed=s) for s in seeds]
    with maybe_pool(n_jobs, initializer=_init_worker, initargs=(gene_to_score, bins)) as map_fn:
        return list(map_fn(_worker, seeds))


# --- CLI ----------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(description='Permute gene scores.')
    parser.add_argument('-i',  '--gene_score_file', type=str, required=True,  help='Input filename')
    parser.add_argument('-bf', '--bin_file',        type=str, required=False, help='Bin filename')
    parser.add_argument('-s',  '--seed',            type=int, required=False, help='Random seed')
    parser.add_argument('-o',  '--output_file',     type=str, required=True,  help='Output filename')
    return parser


def run(args):
    gene_to_score = load_gene_score(args.gene_score_file)
    bins = load_bins(args.bin_file) if args.bin_file else None
    permuted = permute_scores(gene_to_score, bins, seed=args.seed)
    save_gene_score(args.output_file, permuted)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
