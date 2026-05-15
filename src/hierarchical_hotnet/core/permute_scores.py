"""Permute gene scores within degree-preserving bins."""

from typing import Optional

import numpy as np
from hierarchical_hotnet.parallel import maybe_pool
from hierarchical_hotnet.storage import Store


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


_state: dict = {}


def _init_worker(gene_to_score, bins):
    _state['gene_to_score'] = gene_to_score
    _state['bins'] = bins


def _worker(seed):
    return permute_scores(_state['gene_to_score'], _state['bins'], seed=seed)


def permute_scores_many(gene_to_score, bins=None, *, seeds, n_jobs=1, out: Optional[Store] = None):
    """Generate a batch of permuted gene-score dicts.

    Parameters
    ----------
    gene_to_score, bins :
        See :func:`permute_scores`.
    seeds : iterable of int
        One permutation produced per seed.
    n_jobs : int
        ``1`` runs serially. ``-1`` lets the pool pick worker count. The pool
        pickles ``gene_to_score`` and ``bins`` once per worker, not per task.
    out : Store, optional
        If provided, each permutation is written into ``out`` keyed by
        ``str(seed)`` and the populated store is returned. If ``None`` (default),
        the function returns a list of dicts in input order, matching its
        historical behavior. Use a :class:`DiskStore` here to spill permutations
        to disk instead of holding them all in memory at once.

    Returns
    -------
    list[dict] or Store
        A list when ``out is None``; the populated ``out`` store otherwise.
    """
    seeds = list(seeds)
    if out is None:
        if n_jobs == 1:
            return [permute_scores(gene_to_score, bins, seed=s) for s in seeds]
        with maybe_pool(n_jobs, initializer=_init_worker, initargs=(gene_to_score, bins)) as map_fn:
            return list(map_fn(_worker, seeds))

    # Streaming path: drain the (lazy) map iterator into the store one result
    # at a time so peak memory stays at ~one permutation plus the pool's queue.
    if n_jobs == 1:
        for seed in seeds:
            out.put(str(seed), permute_scores(gene_to_score, bins, seed=seed))
    else:
        with maybe_pool(n_jobs, initializer=_init_worker, initargs=(gene_to_score, bins)) as map_fn:
            for seed, result in zip(seeds, map_fn(_worker, seeds)):
                out.put(str(seed), result)
    return out
