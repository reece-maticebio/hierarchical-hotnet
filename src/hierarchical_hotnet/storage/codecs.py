"""Codecs for serializing pipeline artifacts to and from disk.

Each codec pairs an artifact type with a file format. The on-disk formats
match what ``hhnet-*`` CLI commands have always produced, so a run that
spills via :class:`DiskStore` is interoperable with intermediate files
written by the standalone scripts.

Adding a new codec is a matter of implementing the
:class:`hierarchical_hotnet.storage.Codec` protocol — ``extension``,
``write``, ``read``.
"""

from __future__ import annotations

from pathlib import Path

from hierarchical_hotnet.file_io import (
    load_edge_list,
    load_gene_score,
    load_index_gene,
    save_edge_list,
    save_gene_score,
    save_index_gene,
)


class ScoreMapCodec:
    """Codec for ``dict[str, float]`` gene-score maps.

    Matches the TSV format produced by ``hhnet-permute-scores``.
    """

    extension = ".tsv"

    def write(self, value: dict, path: Path) -> None:
        save_gene_score(str(path), value)

    def read(self, path: Path) -> dict:
        # Pass -inf so the codec round-trips every score, including negatives.
        # load_gene_score defaults to threshold=0.0 for CLI ergonomics, which
        # would silently drop negative scores on read.
        return load_gene_score(str(path), score_threshold=float("-inf"))


class HierarchyCodec:
    """Codec for ``(hierarchy_edges, index_to_gene)`` tuples.

    Each hierarchy is two files: ``<key>.edges.tsv`` (the dendrogram edges
    as ``source\\ttarget\\theight`` lines) and ``<key>.genes.tsv`` (the
    surviving index-gene map). This pair matches what
    ``hhnet-construct-hierarchy`` writes. Existence checks key off the
    ``.edges.tsv`` file; the genes file is assumed to live alongside it.
    """

    extension = ".edges.tsv"

    @staticmethod
    def _genes_path(edges_path: Path) -> Path:
        # Replace only the trailing '.edges.tsv' so callers can use any prefix.
        name = edges_path.name
        if not name.endswith(".edges.tsv"):
            raise ValueError(f"path {edges_path!r} does not end with .edges.tsv")
        return edges_path.with_name(name[: -len(".edges.tsv")] + ".genes.tsv")

    def write(self, value, path: Path) -> None:
        T, index_to_gene = value
        save_edge_list(str(path), T)
        save_index_gene(str(self._genes_path(path)), index_to_gene)

    def read(self, path: Path):
        T = load_edge_list(str(path))
        index_to_gene, _ = load_index_gene(str(self._genes_path(path)))
        return T, index_to_gene
