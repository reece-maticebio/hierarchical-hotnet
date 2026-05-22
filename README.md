Hierarchical HotNet
=======================

Hierarchical HotNet is an algorithm for finding hierarchies of altered subnetworks. While originally developed for use with cancer mutation data on protein-protein interaction networks, Hierarchical HotNet supports any application in which scores may be associated with the nodes of a network, i.e., a vertex-weighted graph.

This fork modernizes the original package for Python 3.10+ with a `pip`-installable build (Fortran extension built automatically via `meson-python`) and adds:

* A first-class Python API: every step is importable and operates on Python objects (arrays, dicts) rather than only on files.
* Built-in parallelism: each batch step accepts an `n_jobs` parameter and runs on a `concurrent.futures.ProcessPoolExecutor`.
* An end-to-end `run_pipeline()` that composes the steps.
* **Disk-backed runs.** Passing `workdir=Path(...)` to `run_pipeline()` spills every artifact (similarity matrix, bins, permuted scores, hierarchies) to disk under a fixed layout and streams the large fan-out artifacts (hierarchies) lazily, so peak memory stays bounded even with thousands of permutations. `reuse=True` resumes interrupted runs by reading existing artifacts back instead of recomputing them.
* **Switchable backend.** The Fortran-accelerated paths can be forced or disabled at import time via the `HHNET_BACKEND` env var (`auto` / `fortran` / `python`), with the active choice exposed as `hierarchical_hotnet.backends.BACKEND`.

Setup
------------------------
Hierarchical HotNet is installable as a Python package. The Fortran extension is built automatically by `pip` via the `meson-python` build backend; a Python-only fallback is used if no Fortran compiler is available.

### Requirements
* Python &ge; 3.10 (tested on 3.12)
* A Fortran compiler such as [gfortran](https://gcc.gnu.org/wiki/GFortran) (recommended for performance)
* [GNU parallel](https://www.gnu.org/software/parallel/) (optional, only for `examples/example_commands_parallel.sh`)

Runtime dependencies (`numpy`, `scipy`, `networkx`, `h5py`) are installed automatically. `matplotlib` is an optional extra used by the plotting steps.

### Install
Install directly from the repository with `pip`, pinned to the latest release:

    pip install "hierarchical-hotnet @ git+https://github.com/reece-maticebio/hierarchical-hotnet.git@v0.3.0"
    pip install "hierarchical-hotnet[plot] @ git+https://github.com/reece-maticebio/hierarchical-hotnet.git@v0.3.0"   # with matplotlib for plotting

Drop the `@v0.3.0` suffix to install the current `master` instead of a tagged release.

To work on the package itself, clone it and install in editable mode:

    git clone https://github.com/reece-maticebio/hierarchical-hotnet.git
    cd hierarchical-hotnet
    pip install -e '.[plot]'

Installation exposes both a Python module (`import hierarchical_hotnet as hhn`) and a single CLI entry point (`hhnet`) with subcommands for each pipeline step (listed below; see `hhnet --help`). If no Fortran compiler is found at build time the install still succeeds and the package falls back to the pure-Python implementation, which is slower but otherwise equivalent.

### Backend selection

The Fortran-accelerated routines (SCC, matrix slicing/condensation, statistics aggregation) and their pure-Python equivalents both live behind a single dispatch in `hierarchical_hotnet.backends`. The selection happens once at import time:

| `HHNET_BACKEND` | Behavior |
|---|---|
| unset / `auto` | Try Fortran; warn and fall back to pure Python if the extension wasn't built. |
| `fortran` | Require the Fortran extension (raises on import if missing — useful as a CI guard against silent perf regressions). |
| `python` | Force the pure-Python path (for testing the fallback or profiling). |

The active choice is exposed as `hierarchical_hotnet.backends.BACKEND` (`"fortran"` or `"python"`).

### Testing
To verify the install end-to-end:

    sh examples/example_commands.sh

This runs the full pipeline on a 25-node toy graph and should finish in well under a minute. `examples/example_commands_parallel.sh` does the same with GNU parallel.

For a Python-API test:

    pytest tests/

Algorithm overview
----------------
Given a network and a per-node score, Hierarchical HotNet finds **subnetworks where high-scoring nodes cluster topologically** — not just "top-N scored nodes," and not just "neighbors of top-scored nodes," but coherent regions of the network that are statistically enriched.

The high-level idea:

1. **Diffuse** scores through the network with personalized PageRank, so connected nodes "see" each other even at distance.
2. **Decompose** the score-weighted network into a multiscale hierarchy of strongly connected components, so you don't have to commit to a single clustering resolution.
3. **Compare** the observed hierarchy to a null built from many score permutations to pick the resolution and assess significance.
4. **Combine** results across multiple score sets (or networks) into a consensus.

Input formats
----------------
Three TSV files together define a network with per-node scores. These match the original HotNet2 input formats.

##### Index-to-gene file
Associates each gene with an integer index (used for the edge list and the similarity matrix):

    1   ABC
    2   DEF

##### Edge list file
Defines the network using indices from the index-to-gene file (optional weight column):

    1    2

##### Gene-to-score file
Associates each gene with a score:

    ABC 0.5
    DEF 0.2

Python object formats
----------------
The TSV loaders (`hhn.load_edge_list`, `hhn.load_index_gene`, `hhn.load_gene_score`) produce plain Python objects. If you already have your data as a NetworkX graph, a pandas DataFrame, or similar, you can build these objects directly — no file round-trip needed.

### Data shapes

| Object | Python type | Notes |
|---|---|---|
| `index_to_gene` | `dict[int, str]` | 1-based **consecutive** integer indices `1..N`. Used to align similarity-matrix rows/columns with gene names. |
| `edges` (weighted) | `list[tuple[int, int, float]]` | Indices match `index_to_gene`. Used by `compute_similarity_matrix` and `run_pipeline`. |
| `edges` (unweighted, integer-indexed) | `list[tuple[int, int]]` | Used by `permute_network` / `permute_network_many`. |
| `edges` (gene-labeled, unweighted) | `list[tuple[str, str]]` | Used by `compute_permutation_bins`. |
| `ConsensusInput.edges` | `set[frozenset[str]]` | Undirected gene-labeled edge set used by `perform_consensus`. |
| `gene_to_score` | `dict[str, float]` | Non-negative scores. Genes absent from this dict are dropped from the analysis. |
| `bins` | `list[list[str]]` (or `list[set[str]]`) | Each inner collection is a degree-stratified bin of gene names. |
| `gene_to_score_sets` | iterable of `dict[str, float]` | For `construct_hierarchies` — one hierarchy is built per element. |
| `similarity_matrix` | `np.ndarray`, shape `(N, N)` | Row/column `index - min_index` corresponds to `index_to_gene[index]`. |
| `seeds` | iterable of `int` | RNG seeds for the batch permutation functions. |

**Invariants:**

* Indices in `index_to_gene` and `edges` must be **consecutive integers** with no gaps (`load_index_gene` produces `1..N`). The similarity matrix dimension is `max(index) - min(index) + 1`.
* Scores should be **non-negative**. For p-values, use `-log10(p)` (or pass `log_transform=True` to `construct_hierarchy`) so "more significant" maps to "larger score".
* Genes with no entry in `gene_to_score` are silently excluded from the hierarchy and the score-permutation null.

### Building inputs from a NetworkX graph and pandas Series

```python
import networkx as nx
import pandas as pd
import hierarchical_hotnet as hhn

# G : networkx.Graph with gene-name nodes (optional 'weight' edge attribute)
# scores_df : pandas.DataFrame with columns ['gene', 'score']

genes = sorted(G.nodes())
index_to_gene = {i + 1: gene for i, gene in enumerate(genes)}
gene_to_index = {gene: i + 1 for i, gene in enumerate(genes)}

edges = [
    (gene_to_index[u], gene_to_index[v], float(data.get("weight", 1.0)))
    for u, v, data in G.edges(data=True)
]

gene_to_score = dict(zip(scores_df["gene"], scores_df["score"]))

result = hhn.run_pipeline(
    edges,
    index_to_gene,
    {"my_scores": gene_to_score},
    num_permutations=100,
    n_jobs=4,
)
```

### Building inputs from a pandas edge-list DataFrame

```python
# edges_df : DataFrame with columns ['source', 'target'] and optional 'weight'
genes = sorted(set(edges_df["source"]) | set(edges_df["target"]))
gene_to_index = {gene: i + 1 for i, gene in enumerate(genes)}
index_to_gene = {i + 1: gene for i, gene in enumerate(genes)}

if "weight" in edges_df.columns:
    edges = [
        (gene_to_index[s], gene_to_index[t], float(w))
        for s, t, w in zip(edges_df["source"], edges_df["target"], edges_df["weight"])
    ]
else:
    edges = [
        (gene_to_index[s], gene_to_index[t], 1.0)
        for s, t in zip(edges_df["source"], edges_df["target"])
    ]
```

### Converting between formats

When you need to move between integer- and gene-labeled edges (e.g. you want to call `compute_permutation_bins` after building the similarity matrix), do this:

```python
# integer-indexed → gene-labeled (for compute_permutation_bins)
gene_edges = [(index_to_gene[i], index_to_gene[j]) for i, j, _ in edges]

# gene-labeled → ConsensusInput edges (a set of frozensets)
consensus_edges = {frozenset((u, v)) for u, v in gene_edges}
```

`run_pipeline` does all these conversions for you internally — you only need to do them manually when calling the individual step functions.

Pipeline steps
----------------
Each step has both a CLI subcommand (`hhnet <step>`) and a Python function (importable from `hierarchical_hotnet`). Batch steps that are embarrassingly parallel accept an `n_jobs` parameter.

### 1. Construct similarity matrix
**Purpose.** Compute personalized PageRank `P` where `P[i, j]` is the probability of a random walk landing at `i` when started at `j`. This turns the sparse network into a continuous "topological closeness" measure. The restart probability `β` controls locality; if unspecified it is auto-chosen to balance edge mass.

**CLI:**

    hhnet construct-similarity-matrix \
        -i  network_edge_list.tsv \
        -o  similarity_matrix.h5 \
        -bof beta.txt

**Python:**

```python
import hierarchical_hotnet as hhn

edges = hhn.load_edge_list("network_edge_list.tsv")
P, beta = hhn.compute_similarity_matrix(edges, beta=None)   # auto-pick beta
```

Returns `(similarity_matrix: np.ndarray, beta: float)`.

### 2. Find permutation bins
**Purpose.** Group scored genes by network degree so the score-permutation null can shuffle scores *within* degree bins. This prevents degree-correlated artifacts (hubs are studied more, so they tend to be scored more; without binning, hubs would always appear "enriched" by chance).

**CLI:**

    hhnet find-permutation-bins \
        -igf network_index_gene.tsv \
        -elf network_edge_list.tsv \
        -gsf scores.tsv \
        -ms  1000 \
        -o   score_bins.tsv

**Python:**

```python
index_to_gene, _ = hhn.load_index_gene("network_index_gene.tsv")
edges_idx = hhn.load_edge_list("network_edge_list.tsv", unweighted=True)
gene_edges = [(index_to_gene[i], index_to_gene[j]) for i, j in edges_idx]
gene_to_score = hhn.load_gene_score("scores.tsv")

bins = hhn.compute_permutation_bins(gene_edges, gene_to_score, min_size=1000)
```

Returns `list[list[str]]` — each inner list is a bin of interchangeable gene names.

### 3. Permute scores
**Purpose.** Generate the null distribution by shuffling score values **within** each degree bin. Each seed gives one independent permuted score map.

**CLI (single permutation):**

    hhnet permute-scores \
        -i  scores.tsv \
        -bf score_bins.tsv \
        -s  42 \
        -o  permuted_scores_42.tsv

**Python (single permutation):**

```python
permuted = hhn.permute_scores(gene_to_score, bins, seed=42)
```

**Python (batch, parallel):**

```python
permuted_list = hhn.permute_scores_many(
    gene_to_score, bins,
    seeds=range(1, 101),
    n_jobs=4,
)
```

`n_jobs=1` runs serially; `n_jobs>1` parallelizes across worker processes; `n_jobs=-1` lets the pool pick. Shared inputs (`gene_to_score`, `bins`) are pickled once per worker, not once per task.

### 3b. Permute network (optional)
**Purpose.** Alternative null model: instead of shuffling scores, rewire the network while preserving each node's degree (double-edge swap). Use this when the question is "is the *topology* significant?" rather than "are the *scores* significant?"

**CLI:**

    hhnet permute-network \
        -i edge_list.tsv \
        -s 1 \
        -c \                              # preserve connectivity
        -o permuted_edge_list_1.tsv

**Python:**

```python
permuted_edges = hhn.permute_network(
    edges, seed=1, preserve_connectivity=True, Q=100,
)

permuted_list = hhn.permute_network_many(
    edges, seeds=range(1, 9), n_jobs=4, preserve_connectivity=True,
)
```

Note: neither the default `examples/example_commands.sh` workflow nor `hhn.run_pipeline()` invokes these — both rely on score permutations as the null model, matching the canonical Hierarchical HotNet methodology. Network permutation is exposed for users who want to add it as an additional null themselves: generate permuted edge lists with `permute_network_many`, then run `compute_similarity_matrix` + `construct_hierarchy` on each.

### 4. Construct hierarchies
**Purpose.** Build the dendrogram. Given the similarity matrix `P` and a score map, forms a score-weighted matrix `S[i, j] = P[i, j] * score[j]`, then runs Tarjan's hierarchical decomposition: at each "cut height" `δ`, the strongly connected components of `{edges with weight ≥ δ}` form a clustering. Leaves are individual genes; as `δ` decreases, components merge upward.

You typically build many hierarchies: one for the observed scores and one per permuted score set.

**CLI (single hierarchy):**

    hhnet construct-hierarchy \
        -smf  similarity_matrix.h5 \
        -igf  network_index_gene.tsv \
        -gsf  scores.tsv \
        -helf hierarchy_edge_list.tsv \
        -higf hierarchy_index_gene.tsv

**Python (single):**

```python
T, common_idx_to_gene = hhn.construct_hierarchy(
    P, index_to_gene, gene_to_score=gene_to_score,
)
```

`T` is the dendrogram as a list of `(source, target, height)` edges.

**Python (batch, parallel):**

```python
hierarchies = hhn.construct_hierarchies(
    P, index_to_gene,
    [gene_to_score, *permuted_list],     # observed first, then permutations
    n_jobs=4,
)
observed_T, observed_idx = hierarchies[0]
permuted_results = hierarchies[1:]
```

`construct_hierarchies` pickles `P` only once per worker via the executor's `initializer`, which is essential for real-sized matrices.

### 5. Process hierarchies
**Purpose.** Combine the observed and permuted hierarchies to (a) pick the cut height that maximizes the observed-vs-expected cluster-size ratio, (b) produce the final cluster set, and (c) compute a p-value for that cut.

For each cut height δ, compares the largest observed cluster size to the distribution of largest cluster sizes across permutations. The chosen cut δ* is the height where the ratio is greatest; the p-value is the fraction of permutations that match or exceed the observed maximum ratio.

**CLI:**

    hhnet process-hierarchies \
        -oelf hierarchy_edge_list_0.tsv \
        -oigf hierarchy_index_gene_0.tsv \
        -pelf hierarchy_edge_list_{1..100}.tsv \
        -pigf hierarchy_index_gene_{1..100}.tsv \
        -lsb  10 \
        -cf   clusters.tsv \
        -pf   sizes.pdf \
        -nc   4

**Python:**

```python
result = hhn.process_hierarchies(
    observed_T, observed_idx,
    permuted_Ts, permuted_idx_list,
    lower_size_bound=10,
    upper_size_bound=float("inf"),
    n_jobs=4,
)

result.observed_clusters     # set[frozenset[str]] — final clusters
result.observed_cut_height   # float — chosen δ*
result.observed_cut_ratio    # float — observed/expected at δ*
result.p_value               # float
result.observed_heights, result.observed_sizes
result.distinct_heights, result.min_sizes, result.expected_sizes, result.max_sizes
```

`lower_size_bound` restricts the search to cuts where the observed largest cluster has at least this many nodes — useful for filtering out trivial cuts at small scales.

### 6. Perform consensus
**Purpose.** Combine cluster sets from multiple (network, score) configurations into a single consensus subnetwork. For each pair of co-clustered, network-adjacent genes, count votes across configurations; keep edges with at least `threshold` votes; the connected components of the resulting graph are the consensus.

**CLI:**

    hhnet perform-consensus \
        -cf  clusters_n1_s1.tsv clusters_n1_s2.tsv \
        -igf network_index_gene.tsv network_index_gene.tsv \
        -elf network_edge_list.tsv network_edge_list.tsv \
        -n   network_1 network_1 \
        -s   scores_1 scores_2 \
        -t   2 \
        -cnf consensus_nodes.tsv \
        -cef consensus_edges.tsv

**Python:**

```python
from hierarchical_hotnet import ConsensusInput

# `gene_edges` is the same gene-labeled edge set used elsewhere;
# each `components` is the list of clusters from one configuration.
inputs = [
    ConsensusInput(edges=gene_edges, components=clusters_n1_s1),
    ConsensusInput(edges=gene_edges, components=clusters_n1_s2),
]
consensus = hhn.perform_consensus(inputs, threshold=2)

consensus.nodes   # list[list[str]] — connected components, sorted by size desc
consensus.edges   # list[list[str]] — consensus edges, each as [u, v]
```

Running the full pipeline
----------------
You can drive the whole flow either from the shell (CLI commands wired together with for-loops or GNU parallel) or from Python (`hhn.run_pipeline`).

### Shell
`examples/example_commands.sh` is the canonical reference. It runs:

* one network → similarity matrix
* two score sets, each with degree-stratified bins + 100 score permutations
* observed + 100 permuted hierarchies per score set
* `process_hierarchies` per score set
* consensus across the two

`examples/example_commands_parallel.sh` is the same pipeline parallelized via GNU parallel.

### Python
`hhn.run_pipeline` composes the steps sequentially; parallelism is handled within each step (over permutations / hierarchies), not across score sets.

```python
import hierarchical_hotnet as hhn

index_to_gene, _ = hhn.load_index_gene("network_index_gene.tsv")
edges          = hhn.load_edge_list("network_edge_list.tsv")
scores_1       = hhn.load_gene_score("scores_1.tsv")
scores_2       = hhn.load_gene_score("scores_2.tsv")

result = hhn.run_pipeline(
    edges,
    index_to_gene,
    {"scores_1": scores_1, "scores_2": scores_2},
    num_permutations=100,
    n_jobs=4,                  # parallelism for permute_scores + construct_hierarchies + process_hierarchies
    min_bin_size=1000,
    lower_size_bound=10,
    consensus_threshold=2,
)

result.similarity_matrix              # np.ndarray
result.beta                           # float
result.score_results["scores_1"]      # ProcessHierarchiesResult (cluster set, p-value, etc.)
result.score_results["scores_2"]      # ProcessHierarchiesResult
result.consensus.nodes                # consensus subnetwork node groups
result.consensus.edges                # consensus subnetwork edges
```

`n_jobs` semantics:

| Value | Meaning |
|-------|---------|
| `1` | Serial, no pool created (no overhead). |
| `N` (≥ 2) | `concurrent.futures.ProcessPoolExecutor` with N workers. |
| `-1` | Pool picks worker count (defaults to `os.cpu_count()`). |

For small inputs (the toy example) `n_jobs=1` is usually faster than `n_jobs>1` because pool startup + inter-process serialization dominates the actual work. The break-even point is roughly when each task takes hundreds of milliseconds or more; for real biological networks `construct_hierarchies` is comfortably above that.

### Disk-backed runs (`workdir=`, `reuse=`)

For large permutation counts on real-sized networks, holding every hierarchy in memory is the dominant peak-memory cost. Passing a `workdir` makes `run_pipeline` spill every artifact to disk and stream the hierarchies lazily into the statistics step, so peak memory is bounded regardless of `num_permutations`:

```python
from pathlib import Path

result = hhn.run_pipeline(
    edges, index_to_gene, {"scores_1": scores_1, "scores_2": scores_2},
    num_permutations=1000,
    n_jobs=8,
    workdir=Path("./hhnet_run_2026_05_15"),
)
```

The on-disk layout is fixed and matches the formats the standalone CLI commands produce, so workdir artifacts are interoperable with `hhnet construct-hierarchy` etc.:

```
<workdir>/similarity_matrix.h5
<workdir>/beta.txt
<workdir>/bins/<label>.tsv
<workdir>/permuted_scores/<label>/<seed>.tsv
<workdir>/hierarchies/<label>/<i>.edges.tsv   (+ .genes.tsv)
```

**Resuming an interrupted run.** With `reuse=True`, each artifact path is checked individually; existing files are read back instead of recomputed, missing ones are computed normally. Useful when a long run was killed midway, or when iterating on downstream parameters (the cut bounds, the consensus threshold) without re-doing the expensive hierarchies stage:

```python
# Second invocation picks up wherever the first left off.
result = hhn.run_pipeline(
    edges, index_to_gene, score_sets,
    num_permutations=1000,
    workdir=Path("./hhnet_run_2026_05_15"),
    reuse=True,
)
```

With `reuse=False` (the default), `workdir` is still honored but every stage recomputes from scratch and overwrites the prior artifacts.

Output
----------------
Hierarchical HotNet identifies statistically significant regions of a hierarchical clustering of topologically close, high-scoring genes, and (optionally) performs a consensus across multiple cluster sets.

CLI outputs (TSV / HDF5):

* **Similarity matrix** (`similarity_matrix.h5`): the PPR matrix `P`.
* **Beta** (`beta.txt`): the chosen restart probability.
* **Score bins** (`score_bins.tsv`): degree-stratified bins, one bin per line.
* **Permuted scores / networks** (`scores_{N}.tsv`, `edge_list_{N}.tsv`): per-seed nulls.
* **Hierarchy** (`hierarchy_edge_list_{N}.tsv`, `hierarchy_index_gene_{N}.tsv`): the dendrogram and its gene labels for one score set.
* **Clusters** (`clusters_{network}_{score}.tsv`): final clusters at the chosen cut height, with a header line containing the cut height, p-value, and cluster size statistics.
* **Cluster sizes plot** (`sizes_{network}_{score}.pdf`): observed vs. expected cluster size across all cut heights (requires the `[plot]` extra).
* **Consensus** (`consensus_nodes.tsv`, `consensus_edges.tsv`): the final consensus subnetwork.

Python-API outputs are typed dataclasses (`ProcessHierarchiesResult`, `ConsensusResult`, `PipelineResult`); see each step section above for fields.

Additional information
----------------

### Examples
See the `examples/` directory for example data, scripts, and reference output. The reference `example_consensus_nodes.tsv` and `example_consensus_edges.tsv` are the expected outputs for the toy pipeline; both `examples/example_commands.sh` and `hhn.run_pipeline` reproduce them byte-identically.

### Support
For support with Hierarchical HotNet, please visit the [HotNet Google Group](https://groups.google.com/forum/#!forum/hotnet-users). Please try one of the examples in the `examples/` directory before running Hierarchical HotNet with your own data, and please provide any error messages encountered with these examples to expedite troubleshooting.

### License
See `LICENSE.txt` for license information.

### Citation
If you use Hierarchical HotNet in your work, please cite the following manuscript:

> M.A. Reyna, M.D.M. Leiserson, B.J. Raphael. Hierarchical HotNet: identifying hierarchies of altered subnetworks. [_ECCB/Bioinformatics_ **34**(17):i972-980](https://academic.oup.com/bioinformatics/article/34/17/i972/5093236), 2018.
