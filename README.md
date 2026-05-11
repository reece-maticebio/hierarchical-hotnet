Hierarchical HotNet
=======================

Hierarchical HotNet is an algorithm for finding hierarchies of altered subnetworks.  While originally developed for use with cancer mutation data on protein-protein interaction networks, Hierarchical HotNet supports any application in which scores may be associated with the nodes of a network, i.e., a vertex-weighted graph.

Setup
------------------------
Hierarchical HotNet is installable as a Python package. The Fortran extension
is built automatically by `pip` via the `meson-python` build backend; a Python-
only fallback is used if no Fortran compiler is available.

### Requirements
* Python &ge; 3.10 (tested on 3.12)
* A Fortran compiler such as [gfortran](https://gcc.gnu.org/wiki/GFortran) (recommended for performance)
* [GNU parallel](https://www.gnu.org/software/parallel/) (optional, for the parallel example script)

Runtime dependencies (`numpy`, `scipy`, `networkx`, `h5py`) are installed
automatically. `matplotlib` is an optional extra used by the plotting steps.

### Install
Clone the repository and install with `pip`:

    git clone https://github.com/raphael-group/hierarchical-hotnet.git
    cd hierarchical-hotnet
    pip install .            # core install
    pip install '.[plot]'    # with matplotlib for plotting

Installation exposes the following CLI commands:

* `hhnet-construct-similarity-matrix`
* `hhnet-construct-hierarchy`
* `hhnet-find-permutation-bins`
* `hhnet-permute-scores`
* `hhnet-permute-network`
* `hhnet-process-hierarchies`
* `hhnet-perform-consensus`
* `hhnet-generate-example-graph`

The library itself is importable as `hierarchical_hotnet`.

If no Fortran compiler is found at build time the install still succeeds and
the package falls back to the pure-Python implementation, which is slower but
otherwise equivalent.

### Testing
To test Hierarchical HotNet on an example network with two sets of example scores, please run the following script:

    sh examples/example_commands.sh

This script illustrates the full Hierarchical HotNet pipeline.  It should require less than a minute or two of CPU time, 100 MB of RAM, and 1 MB of storage space.  If this script runs successfully, then Hierarchical HotNet is ready to use.

Alternatively, to run Hierarchical HotNet in parallel on the sample example data, please run the following script:

    sh examples/example_commands_parallel.sh

We hightly recommend running Hierarchical HotNet in parallel.  It should straightforward to modify the above scripts to run Hierarchical HotNet on a compute cluster.

Use
----------------
Hierarchical HotNet requires the use of several scripts on a few input files.

### Input
There are three input files for Hierarchical HotNet that together define a network with scores on the nodes of the network.  For example, the following example defines a network with an edge between the nodes ABC and DEF, which have scores 0.5 and 0.2, respectively.  For convenience, these files use the same format as the input files for HotNet2.

##### Index-to-gene file
This file associates each gene with an index, which we use for the edge list as well as a similarity matrix:

    1   ABC
    2   DEF

##### Edge list file
This file defines a network using the indices in the index-to-gene file:

    1    2

##### Gene-to-score file
This file associates each gene with a score:

    ABC 0.5
    DEF 0.2

### Running
Hierarchical HotNet has several steps:

1. Create a similarity matrix with `hhnet-construct-similarity-matrix`.

2. Create permuted data with `hhnet-find-permutation-bins` and `hhnet-permute-scores` (permuted scores), or with `hhnet-permute-network` (permuted networks). In general, permuting scores is faster than permuting networks.

3. Construct hierarchies on observed and permuted data with `hhnet-construct-hierarchy`.

4. Process the hierarchies with `hhnet-process-hierarchies`.

5. Perform the consensus summarization with `hhnet-perform-consensus`.

See `examples/example_commands.sh` or `examples/example_commands_parallel.sh` for full minimal working examples of Hierarchical HotNet that illustrate the use of each of these commands, including the inputs and outputs for the Hierarchical HotNet pipeline.

### Output
Hierarchical HotNet identifies statistically significant regions of a hierarchical clustering of topologically close, high-scoring genes.  Hierarchical HotNet also performs a consensus across hierarchical clusterings from different networks and gene scores.

Additional information
----------------

### Examples
See the `examples` directory for example data, scripts, and output for Hierarchical HotNet.

### Support
For support with Hierarchical HotNet, please visit the [HotNet Google Group](https://groups.google.com/forum/#!forum/hotnet-users).  Please try one of the examples in the `examples` directory before running Hierarchical HotNet with your own data, and please provide any error messages encountered with these examples to expedite troubleshooting.

### License
See `LICENSE.txt` for license information.

### Citation
If you use Hierarchical HotNet in your work, then please cite the following manuscript:

> M.A. Reyna, M.D.M. Leiserson, B.J. Raphael. Hierarchical HotNet: identifying hierarchies of altered subnetworks. [_ECCB/Bioinformatics_ **34**(17):i972-980](https://academic.oup.com/bioinformatics/article/34/17/i972/5093236), 2018.
