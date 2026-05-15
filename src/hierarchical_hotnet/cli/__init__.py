"""Command-line entry points for Hierarchical HotNet.

The compute modules (``hierarchical_hotnet.construct_similarity_matrix`` and
friends) expose pure library APIs. Their argparse-based CLI wrappers live
here so that importing the library does not pull in argparse machinery, and
so that the CLI surface can be refactored independently of the algorithms.
"""
