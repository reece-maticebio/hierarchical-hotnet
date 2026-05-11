project(
  'hierarchical_hotnet',
  'c',
  version: '0.2.0',
  meson_version: '>=1.2.0',
  default_options: ['warning_level=2'],
)

add_languages('fortran', native: false)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

fortran_module_c = custom_target('fortran_modulemodule.c',
  input: 'src/hierarchical_hotnet/fortran_module.f95',
  output: ['fortran_modulemodule.c', 'fortran_module-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'fortran_module', '--lower'],
)

py.extension_module('fortran_module',
  [
    'src/hierarchical_hotnet/fortran_module.f95',
    fortran_module_c,
  ],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies: py_dep,
  subdir: 'hierarchical_hotnet',
  install: true,
)

py.install_sources(
  [
    'src/hierarchical_hotnet/__init__.py',
    'src/hierarchical_hotnet/_parallel.py',
    'src/hierarchical_hotnet/common.py',
    'src/hierarchical_hotnet/hhio.py',
    'src/hierarchical_hotnet/hierarchical_clustering.py',
    'src/hierarchical_hotnet/hierarchical_clustering_tests.py',
    'src/hierarchical_hotnet/construct_hierarchy.py',
    'src/hierarchical_hotnet/construct_similarity_matrix.py',
    'src/hierarchical_hotnet/find_permutation_bins.py',
    'src/hierarchical_hotnet/generate_example_graph.py',
    'src/hierarchical_hotnet/perform_consensus.py',
    'src/hierarchical_hotnet/permute_network.py',
    'src/hierarchical_hotnet/permute_scores.py',
    'src/hierarchical_hotnet/pipeline.py',
    'src/hierarchical_hotnet/process_hierarchies.py',
  ],
  subdir: 'hierarchical_hotnet',
)
