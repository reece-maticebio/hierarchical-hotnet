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

# Python sources are discovered automatically via tools/list_sources.py.
# Meson only re-runs run_command at configure time, so after ADDING a new .py
# file you must rerun `pip install --no-build-isolation --no-deps -e .`
# (or `meson setup --reconfigure build/cp312`) for it to be picked up.
# Editing an existing file does NOT require a reconfigure.
list_sources = files('tools/list_sources.py')

top_sources = run_command(
  py, list_sources, 'src/hierarchical_hotnet', check: true,
).stdout().strip().split('\n')
py.install_sources(top_sources, subdir: 'hierarchical_hotnet')

subpackages_out = run_command(
  py, list_sources, 'src/hierarchical_hotnet', '--subpackages', check: true,
).stdout().strip()

if subpackages_out != ''
  foreach subpkg : subpackages_out.split('\n')
    sub_sources = run_command(
      py, list_sources, 'src/hierarchical_hotnet/' + subpkg, check: true,
    ).stdout().strip().split('\n')
    py.install_sources(sub_sources, subdir: 'hierarchical_hotnet/' + subpkg)
  endforeach
endif
