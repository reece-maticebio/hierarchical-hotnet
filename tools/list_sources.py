"""List Python source files for meson.build.

Two modes:

  python list_sources.py <package_dir>
      Print sorted .py files directly in <package_dir>, one per line,
      as paths relative to the repo root.

  python list_sources.py <package_dir> --subpackages
      Print sorted names of immediate subdirectories of <package_dir> that
      contain an __init__.py (i.e. are Python subpackages), one per line.

Output is consumed by meson.build via run_command. Keep stdout to one item
per line; never print anything else (no logs, no trailing summary).
"""

from __future__ import annotations

import sys
from pathlib import Path


def list_py_files(package_dir: Path) -> list[str]:
    return sorted(str(p) for p in package_dir.glob("*.py"))


def list_subpackages(package_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in package_dir.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: list_sources.py <package_dir> [--subpackages]", file=sys.stderr)
        return 2

    package_dir = Path(argv[1])
    if not package_dir.is_dir():
        print(f"error: {package_dir} is not a directory", file=sys.stderr)
        return 1

    if "--subpackages" in argv[2:]:
        items = list_subpackages(package_dir)
    else:
        items = list_py_files(package_dir)

    print("\n".join(items))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
