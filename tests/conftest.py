from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "examples" / "data"
REFERENCE_DIR = REPO_ROOT / "examples"


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def reference_dir():
    return REFERENCE_DIR
