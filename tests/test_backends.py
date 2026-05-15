"""Tests for the backend dispatch and Python/Fortran parity.

The dev environment ships with the Fortran extension built, so
``BACKEND == 'fortran'`` is the expected state. The Python fallback is
exercised in a subprocess with ``HHNET_BACKEND=python``; the cross-backend
parity test compares end-to-end pipeline results across the two backends
so the pure-Python path doesn't bit-rot.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hierarchical_hotnet as hhn
from hierarchical_hotnet import backends


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_backend_is_fortran_in_dev_env():
    """Guards against silent fallback to pure Python in the development env."""
    assert backends.BACKEND == "fortran", (
        f"backends.BACKEND={backends.BACKEND!r}; the dev env should ship the "
        "Fortran extension. If you see this fail, the wheel built without "
        "the Fortran extension or HHNET_BACKEND is overriding."
    )


def test_invalid_env_var_value_raises(tmp_path):
    """HHNET_BACKEND=garbage should fail at import time, not silently."""
    script = tmp_path / "import_with_bad_backend.py"
    script.write_text("import hierarchical_hotnet  # should raise during import\n")
    env = {**os.environ, "HHNET_BACKEND": "totally-bogus"}
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "HHNET_BACKEND" in result.stderr


def test_python_backend_matches_fortran_backend(data_dir, tmp_path):
    """End-to-end parity: HHNET_BACKEND=python produces the same PipelineResult.

    Spawns a subprocess so the env var takes effect at backend-import time.
    Compares per-score beta/p_value/cut_size/clusters against an in-process
    run that uses whichever backend the dev env has loaded (Fortran).
    """
    fortran_result = hhn.run_pipeline(
        edges=hhn.load_edge_list(data_dir / "network_1_edge_list.tsv"),
        index_to_gene=hhn.load_index_gene(data_dir / "network_1_index_gene.tsv")[0],
        score_sets={"scores_1": hhn.load_gene_score(data_dir / "scores_1.tsv")},
        num_permutations=20,
        n_jobs=1,
        min_bin_size=1000,
        lower_size_bound=1,
        consensus_threshold=None,
    )
    fortran_summary = {
        "beta": float(fortran_result.beta),
        "p_value": float(fortran_result.score_results["scores_1"].p_value),
        "observed_cut_size": int(fortran_result.score_results["scores_1"].observed_cut_size),
        "observed_clusters": sorted(
            sorted(c) for c in fortran_result.score_results["scores_1"].observed_clusters
        ),
    }

    runner = tmp_path / "run_python_backend.py"
    runner.write_text(textwrap.dedent("""
        import json, sys
        from pathlib import Path
        import hierarchical_hotnet as hhn
        from hierarchical_hotnet import backends
        assert backends.BACKEND == "python", backends.BACKEND
        data = Path(sys.argv[1])
        r = hhn.run_pipeline(
            edges=hhn.load_edge_list(data / "network_1_edge_list.tsv"),
            index_to_gene=hhn.load_index_gene(data / "network_1_index_gene.tsv")[0],
            score_sets={"scores_1": hhn.load_gene_score(data / "scores_1.tsv")},
            num_permutations=20,
            n_jobs=1,
            min_bin_size=1000,
            lower_size_bound=1,
            consensus_threshold=None,
        )
        out = {
            "beta": float(r.beta),
            "p_value": float(r.score_results["scores_1"].p_value),
            "observed_cut_size": int(r.score_results["scores_1"].observed_cut_size),
            "observed_clusters": sorted(
                sorted(c) for c in r.score_results["scores_1"].observed_clusters
            ),
        }
        sys.stdout.write(json.dumps(out))
    """))

    env = {**os.environ, "HHNET_BACKEND": "python"}
    result = subprocess.run(
        [sys.executable, str(runner), str(data_dir)],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"subprocess failed:\nSTDERR:\n{result.stderr}"
    python_summary = json.loads(result.stdout)

    assert python_summary == pytest.approx(fortran_summary)
