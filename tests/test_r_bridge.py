"""End-to-end test for the Python ↔ R bridge.

These tests verify that the containerised R normalisation plugin correctly
round-trips data through an R script (``.ttab → .h5ad → R → .h5ad → .ttab``).

They require ``Rscript`` and the ``anndata`` R package on the host PATH,
conditions that are met inside the ``histoweave-r`` container image. When either
is missing the tests skip gracefully so CI without R still passes.
"""

import shutil

import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, list_methods


def _rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _py_anndata_available() -> bool:
    try:
        import anndata  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _r_anndata_available() -> bool:
    """The R ``anndata`` package must be installed."""
    import subprocess
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(anndata)"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


_r_ready = pytest.mark.skipif(
    not (_rscript_available() and _py_anndata_available() and _r_anndata_available()),
    reason="Rscript, anndata Python, or anndata R not available",
)


def test_r_lognorm_is_registered():
    """Even without R available, the method must be registered."""
    methods = {m["name"] for m in list_methods(category="normalization")}
    assert "r_lognorm" in methods


@_r_ready
def test_r_normalize_preserves_structure():
    data = make_synthetic(n_cells=80, n_genes=20, n_domains=2, seed=42)
    result = create_method("normalization", "r_lognorm", target_sum=5000).run(data)

    assert result.shape == data.shape
    assert np.all(result.X >= 0)                  # log1p >= 0
    assert result.uns.get("r_normalized") is True
    assert result.uns.get("r_target_sum") == 5000


@_r_ready
def test_r_normalize_is_deterministic():
    data = make_synthetic(n_cells=50, n_genes=15, seed=99)
    a = create_method("normalization", "r_lognorm").run(data)
    b = create_method("normalization", "r_lognorm").run(data)
    assert np.allclose(a.X, b.X)
