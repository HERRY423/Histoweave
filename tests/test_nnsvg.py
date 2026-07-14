"""Tests for the nnSVG spatially-variable-gene R-container plugin.

Registration, spec, and input-validation tests run everywhere. The end-to-end
R round-trip is skipped unless ``Rscript`` and the R ``anndata`` + ``nnSVG``
packages are available (they are, inside the ``histoweave-r`` container image).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, get_method, list_methods


def _rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _r_pkg_available(pkg: str) -> bool:
    try:
        r = subprocess.run(
            ["Rscript", "-e", f"library({pkg})"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


_r_ready = pytest.mark.skipif(
    not (
        _rscript_available()
        and _r_pkg_available("anndata")
        and _r_pkg_available("nnSVG")
    ),
    reason="Rscript, anndata R, or nnSVG R package not available",
)


def test_nnsvg_is_registered() -> None:
    methods = {m["name"] for m in list_methods(category="svg")}
    assert "nnsvg" in methods


def test_nnsvg_spec_contract() -> None:
    cls = get_method("svg", "nnsvg")
    assert cls.spec.name == "nnsvg"
    assert cls.spec.language == "container"
    param_names = {p.name for p in cls.spec.params}
    assert {"n_top", "n_neighbors", "order", "assay_name", "seed"} <= param_names


def test_nnsvg_requires_spatial_coordinates() -> None:
    cls = get_method("svg", "nnsvg")
    data = make_synthetic(n_cells=40, n_genes=20, seed=0)
    data.obsm.pop("spatial", None)
    with pytest.raises(ValueError, match="obsm\\['spatial'\\]"):
        cls().run(data)


def test_nnsvg_rejects_non_2d_coordinates() -> None:
    cls = get_method("svg", "nnsvg")
    data = make_synthetic(n_cells=40, n_genes=20, seed=0)
    data.obsm["spatial"] = np.zeros((data.n_obs, 3))
    with pytest.raises(ValueError, match="two-dimensional"):
        cls().run(data)


@_r_ready
def test_nnsvg_end_to_end_writes_ranking() -> None:
    data = make_synthetic(n_cells=120, n_genes=30, n_domains=3, seed=42)
    result = create_method("svg", "nnsvg", n_top=10, n_neighbors=8).run(data)

    for col in ("nnsvg_rank", "nnsvg_LR_stat", "nnsvg_padj"):
        assert col in result.var
    ranks = result.var["nnsvg_rank"].dropna()
    assert ranks.min() == 1
    assert "nnsvg_top_genes" in result.uns
    assert len(result.uns["nnsvg_top_genes"]) <= 10
