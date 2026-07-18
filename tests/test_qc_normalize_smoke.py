"""Smoke tests for QC and normalization builtins."""

import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method


@pytest.mark.parametrize(
    "method",
    ["basic_qc", "library_size_qc", "gene_complexity_qc", "mitochondrial_qc"],
)
def test_qc_methods_run(method: str) -> None:
    data = make_synthetic(n_cells=50, n_genes=20, seed=0)
    out = create_method("qc", method).run(data)
    assert out.n_obs <= data.n_obs
    assert out.provenance[-1]["method"] == method


@pytest.mark.parametrize(
    "method",
    [
        "log1p_cp10k",
        "library_size_scale",
        "sqrt_transform",
        "arcsinh_transform",
        "clr_per_cell",
        "tfidf_l2",
    ],
)
def test_normalization_methods_run(method: str) -> None:
    data = make_synthetic(n_cells=40, n_genes=18, seed=1)
    out = create_method("normalization", method).run(data)
    assert out.shape == data.shape
    assert np.isfinite(np.asarray(out.X, dtype=float)).all()
