"""Smoke tests for spatially variable gene methods."""

import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method


@pytest.mark.parametrize("method", ["morans_i", "gearys_c", "spatial_variance_ratio"])
def test_native_svg_methods(method: str) -> None:
    data = create_method("normalization", "log1p_cp10k").run(
        make_synthetic(n_cells=60, n_genes=25, n_domains=3, seed=0)
    )
    out = create_method("svg", method).run(data)
    assert out.n_vars == data.n_vars
