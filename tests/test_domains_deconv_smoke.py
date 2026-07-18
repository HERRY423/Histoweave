"""Smoke tests for domain detection and deconvolution baselines."""

import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method


@pytest.mark.parametrize(
    "method",
    ["kmeans", "gaussian_mixture", "spectral", "agglomerative", "banksy_py"],
)
def test_domain_methods_write_labels(method: str) -> None:
    data = create_method("normalization", "log1p_cp10k").run(
        make_synthetic(n_cells=80, n_genes=25, n_domains=3, seed=0)
    )
    data.uns["n_domains"] = 3
    out = create_method("domain_detection", method, n_domains=3, random_state=0).run(data)
    assert "domain" in out.obs
    assert out.obs["domain"].nunique() >= 1


def test_marker_deconv_runs_on_synthetic() -> None:
    data = make_synthetic(n_cells=60, n_genes=20, n_domains=3, seed=2)
    data = create_method("normalization", "log1p_cp10k").run(data)
    out = create_method("deconvolution", "marker_deconv").run(data)
    assert out.n_obs == data.n_obs
