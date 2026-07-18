"""Unit tests for synthetic dataset generators."""

from histoweave.datasets import (
    make_developmental_gradient,
    make_mixture_synthetic,
    make_synthetic,
    make_tumor_microenvironment,
)


def test_make_synthetic_shapes() -> None:
    data = make_synthetic(n_cells=30, n_genes=12, n_domains=3, seed=0)
    assert data.n_obs == 30
    assert data.n_vars == 12
    assert "domain_truth" in data.obs


def test_specialized_generators_run() -> None:
    mix = make_mixture_synthetic(seed=0)
    assert mix.n_obs > 0
    tumor = make_tumor_microenvironment(seed=0)
    assert tumor.n_obs > 0
    dev = make_developmental_gradient(seed=0)
    assert dev.n_obs > 0
