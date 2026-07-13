import numpy as np
import pytest

from histoweave.datasets import make_mixture_synthetic, make_synthetic
from histoweave.plugins import create_method


def test_synthetic_has_ground_truth():
    data = make_synthetic(n_cells=200, n_genes=30, n_domains=3, seed=1)
    assert data.shape == (200, 30)
    assert "domain_truth" in data.obs
    assert data.obs["domain_truth"].nunique() == 3
    assert data.spatial.shape == (200, 2)
    assert data.uns["marker_genes"]


def test_mixture_synthetic_has_proportions():
    data = make_mixture_synthetic(n_spots=150, n_genes=24, n_cell_types=3, seed=42)
    assert data.shape == (150, 24)
    truth = data.obsm["proportions_truth"]
    assert truth.shape == (150, 3)
    # Each row sums to ~1 (Dirichlet).
    assert np.allclose(truth.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(truth >= 0)
    assert data.uns["n_cell_types"] == 3


def test_qc_filters_and_adds_metrics():
    data = make_synthetic(seed=2)
    result = create_method("qc", "basic_qc").run(data)
    assert "total_counts" in result.obs
    assert "pct_counts_mito" in result.obs
    # QC should drop the injected low-quality cells.
    assert result.n_obs <= data.n_obs
    assert result.uns["qc"]["n_obs_before"] == data.n_obs



def test_qc_and_normalization_preserve_sparse_matrices():
    sparse = pytest.importorskip("scipy.sparse")
    data = make_synthetic(seed=2)
    data.X = sparse.csr_matrix(data.X)

    qc_result = create_method("qc", "basic_qc").run(data)
    assert sparse.isspmatrix_csr(qc_result.X)

    normalized = create_method("normalization", "log1p_cp10k").run(data)
    assert sparse.isspmatrix_csr(normalized.X)
    assert sparse.isspmatrix_csr(normalized.layers["counts"])
    assert np.all(np.isfinite(normalized.X.data))
    assert np.all(normalized.X.data >= 0)


def test_normalize_is_nonnegative_and_logged():
    data = make_synthetic(seed=3)
    result = create_method("normalization", "log1p_cp10k").run(data)
    assert np.all(result.X >= 0)
    assert result.uns["normalization"]["method"] == "log1p_cp10k"


def test_domain_detection_runs_and_populates_outputs():
    # Unit-level contract for the method: it runs and writes the outputs it promises.
    # Recovery *accuracy* (ARI vs ground truth) is asserted once, at the integration
    # level, in test_benchmark.py — this test deliberately does not re-check it.
    data = make_synthetic(n_cells=400, n_genes=40, n_domains=3, seed=0)
    data = create_method("normalization", "log1p_cp10k").run(data)
    result = create_method("domain_detection", "kmeans", n_pcs=10).run(data)

    # Every observation gets a categorical domain label from the requested k clusters.
    assert "domain" in result.obs
    assert result.obs["domain"].dtype.name == "category"
    assert result.obs["domain"].notna().all()
    assert 1 <= result.obs["domain"].nunique() <= 3
    assert result.n_obs == data.n_obs
    # The PCA embedding is exposed with the requested width.
    assert result.obsm["X_pca"].shape == (data.n_obs, 10)


def test_domain_detection_is_deterministic():
    # Same seed -> identical labels, so downstream reports/benchmarks are reproducible.
    data = make_synthetic(n_cells=200, n_domains=3, seed=0)
    data = create_method("normalization", "log1p_cp10k").run(data)
    a = create_method("domain_detection", "kmeans", random_state=0).run(data)
    b = create_method("domain_detection", "kmeans", random_state=0).run(data)
    assert list(a.obs["domain"]) == list(b.obs["domain"])


def test_annotation_assigns_labels():
    data = make_synthetic(seed=5)
    data = create_method("normalization", "log1p_cp10k").run(data)
    result = create_method("annotation", "marker_score").run(data)
    assert "cell_type" in result.obs
    assert set(result.obs["cell_type"].unique()).issubset(set(data.uns["marker_genes"]))


def test_deconvolution_produces_valid_proportions():
    data = make_mixture_synthetic(n_spots=120, n_cell_types=3, seed=7)
    data = create_method("normalization", "log1p_cp10k").run(data)
    result = create_method("deconvolution", "marker_deconv").run(data)
    prop = result.obsm["proportions"]
    assert prop.shape == (data.n_obs, 3)
    # Valid probability simplex per row.
    assert np.all(prop >= 0)
    assert np.allclose(prop.sum(axis=1), 1.0, atol=1e-6)
    # Non-trivial — the three columns should have some variance.
    assert prop.std(axis=0).min() > 0.01


def test_deconvolution_is_deterministic():
    data = make_mixture_synthetic(n_spots=80, n_cell_types=2, seed=3)
    data = create_method("normalization", "log1p_cp10k").run(data)
    a = create_method("deconvolution", "marker_deconv").run(data)
    b = create_method("deconvolution", "marker_deconv").run(data)
    assert np.allclose(a.obsm["proportions"], b.obsm["proportions"])
