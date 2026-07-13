import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, list_methods, method_coverage_report


def test_method_coverage_release_gates():
    report = method_coverage_report()
    assert report["total_methods"] >= 50
    assert report["ratios"]["production_plus"] > 0.80
    assert report["ratios"]["beta_plus"] == 1.0
    assert report["ratios"]["experimental"] < 0.05
    assert report["counts"]["deep_learning"] >= 10
    assert report["counts"]["image_expression"] >= 4
    assert report["passes_all_targets"] is True


def test_multimodal_and_deep_learning_metadata_are_queryable():
    methods = list_methods()
    multimodal = [
        method for method in methods if {"image", "expression"}.issubset(method["modalities"])
    ]
    deep = [method for method in methods if method["model_family"] == "deep_learning"]
    assert len(multimodal) >= 4
    assert len(deep) >= 10
    assert all(method["is_multimodal"] for method in multimodal)


@pytest.mark.parametrize(
    ("category", "name"),
    [
        ("qc", "library_size_qc"),
        ("qc", "gene_complexity_qc"),
        ("qc", "mitochondrial_qc"),
        ("normalization", "library_size_scale"),
        ("normalization", "clr_per_cell"),
        ("normalization", "sqrt_transform"),
        ("normalization", "arcsinh_transform"),
        ("normalization", "tfidf_l2"),
        ("svg", "gearys_c"),
        ("svg", "spatial_variance_ratio"),
    ],
)
def test_extended_native_methods_smoke(category, name):
    data = make_synthetic(n_cells=24, n_genes=12, seed=7)
    result = create_method(category, name).run(data)
    assert result.n_obs == data.n_obs
    assert result.n_vars == data.n_vars
    assert result.provenance[-1]["method"] == name


def test_expression_and_image_expression_deep_models_train_minimally():
    pytest.importorskip("torch")
    data = make_synthetic(n_cells=16, n_genes=10, seed=8)
    data.images["he"] = np.arange(32 * 32, dtype=float).reshape(32, 32)

    expression = create_method(
        "integration",
        "spatial_autoencoder",
        epochs=1,
        latent_dim=4,
        max_features=8,
    ).run(data)
    multimodal = create_method(
        "integration",
        "image_expression_autoencoder",
        image_key="he",
        patch_size=5,
        epochs=1,
        latent_dim=4,
        max_features=8,
    ).run(data)

    assert expression.obsm["X_spatial_autoencoder"].shape == (16, 4)
    assert multimodal.obsm["X_image_expression_autoencoder"].shape == (16, 4)
    metadata = multimodal.uns["deep_learning"]["image_expression_autoencoder"]
    assert metadata["modalities"] == ["expression", "image"]
    assert np.isfinite(metadata["final_loss"])
