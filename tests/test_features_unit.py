"""Unit tests for dataset feature extraction."""

import numpy as np
import pytest

from histoweave.benchmark.features import (
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)
from histoweave.datasets import make_synthetic


def test_recommendation_features_exclude_domain_labels() -> None:
    assert "n_domains" not in RECOMMENDATION_FEATURE_ORDER
    data = make_synthetic(n_cells=60, n_genes=20, seed=0)
    feats = extract_features(data, include_domain=False)
    for key in ("n_domains", "domain_balance", "domain_spatial_coherence"):
        assert np.isnan(feats[key])


def test_feature_vector_order_stable() -> None:
    data = make_synthetic(n_cells=40, n_genes=15, seed=1)
    feats = extract_features(data, include_domain=False)
    vec = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
    assert vec.shape == (len(RECOMMENDATION_FEATURE_ORDER),)
    assert np.isfinite(vec).sum() >= 10


def test_feature_extraction_is_finite_for_synthetic() -> None:
    data = make_synthetic(n_cells=100, n_genes=30, n_domains=3, seed=2)
    feats = extract_features(data, include_domain=True)
    assert feats["n_obs"] == pytest.approx(100)
    assert 0.0 <= feats["sparsity"] <= 1.0
