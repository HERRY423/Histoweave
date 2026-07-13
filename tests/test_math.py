"""Unit tests for the dependency-free numerical helpers in ``histoweave._math``.

These functions are the numerical core the pipeline, domain detector, and benchmark
all lean on, so they get their own focused tests (independent of any plugin) that pin
down edge cases and mathematical invariants — not just "it runs".
"""

import numpy as np
import pytest

from histoweave._math import (
    adjusted_rand_index,
    kmeans,
    knn_indices,
    neighborhood_mean,
    pca,
    zscore,
)


def _pairwise_sqdist(a: np.ndarray) -> np.ndarray:
    """Full matrix of squared Euclidean distances between the rows of ``a``."""
    diff = a[:, None, :] - a[None, :, :]
    return (diff**2).sum(axis=-1)


# ---------------------------------------------------------------------------
# adjusted_rand_index
# ---------------------------------------------------------------------------
def test_ari_identical_labelings_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_ari_is_invariant_to_relabeling():
    # A pure permutation of cluster ids is the *same* partition -> ARI 1.0.
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0)


def test_ari_handles_non_numeric_labels():
    true = np.array(["a", "a", "b", "b"])
    pred = np.array(["x", "x", "y", "y"])
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0)


def test_ari_complete_disagreement_is_negative():
    # The textbook worst case: every pair that is "together" in one labeling is
    # "apart" in the other. ARI works out to exactly -0.5 here.
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    assert adjusted_rand_index(true, pred) == pytest.approx(-0.5)


def test_ari_is_symmetric():
    true = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 2, 2])
    assert adjusted_rand_index(true, pred) == pytest.approx(
        adjusted_rand_index(pred, true)
    )


def test_ari_independent_labelings_near_zero():
    # Two independent random labelings should have ARI ~ 0 (that is the whole point
    # of the "adjusted" correction over the raw Rand index).
    rng = np.random.default_rng(0)
    true = rng.integers(0, 5, size=2000)
    pred = rng.integers(0, 5, size=2000)
    assert abs(adjusted_rand_index(true, pred)) < 0.05


def test_ari_single_element_is_defined():
    # n == 1 -> no pairs exist (comb2 total is 0); the function must not divide by
    # zero and returns the conventional 1.0.
    assert adjusted_rand_index(np.array([0]), np.array([7])) == pytest.approx(1.0)


def test_ari_both_single_cluster_is_defined():
    # Both labelings put everything in one cluster: max_index == expected, another
    # degenerate branch that must return 1.0 rather than 0/0 = nan.
    ones = np.zeros(5, dtype=int)
    assert adjusted_rand_index(ones, np.full(5, 3)) == pytest.approx(1.0)


def test_ari_returns_python_float():
    val = adjusted_rand_index(np.array([0, 1]), np.array([0, 1]))
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------
def test_pca_output_shape_and_component_clamping():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6))
    assert pca(X, n_components=3).shape == (20, 3)
    # Asking for more components than min(X.shape) clamps rather than erroring.
    assert pca(X, n_components=99).shape == (20, 6)


def test_pca_preserves_total_variance_with_full_components():
    # With the full set of components the scores are just a rotation of the
    # mean-centred data, so their Frobenius energy must equal the centred data's.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 5))
    Xc = X - X.mean(axis=0, keepdims=True)
    scores = pca(X, n_components=5)
    assert np.linalg.norm(scores) ** 2 == pytest.approx(np.linalg.norm(Xc) ** 2)


def test_pca_full_reconstruction_preserves_pairwise_distances():
    # PCA onto all components is distance-preserving (an orthogonal change of basis):
    # pairwise distances in score space match those of the centred input exactly.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 8))
    Xc = X - X.mean(axis=0, keepdims=True)
    scores = pca(X, n_components=8)
    assert np.allclose(_pairwise_sqdist(scores), _pairwise_sqdist(Xc))


def test_pca_components_are_orthogonal():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 6))
    scores = pca(X, n_components=4)
    gram = scores.T @ scores
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.allclose(off_diagonal, 0.0, atol=1e-8)


def test_pca_is_translation_invariant_and_deterministic():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 4))
    shifted = X + np.array([10.0, -3.0, 7.0, 0.5])  # centering removes the offset
    assert np.allclose(pca(X, 3), pca(shifted, 3))
    assert np.array_equal(pca(X, 3), pca(X, 3))  # bit-for-bit reproducible


def test_pca_wide_matrix_matches_svd_geometry():
    # Genes >> observations triggers the Gram/eigh route; it must reproduce the SVD's
    # geometry (energy + pairwise distances) that the distance-preservation test checks.
    rng = np.random.default_rng(30)
    X = rng.normal(size=(8, 40))  # wide: n_features (40) > n_obs (8)
    Xc = X - X.mean(axis=0, keepdims=True)
    scores = pca(X, n_components=8)
    assert scores.shape == (8, 8)
    assert np.linalg.norm(scores) ** 2 == pytest.approx(np.linalg.norm(Xc) ** 2)
    assert np.allclose(_pairwise_sqdist(scores), _pairwise_sqdist(Xc))
    # Components remain orthogonal and the route stays deterministic.
    gram = scores.T @ scores
    assert np.allclose(gram - np.diag(np.diag(gram)), 0.0, atol=1e-8)
    assert np.allclose(pca(X, 8), pca(X, 8))


def test_pca_on_rank_deficient_data_is_zero():
    # All rows identical -> centred data is exactly zero -> zero scores, no NaNs.
    X = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    scores = pca(X, n_components=2)
    assert np.allclose(scores, 0.0)


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------
def test_zscore_standardizes_columns():
    rng = np.random.default_rng(6)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
    Z = zscore(X, axis=0)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-2)


def test_zscore_constant_column_stays_finite():
    # A zero-variance column must not blow up to inf/nan thanks to the eps guard.
    X = np.column_stack([np.full(10, 4.0), np.arange(10.0)])
    Z = zscore(X, axis=0)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 0], 0.0)


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------
def test_kmeans_recovers_well_separated_blobs():
    rng = np.random.default_rng(7)
    centers = np.array([[0.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
    truth = np.repeat([0, 1, 2], 40)
    X = np.repeat(centers, 40, axis=0) + rng.normal(scale=0.5, size=(120, 2))
    labels = kmeans(X, k=3, random_state=0)
    assert adjusted_rand_index(truth, labels) == pytest.approx(1.0)


def test_kmeans_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 3))
    assert np.array_equal(kmeans(X, k=4, random_state=0), kmeans(X, k=4, random_state=0))


def test_kmeans_clamps_k_to_sample_count():
    X = np.arange(10.0).reshape(5, 2)
    labels = kmeans(X, k=10, random_state=0)  # more clusters than points requested
    assert labels.shape == (5,)
    assert set(np.unique(labels)).issubset(set(range(5)))


# ---------------------------------------------------------------------------
# knn_indices / neighborhood_mean
# ---------------------------------------------------------------------------
def test_knn_first_neighbor_is_self():
    rng = np.random.default_rng(9)
    coords = rng.normal(size=(30, 2))
    idx = knn_indices(coords, k=5)
    assert idx.shape == (30, 5)
    assert np.array_equal(idx[:, 0], np.arange(30))


def test_knn_clamps_k_to_point_count():
    coords = np.random.default_rng(10).normal(size=(4, 2))
    assert knn_indices(coords, k=99).shape == (4, 4)


def test_neighborhood_mean_with_k1_is_identity():
    # k=1 means each point's only neighbour is itself, so smoothing is a no-op.
    rng = np.random.default_rng(11)
    coords = rng.normal(size=(20, 2))
    feats = rng.normal(size=(20, 3))
    assert np.allclose(neighborhood_mean(feats, coords, k=1), feats)


def test_neighborhood_mean_preserves_shape_and_averages():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])
    feats = np.array([[1.0], [3.0], [-5.0]])
    # With k=2 the two close points average each other; the far point pairs with its
    # nearest (the closer of the two) neighbour.
    smoothed = neighborhood_mean(feats, coords, k=2)
    assert smoothed.shape == feats.shape
    assert smoothed[0, 0] == pytest.approx(2.0)  # mean(1, 3)
