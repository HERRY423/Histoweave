"""Property-based tests for the numerical core (Hypothesis).

These catch invariant violations that example-based unit tests miss: NaN
propagation, label-permutation symmetry, kNN self-inclusion, and ARI bounds.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from histoweave._math import adjusted_rand_index, knn_indices, neighborhood_mean, zscore

pytestmark = pytest.mark.property

finite_matrix = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(3, 40), st.integers(2, 20)),
    elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)


@st.composite
def unique_coords(draw: st.DrawFn) -> np.ndarray:
    """2-D coordinates with unique rows (avoids degenerate kNN sets)."""
    n = draw(st.integers(4, 40))
    coords = np.column_stack(
        [
            np.arange(n, dtype=float),
            draw(
                arrays(
                    dtype=np.float64,
                    shape=(n,),
                    elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
                )
            ),
        ]
    )
    return coords


label_vector = st.lists(st.integers(0, 6), min_size=2, max_size=80)


@given(finite_matrix)
@settings(max_examples=40, deadline=None)
def test_zscore_columns_have_unit_scale(matrix: np.ndarray) -> None:
    # Skip near-constant columns (std ~ 0 → only eps regularisation).
    scaled = zscore(matrix, axis=0)
    assert scaled.shape == matrix.shape
    assert np.isfinite(scaled).all()
    std = matrix.std(axis=0)
    active = std > 1e-3
    if active.any():
        assert np.allclose(scaled[:, active].mean(axis=0), 0.0, atol=1e-5)
        # Population std after z-score is ~1; eps regularisation keeps it slightly < 1.
        assert np.allclose(scaled[:, active].std(axis=0), 1.0, atol=5e-3)


@given(unique_coords(), st.integers(1, 8))
@settings(max_examples=30, deadline=None)
def test_knn_includes_self_and_unique_neighbours(coords: np.ndarray, k: int) -> None:
    n = coords.shape[0]
    k = min(k, n)
    indices = knn_indices(coords, k)
    assert indices.shape == (n, k)
    # Self is always distance 0 and therefore among the nearest neighbours.
    for i, row in enumerate(indices):
        assert i in set(row.tolist())
        assert len(set(row.tolist())) == k
        assert ((row >= 0) & (row < n)).all()


@given(unique_coords(), st.integers(1, 6))
@settings(max_examples=25, deadline=None)
def test_neighborhood_mean_preserves_shape(coords: np.ndarray, k: int) -> None:
    features = np.arange(coords.shape[0] * 3, dtype=float).reshape(coords.shape[0], 3)
    out = neighborhood_mean(features, coords, k)
    assert out.shape == features.shape
    assert np.isfinite(out).all()


@given(label_vector)
@settings(max_examples=40, deadline=None)
def test_ari_identical_is_one(labels: list[int]) -> None:
    arr = np.asarray(labels)
    assert adjusted_rand_index(arr, arr) == pytest.approx(1.0)


@given(label_vector, st.integers(0, 20))
@settings(max_examples=40, deadline=None)
def test_ari_is_label_permutation_invariant(labels: list[int], offset: int) -> None:
    true = np.asarray(labels)
    # Affine remapping of ids is the same partition.
    pred = true * 3 + offset
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0)


@given(label_vector, label_vector)
@settings(max_examples=40, deadline=None)
def test_ari_is_symmetric_and_bounded(left: list[int], right: list[int]) -> None:
    n = min(len(left), len(right))
    a = np.asarray(left[:n])
    b = np.asarray(right[:n])
    score = adjusted_rand_index(a, b)
    assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9
    assert score == pytest.approx(adjusted_rand_index(b, a))
