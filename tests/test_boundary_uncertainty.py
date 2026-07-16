from __future__ import annotations

import numpy as np
import pytest

from histoweave.benchmark.uncertainty import (
    boundary_mask_from_labels,
    boundary_uncertainty,
    uncertainty_enrichment,
)


def _line_coords(n: int = 6) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=float), np.zeros(n)])


def test_boundary_uncertainty_is_label_permutation_invariant() -> None:
    coords = _line_coords()
    first = np.array(["a", "a", "a", "b", "b", "b"])
    second = np.array([10, 10, 10, 20, 20, 20])
    permuted = np.array([99, 99, 99, -4, -4, -4])

    expected = boundary_uncertainty(coords, {"m1": first, "m2": second}, k=2)
    observed = boundary_uncertainty(coords, {"m1": first, "m2": permuted}, k=2)

    np.testing.assert_allclose(observed.uncertainty, expected.uncertainty)
    np.testing.assert_allclose(
        observed.consensus_boundary_strength,
        expected.consensus_boundary_strength,
    )


def test_uncertainty_localizes_a_boundary_missed_by_one_method() -> None:
    coords = _line_coords()
    late_boundary = np.array([0, 0, 0, 0, 1, 1])
    consensus_boundary = np.array(["x", "x", "x", "y", "y", "y"])
    result = boundary_uncertainty(
        coords,
        {
            "single": late_boundary,
            "spatial": consensus_boundary,
            "supervised": consensus_boundary.copy(),
        },
        k=2,
    )

    assert result.uncertainty[2] > 0
    assert result.uncertainty[3] > 0
    assert result.missed_boundary_score["single"][2] > 0
    assert result.missed_boundary_score["single"][3] > 0
    assert result.missed_boundary_score["spatial"][2] == 0


def test_uncertainty_enrichment_reports_boundary_signal() -> None:
    coords = _line_coords()
    truth = np.array([0, 0, 0, 1, 1, 1])
    result = boundary_uncertainty(
        coords,
        {
            "m1": np.array([0, 0, 0, 0, 1, 1]),
            "m2": truth,
            "m3": truth.copy(),
        },
        k=2,
    )
    boundary = boundary_mask_from_labels(coords, truth, k=2)
    metrics = uncertainty_enrichment(result.uncertainty, boundary, high_quantile=0.5)

    assert metrics["enrichment"] is not None
    assert metrics["enrichment"] > 1.0
    assert metrics["roc_auc"] is not None
    assert metrics["roc_auc"] > 0.5


@pytest.mark.parametrize("k", [0, 6])
def test_boundary_uncertainty_rejects_invalid_k(k: int) -> None:
    coords = _line_coords()
    labels = np.arange(len(coords))
    with pytest.raises(ValueError, match="k must satisfy"):
        boundary_uncertainty(coords, {"a": labels, "b": labels}, k=k)
