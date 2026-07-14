"""Shared scientific input validation for external method adapters.

The external backends generally fail late (often after model initialisation) when
given normalized values instead of counts, NaNs, or malformed coordinates.  These
helpers keep those failures deterministic and in-process without densifying sparse
matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_count_matrix(matrix: Any, *, method: str) -> None:
    """Validate a raw UMI/count matrix without converting sparse data to dense."""

    values = _stored_values(matrix)
    _validate_numeric_values(values, method=method)
    if values.size and not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6):
        raise ValueError(f"{method} requires integer-like raw counts")
    if values.size == 0 or not np.any(values > 0):
        raise ValueError(f"{method} requires at least one positive count")
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError(f"{method} count matrix must be two-dimensional")
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if (row_sums <= 0).any():
        raise ValueError(f"{method} requires a positive library size for every observation")


def validate_nonnegative_matrix(matrix: Any, *, method: str) -> None:
    """Validate finite non-negative expression without densifying sparse data."""

    _validate_numeric_values(_stored_values(matrix), method=method)


def validate_spatial_coordinates(
    coordinates: Any,
    *,
    method: str,
    exact_dimensions: int | None = None,
    minimum_dimensions: int = 2,
) -> np.ndarray:
    """Return a finite two-dimensional coordinate matrix after validation."""

    array = np.asarray(coordinates)
    if array.ndim != 2:
        raise ValueError(f"{method} spatial coordinates must be a two-dimensional matrix")
    if exact_dimensions is not None and array.shape[1] != exact_dimensions:
        raise ValueError(
            f"{method} requires exactly {exact_dimensions} spatial coordinate columns "
            "(two-dimensional coordinates for this backend)"
        )
    if array.shape[1] < minimum_dimensions:
        raise ValueError(
            f"{method} requires at least {minimum_dimensions} spatial coordinate columns"
        )
    try:
        finite = np.isfinite(array).all()
    except TypeError as exc:
        raise ValueError(f"{method} spatial coordinates must be numeric") from exc
    if not finite:
        raise ValueError(f"{method} spatial coordinates must be finite")
    return array


def _stored_values(matrix: Any) -> np.ndarray:
    """Return explicit values for dense or scipy-like sparse matrices."""

    values = matrix.data if hasattr(matrix, "tocsr") and hasattr(matrix, "data") else matrix
    return np.asarray(values)


def _validate_numeric_values(values: np.ndarray, *, method: str) -> None:
    try:
        finite = np.isfinite(values).all()
        nonnegative = not np.any(values < 0)
    except TypeError as exc:
        raise ValueError(f"{method} input must contain numeric values") from exc
    if not finite or not nonnegative:
        raise ValueError(f"{method} input must contain finite non-negative values")
