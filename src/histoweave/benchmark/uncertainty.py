"""Target-free, label-permutation-invariant uncertainty for spatial boundaries.

Cluster identifiers cannot be compared directly across methods because their labels are
arbitrary.  This module instead asks whether each method places a boundary on the same
spatial k-nearest-neighbour edge.  Edge votes are invariant to relabelling.  Their
Bernoulli entropy quantifies cross-method uncertainty and is averaged over incident
edges to produce a per-observation map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._math import knn_indices


@dataclass(frozen=True)
class BoundaryUncertaintyResult:
    """Per-observation uncertainty and method-specific boundary evidence."""

    uncertainty: np.ndarray
    consensus_boundary_strength: np.ndarray
    per_method_boundary_strength: dict[str, np.ndarray]
    missed_boundary_score: dict[str, np.ndarray]
    method_names: tuple[str, ...]
    neighbours: np.ndarray
    k: int
    consensus_min_methods: int

    def summary(self) -> dict[str, Any]:
        """Return a finite JSON-compatible summary of the result."""

        return {
            "n_obs": int(len(self.uncertainty)),
            "k": self.k,
            "method_names": list(self.method_names),
            "consensus_min_methods": self.consensus_min_methods,
            "uncertainty_mean": float(np.mean(self.uncertainty)),
            "uncertainty_max": float(np.max(self.uncertainty)),
            "consensus_boundary_strength_mean": float(
                np.mean(self.consensus_boundary_strength)
            ),
        }


def _validate_inputs(
    coords: np.ndarray,
    predictions: dict[str, np.ndarray],
    k: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    coordinates = np.asarray(coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coords must have shape (n_obs, >=2)")
    if len(coordinates) < 2:
        raise ValueError("at least two observations are required")
    if not np.isfinite(coordinates).all():
        raise ValueError("coords must contain only finite values")
    if len(predictions) < 2:
        raise ValueError("at least two method predictions are required")
    names = list(predictions)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("prediction names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("prediction names must be unique")

    labels: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(coordinates):
            raise ValueError(f"prediction {name!r} must be 1D with n_obs values")
        if any(value is None for value in array.tolist()):
            raise ValueError(f"prediction {name!r} contains missing labels")
        labels[name] = array

    neighbour_count = int(k)
    if neighbour_count < 1 or neighbour_count >= len(coordinates):
        raise ValueError("k must satisfy 1 <= k < n_obs")
    return coordinates, labels, neighbour_count


def _neighbours_without_self(coords: np.ndarray, k: int) -> np.ndarray:
    candidates = knn_indices(coords, k + 1)
    rows: list[np.ndarray] = []
    for index, row in enumerate(candidates):
        without_self = row[row != index]
        if len(without_self) < k:
            raise RuntimeError("kNN search did not return enough non-self neighbours")
        rows.append(without_self[:k])
    return np.vstack(rows)


def _bernoulli_entropy(probability: np.ndarray) -> np.ndarray:
    entropy = np.zeros_like(probability, dtype=float)
    interior = (probability > 0.0) & (probability < 1.0)
    p = probability[interior]
    entropy[interior] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return entropy


def boundary_uncertainty(
    coords: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    k: int = 6,
    consensus_min_methods: int | None = None,
) -> BoundaryUncertaintyResult:
    """Estimate per-observation boundary uncertainty from multiple partitions.

    For every spatial neighbour edge ``(i, j)``, a method casts one binary vote:
    ``prediction[i] != prediction[j]``.  The normalized Bernoulli entropy of the mean
    vote is zero when all methods agree and maximal when they split evenly.  A method's
    missed-boundary score is the fraction of incident edges where it votes "no
    boundary" while a strict method majority votes "boundary".
    """

    coordinates, labels, neighbour_count = _validate_inputs(coords, predictions, k)
    method_names = tuple(labels)
    n_methods = len(method_names)
    support = (
        n_methods // 2 + 1 if consensus_min_methods is None else int(consensus_min_methods)
    )
    if support < 1 or support > n_methods:
        raise ValueError("consensus_min_methods must be between 1 and n_methods")

    neighbours = _neighbours_without_self(coordinates, neighbour_count)
    votes = np.stack(
        [labels[name][:, None] != labels[name][neighbours] for name in method_names],
        axis=0,
    )
    boundary_probability = votes.mean(axis=0)
    uncertainty = _bernoulli_entropy(boundary_probability).mean(axis=1)
    consensus_boundary_strength = boundary_probability.mean(axis=1)
    consensus_edges = votes.sum(axis=0) >= support

    per_method = {
        name: votes[index].mean(axis=1).astype(float)
        for index, name in enumerate(method_names)
    }
    missed = {
        name: ((~votes[index]) & consensus_edges).mean(axis=1).astype(float)
        for index, name in enumerate(method_names)
    }
    return BoundaryUncertaintyResult(
        uncertainty=uncertainty,
        consensus_boundary_strength=consensus_boundary_strength,
        per_method_boundary_strength=per_method,
        missed_boundary_score=missed,
        method_names=method_names,
        neighbours=neighbours,
        k=neighbour_count,
        consensus_min_methods=support,
    )


def boundary_mask_from_labels(coords: np.ndarray, labels: np.ndarray, *, k: int = 6) -> np.ndarray:
    """Return observations incident to at least one label-changing spatial edge."""

    coordinates, validated, neighbour_count = _validate_inputs(
        coords,
        {"labels": np.asarray(labels), "labels_copy": np.asarray(labels)},
        k,
    )
    neighbours = _neighbours_without_self(coordinates, neighbour_count)
    values = validated["labels"]
    return np.any(values[:, None] != values[neighbours], axis=1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks: np.ndarray = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _roc_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return None
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def uncertainty_enrichment(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    *,
    high_quantile: float = 0.8,
) -> dict[str, float | int | None]:
    """Quantify whether high-uncertainty observations enrich a validation mask."""

    values = np.asarray(scores, dtype=float)
    positive = np.asarray(positive_mask, dtype=bool)
    if values.ndim != 1 or positive.ndim != 1 or values.shape != positive.shape:
        raise ValueError("scores and positive_mask must be aligned 1D arrays")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    if not 0.0 < high_quantile < 1.0:
        raise ValueError("high_quantile must be between 0 and 1")

    threshold = float(np.quantile(values, high_quantile))
    high = values >= threshold
    prevalence = float(positive.mean())
    high_prevalence = float(positive[high].mean()) if high.any() else 0.0
    enrichment = high_prevalence / prevalence if prevalence > 0 else None
    recall = float((positive & high).sum() / positive.sum()) if positive.any() else None
    return {
        "n_obs": int(len(values)),
        "n_positive": int(positive.sum()),
        "high_quantile": float(high_quantile),
        "high_threshold": threshold,
        "n_high": int(high.sum()),
        "positive_prevalence": prevalence,
        "positive_prevalence_high": high_prevalence,
        "enrichment": enrichment,
        "recall_at_high": recall,
        "roc_auc": _roc_auc(values, positive),
        "mean_uncertainty_positive": float(values[positive].mean()) if positive.any() else None,
        "mean_uncertainty_negative": float(values[~positive].mean()) if (~positive).any() else None,
    }
