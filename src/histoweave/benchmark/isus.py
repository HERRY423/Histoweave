"""Information-theoretic Spatial Utility Score (ISUS).

ISUS estimates the fraction of label information contributed by coordinates
beyond expression::

    ISUS = I(D; S | E) / I(D; E)

``D`` is a discrete domain label, while expression ``E`` and coordinates ``S``
are continuous.  Mutual information is estimated with the Ross (2014) k-nearest
neighbour estimator and the conditional term is obtained through the chain
rule.  NumPy and SciPy are sufficient; scikit-learn is not required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial import cKDTree
from scipy.special import digamma

ISUS_LOW = 0.1
ISUS_HIGH = 0.3
_MI_DENOMINATOR_EPS = 1e-9


def _as_2d_float(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {array.shape}")
    return array


def _standardize(values: np.ndarray) -> np.ndarray:
    values = _as_2d_float(values, name="continuous values")
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def mi_discrete_continuous(
    continuous: Any,
    discrete: Any,
    *,
    k: int = 3,
    seed: int = 0,
) -> float:
    """Estimate mutual information between discrete and continuous variables.

    The estimator uses a Chebyshev-distance kNN ball within each class and then
    counts neighbours in the full sample, following Ross (2014).  For a class
    with fewer than ``k + 1`` observations, its local ``k`` is reduced.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    values = _as_2d_float(continuous, name="continuous")
    labels = np.asarray(discrete)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if values.shape[0] != labels.shape[0]:
        raise ValueError(
            f"continuous and discrete must have the same number of rows, got "
            f"{values.shape[0]} and {labels.shape[0]}"
        )
    if values.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if not np.isfinite(values).all():
        raise ValueError("continuous values must be finite")
    if pd.isna(labels).any():
        raise ValueError("discrete labels must not be missing")

    codes, unique_labels = pd.factorize(labels, sort=True)
    if len(unique_labels) < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    feature_scale = values.std(axis=0, keepdims=True)
    values = values + 1e-10 * feature_scale * rng.standard_normal(values.shape)
    n_obs = values.shape[0]
    full_tree = cKDTree(values)
    total = 0.0
    usable = 0

    for code in range(len(unique_labels)):
        mask = codes == code
        count = int(mask.sum())
        if count <= 1:
            continue
        local_k = min(k, count - 1)
        subset = values[mask]
        class_tree = cKDTree(subset)
        distances, _ = class_tree.query(subset, k=local_k + 1, p=np.inf)
        radii = np.asarray(distances)[:, local_k]
        neighbourhoods = full_tree.query_ball_point(subset, r=radii, p=np.inf)
        full_counts = np.asarray([len(indices) - 1 for indices in neighbourhoods], dtype=float)
        total += float(
            np.sum(
                digamma(n_obs)
                - digamma(count)
                + digamma(local_k)
                - digamma(np.maximum(full_counts, 1.0))
            )
        )
        usable += count

    if usable == 0:
        return 0.0
    # Singleton classes contribute no estimable local-density term but remain in
    # N, matching the prototype used for the bundled calibration.
    return max(float(total / n_obs), 0.0)


def _dense_pca_scores(matrix: np.ndarray, n_components: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return np.zeros((matrix.shape[0], n_components), dtype=float)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(u[:, :n_components] * singular_values[:n_components], dtype=float)


def _sparse_pca_scores(matrix: Any, n_components: int, seed: int) -> np.ndarray:
    matrix = matrix.astype(float, copy=False)
    n_obs, n_vars = matrix.shape
    mean = np.asarray(matrix.mean(axis=0)).reshape(-1)

    def matvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ vector).reshape(-1) - float(mean @ vector)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix.T @ vector).reshape(-1) - mean * float(vector.sum())

    def matmat(vectors: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ vectors) - np.outer(np.ones(n_obs), mean @ vectors)

    def rmatmat(vectors: np.ndarray) -> np.ndarray:
        return np.asarray(matrix.T @ vectors) - np.outer(mean, vectors.sum(axis=0))

    centered = LinearOperator(
        shape=(n_obs, n_vars),
        dtype=np.dtype(float),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
    )
    u, singular_values, _ = svds(
        centered,
        k=n_components,
        which="LM",
        random_state=seed,
    )
    order = np.argsort(singular_values)[::-1]
    return np.asarray(u[:, order] * singular_values[order], dtype=float)


def _expression_pca(expression: Any, *, n_pcs: int, seed: int) -> tuple[np.ndarray, int]:
    if n_pcs < 1:
        raise ValueError("n_pcs must be at least 1")
    if getattr(expression, "ndim", None) != 2:
        raise ValueError("expression must be a two-dimensional cells-by-genes matrix")
    n_obs, n_vars = expression.shape
    if n_obs < 2 or n_vars < 1:
        raise ValueError("expression requires at least two observations and one feature")
    n_components = min(int(n_pcs), int(n_obs) - 1, int(n_vars))
    if issparse(expression):
        if not np.isfinite(expression.data).all():
            raise ValueError("expression values must be finite")
        if n_components < min(n_obs, n_vars):
            return _sparse_pca_scores(expression, n_components, seed), n_components
        dense = np.asarray(expression.toarray(), dtype=float)
    else:
        dense = np.asarray(expression, dtype=float)
        if not np.isfinite(dense).all():
            raise ValueError("expression values must be finite")
    return _dense_pca_scores(dense, n_components), n_components


def isus_band(value: float | None) -> str:
    """Map an ISUS value to the provisional interpretation bands."""
    if value is None or not math.isfinite(float(value)):
        return "undetermined"
    if value < ISUS_LOW:
        return "expression-sufficient"
    if value <= ISUS_HIGH:
        return "modest-spatial-signal"
    return "spatial-critical"


@dataclass
class ISUSResult:
    dataset: str
    isus: float | None
    i_d_e: float
    i_d_se: float
    i_d_s_given_e: float
    band: str
    n_obs: int
    n_domains: int
    n_pcs: int
    k: int
    estimator: str = "ross-2014-knn-chain-rule"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def safe(value: float | None) -> float | None:
            if value is None:
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        return {
            "dataset": self.dataset,
            "isus": safe(self.isus),
            "i_d_e": safe(self.i_d_e),
            "i_d_se": safe(self.i_d_se),
            "i_d_s_given_e": safe(self.i_d_s_given_e),
            "band": self.band,
            "n_obs": self.n_obs,
            "n_domains": self.n_domains,
            "n_pcs": self.n_pcs,
            "k": self.k,
            "estimator": self.estimator,
            "flags": list(self.flags),
        }


def compute_isus(
    expression: Any,
    spatial: Any,
    domains: Any,
    *,
    dataset: str = "dataset",
    n_pcs: int = 20,
    k: int = 3,
    seed: int = 0,
) -> ISUSResult:
    """Compute ISUS from expression, coordinates, and discrete domain labels."""
    if k < 1:
        raise ValueError("k must be at least 1")
    coordinates = _as_2d_float(spatial, name="spatial")
    labels = np.asarray(domains)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    n_expression = int(expression.shape[0]) if getattr(expression, "shape", None) else -1
    if n_expression != coordinates.shape[0] or labels.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "expression, spatial coordinates, and domain labels must have the same number "
            f"of observations; got {n_expression}, {coordinates.shape[0]}, {labels.shape[0]}"
        )

    valid = np.isfinite(coordinates).all(axis=1) & ~np.asarray(pd.isna(labels), dtype=bool)
    if valid.sum() < 2:
        raise ValueError("fewer than two observations remain after removing missing labels/coords")
    if not valid.all():
        expression = expression[valid]
        coordinates = coordinates[valid]
        labels = labels[valid]

    codes, unique_labels = pd.factorize(labels, sort=True)
    if len(unique_labels) < 2:
        raise ValueError("ISUS requires at least two domain labels")
    scores, n_components = _expression_pca(expression, n_pcs=n_pcs, seed=seed)
    expression_block = _standardize(scores)
    spatial_block = _standardize(coordinates)
    i_d_e = mi_discrete_continuous(expression_block, codes, k=k, seed=seed)
    i_d_se = mi_discrete_continuous(
        np.hstack([spatial_block, expression_block]),
        codes,
        k=k,
        seed=seed,
    )
    conditional = max(float(i_d_se - i_d_e), 0.0)
    value = conditional / i_d_e if i_d_e > _MI_DENOMINATOR_EPS else None

    flags: list[str] = []
    counts = np.bincount(codes)
    if counts.min() <= k:
        flags.append(
            "At least one domain has <= k observations; its local neighbour count was reduced."
        )
    if value is None:
        flags.append("I(D;E) is near zero, so the ISUS ratio is undefined.")
    flags.append(
        "Interpretation thresholds are provisional and require dataset-specific calibration."
    )
    return ISUSResult(
        dataset=str(dataset),
        isus=None if value is None else float(value),
        i_d_e=float(i_d_e),
        i_d_se=float(i_d_se),
        i_d_s_given_e=conditional,
        band=isus_band(value),
        n_obs=int(len(labels)),
        n_domains=int(len(unique_labels)),
        n_pcs=n_components,
        k=int(k),
        flags=flags,
    )


def compute_isus_from_table(
    table: Any,
    *,
    domain_key: str = "domain_truth",
    spatial_key: str = "spatial",
    dataset: str | None = None,
    n_pcs: int = 20,
    k: int = 3,
    seed: int = 0,
) -> ISUSResult:
    """Compute ISUS from a :class:`SpatialTable` or AnnData-like object."""
    if not hasattr(table, "X") or not hasattr(table, "obs"):
        raise TypeError("table must expose X and obs")
    if domain_key not in table.obs:
        raise ValueError(f"obs does not contain domain label column {domain_key!r}")
    coordinates = None
    if hasattr(table, "obsm") and spatial_key in table.obsm:
        coordinates = table.obsm[spatial_key]
    elif spatial_key == "spatial" and getattr(table, "spatial", None) is not None:
        coordinates = table.spatial
    if coordinates is None:
        raise ValueError(f"table does not contain obsm[{spatial_key!r}] coordinates")

    resolved_dataset = dataset
    if resolved_dataset is None:
        uns = getattr(table, "uns", {})
        if hasattr(uns, "get"):
            resolved_dataset = uns.get("dataset_name") or uns.get("dataset") or uns.get("name")
    return compute_isus(
        table.X,
        coordinates,
        np.asarray(table.obs[domain_key]),
        dataset=str(resolved_dataset or "dataset"),
        n_pcs=n_pcs,
        k=k,
        seed=seed,
    )

