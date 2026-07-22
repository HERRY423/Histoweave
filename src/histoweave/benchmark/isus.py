"""Information-theoretic Spatial Utility Score (ISUS).

ISUS estimates the fraction of label information contributed by coordinates
beyond expression::

    ISUS = I(D; S | E) / I(D; E)

``D`` is a discrete domain label, while expression ``E`` and coordinates ``S``
are continuous.  Mutual information is estimated with the Ross (2014) k-nearest
neighbour estimator and the conditional term is obtained through the chain
rule.  NumPy and SciPy are sufficient; scikit-learn is not required.

**Role.** ISUS is a *post-hoc*, label-conditioned descriptor.  It is **not** a
target-free pre-execution predictor of whether a spatial method will improve
domain recovery on an unlabelled query.

**Significance.** Coordinate-shuffle permutation nulls supply Monte Carlo
p-values and Z-scores for residual spatial MI.  When a null is available,
interpretation bands prefer permutation evidence over the legacy absolute
cut-offs :data:`ISUS_LOW` / :data:`ISUS_HIGH` (kept only as a heuristic
fallback).

**Downstream binding.** :func:`fit_isus_gain_calibration` binds ISUS to
observed spatial ARI gain from ``benchmark_long.csv`` (mean over methods of
best spatial-weight ARI minus the ``sw0.0`` baseline).  The resulting map is
honest about reliability when the correlation is weak or underpowered.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial import cKDTree
from scipy.special import digamma

# Legacy absolute cut-offs (heuristic only; prefer permutation bands when n_null>0).
ISUS_LOW = 0.1
ISUS_HIGH = 0.3
# Z-score cut-off for "spatial-critical" under a coordinate-shuffle null.
# Motivated by a ~one-sided 3-sigma residual, not an absolute ISUS fraction.
ISUS_Z_CRITICAL = 3.0
_MI_DENOMINATOR_EPS = 1e-9
_NULL_STD_EPS = 1e-12
# Minimum independent slices before a Spearman calibration is treated as
# even weakly informative for predictor assessment.
_MIN_PREDICTOR_SLICES = 8
_DEFAULT_NULL_PERMUTATIONS = 99


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


def isus_band(
    value: float | None,
    *,
    low: float = ISUS_LOW,
    high: float = ISUS_HIGH,
) -> str:
    """Map an ISUS value with absolute cut-offs (legacy heuristic bands).

    Prefer :func:`isus_band_from_permutation` when a coordinate-shuffle null is
    available; absolute thresholds remain for comparison and for ``n_null=0``.
    """
    if value is None or not math.isfinite(float(value)):
        return "undetermined"
    if value < low:
        return "expression-sufficient"
    if value <= high:
        return "modest-spatial-signal"
    return "spatial-critical"


def isus_band_from_permutation(
    *,
    p_value: float | None,
    z_score: float | None,
    alpha: float = 0.05,
    z_critical: float = ISUS_Z_CRITICAL,
) -> str:
    """Band residual spatial signal using permutation p-value and Z-score.

    * ``not_above_null`` — residual MI does not exceed the coordinate-shuffle null
    * ``modest-spatial-signal`` — significant residual, Z below ``z_critical``
    * ``spatial-critical`` — significant residual with Z ≥ ``z_critical``
    * ``undetermined`` — null statistics unavailable
    """
    if p_value is None or not math.isfinite(float(p_value)):
        return "undetermined"
    if float(p_value) >= alpha:
        return "not_above_null"
    if z_score is not None and math.isfinite(float(z_score)) and float(z_score) >= z_critical:
        return "spatial-critical"
    return "modest-spatial-signal"


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _isus_ratio(conditional: float, i_d_e: float) -> float | None:
    if i_d_e > _MI_DENOMINATOR_EPS:
        return float(conditional / i_d_e)
    return None


def _empirical_one_sided_pvalue(observed: float, null_samples: np.ndarray) -> float:
    """Right-tail Monte Carlo p-value with the +1 continuity correction."""
    null = np.asarray(null_samples, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not math.isfinite(observed):
        return float("nan")
    return float((1.0 + np.sum(null >= observed)) / (1.0 + null.size))


def _z_score(observed: float | None, null_samples: np.ndarray) -> float | None:
    """Standard score of ``observed`` under the empirical null."""
    if observed is None or not math.isfinite(float(observed)):
        return None
    null = np.asarray(null_samples, dtype=float)
    null = null[np.isfinite(null)]
    if null.size < 2:
        return None
    mean = float(np.mean(null))
    std = float(np.std(null, ddof=1))
    if std < _NULL_STD_EPS:
        # Degenerate null: infinite Z if above mean, else 0.
        if float(observed) > mean + _NULL_STD_EPS:
            return float("inf")
        if float(observed) < mean - _NULL_STD_EPS:
            return float("-inf")
        return 0.0
    return float((float(observed) - mean) / std)


def _null_quantile(null_samples: np.ndarray, q: float) -> float | None:
    null = np.asarray(null_samples, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not 0.0 <= q <= 1.0:
        return None
    return float(np.quantile(null, q))


def _coordinate_shuffle_null(
    expression_block: np.ndarray,
    spatial_block: np.ndarray,
    codes: np.ndarray,
    *,
    i_d_e: float,
    k: int,
    seed: int,
    n_null: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Null ISUS and I(D;S|E) by shuffling coordinate rows (labels/expression fixed)."""
    if n_null < 1:
        return np.array([], dtype=float), np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    n_obs = spatial_block.shape[0]
    null_isus = np.empty(n_null, dtype=float)
    null_cond = np.empty(n_null, dtype=float)
    for i in range(n_null):
        order = rng.permutation(n_obs)
        shuffled = spatial_block[order]
        i_d_se = mi_discrete_continuous(
            np.hstack([shuffled, expression_block]),
            codes,
            k=k,
            seed=seed + 1 + i,
        )
        conditional = max(float(i_d_se - i_d_e), 0.0)
        ratio = _isus_ratio(conditional, i_d_e)
        null_cond[i] = conditional
        null_isus[i] = float("nan") if ratio is None else float(ratio)
    return null_isus, null_cond


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
    # Coordinate-shuffle null calibration (type-I control for spatial residual).
    n_null: int = 0
    null_mean_isus: float | None = None
    null_std_isus: float | None = None
    null_mean_i_d_s_given_e: float | None = None
    null_std_i_d_s_given_e: float | None = None
    p_value_isus: float | None = None
    p_value_i_d_s_given_e: float | None = None
    z_score_isus: float | None = None
    z_score_i_d_s_given_e: float | None = None
    significant: bool | None = None
    alpha: float | None = None
    null_control: str | None = None
    # Absolute ISUS-scale thresholds implied by this dataset's null (not global 0.1/0.3).
    threshold_significant_isus: float | None = None
    threshold_critical_isus: float | None = None
    band_heuristic: str | None = None
    band_source: str = "heuristic_absolute"
    # Optional binding to benchmark_long spatial ARI gain (attached by calibration).
    expected_spatial_ari_gain: float | None = None
    expected_spatial_ari_gain_low: float | None = None
    expected_spatial_ari_gain_high: float | None = None
    gain_prediction_reliability: str | None = None
    gain_prediction_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def _z_json(value: float | None) -> float | None | str:
            if value is None:
                return None
            if math.isinf(value):
                return "inf" if value > 0 else "-inf"
            return value if math.isfinite(value) else None

        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "isus": _finite_or_none(self.isus),
            "i_d_e": _finite_or_none(self.i_d_e),
            "i_d_se": _finite_or_none(self.i_d_se),
            "i_d_s_given_e": _finite_or_none(self.i_d_s_given_e),
            "band": self.band,
            "band_heuristic": self.band_heuristic or isus_band(self.isus),
            "band_source": self.band_source,
            "n_obs": self.n_obs,
            "n_domains": self.n_domains,
            "n_pcs": self.n_pcs,
            "k": self.k,
            "estimator": self.estimator,
            "flags": list(self.flags),
            "role": "posthoc_label_conditioned_descriptor",
            "not_a_pre_predictor": True,
            "thresholds": {
                "heuristic_low": ISUS_LOW,
                "heuristic_high": ISUS_HIGH,
                "z_critical": ISUS_Z_CRITICAL,
                "null_significant_isus": _finite_or_none(self.threshold_significant_isus),
                "null_critical_isus": _finite_or_none(self.threshold_critical_isus),
            },
        }
        if self.n_null > 0:
            payload.update(
                {
                    "n_null": int(self.n_null),
                    "null_control": self.null_control,
                    "null_mean_isus": _finite_or_none(self.null_mean_isus),
                    "null_std_isus": _finite_or_none(self.null_std_isus),
                    "null_mean_i_d_s_given_e": _finite_or_none(self.null_mean_i_d_s_given_e),
                    "null_std_i_d_s_given_e": _finite_or_none(self.null_std_i_d_s_given_e),
                    "p_value_isus": _finite_or_none(self.p_value_isus),
                    "p_value_i_d_s_given_e": _finite_or_none(self.p_value_i_d_s_given_e),
                    "z_score_isus": _z_json(self.z_score_isus),
                    "z_score_i_d_s_given_e": _z_json(self.z_score_i_d_s_given_e),
                    "significant": self.significant,
                    "alpha": self.alpha,
                }
            )
        if self.expected_spatial_ari_gain is not None or self.gain_prediction_source:
            gain: dict[str, float | str | None] = {
                "expected_spatial_ari_gain": _finite_or_none(self.expected_spatial_ari_gain),
                "expected_spatial_ari_gain_low": _finite_or_none(
                    self.expected_spatial_ari_gain_low
                ),
                "expected_spatial_ari_gain_high": _finite_or_none(
                    self.expected_spatial_ari_gain_high
                ),
                "reliability": self.gain_prediction_reliability,
                "source": self.gain_prediction_source,
            }
            payload["downstream_gain"] = gain
        return payload


@dataclass(frozen=True)
class ISUSPredictorAssessment:
    """Whether ISUS tracks observed spatial-method gain (pre-predictor audit)."""

    n_slices: int
    spearman_rho: float | None
    spearman_pvalue: float | None
    predictor_status: str
    failure_reasons: list[str]
    statistical_gaps: list[str]
    min_slices_recommended: int = _MIN_PREDICTOR_SLICES

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_slices": self.n_slices,
            "spearman_rho_isus_vs_spatial_gain": _finite_or_none(self.spearman_rho),
            "spearman_pvalue": _finite_or_none(self.spearman_pvalue),
            "predictor_status": self.predictor_status,
            "failure_reasons": list(self.failure_reasons),
            "statistical_gaps": list(self.statistical_gaps),
            "min_slices_recommended": self.min_slices_recommended,
            "note": (
                "ISUS is post-hoc and label-conditioned. A non-significant or "
                "non-positive correlation with spatial ARI gain means it fails as "
                "a pre-execution predictor of method improvement."
            ),
        }


def assess_isus_predictor(
    records: Sequence[Mapping[str, Any]],
    *,
    isus_key: str = "isus",
    gain_key: str = "spatial_ari_gain",
    alpha: float = 0.05,
    min_slices: int = _MIN_PREDICTOR_SLICES,
) -> ISUSPredictorAssessment:
    """Quantify ISUS's failure (or provisional support) as a method-gain predictor.

    The assessment uses Spearman rank correlation between per-slice ISUS and
    observed spatial ARI gain.  Status values:

    * ``failed`` — correlation is non-positive or not significant at ``alpha``
    * ``underpowered`` — fewer than ``min_slices`` finite pairs
    * ``supported`` — significant positive correlation (still not causal proof)
    * ``undefined`` — fewer than two finite pairs
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min_slices < 2:
        raise ValueError("min_slices must be at least 2")

    isus_arr = np.array(
        [np.nan if r.get(isus_key) is None else float(r[isus_key]) for r in records],
        dtype=float,
    )
    gain_arr = np.array(
        [np.nan if r.get(gain_key) is None else float(r[gain_key]) for r in records],
        dtype=float,
    )
    ok = np.isfinite(isus_arr) & np.isfinite(gain_arr)
    n_ok = int(ok.sum())
    reasons: list[str] = []
    gaps: list[str] = []

    rho: float | None = None
    pvalue: float | None = None
    if n_ok < 2:
        status = "undefined"
        reasons.append("fewer_than_two_finite_pairs")
        gaps.append("Need at least two slices with finite ISUS and spatial ARI gain.")
        return ISUSPredictorAssessment(
            n_slices=n_ok,
            spearman_rho=None,
            spearman_pvalue=None,
            predictor_status=status,
            failure_reasons=reasons,
            statistical_gaps=gaps,
            min_slices_recommended=min_slices,
        )

    from scipy.stats import spearmanr

    if np.unique(isus_arr[ok]).size < 2 or np.unique(gain_arr[ok]).size < 2:
        status = "undefined"
        reasons.append("zero_variance_in_isus_or_gain")
        gaps.append(
            "ISUS or spatial ARI gain is constant across slices; rank correlation is undefined."
        )
        return ISUSPredictorAssessment(
            n_slices=n_ok,
            spearman_rho=None,
            spearman_pvalue=None,
            predictor_status=status,
            failure_reasons=reasons,
            statistical_gaps=gaps,
            min_slices_recommended=min_slices,
        )

    result = spearmanr(isus_arr[ok], gain_arr[ok])
    rho = float(result.correlation)
    pvalue = float(result.pvalue)
    if not math.isfinite(rho):
        rho = None
    if not math.isfinite(pvalue):
        pvalue = None

    underpowered = n_ok < min_slices
    if underpowered:
        reasons.append("sample_size_below_minimum")
        gaps.append(
            f"Only {n_ok} slices; recommend >= {min_slices} independent study/donor units "
            "before treating the correlation as informative."
        )

    if rho is None or pvalue is None:
        status = "undefined"
        reasons.append("correlation_undefined")
    else:
        if rho <= 0.0:
            reasons.append("correlation_nonpositive")
        if pvalue >= alpha:
            reasons.append("correlation_not_significant")
        # Supported only when adequately powered, positive, and significant.
        if not underpowered and rho > 0.0 and pvalue < alpha:
            status = "supported"
        elif underpowered and rho > 0.0 and pvalue < alpha:
            # Positive signal but n too small to treat as a validated predictor.
            status = "underpowered"
        elif underpowered:
            status = "underpowered"
        else:
            status = "failed"

    # Always surface residual statistical gaps for honest reporting.
    gaps.append(
        "Even a supported correlation does not license target-free pre-execution use; "
        "ISUS requires trusted domain labels."
    )
    if status in {"failed", "underpowered", "undefined"}:
        gaps.append(
            "Do not use ISUS bands as a gate for choosing spatial methods on unlabelled queries."
        )

    return ISUSPredictorAssessment(
        n_slices=n_ok,
        spearman_rho=rho,
        spearman_pvalue=pvalue,
        predictor_status=status,
        failure_reasons=reasons,
        statistical_gaps=gaps,
        min_slices_recommended=min_slices,
    )


def extract_spatial_ari_gains_from_long(
    source: str | Path | Sequence[Mapping[str, Any]],
    *,
    dataset_key: str = "dataset",
    config_key: str = "config",
    method_key: str = "method",
    ari_key: str = "ari",
) -> dict[str, float]:
    """Mean spatial ARI gain per dataset from a ``benchmark_long`` table.

    For each method, gain is ``max(ARI at sw>0) - ARI at sw0.0`` (seed-averaged
    first).  The dataset-level value is the mean of those method-level gains.
    Configs that cannot be paired with a ``@sw0.0`` baseline are skipped.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open(encoding="utf-8", newline="") as handle:
            rows: list[Mapping[str, Any]] = list(csv.DictReader(handle))
    else:
        rows = list(source)

    # dataset -> "method|config" -> list of ARI
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        ds = str(row.get(dataset_key, "")).strip()
        cfg = str(row.get(config_key, "")).strip()
        if not ds or not cfg:
            continue
        base = str(row.get(method_key) or cfg.split("@", 1)[0]).strip()
        try:
            ari = float(row[ari_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(ari):
            continue
        buckets.setdefault(ds, {}).setdefault(f"{base}|{cfg}", []).append(ari)

    gains: dict[str, float] = {}
    for ds, cfg_map in buckets.items():
        means = {key: float(np.mean(vals)) for key, vals in cfg_map.items()}
        per_method: list[float] = []
        bases = {key.split("|", 1)[0] for key in means}
        for base in bases:
            sw0 = means.get(f"{base}|{base}@sw0.0")
            spatial = [
                means[key]
                for key in means
                if key.startswith(f"{base}|") and not key.endswith("@sw0.0")
            ]
            if sw0 is not None and spatial:
                per_method.append(max(spatial) - sw0)
        if per_method:
            gains[ds] = float(np.mean(per_method))
    return gains


@dataclass(frozen=True)
class ExpectedSpatialGain:
    """ISUS-conditioned expectation of spatial ARI gain from a fitted map."""

    isus: float | None
    expected_spatial_ari_gain: float | None
    expected_spatial_ari_gain_low: float | None
    expected_spatial_ari_gain_high: float | None
    reliability: str
    source: str
    residual_std: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "isus": _finite_or_none(self.isus),
            "expected_spatial_ari_gain": _finite_or_none(self.expected_spatial_ari_gain),
            "expected_spatial_ari_gain_low": _finite_or_none(self.expected_spatial_ari_gain_low),
            "expected_spatial_ari_gain_high": _finite_or_none(self.expected_spatial_ari_gain_high),
            "reliability": self.reliability,
            "source": self.source,
            "residual_std": _finite_or_none(self.residual_std),
            "notes": list(self.notes),
        }


@dataclass
class ISUSGainCalibration:
    """Linear map from ISUS to observed spatial ARI gain (benchmark_long).

    Fit is ordinary least squares with leave-one-out residual scale for
    intervals.  Reliability is inherited from :func:`assess_isus_predictor`
    so a weak or underpowered correlation cannot silently become a strong claim.
    """

    n_slices: int
    slope: float | None
    intercept: float | None
    residual_std: float | None
    loo_rmse: float | None
    r_squared: float | None
    reliability: str
    predictor: ISUSPredictorAssessment
    per_slice: list[dict[str, Any]] = field(default_factory=list)
    empirical_points: list[dict[str, Any]] = field(default_factory=list)
    formula: str = "spatial_ari_gain ~ intercept + slope * isus"
    source: str = "benchmark_long_sw_delta"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_slices": self.n_slices,
            "slope": _finite_or_none(self.slope),
            "intercept": _finite_or_none(self.intercept),
            "residual_std": _finite_or_none(self.residual_std),
            "loo_rmse": _finite_or_none(self.loo_rmse),
            "r_squared": _finite_or_none(self.r_squared),
            "reliability": self.reliability,
            "formula": self.formula,
            "source": self.source,
            "predictor": self.predictor.to_dict(),
            "per_slice": list(self.per_slice),
            "empirical_points": list(self.empirical_points),
            "note": (
                "Maps post-hoc ISUS to observed mean spatial ARI gain "
                "(best sw>0 minus sw0.0, averaged over methods). Reliability "
                "follows the Spearman audit; do not treat low-reliability maps "
                "as pre-execution decisions for unlabelled queries."
            ),
        }

    def predict(self, isus: float | None, *, z: float = 1.96) -> ExpectedSpatialGain:
        return predict_expected_spatial_ari_gain(isus, self, z=z)


def _fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return intercept, slope, residual_std, r_squared for simple OLS."""
    n = int(x.size)
    if n < 2:
        raise ValueError("OLS requires at least two points")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    ss_xx = float(np.sum((x - x_mean) ** 2))
    if ss_xx < _NULL_STD_EPS:
        # Constant ISUS: intercept-only model.
        resid = y - y_mean
        residual_std = float(np.sqrt(np.sum(resid**2) / max(n - 1, 1)))
        return y_mean, 0.0, residual_std, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    resid = y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    residual_std = float(np.sqrt(ss_res / max(n - 2, 1)))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > _NULL_STD_EPS else 0.0
    return intercept, slope, residual_std, float(r_squared)


def fit_isus_gain_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    isus_key: str = "isus",
    gain_key: str = "spatial_ari_gain",
    alpha: float = 0.05,
    min_slices: int = _MIN_PREDICTOR_SLICES,
) -> ISUSGainCalibration:
    """Fit ISUS → spatial ARI gain using paired slice records.

    Each record should contain ISUS and the observed spatial ARI gain extracted
    from ``benchmark_long.csv`` (see :func:`extract_spatial_ari_gains_from_long`).
    """
    predictor = assess_isus_predictor(
        records,
        isus_key=isus_key,
        gain_key=gain_key,
        alpha=alpha,
        min_slices=min_slices,
    )
    pairs: list[tuple[float, float, Mapping[str, Any]]] = []
    for row in records:
        try:
            isus_v = row.get(isus_key)
            gain_v = row.get(gain_key)
            if isus_v is None or gain_v is None:
                continue
            isus_f = float(isus_v)
            gain_f = float(gain_v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(isus_f) and math.isfinite(gain_f):
            pairs.append((isus_f, gain_f, row))

    if len(pairs) < 2:
        return ISUSGainCalibration(
            n_slices=len(pairs),
            slope=None,
            intercept=None,
            residual_std=None,
            loo_rmse=None,
            r_squared=None,
            reliability="undefined",
            predictor=predictor,
            per_slice=[],
            empirical_points=[],
        )

    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    intercept, slope, residual_std, r_squared = _fit_ols(x, y)

    # Leave-one-out residuals for a more honest predictive scale.
    loo_errors: list[float] = []
    for i in range(len(pairs)):
        mask = np.ones(len(pairs), dtype=bool)
        mask[i] = False
        if int(mask.sum()) < 2:
            continue
        inter_i, slope_i, _, _ = _fit_ols(x[mask], y[mask])
        pred_i = inter_i + slope_i * x[i]
        loo_errors.append(float(y[i] - pred_i))
    loo_rmse = float(np.sqrt(np.mean(np.square(loo_errors)))) if loo_errors else residual_std

    reliability = {
        "supported": "moderate",
        "underpowered": "low",
        "failed": "unsupported",
        "undefined": "undefined",
    }.get(predictor.predictor_status, "unsupported")

    per_slice: list[dict[str, Any]] = []
    for isus_f, gain_f, row in pairs:
        fitted = intercept + slope * isus_f
        entry = {
            "dataset": str(row.get("dataset", "")),
            "isus": isus_f,
            "spatial_ari_gain": gain_f,
            "fitted_spatial_ari_gain": fitted,
            "residual": gain_f - fitted,
        }
        per_slice.append(entry)

    order = np.argsort(x)
    empirical_points = [{"isus": float(x[i]), "spatial_ari_gain": float(y[i])} for i in order]

    return ISUSGainCalibration(
        n_slices=len(pairs),
        slope=slope,
        intercept=intercept,
        residual_std=residual_std,
        loo_rmse=loo_rmse,
        r_squared=r_squared,
        reliability=reliability,
        predictor=predictor,
        per_slice=per_slice,
        empirical_points=empirical_points,
    )


def predict_expected_spatial_ari_gain(
    isus: float | None,
    calibration: ISUSGainCalibration,
    *,
    z: float = 1.96,
) -> ExpectedSpatialGain:
    """Apply a fitted gain map to a new (or held-out) ISUS value."""
    notes: list[str] = []
    if isus is None or not math.isfinite(float(isus)):
        return ExpectedSpatialGain(
            isus=None,
            expected_spatial_ari_gain=None,
            expected_spatial_ari_gain_low=None,
            expected_spatial_ari_gain_high=None,
            reliability="undefined",
            source=calibration.source,
            residual_std=calibration.loo_rmse or calibration.residual_std,
            notes=("ISUS is undefined; gain cannot be predicted.",),
        )
    if calibration.slope is None or calibration.intercept is None:
        return ExpectedSpatialGain(
            isus=float(isus),
            expected_spatial_ari_gain=None,
            expected_spatial_ari_gain_low=None,
            expected_spatial_ari_gain_high=None,
            reliability="undefined",
            source=calibration.source,
            notes=("Gain calibration is not fitted.",),
        )

    expected = float(calibration.intercept + calibration.slope * float(isus))
    scale = calibration.loo_rmse
    if scale is None:
        scale = calibration.residual_std
    low = high = None
    if scale is not None and math.isfinite(scale) and z > 0:
        low = expected - float(z) * float(scale)
        high = expected + float(z) * float(scale)

    if calibration.reliability in {"unsupported", "low", "undefined"}:
        notes.append(
            f"Map reliability is {calibration.reliability} "
            f"(predictor_status={calibration.predictor.predictor_status}); "
            "treat the expected gain as exploratory only."
        )
    notes.append(
        "Expected gain is calibrated on labelled benchmark slices and remains "
        "post-hoc; it is not a target-free pre-execution decision rule."
    )
    return ExpectedSpatialGain(
        isus=float(isus),
        expected_spatial_ari_gain=expected,
        expected_spatial_ari_gain_low=low,
        expected_spatial_ari_gain_high=high,
        reliability=calibration.reliability,
        source="linear_ols_loo",
        residual_std=scale,
        notes=tuple(notes),
    )


def attach_gain_prediction(
    result: ISUSResult,
    calibration: ISUSGainCalibration,
    *,
    z: float = 1.96,
) -> ISUSResult:
    """Return a copy of ``result`` with downstream gain fields filled in."""
    prediction = calibration.predict(result.isus, z=z)
    return ISUSResult(
        dataset=result.dataset,
        isus=result.isus,
        i_d_e=result.i_d_e,
        i_d_se=result.i_d_se,
        i_d_s_given_e=result.i_d_s_given_e,
        band=result.band,
        n_obs=result.n_obs,
        n_domains=result.n_domains,
        n_pcs=result.n_pcs,
        k=result.k,
        estimator=result.estimator,
        flags=list(result.flags),
        n_null=result.n_null,
        null_mean_isus=result.null_mean_isus,
        null_std_isus=result.null_std_isus,
        null_mean_i_d_s_given_e=result.null_mean_i_d_s_given_e,
        null_std_i_d_s_given_e=result.null_std_i_d_s_given_e,
        p_value_isus=result.p_value_isus,
        p_value_i_d_s_given_e=result.p_value_i_d_s_given_e,
        z_score_isus=result.z_score_isus,
        z_score_i_d_s_given_e=result.z_score_i_d_s_given_e,
        significant=result.significant,
        alpha=result.alpha,
        null_control=result.null_control,
        threshold_significant_isus=result.threshold_significant_isus,
        threshold_critical_isus=result.threshold_critical_isus,
        band_heuristic=result.band_heuristic,
        band_source=result.band_source,
        expected_spatial_ari_gain=prediction.expected_spatial_ari_gain,
        expected_spatial_ari_gain_low=prediction.expected_spatial_ari_gain_low,
        expected_spatial_ari_gain_high=prediction.expected_spatial_ari_gain_high,
        gain_prediction_reliability=prediction.reliability,
        gain_prediction_source=prediction.source,
    )


def compute_isus(
    expression: Any,
    spatial: Any,
    domains: Any,
    *,
    dataset: str = "dataset",
    n_pcs: int = 20,
    k: int = 3,
    seed: int = 0,
    n_null: int = 0,
    alpha: float = 0.05,
) -> ISUSResult:
    """Compute ISUS from expression, coordinates, and discrete domain labels.

    Parameters
    ----------
    n_null
        Number of coordinate-shuffle null draws.  When ``n_null > 0``, the result
        includes Monte Carlo p-values for ISUS and ``I(D;S|E)``.  Default ``0``
        keeps the cheap point estimate used by the decision-card path.
    alpha
        Significance level for the optional null calibration.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if n_null < 0:
        raise ValueError("n_null must be non-negative")
    if n_null > 0 and not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1) when n_null > 0")
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
    value = _isus_ratio(conditional, i_d_e)

    flags: list[str] = []
    counts = np.bincount(codes)
    if counts.min() <= k:
        flags.append(
            "At least one domain has <= k observations; its local neighbour count was reduced."
        )
    if value is None:
        flags.append("I(D;E) is near zero, so the ISUS ratio is undefined.")
    flags.append(
        "ISUS is post-hoc (requires domain labels) and is not a pre-execution predictor "
        "of spatial-method ARI gain."
    )

    heuristic_band = isus_band(value)
    band = heuristic_band
    band_source = "heuristic_absolute"

    null_mean_isus: float | None = None
    null_std_isus: float | None = None
    null_mean_cond: float | None = None
    null_std_cond: float | None = None
    p_isus: float | None = None
    p_cond: float | None = None
    z_isus: float | None = None
    z_cond: float | None = None
    significant: bool | None = None
    null_control: str | None = None
    resolved_alpha: float | None = None
    thr_sig: float | None = None
    thr_crit: float | None = None

    if n_null > 0:
        null_control = "coordinate_shuffle"
        resolved_alpha = float(alpha)
        null_isus, null_cond = _coordinate_shuffle_null(
            expression_block,
            spatial_block,
            codes,
            i_d_e=float(i_d_e),
            k=k,
            seed=seed + 17,
            n_null=n_null,
        )
        finite_isus = null_isus[np.isfinite(null_isus)]
        finite_cond = null_cond[np.isfinite(null_cond)]
        if finite_isus.size:
            null_mean_isus = float(np.mean(finite_isus))
            null_std_isus = float(np.std(finite_isus, ddof=1)) if finite_isus.size >= 2 else 0.0
        if finite_cond.size:
            null_mean_cond = float(np.mean(finite_cond))
            null_std_cond = float(np.std(finite_cond, ddof=1)) if finite_cond.size >= 2 else 0.0
        if value is not None:
            p_isus = _empirical_one_sided_pvalue(float(value), null_isus)
            z_isus = _z_score(float(value), null_isus)
        p_cond = _empirical_one_sided_pvalue(float(conditional), null_cond)
        z_cond = _z_score(float(conditional), null_cond)
        # Dataset-specific absolute thresholds from this null (replace global 0.1/0.3).
        thr_sig = _null_quantile(null_isus, 1.0 - alpha)
        if null_mean_isus is not None and null_std_isus is not None:
            thr_crit = float(null_mean_isus + ISUS_Z_CRITICAL * null_std_isus)

        # Prefer residual MI p-value: defined even when I(D;E)≈0.
        p_for_decision = p_cond if p_cond is not None and math.isfinite(p_cond) else p_isus
        z_for_band = z_cond if z_cond is not None else z_isus
        if p_for_decision is not None and math.isfinite(p_for_decision):
            significant = bool(p_for_decision < alpha)
            band = isus_band_from_permutation(
                p_value=p_for_decision,
                z_score=z_for_band,
                alpha=alpha,
                z_critical=ISUS_Z_CRITICAL,
            )
            band_source = "permutation_z"
            z_str = (
                "NA"
                if z_for_band is None
                else ("inf" if math.isinf(z_for_band) else f"{z_for_band:.3g}")
            )
            if not significant:
                flags.append(
                    f"Coordinate-shuffle null: residual spatial MI not significant "
                    f"at alpha={alpha:g} (p={p_for_decision:.4g}, Z={z_str})."
                )
            else:
                flags.append(
                    f"Coordinate-shuffle null: residual spatial MI exceeds chance "
                    f"at alpha={alpha:g} (p={p_for_decision:.4g}, Z={z_str}); "
                    "validates the descriptor, not method-gain prediction."
                )
            flags.append(
                f"Primary band uses permutation evidence (Z_critical={ISUS_Z_CRITICAL:g}); "
                f"heuristic band under absolute cut-offs 0.1/0.3 is {heuristic_band!r}."
            )
        else:
            flags.append("Coordinate-shuffle null could not produce a finite p-value.")
    else:
        flags.append(
            "No permutation null requested (n_null=0); band uses legacy absolute "
            f"cut-offs ISUS_LOW={ISUS_LOW} / ISUS_HIGH={ISUS_HIGH} and is subjective."
        )

    return ISUSResult(
        dataset=str(dataset),
        isus=None if value is None else float(value),
        i_d_e=float(i_d_e),
        i_d_se=float(i_d_se),
        i_d_s_given_e=conditional,
        band=band,
        n_obs=int(len(labels)),
        n_domains=int(len(unique_labels)),
        n_pcs=n_components,
        k=int(k),
        flags=flags,
        n_null=int(n_null),
        null_mean_isus=null_mean_isus,
        null_std_isus=null_std_isus,
        null_mean_i_d_s_given_e=null_mean_cond,
        null_std_i_d_s_given_e=null_std_cond,
        p_value_isus=p_isus,
        p_value_i_d_s_given_e=p_cond,
        z_score_isus=z_isus,
        z_score_i_d_s_given_e=z_cond,
        significant=significant,
        alpha=resolved_alpha,
        null_control=null_control,
        threshold_significant_isus=thr_sig,
        threshold_critical_isus=thr_crit,
        band_heuristic=heuristic_band,
        band_source=band_source,
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
    n_null: int = 0,
    alpha: float = 0.05,
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
        n_null=n_null,
        alpha=alpha,
    )
