"""Non-oracle domain-count (K) estimation for realistic benchmarking.

Most domain-detection benchmarks historically inject the true number of
domains (``n_domains = domain_truth.nunique()``) — the so-called *oracle K*
problem.  Real analyses never know K a priori.  This module:

1. Estimates K from expression geometry alone (silhouette / BIC-GMM / gap).
2. Builds ``extra_params_factory`` callables for landscape / harness use.
3. Supports dual-track scoring (oracle vs estimated) so papers can report
   the sensitivity of leaderboards to the oracle-K leak.

Default scientific policy is ``estimate``; ``oracle`` is opt-in and should
be gated by :attr:`TaskContract.allow_oracle_k`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .._math import kmeans, pca, zscore
from ..data import SpatialTable

logger = logging.getLogger(__name__)

KPolicy = Literal["oracle", "estimate", "fixed", "dual"]
KEstimator = Literal["silhouette", "bic_gmm", "gap"]


@dataclass(frozen=True)
class KSelectionResult:
    """Outcome of a single K-estimation run."""

    k: int
    method: str
    scores: dict[int, float]
    k_range: tuple[int, int]
    n_obs: int
    n_pcs: int
    notes: str = ""
    oracle_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = {str(k): float(v) for k, v in self.scores.items()}
        payload["k_range"] = list(self.k_range)
        return payload


@dataclass
class DualTrackKReport:
    """Side-by-side oracle vs estimated K comparison for one landscape cell."""

    dataset: str
    oracle_k: int
    estimated_k: int
    estimator: str
    k_match: bool
    selection: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "oracle_k": self.oracle_k,
            "estimated_k": self.estimated_k,
            "estimator": self.estimator,
            "k_match": self.k_match,
            "selection": dict(self.selection),
        }


def estimate_n_domains(
    data: SpatialTable,
    *,
    method: KEstimator = "silhouette",
    k_min: int = 2,
    k_max: int | None = None,
    n_pcs: int = 15,
    random_state: int = 0,
    max_obs: int = 4000,
) -> KSelectionResult:
    """Estimate the number of spatial domains without using ground truth.

    Parameters
    ----------
    data
        Expression table.  Only ``X`` is used (and optionally subsampled).
    method
        ``"silhouette"`` (default) — max mean silhouette of k-means on PCA.
        ``"bic_gmm"`` — minimum BIC of a diagonal Gaussian mixture (numpy).
        ``"gap"`` — Tibshirani gap statistic with uniform reference.
    k_min, k_max
        Inclusive search range.  Default max is ``min(12, sqrt(n_obs))``.
    n_pcs
        PCA dimensionality before clustering.
    random_state
        Seed for PCA / k-means / gap reference draws.
    max_obs
        Cap on observations used for the search (speed on large slides).
    """
    raw_x = data.X
    if hasattr(raw_x, "toarray") and not isinstance(raw_x, np.ndarray):
        raw_x = raw_x.toarray()
    X = np.asarray(raw_x, dtype=float)
    n_obs = int(X.shape[0])
    if n_obs < 4:
        raise ValueError("estimate_n_domains needs at least 4 observations")

    if k_max is None:
        k_max = int(min(12, max(k_min + 1, int(np.sqrt(n_obs)))))
    k_max = int(max(k_min, min(k_max, n_obs - 1)))

    # Subsample for speed while preserving determinism.
    rng = np.random.default_rng(random_state)
    if n_obs > max_obs:
        idx = np.sort(rng.choice(n_obs, size=max_obs, replace=False))
        X = X[idx]
        n_obs = int(X.shape[0])

    feats = zscore(X)
    n_comp = int(min(n_pcs, n_obs - 1, feats.shape[1]))
    embedding = pca(feats, n_comp, random_state=random_state)

    if method == "silhouette":
        scores, best = _select_by_silhouette(embedding, k_min, k_max, random_state)
        notes = "max mean silhouette on PCA+k-means"
    elif method == "bic_gmm":
        scores, best = _select_by_bic_gmm(embedding, k_min, k_max, random_state)
        notes = "min BIC of diagonal Gaussian mixture"
    elif method == "gap":
        scores, best = _select_by_gap(embedding, k_min, k_max, random_state)
        notes = "gap statistic vs uniform reference (Tibshirani)"
    else:
        raise ValueError(f"unknown K estimator: {method!r}")

    oracle_k = None
    for key in ("domain_truth", "domain", "layer_guess"):
        if key in data.obs.columns:
            oracle_k = int(data.obs[key].nunique())
            break
    if oracle_k is None and data.uns.get("n_domains"):
        oracle_k = int(data.uns["n_domains"])

    return KSelectionResult(
        k=int(best),
        method=method,
        scores={int(k): float(v) for k, v in scores.items()},
        k_range=(int(k_min), int(k_max)),
        n_obs=n_obs,
        n_pcs=n_comp,
        notes=notes,
        oracle_k=oracle_k,
    )


def oracle_n_domains(
    data: SpatialTable,
    *,
    truth_key: str = "domain_truth",
) -> int:
    """Read true domain count (oracle).  Prefer estimate in real protocols."""
    if truth_key in data.obs.columns:
        return int(data.obs[truth_key].nunique())
    if data.uns.get("n_domains"):
        return int(data.uns["n_domains"])
    raise ValueError(f"oracle_n_domains: neither obs[{truth_key!r}] nor uns['n_domains'] present")


def make_domain_k_factory(
    *,
    policy: KPolicy = "estimate",
    fixed_k: int | None = None,
    estimator: KEstimator = "silhouette",
    truth_key: str = "domain_truth",
    allow_oracle_k: bool = False,
    random_state: int = 0,
    store: dict[str, KSelectionResult] | None = None,
) -> Callable[[SpatialTable], dict[str, Any]]:
    """Build an ``extra_params_factory`` that injects ``n_domains`` under *policy*.

    Parameters
    ----------
    policy
        ``estimate`` (default scientific path), ``oracle`` (requires
        ``allow_oracle_k=True``), ``fixed`` (uses *fixed_k*), or ``dual``
        (estimates K but also records the oracle for ablation reports).
    allow_oracle_k
        Hard gate: oracle / dual that *uses* oracle K for method params only
        proceeds when True.  Dual with estimate as the *param* source is
        always allowed; the oracle is stored only for reporting.
    store
        Optional dict filled with per-call :class:`KSelectionResult` keyed by
        ``id(data)`` (callers should re-key by dataset name after the run).
    """
    if policy == "fixed":
        if fixed_k is None or int(fixed_k) < 1:
            raise ValueError("policy='fixed' requires a positive fixed_k")
    if policy == "oracle" and not allow_oracle_k:
        raise ValueError(
            "policy='oracle' is disabled by default (oracle-K leak).  "
            "Pass allow_oracle_k=True only for controlled ablations, or use "
            "policy='estimate' / 'dual'."
        )

    cache: dict[int, KSelectionResult] = {}

    def factory(data: SpatialTable) -> dict[str, Any]:
        if policy == "fixed":
            assert fixed_k is not None
            return {"n_domains": int(fixed_k)}

        if policy == "oracle":
            k = oracle_n_domains(data, truth_key=truth_key)
            return {"n_domains": k}

        # estimate or dual → estimate for the method parameter
        key = id(data)
        if key not in cache:
            cache[key] = estimate_n_domains(data, method=estimator, random_state=random_state)
            if store is not None:
                # Caller should rename keys; store by temporary id for now.
                store[str(key)] = cache[key]
        result = cache[key]
        if policy == "dual":
            # Still use estimated K for methods; oracle recorded in result.
            try:
                result = KSelectionResult(
                    k=result.k,
                    method=result.method,
                    scores=result.scores,
                    k_range=result.k_range,
                    n_obs=result.n_obs,
                    n_pcs=result.n_pcs,
                    notes=result.notes + "; dual-track (params use estimate)",
                    oracle_k=oracle_n_domains(data, truth_key=truth_key),
                )
                cache[key] = result
                if store is not None:
                    store[str(key)] = result
            except ValueError:
                pass
        return {"n_domains": int(result.k)}

    return factory


def compare_k_policies(
    data: SpatialTable,
    *,
    estimator: KEstimator = "silhouette",
    truth_key: str = "domain_truth",
    random_state: int = 0,
) -> DualTrackKReport:
    """Report oracle vs estimated K for one dataset (no method runs)."""
    selection = estimate_n_domains(data, method=estimator, random_state=random_state)
    oracle = oracle_n_domains(data, truth_key=truth_key)
    return DualTrackKReport(
        dataset=str(data.uns.get("dataset_name", "dataset")),
        oracle_k=oracle,
        estimated_k=selection.k,
        estimator=estimator,
        k_match=oracle == selection.k,
        selection=selection.to_dict(),
    )


# ---------------------------------------------------------------------------
# Estimators (pure NumPy)
# ---------------------------------------------------------------------------
def _select_by_silhouette(
    embedding: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[dict[int, float], int]:
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -np.inf
    for k in range(k_min, k_max + 1):
        labels = kmeans(embedding, k, random_state=random_state)
        score = float(_mean_silhouette(embedding, labels))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return scores, best_k


def _mean_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (euclidean).  Returns -1 if undefined."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= X.shape[0]:
        return -1.0
    # Pairwise squared distances once.
    # For n up to a few thousand this is fine; larger tables are subsampled.
    n = X.shape[0]
    # Use chunked computation to limit memory: silhouette via cluster means.
    # Classical O(n²) path is OK for max_obs=4000 (~128 MB floats).
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))
    np.fill_diagonal(dist, 0.0)

    sil = np.zeros(n, dtype=float)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            sil[i] = 0.0
            continue
        a = float(dist[i, same].mean())
        b = np.inf
        for lab in unique:
            if lab == labels[i]:
                continue
            mask = labels == lab
            if mask.any():
                b = min(b, float(dist[i, mask].mean()))
        if not np.isfinite(b):
            sil[i] = 0.0
            continue
        denom = max(a, b)
        sil[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(np.mean(sil))


def _select_by_bic_gmm(
    embedding: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[dict[int, float], int]:
    """Diagonal-covariance GMM BIC (lower is better → scores stored as -BIC)."""
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -np.inf  # maximise -BIC
    for k in range(k_min, k_max + 1):
        bic = _diagonal_gmm_bic(embedding, k, random_state)
        score = -float(bic)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return scores, best_k


def _diagonal_gmm_bic(X: np.ndarray, k: int, random_state: int) -> float:
    """Hard-assignment diagonal GMM BIC after k-means init."""
    n, d = X.shape
    labels = kmeans(X, k, random_state=random_state)
    log_lik = 0.0
    n_params = k * (d + d) + (k - 1)  # means + diag vars + mixing weights
    for c in range(k):
        members = X[labels == c]
        n_c = members.shape[0]
        if n_c < 2:
            # Degenerate cluster — heavy penalty.
            return 1e18
        mean = members.mean(axis=0)
        var = members.var(axis=0) + 1e-6
        # log N(x | mean, diag(var))
        const = -0.5 * (d * np.log(2 * np.pi) + np.log(var).sum())
        quad = -0.5 * (((members - mean) ** 2) / var).sum(axis=1)
        # Mix weight = n_c / n
        log_lik += float((const + quad + np.log(n_c / n)).sum())
    bic = -2.0 * log_lik + n_params * np.log(n)
    return float(bic)


def _select_by_gap(
    embedding: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    n_refs: int = 5,
) -> tuple[dict[int, float], int]:
    """Gap statistic; scores are gap(k); pick smallest k with gap(k) >= gap(k+1)-s(k+1)."""
    rng = np.random.default_rng(random_state)
    lo = embedding.min(axis=0)
    hi = embedding.max(axis=0)
    gaps: dict[int, float] = {}
    sk: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = kmeans(embedding, k, random_state=random_state)
        w_k = _cluster_dispersion(embedding, labels)
        ref_logs = []
        for r in range(n_refs):
            ref = rng.uniform(lo, hi, size=embedding.shape)
            ref_labels = kmeans(ref, k, random_state=random_state + 1000 + r)
            ref_logs.append(np.log(_cluster_dispersion(ref, ref_labels) + 1e-12))
        ref_logs_arr = np.asarray(ref_logs, dtype=float)
        gap = float(ref_logs_arr.mean() - np.log(w_k + 1e-12))
        # s_k = sd * sqrt(1 + 1/B)
        sd = float(ref_logs_arr.std(ddof=1)) if n_refs > 1 else 0.0
        sk[k] = sd * np.sqrt(1.0 + 1.0 / n_refs)
        gaps[k] = gap

    best_k = k_max
    for k in range(k_min, k_max):
        if gaps[k] >= gaps[k + 1] - sk[k + 1]:
            best_k = k
            break
    return gaps, best_k


def _cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for lab in np.unique(labels):
        members = X[labels == lab]
        if members.shape[0] == 0:
            continue
        center = members.mean(axis=0)
        total += float(((members - center) ** 2).sum())
    return total
