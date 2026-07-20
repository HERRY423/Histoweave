"""Non-oracle domain-count (K) estimation for realistic benchmarking.

Most domain-detection benchmarks historically inject the true number of
domains (``n_domains = domain_truth.nunique()``) — the so-called *oracle K*
problem.  Real analyses never know K a priori.  This module:

1. Estimates K **without** ground-truth labels (blind / non-oracle path).
2. Supports pure-expression *and* spatial-aware geometries so estimation is
   not blind to coordinates when they exist.
3. Builds ``extra_params_factory`` callables for landscape / harness use.
4. Supports dual-track scoring (oracle vs estimated) so papers can report
   the sensitivity of leaderboards to the oracle-K leak.

Default scientific policy is ``estimate``; ``oracle`` is opt-in and should
be gated by :attr:`TaskContract.allow_oracle_k`.

Blind-mode estimators
---------------------
Expression-only (legacy; geometry ignores coordinates):

* ``silhouette`` — max mean silhouette of k-means on PCA
* ``bic_gmm`` — min BIC of a diagonal Gaussian mixture
* ``gap`` — Tibshirani gap statistic with uniform reference
* ``calinski_harabasz`` — max CH variance-ratio index
* ``davies_bouldin`` — min Davies–Bouldin index

Spatial-aware (use coordinates; required for realistic spatial domain K):

* ``spatial_silhouette`` — silhouette on neighbourhood-smoothed expression PCA
* ``spatial_coherence`` — maximise spatial kNN label agreement of clusters
* ``ensemble`` (**default for ``k_policy='estimate'``**) — majority vote across
  expression + spatial criteria when coordinates are present; expression-only
  fallback when they are not
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .._math import kmeans, knn_indices, neighborhood_mean, pca, zscore
from ..data import SpatialTable

logger = logging.getLogger(__name__)

KPolicy = Literal["oracle", "estimate", "fixed", "dual"]
KEstimator = Literal[
    "silhouette",
    "bic_gmm",
    "gap",
    "calinski_harabasz",
    "davies_bouldin",
    "spatial_silhouette",
    "spatial_coherence",
    "ensemble",
]
EmbeddingGeometry = Literal["expression", "spatial_smooth", "joint"]

# Expression-only estimators (do not require coordinates).
_EXPRESSION_ESTIMATORS: tuple[str, ...] = (
    "silhouette",
    "bic_gmm",
    "gap",
    "calinski_harabasz",
    "davies_bouldin",
)
# Estimators that hard-require spatial coordinates.
_SPATIAL_REQUIRED: tuple[str, ...] = ("spatial_silhouette", "spatial_coherence")
# Members of the ensemble when coordinates are available.
# Expression-only silhouette is retained at low weight (see _estimate_ensemble)
# so spatial criteria can move K away from the trivial two-cluster mode.
_ENSEMBLE_WITH_SPATIAL: tuple[str, ...] = (
    "silhouette",
    "bic_gmm",
    "calinski_harabasz",
    "spatial_silhouette",
    "spatial_coherence",
)
# Members of the ensemble without coordinates.
_ENSEMBLE_EXPRESSION_ONLY: tuple[str, ...] = (
    "silhouette",
    "bic_gmm",
    "calinski_harabasz",
    "davies_bouldin",
)


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
    geometry: str = "expression"
    spatial_used: bool = False
    component_votes: dict[str, int] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = {str(k): float(v) for k, v in self.scores.items()}
        payload["k_range"] = list(self.k_range)
        payload["component_votes"] = {str(m): int(v) for m, v in self.component_votes.items()}
        payload["flags"] = list(self.flags)
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


def _coords_from_table(data: SpatialTable) -> np.ndarray | None:
    """Return finite spatial coordinates or ``None`` if unavailable."""
    coords = getattr(data, "spatial", None)
    if coords is None and hasattr(data, "obsm"):
        obsm = data.obsm
        if obsm is not None and "spatial" in obsm:
            coords = obsm["spatial"]
    if coords is None:
        return None
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 1:
        return None
    if not np.isfinite(array).all():
        # Drop non-finite rows only when building the shared index later.
        if not np.isfinite(array).any():
            return None
    return array


def _expression_matrix(data: SpatialTable) -> np.ndarray:
    raw_x = data.X
    if hasattr(raw_x, "toarray") and not isinstance(raw_x, np.ndarray):
        raw_x = raw_x.toarray()
    return np.asarray(raw_x, dtype=float)


def _build_embedding(
    X: np.ndarray,
    coords: np.ndarray | None,
    *,
    geometry: EmbeddingGeometry,
    n_pcs: int,
    random_state: int,
    knn: int,
    spatial_weight: float,
) -> tuple[np.ndarray, int, str]:
    """Return (embedding, n_components_used, geometry_resolved)."""
    feats = zscore(X)
    n_obs = int(feats.shape[0])
    n_comp = int(min(n_pcs, n_obs - 1, feats.shape[1]))
    if n_comp < 1:
        raise ValueError("embedding requires at least one PCA component")

    if geometry == "expression" or coords is None:
        embedding = pca(feats, n_comp, random_state=random_state)
        return embedding, n_comp, "expression"

    # Align coords to the same rows (caller already subsampled both together).
    coords_z = zscore(np.asarray(coords, dtype=float))
    knn_use = int(min(max(knn, 2), n_obs))

    if geometry == "spatial_smooth":
        smoothed = neighborhood_mean(feats, coords, knn_use)
        # BANKSY-style blend: keep some raw signal so pure spatial blobs
        # without expression structure still do not collapse.
        blended = (1.0 - float(spatial_weight)) * feats + float(spatial_weight) * smoothed
        embedding = pca(blended, n_comp, random_state=random_state)
        return embedding, n_comp, "spatial_smooth"

    if geometry == "joint":
        expr_emb = pca(feats, n_comp, random_state=random_state)
        # Scale coordinates so they contribute without drowning expression PCs.
        # Each block is z-scored; spatial block reweighted by spatial_weight.
        spatial_block = coords_z * float(spatial_weight) * np.sqrt(max(n_comp, 1))
        embedding = np.hstack([expr_emb, spatial_block])
        return embedding, n_comp, "joint"

    raise ValueError(f"unknown embedding geometry: {geometry!r}")


def _default_geometry_for_method(
    method: str,
    *,
    has_spatial: bool,
    geometry: EmbeddingGeometry | None,
) -> EmbeddingGeometry:
    if geometry is not None:
        return geometry
    if method == "spatial_silhouette":
        return "spatial_smooth" if has_spatial else "expression"
    if method == "spatial_coherence":
        # Cluster on spatially-smoothed expression so domains align with space.
        return "spatial_smooth" if has_spatial else "expression"
    if method == "ensemble":
        return "expression"  # components choose their own geometries
    return "expression"


def estimate_n_domains(
    data: SpatialTable,
    *,
    method: KEstimator = "ensemble",
    k_min: int = 2,
    k_max: int | None = None,
    n_pcs: int = 15,
    random_state: int = 0,
    max_obs: int = 4000,
    geometry: EmbeddingGeometry | None = None,
    knn: int = 6,
    spatial_weight: float = 0.5,
) -> KSelectionResult:
    """Estimate the number of spatial domains without using ground truth.

    Parameters
    ----------
    data
        Expression table.  Coordinates in ``obsm['spatial']`` / ``.spatial``
        enable spatial-aware estimators; without them the path falls back to
        expression geometry.
    method
        See module docstring.  Default ``"ensemble"`` reduces single-criterion
        blindness in blind (non-oracle) runs.
    k_min, k_max
        Inclusive search range.  Default max is ``min(12, sqrt(n_obs))``.
    n_pcs
        PCA dimensionality before clustering.
    random_state
        Seed for PCA / k-means / gap reference draws.
    max_obs
        Cap on observations used for the search (speed on large slides).
    geometry
        Embedding geometry for single estimators: ``expression`` (default for
        legacy methods), ``spatial_smooth`` (neighbourhood blend + PCA), or
        ``joint`` (PCA scores ⊕ scaled coordinates).  ``None`` picks a sensible
        default per method.  Ignored by ``ensemble`` (each member has its own).
    knn
        Neighbourhood size for spatial smoothing / coherence.
    spatial_weight
        Blend weight in ``[0, 1]`` for spatial_smooth / joint geometries.
    """
    if not 0.0 <= float(spatial_weight) <= 1.0:
        raise ValueError("spatial_weight must be in [0, 1]")
    if knn < 1:
        raise ValueError("knn must be at least 1")
    # Spatial-only estimators need a stronger neighbourhood blend than the
    # generic default; otherwise silhouette on weakly smoothed PCA still
    # collapses to the expression two-cluster mode on layered Visium.
    if method in {"spatial_silhouette", "spatial_coherence", "ensemble"}:
        spatial_weight = max(float(spatial_weight), 0.75)

    X = _expression_matrix(data)
    coords_full = _coords_from_table(data)
    n_obs = int(X.shape[0])
    if n_obs < 4:
        raise ValueError("estimate_n_domains needs at least 4 observations")
    if coords_full is not None and coords_full.shape[0] != n_obs:
        raise ValueError(
            f"spatial coordinates length {coords_full.shape[0]} != n_obs {n_obs}"
        )

    if k_max is None:
        k_max = int(min(12, max(k_min + 1, int(np.sqrt(n_obs)))))
    k_max = int(max(k_min, min(k_max, n_obs - 1)))

    rng = np.random.default_rng(random_state)
    idx = None
    if n_obs > max_obs:
        idx = np.sort(rng.choice(n_obs, size=max_obs, replace=False))
        X = X[idx]
        if coords_full is not None:
            coords_full = coords_full[idx]
        n_obs = int(X.shape[0])

    # Drop rows with non-finite coordinates when spatial path is requested.
    coords = coords_full
    if coords is not None:
        finite = np.isfinite(coords).all(axis=1)
        if not finite.all():
            X = X[finite]
            coords = coords[finite]
            n_obs = int(X.shape[0])
            if n_obs < 4:
                coords = None

    has_spatial = coords is not None
    flags: list[str] = []

    if method in _SPATIAL_REQUIRED and not has_spatial:
        flags.append(
            f"method={method!r} requires spatial coordinates; falling back to silhouette "
            "on expression PCA."
        )
        method = "silhouette"

    if method == "ensemble":
        return _estimate_ensemble(
            X,
            coords,
            k_min=k_min,
            k_max=k_max,
            n_pcs=n_pcs,
            random_state=random_state,
            knn=knn,
            spatial_weight=spatial_weight,
            data=data,
            flags=flags,
        )

    resolved_geometry = _default_geometry_for_method(
        method, has_spatial=has_spatial, geometry=geometry
    )
    if resolved_geometry != "expression" and not has_spatial:
        flags.append(
            f"geometry={resolved_geometry!r} requested without coordinates; using expression."
        )
        resolved_geometry = "expression"

    embedding, n_comp, geo_used = _build_embedding(
        X,
        coords,
        geometry=resolved_geometry,
        n_pcs=n_pcs,
        random_state=random_state,
        knn=knn,
        spatial_weight=spatial_weight,
    )

    scores, best, notes = _run_single_estimator(
        method,
        embedding,
        coords=coords,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
        knn=knn,
    )

    oracle_k = _peek_oracle_k(data)
    spatial_used = geo_used != "expression" or method in _SPATIAL_REQUIRED
    if method == "spatial_coherence":
        spatial_used = has_spatial

    if geo_used == "expression" and method in _EXPRESSION_ESTIMATORS:
        flags.append(
            "Expression-only geometry: coordinates were not used for this estimator. "
            "Prefer method='ensemble' or 'spatial_silhouette' for blind spatial domain K."
        )

    return KSelectionResult(
        k=int(best),
        method=method,
        scores={int(k): float(v) for k, v in scores.items()},
        k_range=(int(k_min), int(k_max)),
        n_obs=n_obs,
        n_pcs=n_comp,
        notes=notes,
        oracle_k=oracle_k,
        geometry=geo_used,
        spatial_used=bool(spatial_used and has_spatial),
        flags=tuple(flags),
    )


def _peek_oracle_k(data: SpatialTable) -> int | None:
    """Record oracle K for dual-track reporting only — never used as the estimate."""
    for key in ("domain_truth", "domain", "layer_guess"):
        if key in data.obs.columns:
            return int(data.obs[key].nunique())
    if data.uns.get("n_domains"):
        return int(data.uns["n_domains"])
    return None


def _run_single_estimator(
    method: str,
    embedding: np.ndarray,
    *,
    coords: np.ndarray | None,
    k_min: int,
    k_max: int,
    random_state: int,
    knn: int,
) -> tuple[dict[int, float], int, str]:
    if method == "silhouette":
        scores, best = _select_by_silhouette(embedding, k_min, k_max, random_state)
        return scores, best, "max mean silhouette on embedding+k-means"
    if method == "bic_gmm":
        scores, best = _select_by_bic_gmm(embedding, k_min, k_max, random_state)
        return scores, best, "min BIC of diagonal Gaussian mixture"
    if method == "gap":
        scores, best = _select_by_gap(embedding, k_min, k_max, random_state)
        return scores, best, "gap statistic vs uniform reference (Tibshirani)"
    if method == "calinski_harabasz":
        scores, best = _select_by_calinski_harabasz(embedding, k_min, k_max, random_state)
        return scores, best, "max Calinski-Harabasz variance ratio on embedding"
    if method == "davies_bouldin":
        scores, best = _select_by_davies_bouldin(embedding, k_min, k_max, random_state)
        return scores, best, "min Davies-Bouldin index on embedding (stored as -DB)"
    if method == "spatial_silhouette":
        scores, best = _select_by_silhouette(embedding, k_min, k_max, random_state)
        return scores, best, "max silhouette on spatially-smoothed expression PCA"
    if method == "spatial_coherence":
        if coords is None:
            raise ValueError("spatial_coherence requires coordinates")
        scores, best = _select_by_spatial_coherence(
            embedding, coords, k_min, k_max, random_state, knn=knn
        )
        return (
            scores,
            best,
            "max chance-corrected spatial kNN label kappa of clusters "
            "on spatially-smoothed embedding",
        )
    raise ValueError(f"unknown K estimator: {method!r}")


def _normalize_score_curve(scores: dict[int, float]) -> dict[int, float]:
    """Map a higher-is-better curve onto [0, 1] (constant curves → 0.5)."""
    if not scores:
        return {}
    vals = np.asarray(list(scores.values()), dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {int(k): 0.0 for k in scores}
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo < 1e-12:
        return {int(k): 0.5 for k in scores}
    return {
        int(k): float((float(v) - lo) / (hi - lo)) if math.isfinite(float(v)) else 0.0
        for k, v in scores.items()
    }


def _estimate_ensemble(
    X: np.ndarray,
    coords: np.ndarray | None,
    *,
    k_min: int,
    k_max: int,
    n_pcs: int,
    random_state: int,
    knn: int,
    spatial_weight: float,
    data: SpatialTable,
    flags: list[str],
) -> KSelectionResult:
    has_spatial = coords is not None
    members = list(_ENSEMBLE_WITH_SPATIAL if has_spatial else _ENSEMBLE_EXPRESSION_ONLY)
    # Spatial / BIC members dominate; pure silhouette is heavily down-weighted
    # because on layered Visium it routinely collapses to k=2.
    weights = {
        "silhouette": 0.35,
        "bic_gmm": 1.6,
        "calinski_harabasz": 0.8,
        "davies_bouldin": 0.8,
        "spatial_silhouette": 1.8,
        "spatial_coherence": 2.0,
    }
    if not has_spatial:
        flags.append(
            "ensemble: no spatial coordinates — score-average among expression-only estimators."
        )
    else:
        flags.append(
            "ensemble: weighted average of min-max-normalised score curves over "
            f"{', '.join(members)} (spatial members up-weighted)."
        )

    # Shared embeddings.
    expr_emb, n_comp, _ = _build_embedding(
        X,
        coords,
        geometry="expression",
        n_pcs=n_pcs,
        random_state=random_state,
        knn=knn,
        spatial_weight=spatial_weight,
    )
    smooth_emb = None
    # Stronger spatial blend for spatial members (layered tissues).
    spatial_blend = max(float(spatial_weight), 0.75)
    if has_spatial:
        smooth_emb, _, _ = _build_embedding(
            X,
            coords,
            geometry="spatial_smooth",
            n_pcs=n_pcs,
            random_state=random_state,
            knn=knn,
            spatial_weight=spatial_blend,
        )

    component_votes: dict[str, int] = {}
    component_scores: dict[str, dict[int, float]] = {}
    for member in members:
        if member in {"spatial_silhouette", "spatial_coherence"}:
            assert smooth_emb is not None
            emb = smooth_emb
        else:
            emb = expr_emb
        try:
            scores, best, _notes = _run_single_estimator(
                member,
                emb,
                coords=coords,
                k_min=k_min,
                k_max=k_max,
                random_state=random_state,
                knn=knn,
            )
        except (ValueError, FloatingPointError) as exc:
            flags.append(f"ensemble member {member!r} failed: {exc}")
            continue
        component_votes[member] = int(best)
        component_scores[member] = scores

    if not component_votes:
        raise RuntimeError("ensemble: every member failed")

    all_ks = list(range(k_min, k_max + 1))
    agg: dict[int, float] = {k: 0.0 for k in all_ks}
    weight_sum = 0.0
    for member, scores in component_scores.items():
        normed = _normalize_score_curve(scores)
        w = float(weights.get(member, 1.0))
        weight_sum += w
        for k in all_ks:
            agg[k] += w * float(normed.get(k, 0.0))
    if weight_sum > 0:
        agg = {k: v / weight_sum for k, v in agg.items()}

    # Consensus = argmax of the aggregated curve.
    # Spatial refinement: among the top-3 aggregate K values, prefer the one
    # with highest chance-corrected spatial coherence when available — this
    # breaks expression-only ties that collapse to k=2 on layered tissues.
    ordered_ks = sorted(all_ks, key=lambda kk: agg[kk], reverse=True)
    top = ordered_ks[: min(3, len(ordered_ks))]
    consensus_k = int(top[0])
    if has_spatial and "spatial_coherence" in component_scores and len(top) > 1:
        sc_curve = component_scores["spatial_coherence"]
        consensus_k = int(max(top, key=lambda kk: sc_curve.get(kk, -np.inf)))
        flags.append(
            f"ensemble spatial refinement among top-{len(top)} aggregate K={top} "
            f"→ k={consensus_k} by spatial_coherence."
        )

    vote_counts = dict(Counter(component_votes.values()))
    notes = (
        f"ensemble weighted score-average over {len(component_votes)} estimators; "
        f"consensus k={consensus_k} (member modes={vote_counts})"
    )
    return KSelectionResult(
        k=consensus_k,
        method="ensemble",
        scores={int(k): float(v) for k, v in agg.items()},
        k_range=(int(k_min), int(k_max)),
        n_obs=int(X.shape[0]),
        n_pcs=n_comp,
        notes=notes,
        oracle_k=_peek_oracle_k(data),
        geometry="ensemble",
        spatial_used=has_spatial,
        component_votes=component_votes,
        flags=tuple(flags),
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
    estimator: KEstimator = "ensemble",
    truth_key: str = "domain_truth",
    allow_oracle_k: bool = False,
    random_state: int = 0,
    store: dict[str, KSelectionResult] | None = None,
    geometry: EmbeddingGeometry | None = None,
    knn: int = 6,
    spatial_weight: float = 0.5,
) -> Callable[[SpatialTable], dict[str, Any]]:
    """Build an ``extra_params_factory`` that injects ``n_domains`` under *policy*.

    Parameters
    ----------
    policy
        ``estimate`` (default scientific path), ``oracle`` (requires
        ``allow_oracle_k=True``), ``fixed`` (uses *fixed_k*), or ``dual``
        (estimates K but also records the oracle for ablation reports).
    estimator
        Blind K estimator.  Default ``"ensemble"`` combines expression and
        spatial criteria when coordinates exist.
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
            cache[key] = estimate_n_domains(
                data,
                method=estimator,
                random_state=random_state,
                geometry=geometry,
                knn=knn,
                spatial_weight=spatial_weight,
            )
            if store is not None:
                store[str(key)] = cache[key]
        result = cache[key]
        if policy == "dual":
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
                    geometry=result.geometry,
                    spatial_used=result.spatial_used,
                    component_votes=dict(result.component_votes),
                    flags=result.flags,
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
    estimator: KEstimator = "ensemble",
    truth_key: str = "domain_truth",
    random_state: int = 0,
    geometry: EmbeddingGeometry | None = None,
) -> DualTrackKReport:
    """Report oracle vs estimated K for one dataset (no method runs)."""
    selection = estimate_n_domains(
        data,
        method=estimator,
        random_state=random_state,
        geometry=geometry,
    )
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
    n = X.shape[0]
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
    best_score = -np.inf
    for k in range(k_min, k_max + 1):
        bic = _diagonal_gmm_bic(embedding, k, random_state)
        score = -float(bic)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return scores, best_k


def _diagonal_gmm_bic(X: np.ndarray, k: int, random_state: int) -> float:
    """Hard-assignment diagonal GMM BIC after k-means init.

    Empty / singleton clusters are reseeded from the largest cluster rather than
    returning a catastrophic BIC that collapses the ensemble score scale.
    """
    n, d = X.shape
    labels = kmeans(X, k, random_state=random_state)
    labels = _reseed_tiny_clusters(X, labels, k, random_state=random_state)
    log_lik = 0.0
    n_params = k * (d + d) + (k - 1)
    for c in range(k):
        members = X[labels == c]
        n_c = members.shape[0]
        if n_c < 2:
            # Still degenerate after reseed — mild finite penalty, not 1e18.
            return float(1e6 + 1e3 * k)
        mean = members.mean(axis=0)
        var = members.var(axis=0) + 1e-6
        const = -0.5 * (d * np.log(2 * np.pi) + np.log(var).sum())
        quad = -0.5 * (((members - mean) ** 2) / var).sum(axis=1)
        log_lik += float((const + quad + np.log(n_c / n)).sum())
    bic = -2.0 * log_lik + n_params * np.log(n)
    return float(bic)


def _reseed_tiny_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Move singleton/empty clusters to farthest points of the largest cluster."""
    labels = np.asarray(labels, dtype=int).copy()
    rng = np.random.default_rng(random_state + 17)
    for _ in range(k):  # at most k repair passes
        counts = np.bincount(labels, minlength=k)
        tiny = np.where(counts < 2)[0]
        if tiny.size == 0:
            break
        large = int(np.argmax(counts))
        large_idx = np.flatnonzero(labels == large)
        if large_idx.size < 3:
            break
        center = X[large_idx].mean(axis=0)
        dist = ((X[large_idx] - center) ** 2).sum(axis=1)
        order = np.argsort(-dist)
        for t, pos in zip(tiny, order, strict=False):
            labels[large_idx[int(pos)]] = int(t)
    # If still empty, random reassignment of distinct points.
    counts = np.bincount(labels, minlength=k)
    for t in np.where(counts < 1)[0]:
        donor = int(rng.integers(0, X.shape[0]))
        labels[donor] = int(t)
    return labels


def _select_by_gap(
    embedding: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    n_refs: int = 5,
) -> tuple[dict[int, float], int]:
    """Gap statistic; pick smallest k with gap(k) >= gap(k+1)-s(k+1)."""
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


def _select_by_calinski_harabasz(
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
        score = float(_calinski_harabasz(embedding, labels))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return scores, best_k


def _calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = X.shape[0]
    k = int(unique.size)
    if k < 2 or k >= n:
        return -1.0
    overall = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for lab in unique:
        members = X[labels == lab]
        n_c = members.shape[0]
        if n_c == 0:
            continue
        center = members.mean(axis=0)
        between += n_c * float(((center - overall) ** 2).sum())
        within += float(((members - center) ** 2).sum())
    if within <= 0.0:
        return 1e6 if between > 0 else 0.0
    return float((between / (k - 1)) / (within / (n - k)))


def _select_by_davies_bouldin(
    embedding: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[dict[int, float], int]:
    """Lower DB is better; store ``-DB`` so higher scores remain preferred."""
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -np.inf
    for k in range(k_min, k_max + 1):
        labels = kmeans(embedding, k, random_state=random_state)
        db = float(_davies_bouldin(embedding, labels))
        score = -db
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return scores, best_k


def _davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    k = int(unique.size)
    if k < 2:
        return 1e6
    centers = []
    scatters = []
    for lab in unique:
        members = X[labels == lab]
        if members.shape[0] == 0:
            centers.append(np.zeros(X.shape[1]))
            scatters.append(0.0)
            continue
        center = members.mean(axis=0)
        centers.append(center)
        scatters.append(float(np.linalg.norm(members - center, axis=1).mean()))
    centers_arr = np.asarray(centers, dtype=float)
    db_sum = 0.0
    for i in range(k):
        worst = 0.0
        for j in range(k):
            if i == j:
                continue
            sep = float(np.linalg.norm(centers_arr[i] - centers_arr[j]))
            if sep <= 0.0:
                ratio = 1e6
            else:
                ratio = (scatters[i] + scatters[j]) / sep
            worst = max(worst, ratio)
        db_sum += worst
    return float(db_sum / k)


def _select_by_spatial_coherence(
    embedding: np.ndarray,
    coords: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    *,
    knn: int = 6,
) -> tuple[dict[int, float], int]:
    """Maximise chance-corrected spatial kNN label agreement of clusters.

    Raw neighbour-agreement is maximised by trivial coarse partitions (k=2).
    We therefore score Cohen-style excess agreement::

        kappa = (p_obs - p_exp) / (1 - p_exp)

    where ``p_exp = sum_c freq_c^2`` is the chance that two independent draws
    share a label under the empirical cluster sizes.  Clusters are formed on
    the expression embedding; the score uses only coordinates + labels.
    """
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -np.inf
    n_obs = int(coords.shape[0])
    knn_use = int(min(max(knn, 2), n_obs))
    neighbours = knn_indices(coords, knn_use)
    if neighbours.shape[1] > 0:
        if np.all(neighbours[:, 0] == np.arange(n_obs)):
            if neighbours.shape[1] == 1:
                raise ValueError("spatial_coherence needs knn>=2 (self only)")
            neighbours = neighbours[:, 1:]

    for k in range(k_min, k_max + 1):
        labels = kmeans(embedding, k, random_state=random_state)
        score = float(_spatial_label_kappa(labels, neighbours))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return scores, best_k


def _spatial_label_agreement(labels: np.ndarray, neighbours: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = labels.shape[0]
    if n == 0 or neighbours.size == 0:
        return 0.0
    nbr_labels = labels[neighbours]
    same = nbr_labels == labels[:, None]
    return float(same.mean())


def _spatial_label_kappa(labels: np.ndarray, neighbours: np.ndarray) -> float:
    """Chance-corrected spatial neighbour label agreement (Cohen-style)."""
    labels = np.asarray(labels)
    n = labels.shape[0]
    if n == 0 or neighbours.size == 0:
        return 0.0
    p_obs = _spatial_label_agreement(labels, neighbours)
    _, counts = np.unique(labels, return_counts=True)
    freqs = counts.astype(float) / float(n)
    p_exp = float(np.sum(freqs * freqs))
    if p_exp >= 1.0 - 1e-12:
        return 0.0
    return float((p_obs - p_exp) / (1.0 - p_exp))
