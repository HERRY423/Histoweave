"""Digital-twin synthetic datasets with feature-matched, planted ground truth.

A *digital twin* is a synthetic SpatialTable that preserves the target-free
statistical fingerprint of a real (often unlabelled) sample — sparsity, library
size distribution, Moran's I, Hopkins tendency, effective rank, and the rest of
:data:`~histoweave.benchmark.features.RECOMMENDATION_FEATURE_ORDER` — while
carrying **known planted labels** for method benchmarking.

Workflow
--------
1. Extract the target-free feature vector from the real dataset.
2. Plant spatially coherent domain labels on (subsampled) real coordinates.
3. Generate domain-conditioned count expression and calibrate it so the twin
   matches the real feature vector in L2 distance on a z-scored feature space.
4. Return the twin together with a match-quality report.

The twin is then used by :mod:`histoweave.benchmark.digital_twin` to rank methods
with ARI against planted truth and transfer that ranking as a prediction for the
real sample (which has no ground truth).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data import Provenance, SpatialTable

# Canonical 13-dimension twin-matching schema (subset of the full target-free
# recommendation vector).  Size-like quantities (n_obs, n_vars, aspect_ratio)
# are matched by construction rather than optimised over.
# NOTE: Do not import histoweave.benchmark at module level — that package
# re-exports digital-twin validation and would create a circular import.
TWIN_MATCH_FEATURES: tuple[str, ...] = (
    "sparsity",
    "mean_nonzero",
    "library_mean",
    "library_cv",
    "expression_entropy",
    "spatial_autocorrelation",
    "cluster_tendency",
    "mean_nn_distance",
    "spatial_density_cv",
    "spatial_entropy",
    "effective_rank_90",
    "effective_rank_95",
    "sv_entropy",
)

DIGITAL_TWIN_SCHEMA_VERSION = 1


@dataclass
class FeatureMatchReport:
    """How well the twin matches the real sample on each matched dimension."""

    feature_order: list[str]
    real_features: dict[str, float]
    twin_features: dict[str, float]
    absolute_errors: dict[str, float]
    relative_errors: dict[str, float]
    match_l2: float
    match_cosine: float
    n_trials: int
    best_trial: int
    generator_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Digital-twin match: L2={self.match_l2:.4f}  "
            f"cosine={self.match_cosine:.4f}  "
            f"(trial {self.best_trial}/{self.n_trials})",
        ]
        for key in self.feature_order:
            r = self.real_features.get(key, float("nan"))
            t = self.twin_features.get(key, float("nan"))
            e = self.absolute_errors.get(key, float("nan"))
            lines.append(f"  {key:<28} real={r:>10.4g}  twin={t:>10.4g}  |Δ|={e:>8.4g}")
        return "\n".join(lines)


@dataclass
class DigitalTwinResult:
    """Output of :func:`make_digital_twin`."""

    twin: SpatialTable
    real_features: dict[str, float]
    twin_features: dict[str, float]
    match: FeatureMatchReport
    planted_truth_key: str = "domain_truth"
    schema_version: int = DIGITAL_TWIN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planted_truth_key": self.planted_truth_key,
            "n_obs": int(self.twin.n_obs),
            "n_vars": int(self.twin.n_vars),
            "n_domains": int(self.twin.uns.get("n_domains", 0)),
            "real_features": _json_floats(self.real_features),
            "twin_features": _json_floats(self.twin_features),
            "match": match_report_json_safe(self.match),
        }


def make_digital_twin(
    data: SpatialTable,
    *,
    seed: int = 0,
    n_domains: int | None = None,
    max_cells: int | None = 1500,
    max_genes: int | None = 500,
    n_trials: int = 16,
    match_features: list[str] | tuple[str, ...] | None = None,
    preserve_coordinates: bool = True,
) -> DigitalTwinResult:
    """Build a feature-matched synthetic twin of ``data`` with planted labels.

    Parameters
    ----------
    data
        Real spatial dataset, typically **without** ground-truth labels.
    seed
        Master RNG seed for deterministic twin generation.
    n_domains
        Number of planted domains.  When *None*, a heuristic estimates 3–8
        domains from sample size and Hopkins cluster tendency.
    max_cells, max_genes
        Caps for large real datasets (keeps twin generation CI-friendly).
        Pass ``None`` to use the full real dimensions.
    n_trials
        Random-search budget over generator knobs (noise, lift, domain count
        jitter, calibration strength).
    match_features
        Feature names to match (default: the 13-dim :data:`TWIN_MATCH_FEATURES`).
    preserve_coordinates
        When *True* (default), reuse (subsampled) real spatial coordinates so
        spatial geometry and Moran's I are inherited rather than re-simulated.

    Returns
    -------
    DigitalTwinResult
        Twin table (``obs['domain_truth']`` planted) plus match report.
    """
    # Lazy import avoids histoweave.benchmark ↔ datasets.digital_twin cycles.
    from ..benchmark.features import (
        RECOMMENDATION_FEATURE_ORDER,
        extract_features,
        feature_vector,
    )

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if data.n_obs < 10:
        raise ValueError("digital twin requires at least 10 observations")
    if data.n_vars < 5:
        raise ValueError("digital twin requires at least 5 genes")

    features_to_match = list(match_features or TWIN_MATCH_FEATURES)
    for name in features_to_match:
        if name not in RECOMMENDATION_FEATURE_ORDER:
            raise ValueError(
                f"unknown match feature {name!r}; "
                f"expected one of {list(RECOMMENDATION_FEATURE_ORDER)}"
            )

    real_feats = extract_features(data, include_domain=False)
    real_vec = feature_vector(real_feats, order=features_to_match)

    n_cells = data.n_obs if max_cells is None else min(data.n_obs, int(max_cells))
    n_genes = data.n_vars if max_genes is None else min(data.n_vars, int(max_genes))
    n_cells = max(n_cells, 10)
    n_genes = max(n_genes, 5)

    rng = np.random.default_rng(seed)
    coords_pool = _coords_or_fallback(data, rng)
    obs_idx = rng.choice(data.n_obs, size=n_cells, replace=False)
    obs_idx.sort()
    coords = np.asarray(coords_pool[obs_idx], dtype=float)

    # Gene subset: prefer highest-variance genes when the matrix is large.
    gene_idx = _select_genes(data, n_genes, rng)
    target_sparsity = float(real_feats.get("sparsity", 0.7))
    target_lib_mean = float(real_feats.get("library_mean", 500.0))
    target_lib_cv = float(real_feats.get("library_cv", 0.5))
    target_mean_nz = float(real_feats.get("mean_nonzero", 2.0))

    base_k = n_domains if n_domains is not None else _estimate_n_domains(real_feats, n_cells)
    base_k = int(np.clip(base_k, 2, max(2, n_cells // 10)))

    best: tuple[float, int, SpatialTable, dict[str, Any]] | None = None
    # Feature-space standardisation anchors (use real vector + unit scale fallback).
    scale = np.maximum(np.abs(real_vec), 1e-3)

    for trial in range(n_trials):
        trial_rng = np.random.default_rng(seed + 17 * (trial + 1))
        k = int(np.clip(base_k + trial_rng.integers(-1, 2), 2, max(2, n_cells // 10)))
        noise = float(trial_rng.uniform(0.12, 0.55))
        lift = float(trial_rng.uniform(3.0, 12.0))
        markers_per = int(trial_rng.integers(3, max(4, min(8, n_genes // k))))
        twin = _generate_twin_candidate(
            coords=coords,
            n_genes=n_genes,
            n_domains=k,
            noise=noise,
            marker_gene_lift=lift,
            markers_per_domain=markers_per,
            target_sparsity=target_sparsity,
            target_lib_mean=target_lib_mean,
            target_lib_cv=target_lib_cv,
            target_mean_nz=target_mean_nz,
            preserve_coordinates=preserve_coordinates,
            seed=int(trial_rng.integers(0, 2**31 - 1)),
            gene_names=[str(data.var_names[i]) for i in gene_idx]
            if hasattr(data, "var_names")
            else None,
        )
        twin_feats = extract_features(twin, include_domain=False)
        twin_vec = feature_vector(twin_feats, order=features_to_match)
        dist = float(np.linalg.norm((twin_vec - real_vec) / scale))
        params = {
            "n_domains": k,
            "noise": noise,
            "marker_gene_lift": lift,
            "markers_per_domain": markers_per,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "preserve_coordinates": preserve_coordinates,
        }
        if best is None or dist < best[0]:
            best = (dist, trial, twin, params)

    assert best is not None
    _, best_trial, best_twin, best_params = best
    twin_feats = extract_features(best_twin, include_domain=False)
    twin_vec = feature_vector(twin_feats, order=features_to_match)
    match = _build_match_report(
        features_to_match,
        real_feats,
        twin_feats,
        real_vec,
        twin_vec,
        n_trials=n_trials,
        best_trial=best_trial,
        generator_params=best_params,
    )

    # Stamp twin provenance and real-feature snapshot for auditability.
    best_twin.uns["digital_twin"] = {
        "schema_version": DIGITAL_TWIN_SCHEMA_VERSION,
        "source_n_obs": int(data.n_obs),
        "source_n_vars": int(data.n_vars),
        "source_assay": str(data.uns.get("assay", data.uns.get("platform", "unknown"))),
        "match_l2": match.match_l2,
        "match_cosine": match.match_cosine,
        "match_features": features_to_match,
        "generator_params": best_params,
        "seed": seed,
    }
    best_twin.record(
        Provenance(
            step="ingestion",
            method="make_digital_twin",
            method_version=f"0.{DIGITAL_TWIN_SCHEMA_VERSION}.0",
            params={"seed": seed, "n_trials": n_trials, **best_params},
        )
    )
    return DigitalTwinResult(
        twin=best_twin,
        real_features=real_feats,
        twin_features=twin_feats,
        match=match,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _json_floats(mapping: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in mapping.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            out[key] = None
            continue
        out[key] = v if np.isfinite(v) else None
    return out


def match_report_json_safe(report: FeatureMatchReport) -> dict[str, Any]:
    """JSON-serialisable match report (NaNs → null)."""
    payload = report.to_dict()
    for key in ("real_features", "twin_features", "absolute_errors", "relative_errors"):
        payload[key] = _json_floats(payload.get(key) or {})
    for key in ("match_l2", "match_cosine"):
        try:
            v = float(payload[key])
            payload[key] = v if np.isfinite(v) else None
        except (TypeError, ValueError, KeyError):
            payload[key] = None
    return payload


def _coords_or_fallback(data: SpatialTable, rng: np.random.Generator) -> np.ndarray:
    coords = data.spatial
    if coords is not None and len(coords) == data.n_obs:
        return np.asarray(coords, dtype=float)
    # Unit square fallback when the real table lacks spatial coords.
    return rng.uniform(0.0, 100.0, size=(data.n_obs, 2))


def _select_genes(data: SpatialTable, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    n_genes = min(n_genes, data.n_vars)
    if n_genes >= data.n_vars:
        return np.arange(data.n_vars)
    X = data.X
    try:
        if hasattr(X, "tocsc"):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            var = np.maximum(mean_sq - mean * mean, 0.0)
        else:
            var = np.asarray(X, dtype=float).var(axis=0)
        order = np.argsort(var)[::-1]
        return np.sort(order[:n_genes])
    except Exception:
        return np.sort(rng.choice(data.n_vars, size=n_genes, replace=False))


def _estimate_n_domains(features: dict[str, float], n_cells: int) -> int:
    """Heuristic domain count from Hopkins tendency and sample size."""
    hopkins = features.get("cluster_tendency", float("nan"))
    # Hopkins near 0.5 → random; lower → more clustered (our implementation).
    if np.isfinite(hopkins):
        if hopkins < 0.35:
            base = 6
        elif hopkins < 0.45:
            base = 5
        elif hopkins < 0.55:
            base = 4
        else:
            base = 3
    else:
        base = 4
    # Cap by sample size so ARI remains informative.
    return int(np.clip(base, 2, max(2, min(8, n_cells // 25))))


def _plant_domain_labels(
    coords: np.ndarray,
    n_domains: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spatially coherent Voronoi labels from random centroids."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    centroids = lo + rng.uniform(0.15, 0.85, size=(n_domains, coords.shape[1])) * span
    dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1).astype(int)


def _generate_twin_candidate(
    *,
    coords: np.ndarray,
    n_genes: int,
    n_domains: int,
    noise: float,
    marker_gene_lift: float,
    markers_per_domain: int,
    target_sparsity: float,
    target_lib_mean: float,
    target_lib_cv: float,
    target_mean_nz: float,
    preserve_coordinates: bool,
    seed: int,
    gene_names: list[str] | None,
) -> SpatialTable:
    rng = np.random.default_rng(seed)
    n_cells = coords.shape[0]
    domain = _plant_domain_labels(coords, n_domains, rng)

    if not preserve_coordinates:
        # Mild isotropic jitter only — still based on real layout scale.
        scale = float(np.median(np.std(coords, axis=0))) or 1.0
        coords = coords + rng.normal(0.0, 0.02 * scale, size=coords.shape)

    n_markers = min(n_domains * markers_per_domain, n_genes)
    if gene_names is None or len(gene_names) != n_genes:
        gene_names = [f"twin_gene_{i:04d}" for i in range(n_genes)]
    else:
        gene_names = list(gene_names)

    base_rate = rng.uniform(0.4, 1.6, size=n_genes)
    X = np.asarray(rng.poisson(lam=np.broadcast_to(base_rate, (n_cells, n_genes))), dtype=float)

    marker_genes: dict[str, list[str]] = {}
    for d in range(n_domains):
        start = d * markers_per_domain
        stop = min(start + markers_per_domain, n_markers)
        block = list(range(start, stop))
        if not block:
            break
        marker_genes[f"domain_{d}"] = [gene_names[i] for i in block]
        in_domain = domain == d
        lift = np.full(len(block), marker_gene_lift)
        X[np.ix_(in_domain, block)] += rng.poisson(
            lam=np.broadcast_to(lift, (int(in_domain.sum()), len(block)))
        )

    X *= rng.lognormal(mean=0.0, sigma=noise, size=X.shape)
    X = np.rint(np.clip(X, 0.0, None)).astype(float)
    X = _calibrate_expression(
        X,
        target_sparsity=target_sparsity,
        target_lib_mean=target_lib_mean,
        target_lib_cv=target_lib_cv,
        target_mean_nz=target_mean_nz,
        rng=rng,
    )

    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical([f"domain_{d}" for d in domain])},
        index=[f"twin_cell_{i:05d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    return SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": np.asarray(coords, dtype=float)},
        uns={
            "marker_genes": marker_genes,
            "n_domains": int(n_domains),
            "assay": "digital_twin",
            "platform": "synthetic_twin",
        },
    )


def _calibrate_expression(
    X: np.ndarray,
    *,
    target_sparsity: float,
    target_lib_mean: float,
    target_lib_cv: float,
    target_mean_nz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Post-process counts toward target sparsity / library-size statistics."""
    X = np.asarray(X, dtype=float)
    n_cells, n_genes = X.shape
    n_total = max(n_cells * n_genes, 1)

    # 1. Library-size: scale rows so mean ≈ target, then re-impose CV.
    lib = X.sum(axis=1)
    lib_mean = float(lib.mean()) + 1e-8
    scale = (target_lib_mean / lib_mean) if target_lib_mean > 0 else 1.0
    X = X * scale
    lib = X.sum(axis=1)
    # Stretch library sizes toward target CV via log-normal multipliers.
    current_cv = float(lib.std() / (lib.mean() + 1e-8))
    if target_lib_cv > 0 and current_cv > 1e-8:
        # Multiplicative residual to push CV.
        sigma = float(np.clip(target_lib_cv / max(current_cv, 0.1) * 0.25, 0.05, 0.8))
        mult = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n_cells)
        # Renormalise so mean library is preserved.
        mult *= (n_cells / mult.sum()) * (target_lib_mean / (float((lib * mult).mean()) + 1e-8))
        X = X * mult[:, None]

    X = np.rint(np.clip(X, 0.0, None)).astype(float)

    # 2. Sparsity: randomly zero entries until close to target.
    target_sparsity = float(np.clip(target_sparsity, 0.0, 0.995))
    nonzero = X > 0
    current_sparsity = 1.0 - float(nonzero.sum()) / n_total
    if current_sparsity < target_sparsity:
        # Need more zeros: zero out a fraction of current non-zeros.
        nz_idx = np.argwhere(nonzero)
        n_drop = int((target_sparsity - current_sparsity) * n_total)
        n_drop = min(n_drop, max(0, len(nz_idx) - n_cells))  # keep ≥1 count/cell when possible
        if n_drop > 0:
            pick = rng.choice(len(nz_idx), size=n_drop, replace=False)
            for i, j in nz_idx[pick]:
                X[i, j] = 0.0
    elif current_sparsity > target_sparsity + 0.02:
        # Need more non-zeros: fill some zeros with small positive counts.
        z_idx = np.argwhere(~nonzero)
        n_fill = int((current_sparsity - target_sparsity) * n_total)
        n_fill = min(n_fill, len(z_idx))
        if n_fill > 0:
            fill_val = max(1.0, target_mean_nz)
            pick = rng.choice(len(z_idx), size=n_fill, replace=False)
            for i, j in z_idx[pick]:
                X[i, j] = fill_val

    # 3. Soft-adjust mean non-zero magnitude.
    nz = X[X > 0]
    if nz.size and target_mean_nz > 0:
        factor = target_mean_nz / (float(nz.mean()) + 1e-8)
        factor = float(np.clip(factor, 0.25, 4.0))
        X = np.where(X > 0, np.rint(X * factor), 0.0)

    # Ensure no empty cells (library size ≥ 1).
    lib = X.sum(axis=1)
    empty = np.where(lib <= 0)[0]
    if len(empty):
        gene = int(rng.integers(0, n_genes))
        X[empty, gene] = max(1.0, target_mean_nz)

    return X.astype(float)


def _build_match_report(
    feature_order: list[str],
    real_feats: dict[str, float],
    twin_feats: dict[str, float],
    real_vec: np.ndarray,
    twin_vec: np.ndarray,
    *,
    n_trials: int,
    best_trial: int,
    generator_params: dict[str, Any],
) -> FeatureMatchReport:
    abs_err = {
        k: float(abs(twin_feats.get(k, float("nan")) - real_feats.get(k, float("nan"))))
        for k in feature_order
    }
    rel_err: dict[str, float] = {}
    for k in feature_order:
        r = float(real_feats.get(k, float("nan")))
        t = float(twin_feats.get(k, float("nan")))
        denom = abs(r) if abs(r) > 1e-8 else 1.0
        if np.isfinite(t) and np.isfinite(r):
            rel_err[k] = float(abs(t - r) / denom)
        else:
            rel_err[k] = float("nan")

    scale = np.maximum(np.abs(real_vec), 1e-3)
    match_l2 = float(np.linalg.norm((twin_vec - real_vec) / scale))
    # Cosine similarity in the same scaled space (map to [0, 1]).
    a = real_vec / scale
    b = twin_vec / scale
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na > 0 and nb > 0:
        cos = float(np.dot(a, b) / (na * nb))
        match_cosine = float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
    else:
        match_cosine = 0.0

    return FeatureMatchReport(
        feature_order=list(feature_order),
        real_features={k: float(real_feats.get(k, float("nan"))) for k in feature_order},
        twin_features={k: float(twin_feats.get(k, float("nan"))) for k in feature_order},
        absolute_errors=abs_err,
        relative_errors=rel_err,
        match_l2=match_l2,
        match_cosine=match_cosine,
        n_trials=n_trials,
        best_trial=best_trial,
        generator_params=dict(generator_params),
    )
