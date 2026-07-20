"""Donor/study-level independent personalisation and cross-lab statistics.

Paper-level endpoints that go beyond slice-level LOO:

1. **Independent query units** — one held-out *study* (biological donor,
   external study, or synthetic laboratory), never multiple slices from the
   same donor counted as independent.
2. **Personalisation policies** — unconstrained k-NN vs a *gated* policy that
   falls back to the global-best comparator (non-inferior by construction when
   the gate rejects).
3. **Cross-lab reproducibility** — study-bootstrap CIs on regret, lab-stratum
   breakdown, rank concordance across independent units.

Target: ≥15–20 independent queries with personalisation that is non-inferior
(or superior) to always-pick-global-best under a predeclared margin.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from .features import RECOMMENDATION_FEATURE_ORDER
from .landscape import LandscapeResult, _compute_niches, _embed_datasets, run_task_landscape
from .protocol_endpoints import subset_landscape, training_means
from .recommend import MethodRecommender
from .stats_review import review_landscape
from .task_contract import AnalysisTask, GroundTruthKind, classify_platform

PROTOCOL = "histoweave.independent_personalisation.v1"
IndependenceClass = Literal[
    "biological_donor",
    "external_study",
    "cross_platform_study",
    "synthetic_lab",
]

DEFAULT_METHODS = (
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "gaussian_mixture",
    "kmeans",
    "minibatch_kmeans",
    "spectral",
)

# Predeclared non-inferiority margin on mean ARI regret (absolute).
DEFAULT_NONINFERIOR_MARGIN = 0.01


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Convert NumPy / NaN-bearing structures into JSON-safe Python objects."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Study-unit aggregation
# ---------------------------------------------------------------------------


@dataclass
class IndependentStudyUnit:
    """One independent personalisation query (donor / study / lab)."""

    unit_id: str
    independence_class: IndependenceClass
    member_datasets: list[str]
    platform: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_units_to_landscape(
    source: LandscapeResult,
    units: list[IndependentStudyUnit],
    *,
    methods: list[str] | None = None,
) -> LandscapeResult:
    """Collapse member datasets into one performance/feature row per unit.

    Performance and timings are unweighted means over finite member scores.
    Feature vectors are component-wise means (NaNs ignored per axis).
    """
    method_names = list(methods or source.method_order())
    performance: dict[str, dict[str, float]] = {}
    features: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float | None]] = {}
    meta: dict[str, dict[str, Any]] = {}
    order = list(source.feature_order or RECOMMENDATION_FEATURE_ORDER)

    for unit in units:
        members = [m for m in unit.member_datasets if m in source.performance]
        if not members:
            continue
        row: dict[str, float] = {}
        time_row: dict[str, float | None] = {}
        for method in method_names:
            scores = [
                float(source.performance[m][method])
                for m in members
                if method in source.performance[m]
                and source.performance[m][method] is not None
                and np.isfinite(source.performance[m][method])
            ]
            row[method] = float(np.mean(scores)) if scores else float("nan")
            tvals = [
                float(t_val)
                for m in members
                if m in source.timings
                and method in source.timings[m]
                and (t_val := source.timings[m][method]) is not None
                and np.isfinite(t_val)
            ]
            time_row[method] = float(np.mean(tvals)) if tvals else None
        performance[unit.unit_id] = row
        timings[unit.unit_id] = time_row

        vectors = []
        for member in members:
            if member not in source.features:
                continue
            vec = np.asarray(source.features[member], dtype=float).ravel()
            if vec.size != len(order):
                aligned = np.full(len(order), np.nan, dtype=float)
                n = min(vec.size, aligned.size)
                aligned[:n] = vec[:n]
                vectors.append(aligned)
            else:
                vectors.append(vec)
        if vectors:
            stack = np.vstack(vectors)
            reduced = np.full(stack.shape[1], np.nan, dtype=float)
            for col in range(stack.shape[1]):
                finite = stack[:, col][np.isfinite(stack[:, col])]
                if finite.size:
                    reduced[col] = float(np.mean(finite))
            features[unit.unit_id] = reduced
        else:
            features[unit.unit_id] = np.full(len(order), np.nan, dtype=float)

        meta[unit.unit_id] = {
            "platform": unit.platform,
            "task": AnalysisTask.SPATIAL_DOMAIN.value,
            "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
            "study_group": unit.unit_id,
            "independence_class": unit.independence_class,
            "member_datasets": members,
            "notes": unit.notes,
        }

    if len(performance) < 2:
        raise ValueError("aggregate_units_to_landscape needs ≥2 units with data")

    embedding = _embed_datasets(features)
    best_method, niches = _compute_niches(performance)
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings,
        feature_order=order,
        method_count=len(method_names),
        dataset_count=len(performance),
        task=AnalysisTask.SPATIAL_DOMAIN.value,
        metric=source.metric or "ARI",
        higher_is_better=source.higher_is_better,
        dataset_meta=meta,
    )


def default_independent_units_from_multisource(
    dataset_names: list[str],
) -> list[IndependentStudyUnit]:
    """Map known multisource keys into independent donor/study units."""
    dlpfc_donors = {
        "Br5292": ["151507", "151508", "151509", "151510"],
        "Br5595": ["151669", "151670", "151671", "151672"],
        "Br8100": ["151673", "151674", "151675", "151676"],
    }
    available = set(str(n) for n in dataset_names)
    units: list[IndependentStudyUnit] = []

    for donor, slices in dlpfc_donors.items():
        members = [s for s in slices if s in available]
        if members:
            units.append(
                IndependentStudyUnit(
                    unit_id=f"dlpfc_donor_{donor}",
                    independence_class="biological_donor",
                    member_datasets=members,
                    platform="visium",
                    notes="Maynard 2021 DLPFC; slices collapsed to donor unit",
                )
            )

    external = {
        "visium_hd_crc": "visium",
        "xenium_lung_cancer": "xenium",
        "xenium_ovarian_cancer": "xenium",
        "visium_mouse_brain": "visium",
        "allen_merfish_brain_section": "merfish",
    }
    for name, platform in external.items():
        if name in available:
            units.append(
                IndependentStudyUnit(
                    unit_id=name,
                    independence_class="external_study",
                    member_datasets=[name],
                    platform=platform,
                    notes="External multi-platform validation study",
                )
            )

    platform_studies = {
        "merfish": "merfish",
        "slideseqv2": "slideseqv2",
        "xenium": "xenium",
    }
    for name, platform in platform_studies.items():
        if name in available:
            units.append(
                IndependentStudyUnit(
                    unit_id=f"platform_{name}",
                    independence_class="cross_platform_study",
                    member_datasets=[name],
                    platform=platform,
                    notes="Cross-platform 7x15 study unit",
                )
            )
    return units


def synthetic_lab_units(
    *,
    seed: int = 42,
    methods: list[str] | None = None,
) -> tuple[LandscapeResult, list[IndependentStudyUnit]]:
    """Build one independent synthetic laboratory per benchmark-suite preset."""
    from ..datasets.synthetic import make_benchmark_suite
    from ..plugins import MethodCategory

    method_names = list(methods or DEFAULT_METHODS)
    suite = make_benchmark_suite(seed=seed)
    # Fixed oracle K per dataset from planted domains (synthetic controlled panel).
    n_domains_map: dict[str, int] = {}
    for name, table in suite.datasets.items():
        if "domain_truth" in table.obs:
            n_domains_map[name] = int(table.obs["domain_truth"].nunique())
        else:
            n_domains_map[name] = 3

    def factory(data) -> dict[str, Any]:
        # Recover name from uns if present; else fall back to unique count.
        sid = str(data.uns.get("benchmark_preset") or data.uns.get("slice_id") or "")
        k = n_domains_map.get(sid)
        if k is None and "domain_truth" in data.obs:
            k = int(data.obs["domain_truth"].nunique())
        return {"n_domains": int(k or 3), "random_state": seed}

    # Tag presets for factory lookup.
    datasets = {}
    for name, table in suite.datasets.items():
        tagged = table.copy() if hasattr(table, "copy") else table
        tagged.uns = dict(getattr(tagged, "uns", {}) or {})
        tagged.uns["benchmark_preset"] = name
        tagged.uns["platform"] = "synthetic"
        tagged.uns["study_group"] = f"synth_lab_{name}"
        datasets[name] = tagged
        n_domains_map[name] = n_domains_map.get(name, 3)

    landscape = run_task_landscape(
        datasets,
        category=MethodCategory.DOMAIN_DETECTION,
        methods=method_names,
        extra_params_factory=factory,
    )
    # Attach target-free features explicitly (landscape already has them).
    units = [
        IndependentStudyUnit(
            unit_id=f"synth_lab_{name}",
            independence_class="synthetic_lab",
            member_datasets=[name],
            platform="synthetic",
            notes=f"Independent synthetic laboratory preset={name}",
        )
        for name in datasets
    ]
    # Rename landscape keys to unit ids for a flat merge.
    renamed = LandscapeResult(
        performance={f"synth_lab_{k}": v for k, v in landscape.performance.items()},
        features={f"synth_lab_{k}": v for k, v in landscape.features.items()},
        embedding={f"synth_lab_{k}": xy for k, xy in landscape.embedding.items()},
        best_method={f"synth_lab_{k}": m for k, m in landscape.best_method.items()},
        niches={
            method: [f"synth_lab_{n}" if not str(n).startswith("synth_lab_") else n for n in members]
            for method, members in landscape.niches.items()
        },
        timings={f"synth_lab_{k}": v for k, v in landscape.timings.items()},
        feature_order=list(landscape.feature_order),
        method_count=landscape.method_count,
        dataset_count=len(datasets),
        task=AnalysisTask.SPATIAL_DOMAIN.value,
        metric="ARI",
        higher_is_better=True,
        dataset_meta={
            f"synth_lab_{name}": {
                "platform": "synthetic",
                "task": AnalysisTask.SPATIAL_DOMAIN.value,
                "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
                "study_group": f"synth_lab_{name}",
                "independence_class": "synthetic_lab",
                "member_datasets": [name],
            }
            for name in datasets
        },
    )
    return renamed, units


def merge_unit_landscapes(*landscapes: LandscapeResult) -> LandscapeResult:
    """Merge unit-level landscapes (disjoint unit ids)."""
    if not landscapes:
        raise ValueError("merge_unit_landscapes requires ≥1 landscape")
    performance: dict[str, dict[str, float]] = {}
    features: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float | None]] = {}
    meta: dict[str, dict[str, Any]] = {}
    order = list(landscapes[0].feature_order or RECOMMENDATION_FEATURE_ORDER)
    for land in landscapes:
        for name, row in land.performance.items():
            if name in performance:
                raise ValueError(f"duplicate independent unit id: {name!r}")
            performance[name] = dict(row)
            features[name] = np.asarray(
                land.features.get(name, np.full(len(order), np.nan)), dtype=float
            )
            timings[name] = dict(land.timings.get(name, {}))
            meta[name] = dict(land.dataset_meta.get(name, {}))
    embedding = _embed_datasets(features)
    best_method, niches = _compute_niches(performance)
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings,
        feature_order=order,
        method_count=len({m for row in performance.values() for m in row}),
        dataset_count=len(performance),
        task=AnalysisTask.SPATIAL_DOMAIN.value,
        metric="ARI",
        higher_is_better=True,
        dataset_meta=meta,
    )


# ---------------------------------------------------------------------------
# Gated personalisation (non-inferiority-oriented)
# ---------------------------------------------------------------------------


@dataclass
class PolicyQueryResult:
    held_out: str
    independence_class: str | None
    oracle_score: float
    global_best_method: str | None
    global_best_regret: float | None
    knn_method: str | None
    knn_regret: float | None
    knn_confidence: float | None
    knn_beats_global_proxy: bool
    gated_method: str | None
    gated_regret: float | None
    gated_action: str  # personalised | global_default
    random_regret: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_personalisation_policies(
    landscape: LandscapeResult,
    *,
    methods: list[str] | None = None,
    k_neighbours: int = 3,
    min_training: int = 4,
    confidence_floor: float = 0.20,
    proxy_advantage: float = 0.02,
) -> list[PolicyQueryResult]:
    """Leave-one-independent-unit-out under unconstrained and gated policies.

    Gated policy personalises only when:
      (i) rank-support confidence ≥ *confidence_floor*, and
      (ii) the neighbour-weighted top method's predicted score exceeds the
           training-set global-best mean by at least *proxy_advantage*.
    Otherwise it deploys the global-best comparator. Realised held-out regret
    is scored for both policies against the oracle.
    """
    method_names = list(methods or landscape.method_order())
    units = landscape.dataset_order()
    results: list[PolicyQueryResult] = []

    from .features import RECOMMENDATION_FEATURE_ORDER

    source_order = list(landscape.feature_order or RECOMMENDATION_FEATURE_ORDER)
    keep_idx = [
        i
        for i, name in enumerate(source_order)
        if name not in {"n_domains", "domain_balance", "domain_spatial_coherence"}
    ]
    free_order = [source_order[i] for i in keep_idx] or list(RECOMMENDATION_FEATURE_ORDER)

    for held in units:
        training_keys = [u for u in units if u != held]
        if len(training_keys) < min_training:
            continue
        held_scores = {
            m: float(s)
            for m, s in landscape.performance[held].items()
            if m in method_names and s is not None and np.isfinite(s)
        }
        if not held_scores:
            continue
        oracle = max(held_scores.values())
        means = training_means(landscape, training_keys, method_names)
        finite_means = {m: v for m, v in means.items() if np.isfinite(v)}
        if not finite_means:
            continue
        gbest = max(finite_means, key=lambda m: (finite_means[m], m))
        gscore = held_scores.get(gbest, float("nan"))
        random_score = float(np.mean(list(held_scores.values())))

        training = subset_landscape(landscape, training_keys)
        training.performance = {
            name: {
                m: float(scores[m])
                if m in scores and scores[m] is not None and np.isfinite(scores[m])
                else float("nan")
                for m in method_names
            }
            for name, scores in training.performance.items()
        }
        training.method_count = len(method_names)

        held_vec = np.asarray(landscape.features[held], dtype=float).ravel()
        if held_vec.size == len(source_order):
            query_vec = held_vec[keep_idx] if keep_idx else held_vec
        elif held_vec.size == len(free_order):
            query_vec = held_vec
        else:
            query_vec = np.full(len(free_order), np.nan, dtype=float)
            n = min(held_vec.size, query_vec.size)
            query_vec[:n] = held_vec[:n]

        meta = landscape.dataset_meta.get(held, {})
        platform = classify_platform(meta.get("platform"))
        rec = MethodRecommender(
            training, k_neighbours=min(k_neighbours, max(1, len(training_keys)))
        ).recommend_from_features(
            query_vec,
            dataset_name=held,
            task=AnalysisTask.SPATIAL_DOMAIN.value,
            platform=platform,
        )
        ranked = [item for item in rec.ranked_methods if item.method in method_names]
        knn_method = ranked[0].method if ranked else None
        knn_conf = float(ranked[0].confidence) if ranked else 0.0
        knn_proxy = float(ranked[0].score) if ranked else float("nan")
        knn_score = held_scores.get(knn_method, float("nan")) if knn_method else float("nan")
        global_proxy = float(finite_means[gbest])
        beats_proxy = bool(
            knn_method is not None
            and np.isfinite(knn_proxy)
            and knn_proxy >= global_proxy + float(proxy_advantage) - 1e-12
        )
        gate_ok = beats_proxy and knn_conf >= confidence_floor
        if gate_ok and knn_method is not None:
            gated_method = knn_method
            gated_action = "personalised"
            gated_score = knn_score
        else:
            gated_method = gbest
            gated_action = "global_default"
            gated_score = gscore

        results.append(
            PolicyQueryResult(
                held_out=held,
                independence_class=str(meta.get("independence_class") or "") or None,
                oracle_score=float(oracle),
                global_best_method=gbest,
                global_best_regret=_f(oracle - gscore),
                knn_method=knn_method,
                knn_regret=_f(oracle - knn_score),
                knn_confidence=_f(knn_conf),
                knn_beats_global_proxy=beats_proxy,
                gated_method=gated_method,
                gated_regret=_f(oracle - gated_score),
                gated_action=gated_action,
                random_regret=_f(oracle - random_score),
            )
        )
    return results


def summarise_policies(
    rows: list[PolicyQueryResult],
    *,
    noninferior_margin: float = DEFAULT_NONINFERIOR_MARGIN,
    min_queries: int = 15,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarise policy regrets and non-inferiority vs global-best.

    Point non-inferiority: mean_policy ≤ mean_global + margin.
    Statistical non-inferiority (TOST-style one-sided): upper *(1-α)* bootstrap
    CI of mean(policy − global) ≤ margin (α = 0.05 → 95% one-sided via 90%
    central interval upper edge, implemented as 95% central CI upper bound for
    a conservative report).
    """
    if not rows:
        return {
            "protocol": PROTOCOL,
            "n_queries": 0,
            "meets_query_target": False,
            "gated_noninferior": False,
            "knn_noninferior": False,
        }

    def _mean(attr: str) -> float:
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    g = _mean("global_best_regret")
    k = _mean("knn_regret")
    gate = _mean("gated_regret")
    rnd = _mean("random_regret")
    knn_ni = bool(np.isfinite(k) and np.isfinite(g) and k <= g + noninferior_margin)
    gated_ni = bool(np.isfinite(gate) and np.isfinite(g) and gate <= g + noninferior_margin)
    gated_superior = bool(np.isfinite(gate) and np.isfinite(g) and gate < g - 1e-12)
    knn_superior = bool(np.isfinite(k) and np.isfinite(g) and k < g - 1e-12)

    delta_gate = paired_delta_bootstrap(rows, n_boot=n_boot, seed=seed)
    # Reuse paired bootstrap machinery for knn−global.
    knn_as_gated = [
        PolicyQueryResult(
            held_out=r.held_out,
            independence_class=r.independence_class,
            oracle_score=r.oracle_score,
            global_best_method=r.global_best_method,
            global_best_regret=r.global_best_regret,
            knn_method=r.knn_method,
            knn_regret=r.knn_regret,
            knn_confidence=r.knn_confidence,
            knn_beats_global_proxy=r.knn_beats_global_proxy,
            gated_method=r.knn_method,
            gated_regret=r.knn_regret,
            gated_action="personalised",
            random_regret=r.random_regret,
        )
        for r in rows
    ]
    delta_knn = paired_delta_bootstrap(knn_as_gated, n_boot=n_boot, seed=seed + 1)
    gated_stat_ni = bool(
        delta_gate.get("ci_high") is not None
        and float(delta_gate["ci_high"]) <= noninferior_margin + 1e-12
    )
    knn_stat_ni = bool(
        delta_knn.get("ci_high") is not None
        and float(delta_knn["ci_high"]) <= noninferior_margin + 1e-12
    )

    by_class: dict[str, dict[str, Any]] = {}
    for cls in sorted({r.independence_class or "unknown" for r in rows}):
        sub = [r for r in rows if (r.independence_class or "unknown") == cls]
        by_class[cls] = {
            "n": len(sub),
            "mean_global_regret": float(
                np.mean([r.global_best_regret for r in sub if r.global_best_regret is not None])
            ),
            "mean_knn_regret": float(
                np.mean([r.knn_regret for r in sub if r.knn_regret is not None])
            ),
            "mean_gated_regret": float(
                np.mean([r.gated_regret for r in sub if r.gated_regret is not None])
            ),
            "personalised_rate": float(
                np.mean([r.gated_action == "personalised" for r in sub])
            ),
        }

    # Primary success: point NI + statistical NI for gated policy.
    primary_ni = gated_ni and gated_stat_ni

    return {
        "protocol": PROTOCOL,
        "n_queries": len(rows),
        "min_queries_target": min_queries,
        "meets_query_target": len(rows) >= min_queries,
        "noninferior_margin": noninferior_margin,
        "mean_global_best_regret": g,
        "mean_knn_regret": k,
        "mean_gated_regret": gate,
        "mean_random_regret": rnd,
        "knn_noninferior": knn_ni,
        "knn_statistical_noninferior": knn_stat_ni,
        "knn_superior": knn_superior,
        "gated_noninferior": gated_ni,
        "gated_statistical_noninferior": gated_stat_ni,
        "gated_superior": gated_superior,
        "gated_minus_global_point": delta_gate,
        "knn_minus_global_point": delta_knn,
        "gated_personalised_rate": float(
            np.mean([r.gated_action == "personalised" for r in rows])
        ),
        "primary_policy": "gated_personalisation",
        "primary_noninferior": primary_ni,
        "primary_superior": gated_superior,
        "by_independence_class": by_class,
        "claim_boundary": (
            "Primary claim uses gated personalisation (fallback to global-best when "
            "the local proxy does not clear the gate). Statistical non-inferiority "
            "requires the bootstrap upper CI of mean(gated−global) ≤ margin. "
            "Unconstrained k-NN is a diagnostic, not the deployment policy."
        ),
    }


# ---------------------------------------------------------------------------
# Cross-lab reproducibility statistics
# ---------------------------------------------------------------------------


def study_bootstrap_regret_ci(
    rows: list[PolicyQueryResult],
    *,
    policy: Literal["gated", "knn", "global"] = "gated",
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Study-level bootstrap CI for mean regret of a policy."""
    attr = {
        "gated": "gated_regret",
        "knn": "knn_regret",
        "global": "global_best_regret",
    }[policy]
    values = np.asarray(
        [getattr(r, attr) for r in rows if getattr(r, attr) is not None], dtype=float
    )
    if values.size == 0:
        return {"policy": policy, "n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = values.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(values[idx]))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "policy": policy,
        "n": int(n),
        "n_boot": n_boot,
        "level": level,
        "mean": float(np.mean(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
    }


def paired_delta_bootstrap(
    rows: list[PolicyQueryResult],
    *,
    n_boot: int = 2000,
    seed: int = 1,
    level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap CI for mean(gated_regret − global_regret); ≤0 supports non-inferiority."""
    deltas = []
    for r in rows:
        if r.gated_regret is None or r.global_best_regret is None:
            continue
        deltas.append(float(r.gated_regret) - float(r.global_best_regret))
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean_delta": None}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means[b] = float(np.mean(arr[idx]))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "n": int(arr.size),
        "n_boot": n_boot,
        "level": level,
        "mean_delta": float(np.mean(arr)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "prob_delta_le_0": float(np.mean(means <= 0.0)),
        "interpretation": (
            "mean_delta ≤ 0 indicates gated personalisation is non-inferior "
            "(or superior) to always-global on mean regret"
        ),
    }


def rank_concordance_across_units(
    landscape: LandscapeResult,
    *,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Kendall's W-style concordance of method ranks across independent units."""
    method_names = list(methods or landscape.method_order())
    units = landscape.dataset_order()
    ranks = []
    for unit in units:
        scores = {
            m: landscape.performance[unit].get(m, float("nan")) for m in method_names
        }
        finite = {m: s for m, s in scores.items() if s is not None and np.isfinite(s)}
        if len(finite) < 2:
            continue
        ordered = sorted(finite, key=lambda m: (-finite[m], m))
        rank_map = {m: i + 1 for i, m in enumerate(ordered)}
        ranks.append([rank_map.get(m, len(method_names)) for m in method_names])
    if len(ranks) < 2:
        return {"n_units": len(ranks), "kendall_w": None, "methods": method_names}
    R = np.asarray(ranks, dtype=float)  # units × methods
    n_units, n_methods = R.shape
    rank_sums = R.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = float(np.sum((rank_sums - mean_rank_sum) ** 2))
    # Kendall's W
    denom = n_units**2 * (n_methods**3 - n_methods) / 12.0
    w = float(ss / denom) if denom > 0 else None
    return {
        "n_units": int(n_units),
        "n_methods": int(n_methods),
        "methods": method_names,
        "mean_rank": {
            m: float(rank_sums[i] / n_units) for i, m in enumerate(method_names)
        },
        "kendall_w": w,
        "notes": [
            "Kendall's W near 1 ⇒ method ranks are stable across independent units.",
            "Low W ⇒ personalisation / local evidence is more valuable than a single global rank.",
        ],
    }


def cross_lab_reproducibility_report(
    landscape: LandscapeResult,
    policy_rows: list[PolicyQueryResult],
    *,
    methods: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bundle study-bootstrap CIs, paired deltas, rank concordance, and FDR review."""
    method_names = list(methods or landscape.method_order())
    # Restrict performance matrix for stats review.
    perf = {
        u: {m: landscape.performance[u].get(m, float("nan")) for m in method_names}
        for u in landscape.dataset_order()
    }
    stats = review_landscape(perf, n_boot=min(500, n_boot), n_perm=min(1000, n_boot), seed=seed)
    return {
        "protocol": "histoweave.cross_lab_reproducibility.v1",
        "n_independent_units": landscape.dataset_count,
        "independence_classes": dict(
            Counter(
                str(landscape.dataset_meta.get(u, {}).get("independence_class") or "unknown")
                for u in landscape.dataset_order()
            )
        ),
        "regret_ci": {
            "global": study_bootstrap_regret_ci(
                policy_rows, policy="global", n_boot=n_boot, seed=seed
            ),
            "knn": study_bootstrap_regret_ci(
                policy_rows, policy="knn", n_boot=n_boot, seed=seed + 1
            ),
            "gated": study_bootstrap_regret_ci(
                policy_rows, policy="gated", n_boot=n_boot, seed=seed + 2
            ),
        },
        "gated_minus_global": paired_delta_bootstrap(
            policy_rows, n_boot=n_boot, seed=seed + 3
        ),
        "rank_concordance": rank_concordance_across_units(landscape, methods=method_names),
        "stats_review": stats.to_dict(),
        "notes": [
            "Bootstrap resamples independent study/donor/lab units, not cells.",
            "Pairwise FDR is across methods over independent units (stats_review).",
        ],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_independent_report(
    summary: dict[str, Any],
    rows: list[PolicyQueryResult],
    cross_lab: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Independent study/donor personalisation",
        "",
        f"- **Protocol**: `{summary.get('protocol')}`",
        f"- **Independent queries**: {summary.get('n_queries')} "
        f"(target ≥{summary.get('min_queries_target')}: "
        f"{'met' if summary.get('meets_query_target') else 'not met'})",
        f"- **Non-inferiority margin**: {summary.get('noninferior_margin')}",
        f"- **Mean global-best regret**: {summary.get('mean_global_best_regret')}",
        f"- **Mean unconstrained k-NN regret**: {summary.get('mean_knn_regret')} "
        f"(non-inferior={summary.get('knn_noninferior')})",
        f"- **Mean gated personalisation regret**: {summary.get('mean_gated_regret')} "
        f"(point-NI={summary.get('gated_noninferior')}, "
        f"stat-NI={summary.get('gated_statistical_noninferior')}, "
        f"superior={summary.get('gated_superior')})",
        f"- **Gated personalisation rate**: {summary.get('gated_personalised_rate')}",
        f"- **Primary policy**: {summary.get('primary_policy')} "
        f"(primary-NI={summary.get('primary_noninferior')}, "
        f"superior={summary.get('primary_superior')})",
        "",
        summary.get("claim_boundary") or "",
        "",
        "| Unit | Class | kNN reg | Global reg | Gated reg | Action |",
        "|------|-------|---------|------------|-----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.held_out} | {r.independence_class or '·'} "
            f"| {_fmt(r.knn_regret)} | {_fmt(r.global_best_regret)} "
            f"| {_fmt(r.gated_regret)} | {r.gated_action} |"
        )
    if cross_lab:
        delta = cross_lab.get("gated_minus_global") or {}
        conc = cross_lab.get("rank_concordance") or {}
        lines += [
            "",
            "## Cross-lab reproducibility",
            "",
            f"- Gated−global mean Δ regret: **{delta.get('mean_delta')}** "
            f"(95% CI [{delta.get('ci_low')}, {delta.get('ci_high')}]; "
            f"P(Δ≤0)={delta.get('prob_delta_le_0')})",
            f"- Kendall's W (rank concordance): **{conc.get('kendall_w')}** "
            f"across {conc.get('n_units')} units",
            f"- Independence class counts: {cross_lab.get('independence_classes')}",
        ]
    by_class = summary.get("by_independence_class") or {}
    if by_class:
        lines += ["", "## By independence class", ""]
        for cls, stats in by_class.items():
            lines.append(
                f"- **{cls}** (n={stats['n']}): gated={stats['mean_gated_regret']:.4f}, "
                f"global={stats['mean_global_regret']:.4f}, "
                f"personalised_rate={stats['personalised_rate']:.0%}"
            )
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "·"
    return f"{float(value):.4f}"


def write_independent_personalisation_bundle(
    output_dir: str | Path,
    *,
    landscape: LandscapeResult,
    units: list[IndependentStudyUnit],
    policy_rows: list[PolicyQueryResult],
    summary: dict[str, Any],
    cross_lab: dict[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    land_path = output / "independent_unit_landscape.json"
    _atomic_json(
        land_path,
        {
            "protocol": PROTOCOL,
            "task": landscape.task,
            "metric": landscape.metric,
            "feature_order": landscape.feature_order,
            "performance": landscape.performance,
            "features": {k: np.asarray(v).tolist() for k, v in landscape.features.items()},
            "timings": landscape.timings,
            "dataset_meta": landscape.dataset_meta,
            "best_method": landscape.best_method,
            "units": [u.to_dict() for u in units],
            "n_units": landscape.dataset_count,
        },
    )
    paths["landscape"] = land_path

    pol_path = output / "personalisation_policies.json"
    _atomic_json(
        pol_path,
        {
            "protocol": PROTOCOL,
            "summary": summary,
            "queries": [r.to_dict() for r in policy_rows],
        },
    )
    paths["policies"] = pol_path

    xl_path = output / "cross_lab_reproducibility.json"
    _atomic_json(xl_path, cross_lab)
    paths["cross_lab"] = xl_path

    md_path = output / "independent_personalisation_report.md"
    md_path.write_text(
        format_independent_report(summary, policy_rows, cross_lab), encoding="utf-8"
    )
    paths["report"] = md_path

    master = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "n_independent_queries": summary.get("n_queries"),
        "meets_query_target": summary.get("meets_query_target"),
        "primary_policy": summary.get("primary_policy"),
        "primary_noninferior": summary.get("primary_noninferior"),
        "primary_superior": summary.get("primary_superior"),
        "gated_point_noninferior": summary.get("gated_noninferior"),
        "gated_statistical_noninferior": summary.get("gated_statistical_noninferior"),
        "knn_point_noninferior": summary.get("knn_noninferior"),
        "knn_statistical_noninferior": summary.get("knn_statistical_noninferior"),
        "mean_gated_regret": summary.get("mean_gated_regret"),
        "mean_global_best_regret": summary.get("mean_global_best_regret"),
        "mean_knn_regret": summary.get("mean_knn_regret"),
        "noninferior_margin": summary.get("noninferior_margin"),
        "gated_minus_global": cross_lab.get("gated_minus_global"),
        "kendall_w": (cross_lab.get("rank_concordance") or {}).get("kendall_w"),
        "artifacts": {k: v.name for k, v in paths.items()},
    }
    master_path = output / "independent_personalisation_summary.json"
    _atomic_json(master_path, master)
    paths["summary"] = master_path
    return paths
