"""Falsifiable paper-level evaluation endpoints (decision-protocol.md).

Implements five predeclared endpoints:

1. **Study-grouped personalisation** — leave-one-study/slice-out regret vs
   global-best on ≥15–20 independent queries.
2. **Selective regret–coverage** — abstention by rank-support threshold.
3. **Pareto membership stability** — bootstrap frontier inclusion probability.
4. **Unified-resource SOTA comparison** — matched timeout / status filters.
5. **Oracle-K leakage impact** — mean ARI(oracle) − mean ARI(estimate) on
   dual-track SOTA long tables (non-oracle K drop).

These are analysis utilities over existing landscapes and long-format CSVs;
they do not invent biological ground truth.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .landscape import LandscapeResult
from .pareto import OBJECTIVE_DIRECTIONS, ObjectiveTable, analyze_dataset, pareto_frontier
from .recommend import MethodRecommender
from .task_contract import AnalysisTask, classify_platform

PROTOCOL_STUDY = "study_grouped_holdout"
PROTOCOL_SELECTIVE = "histoweave.selective_regret_coverage.v1"
PROTOCOL_PARETO_STAB = "histoweave.pareto_stability.v1"
PROTOCOL_SOTA_RESOURCE = "histoweave.sota_unified_resource.v1"
PROTOCOL_ORACLE_K_LEAK = "histoweave.oracle_k_leakage.v1"

DEFAULT_CONFIDENCE_THRESHOLDS = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
)


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def subset_landscape(landscape: LandscapeResult, names: list[str]) -> LandscapeResult:
    """Return a landscape restricted to the named datasets."""
    keep = [name for name in names if name in landscape.performance]
    return LandscapeResult(
        performance={name: dict(landscape.performance[name]) for name in keep},
        features={
            name: landscape.features[name].copy() for name in keep if name in landscape.features
        },
        embedding={name: landscape.embedding.get(name, (0.0, 0.0)) for name in keep},
        best_method={name: landscape.best_method.get(name, "?") for name in keep},
        niches={
            method: [name for name in members if name in keep]
            for method, members in landscape.niches.items()
        },
        timings={name: dict(landscape.timings.get(name, {})) for name in keep},
        feature_order=list(landscape.feature_order),
        method_count=landscape.method_count,
        dataset_count=len(keep),
        task=landscape.task,
        metric=landscape.metric,
        higher_is_better=landscape.higher_is_better,
        dataset_meta={
            name: dict(landscape.dataset_meta[name])
            for name in keep
            if name in landscape.dataset_meta
        },
    )


def training_means(
    landscape: LandscapeResult,
    training_keys: list[str],
    methods: list[str] | None = None,
) -> dict[str, float]:
    """Mean finite score per method on the training keys."""
    if methods is None:
        methods = landscape.method_order()
    means: dict[str, float] = {}
    for method in methods:
        values = [landscape.performance[key].get(method, float("nan")) for key in training_keys]
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        means[method] = float(np.mean(finite)) if finite else float("nan")
    return means


@dataclass
class StudyGroupedQuery:
    held_out: str
    training_sets: list[str]
    oracle_methods: list[str]
    oracle_score: float
    selected_method: str | None
    selected_score: float | None
    recommended_methods: list[str]
    top1_hit: bool
    top3_hit: bool
    selection_regret: float | None
    global_best_method: str | None
    global_best_score: float | None
    global_best_regret: float | None
    random_expected_score: float | None
    random_expected_regret: float | None
    knn_beats_global: bool
    knn_beats_random: bool
    confidence: float | None = None
    platform: str | None = None
    study_group: str | None = None
    neighbours: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StudyGroupedSummary:
    n_queries: int
    n_training_pool: int
    methods: list[str]
    top1_accuracy: float
    top3_accuracy: float
    mean_selection_regret: float
    median_selection_regret: float
    mean_global_best_regret: float
    mean_random_regret: float
    knn_beats_global_rate: float
    knn_beats_random_rate: float
    beats_global_best: bool
    regret_delta_vs_global: float
    protocol: str = PROTOCOL_STUDY
    min_queries_target: int = 20
    meets_query_target: bool = False
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def leave_one_study_out(
    landscape: LandscapeResult,
    *,
    query_names: list[str] | None = None,
    methods: list[str] | None = None,
    k_neighbours: int = 3,
    min_training: int = 4,
) -> tuple[list[StudyGroupedQuery], StudyGroupedSummary]:
    """Leave-one-study/slice-out recommendation validation on a landscape.

    Uses stored target-free feature rows for held-out queries (no table reload).
    """
    pool = list(query_names or landscape.dataset_order())
    method_names = list(methods or landscape.method_order())
    if len(pool) < 2:
        raise ValueError("leave_one_study_out requires at least two datasets")

    # Align feature vectors to the recommender's target-free schema.
    from .features import RECOMMENDATION_FEATURE_ORDER

    source_order = list(landscape.feature_order or RECOMMENDATION_FEATURE_ORDER)
    keep_idx = [
        index
        for index, name in enumerate(source_order)
        if name not in {"n_domains", "domain_balance", "domain_spatial_coherence"}
    ]
    free_order = [source_order[index] for index in keep_idx]
    if not free_order:
        free_order = list(RECOMMENDATION_FEATURE_ORDER)
        keep_idx = list(range(len(free_order)))

    queries: list[StudyGroupedQuery] = []
    for held in pool:
        training_keys = [name for name in landscape.dataset_order() if name != held]
        if len(training_keys) < min_training:
            continue
        if held not in landscape.performance:
            continue
        held_features = landscape.features.get(held)
        if held_features is None:
            continue
        held_vec = np.asarray(held_features, dtype=float).ravel()
        if held_vec.size == len(source_order):
            query_vec = held_vec[keep_idx]
        elif held_vec.size == len(free_order):
            query_vec = held_vec
        else:
            # Pad / truncate conservatively.
            query_vec = np.full(len(free_order), np.nan, dtype=float)
            n = min(held_vec.size, query_vec.size)
            query_vec[:n] = held_vec[:n]

        training = subset_landscape(landscape, training_keys)
        if methods is not None:
            restricted: dict[str, dict[str, float]] = {}
            for name, scores in training.performance.items():
                restricted[name] = {
                    method: float(scores[method])
                    if method in scores
                    and scores[method] is not None
                    and np.isfinite(scores[method])
                    else float("nan")
                    for method in method_names
                }
            training.performance = restricted
            training.method_count = len(method_names)

        recommender = MethodRecommender(
            training,
            k_neighbours=min(k_neighbours, max(1, len(training_keys))),
        )
        meta = landscape.dataset_meta.get(held, {})
        platform = classify_platform(meta.get("platform") or meta.get("assay"))
        recommendation = recommender.recommend_from_features(
            query_vec,
            dataset_name=held,
            task=landscape.task or AnalysisTask.SPATIAL_DOMAIN.value,
            platform=platform,
        )
        ranked = [item.method for item in recommendation.ranked_methods]
        # Keep only methods present on the held-out row when a method filter is set.
        if methods is not None:
            ranked = [name for name in ranked if name in method_names] or ranked

        held_scores = {
            method: float(score)
            for method, score in landscape.performance[held].items()
            if (methods is None or method in method_names)
            and score is not None
            and np.isfinite(score)
        }
        if not held_scores:
            continue
        oracle_score = max(held_scores.values())
        oracle_methods = sorted(
            method
            for method, score in held_scores.items()
            if np.isclose(score, oracle_score, atol=1e-12, rtol=0.0)
        )
        selected = ranked[0] if ranked else None
        selected_score = held_scores.get(selected) if selected is not None else None
        means = training_means(landscape, training_keys, method_names)
        finite_means = {method: score for method, score in means.items() if np.isfinite(score)}
        global_best = (
            max(finite_means, key=lambda method: (finite_means[method], method))
            if finite_means
            else None
        )
        global_score = held_scores.get(global_best) if global_best else None
        random_score = float(np.mean(list(held_scores.values())))
        confidence = (
            float(recommendation.ranked_methods[0].confidence)
            if recommendation.ranked_methods
            else None
        )
        sel = selected_score if selected_score is not None else float("nan")
        gsc = global_score if global_score is not None else float("nan")
        queries.append(
            StudyGroupedQuery(
                held_out=held,
                training_sets=training_keys,
                oracle_methods=oracle_methods,
                oracle_score=float(oracle_score),
                selected_method=selected,
                selected_score=_f(selected_score),
                recommended_methods=ranked[:3],
                top1_hit=selected in oracle_methods if selected else False,
                top3_hit=bool(set(ranked[:3]) & set(oracle_methods)),
                selection_regret=_f(oracle_score - sel),
                global_best_method=global_best,
                global_best_score=_f(global_score),
                global_best_regret=_f(oracle_score - gsc),
                random_expected_score=_f(random_score),
                random_expected_regret=_f(oracle_score - random_score),
                knn_beats_global=bool(
                    selected is not None
                    and np.isfinite(sel)
                    and np.isfinite(gsc)
                    and sel >= gsc - 1e-9
                ),
                knn_beats_random=bool(
                    selected is not None and np.isfinite(sel) and sel > random_score
                ),
                confidence=_f(confidence),
                platform=platform,
                study_group=str(meta.get("study_group") or meta.get("donor") or held),
                neighbours=list(recommendation.neighbours),
            )
        )

    summary = summarise_study_grouped(
        queries, n_training_pool=len(landscape.dataset_order()), methods=method_names
    )
    return queries, summary


def summarise_study_grouped(
    queries: list[StudyGroupedQuery],
    *,
    n_training_pool: int,
    methods: list[str],
    min_queries_target: int = 20,
) -> StudyGroupedSummary:
    if not queries:
        return StudyGroupedSummary(
            n_queries=0,
            n_training_pool=n_training_pool,
            methods=methods,
            top1_accuracy=0.0,
            top3_accuracy=0.0,
            mean_selection_regret=float("nan"),
            median_selection_regret=float("nan"),
            mean_global_best_regret=float("nan"),
            mean_random_regret=float("nan"),
            knn_beats_global_rate=0.0,
            knn_beats_random_rate=0.0,
            beats_global_best=False,
            regret_delta_vs_global=float("nan"),
            min_queries_target=min_queries_target,
            meets_query_target=False,
            caveats=["No successful leave-one-out queries were produced."],
        )
    regrets = [float(q.selection_regret) for q in queries if q.selection_regret is not None]
    global_regrets = [
        float(q.global_best_regret) for q in queries if q.global_best_regret is not None
    ]
    random_regrets = [
        float(q.random_expected_regret) for q in queries if q.random_expected_regret is not None
    ]
    mean_sel = float(np.mean(regrets)) if regrets else float("nan")
    mean_glb = float(np.mean(global_regrets)) if global_regrets else float("nan")
    mean_rnd = float(np.mean(random_regrets)) if random_regrets else float("nan")
    beats = bool(np.isfinite(mean_sel) and np.isfinite(mean_glb) and mean_sel <= mean_glb + 1e-12)
    caveats: list[str] = []
    if len(queries) < min_queries_target:
        caveats.append(
            f"Only {len(queries)} queries; personalisation claims require ≥{min_queries_target}."
        )
    if not beats:
        caveats.append(
            "Mean selection regret does not beat the global-best comparator "
            "(fallback / global_default remains justified)."
        )
    else:
        caveats.append(
            "k-NN mean regret is non-inferior to global-best on this multi-source holdout."
        )
    return StudyGroupedSummary(
        n_queries=len(queries),
        n_training_pool=n_training_pool,
        methods=methods,
        top1_accuracy=float(np.mean([q.top1_hit for q in queries])),
        top3_accuracy=float(np.mean([q.top3_hit for q in queries])),
        mean_selection_regret=mean_sel,
        median_selection_regret=float(np.median(regrets)) if regrets else float("nan"),
        mean_global_best_regret=mean_glb,
        mean_random_regret=mean_rnd,
        knn_beats_global_rate=float(np.mean([q.knn_beats_global for q in queries])),
        knn_beats_random_rate=float(np.mean([q.knn_beats_random for q in queries])),
        beats_global_best=beats,
        regret_delta_vs_global=float(mean_sel - mean_glb)
        if np.isfinite(mean_sel) and np.isfinite(mean_glb)
        else float("nan"),
        min_queries_target=min_queries_target,
        meets_query_target=len(queries) >= min_queries_target,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Selective regret–coverage
# ---------------------------------------------------------------------------


@dataclass
class SelectivePoint:
    threshold: float
    coverage: float
    n_accepted: int
    n_total: int
    mean_regret_accepted: float | None
    mean_regret_abstain_as_global: float | None
    mean_regret_always_personalised: float | None
    mean_regret_always_global: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def selective_regret_coverage(
    queries: list[StudyGroupedQuery] | list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Compute selective-prediction curves for personalised recommendations.

    When confidence < threshold the protocol *abstains* from personalisation and
    uses the global-best action instead. Coverage is the fraction of queries
    where personalisation is retained.
    """
    rows: list[dict[str, Any]] = []
    for item in queries:
        row = item.to_dict() if isinstance(item, StudyGroupedQuery) else dict(item)
        conf = _f(row.get("confidence"))
        sel_regret = _f(row.get("selection_regret"))
        glb_regret = _f(row.get("global_best_regret"))
        if sel_regret is None or glb_regret is None:
            continue
        rows.append(
            {
                "confidence": conf if conf is not None else 0.0,
                "selection_regret": sel_regret,
                "global_best_regret": glb_regret,
            }
        )
    n_total = len(rows)
    always_p = float(np.mean([r["selection_regret"] for r in rows])) if rows else None
    always_g = float(np.mean([r["global_best_regret"] for r in rows])) if rows else None
    # Include a pure-abstention sentinel above any finite confidence.
    resolved_thresholds = list(thresholds) + [float("inf")]
    curve: list[SelectivePoint] = []
    for threshold in resolved_thresholds:
        accepted = [r for r in rows if r["confidence"] >= threshold]
        n_acc = len(accepted)
        if n_acc:
            mean_acc = float(np.mean([r["selection_regret"] for r in accepted]))
        else:
            mean_acc = None
        # Hybrid policy: personalise if accepted else global default.
        hybrid = []
        for row in rows:
            if row["confidence"] >= threshold:
                hybrid.append(row["selection_regret"])
            else:
                hybrid.append(row["global_best_regret"])
        mean_hybrid = float(np.mean(hybrid)) if hybrid else None
        curve.append(
            SelectivePoint(
                threshold=float(threshold) if math.isfinite(threshold) else float("inf"),
                coverage=float(n_acc / n_total) if n_total else 0.0,
                n_accepted=n_acc,
                n_total=n_total,
                mean_regret_accepted=mean_acc,
                mean_regret_abstain_as_global=mean_hybrid,
                mean_regret_always_personalised=always_p,
                mean_regret_always_global=always_g,
            )
        )

    # Recommended operating point: minimal hybrid regret; ties → higher coverage.
    best = None
    for point in curve:
        if point.mean_regret_abstain_as_global is None:
            continue
        if best is None:
            best = point
            continue
        assert best.mean_regret_abstain_as_global is not None
        best_regret: float = best.mean_regret_abstain_as_global
        if point.mean_regret_abstain_as_global < best_regret - 1e-12:
            best = point
        elif (
            abs(point.mean_regret_abstain_as_global - best_regret) <= 1e-12
            and point.coverage > best.coverage
        ):
            best = point

    # JSON cannot encode Infinity; serialise abstention sentinel as null threshold.
    curve_payload = []
    for point in curve:
        payload = point.to_dict()
        if not math.isfinite(payload["threshold"]):
            payload["threshold"] = None
            payload["label"] = "always_global_default"
        curve_payload.append(payload)

    rec_threshold: float | None
    if best is None:
        rec_threshold = None
    elif not math.isfinite(best.threshold):
        rec_threshold = None
    else:
        rec_threshold = best.threshold

    return {
        "protocol": PROTOCOL_SELECTIVE,
        "n_queries": n_total,
        "curve": curve_payload,
        "recommended_threshold": rec_threshold,
        "recommended_coverage": best.coverage if best else None,
        "recommended_hybrid_regret": best.mean_regret_abstain_as_global if best else None,
        "recommended_policy": (
            "always_global_default"
            if best is not None and not math.isfinite(best.threshold)
            else "confidence_gated_personalisation"
        ),
        "notes": [
            "Abstention replaces personalised action with the global-best comparator.",
            "Confidence is the recommender rank-support heuristic, not a calibrated probability.",
            "A null recommended_threshold means pure global-default (full abstention) is optimal.",
        ],
    }


# ---------------------------------------------------------------------------
# Pareto stability
# ---------------------------------------------------------------------------


def _bootstrap_objective_table(
    seed_scores: dict[str, list[float]],
    seed_times: dict[str, list[float]],
    *,
    dataset: str,
    rng: np.random.Generator,
) -> ObjectiveTable:
    points: dict[str, dict[str, float | None]] = {}
    for config, values in seed_scores.items():
        if not values:
            continue
        idx = rng.integers(0, len(values), size=len(values))
        accuracy = float(np.mean([values[i] for i in idx]))
        row: dict[str, float | None] = {"accuracy": accuracy}
        times: list[float] = seed_times.get(config, [])
        if times:
            t_idx = rng.integers(0, len(times), size=len(times))
            row["speed"] = float(np.mean([times[i] for i in t_idx]))
        points[config] = row
    directions = {
        name: OBJECTIVE_DIRECTIONS[name]
        for name in ("accuracy", "speed")
        if any(row.get(name) is not None for row in points.values())
    }
    return ObjectiveTable(dataset=dataset, table=points, directions=directions)


def pareto_stability_from_long_csv(
    path: str | Path,
    *,
    n_boot: int = 200,
    seed: int = 0,
    dataset_col: str = "dataset",
    config_col: str | None = None,
    score_col: str = "ari",
    seconds_col: str = "seconds",
    status_col: str = "status",
) -> dict[str, Any]:
    """Bootstrap frontier inclusion probability from multi-seed long CSV rows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if dataset_col not in fields or score_col not in fields:
            raise ValueError(f"{path} must contain {dataset_col!r} and {score_col!r}")
        resolved_config = (
            config_col
            if config_col and config_col in fields
            else ("config" if "config" in fields else "method")
        )
        if resolved_config not in fields:
            raise ValueError(f"{path} needs a config or method column")
        for row in reader:
            status = str(row.get(status_col, "") or "").strip().lower()
            if status in {"failed", "error", "timeout", "oom", "skipped"}:
                continue
            dataset = str(row.get(dataset_col, "") or "").strip()
            config = str(row.get(resolved_config, "") or "").strip()
            accuracy = _f(row.get(score_col))
            if not dataset or not config or accuracy is None:
                continue
            scores[dataset][config].append(accuracy)
            seconds = _f(row.get(seconds_col)) if seconds_col in fields else None
            if seconds is not None:
                times[dataset][config].append(seconds)

    if not scores:
        raise ValueError(f"{path} produced no finite seed rows")

    rng = np.random.default_rng(seed)
    per_dataset: dict[str, dict[str, Any]] = {}
    global_counts: Counter[str] = Counter()
    global_trials = 0

    for dataset, config_scores in sorted(scores.items()):
        point_table = ObjectiveTable(
            dataset=dataset,
            table={
                config: {
                    "accuracy": float(np.mean(values)),
                    **(
                        {"speed": float(np.mean(times[dataset][config]))}
                        if times[dataset].get(config)
                        else {}
                    ),
                }
                for config, values in config_scores.items()
                if values
            },
        )
        point_result = analyze_dataset(point_table)
        inclusion: Counter[str] = Counter()
        for _ in range(n_boot):
            boot_table = _bootstrap_objective_table(
                config_scores,
                times[dataset],
                dataset=dataset,
                rng=rng,
            )
            frontier = pareto_frontier(boot_table.table, boot_table.directions)
            inclusion.update(frontier)
            global_counts.update(frontier)
            global_trials += 1
        n_configs = len(config_scores)
        per_dataset[dataset] = {
            "n_configs": n_configs,
            "point_frontier": point_result.frontier,
            "point_knee": point_result.knee,
            "inclusion_probability": {
                config: float(inclusion[config] / n_boot) for config in sorted(inclusion)
            },
            "stable_frontier": sorted(
                config for config, count in inclusion.items() if count / n_boot >= 0.5
            ),
            "mean_frontier_size": float(
                np.mean(list(inclusion.values())) / max(n_boot, 1) * n_configs
            )
            if inclusion
            else 0.0,
        }

    inclusion_global = {
        config: float(count / max(global_trials, 1))
        for config, count in sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))
    }
    return {
        "protocol": PROTOCOL_PARETO_STAB,
        "n_boot": n_boot,
        "seed": seed,
        "n_datasets": len(per_dataset),
        "datasets": per_dataset,
        "global_inclusion_probability": inclusion_global,
        "notes": [
            "Inclusion probability is the fraction of bootstrap resamples of seed-level "
            "replicates in which a configuration remains non-dominated.",
            "A configuration with inclusion ≥ 0.5 is reported as stable-frontier.",
        ],
    }


# ---------------------------------------------------------------------------
# Unified-resource SOTA comparison
# ---------------------------------------------------------------------------


def sota_unified_resource_compare(
    path: str | Path,
    *,
    timeout_seconds: float | None = None,
    max_seconds: float | None = None,
    require_status: tuple[str, ...] = ("success",),
    baseline_csv: str | Path | None = None,
    resource_label: str = "cpu_default",
) -> dict[str, Any]:
    """Compare SOTA (and optional baselines) under a shared resource filter.

    Cells exceeding ``max_seconds`` (or the env default timeout) are treated as
    resource violations rather than accuracy wins.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    budget = max_seconds
    if budget is None:
        budget = timeout_seconds
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    if baseline_csv is not None:
        base_path = Path(baseline_csv)
        if base_path.is_file():
            with base_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    payload = dict(row)
                    payload.setdefault("family", "baseline")
                    rows.append(payload)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method") or row.get("config") or "").strip()
        dataset = str(row.get("dataset") or "").strip()
        status = str(row.get("status") or "success").strip().lower()
        ari = _f(row.get("ari"))
        seconds = _f(row.get("seconds"))
        if not method or not dataset:
            continue
        if status not in require_status and status not in {"", "success", "ok", "completed"}:
            # Allow baseline CSVs without a status column (empty → success).
            if status:
                rejected.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reason": f"status={status}",
                        "ari": ari,
                        "seconds": seconds,
                    }
                )
                continue
        if budget is not None and seconds is not None and seconds > budget:
            rejected.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "reason": f"exceeded_budget_{budget}s",
                    "ari": ari,
                    "seconds": seconds,
                }
            )
            continue
        if ari is None:
            rejected.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "reason": "missing_ari",
                    "ari": None,
                    "seconds": seconds,
                }
            )
            continue
        accepted.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": row.get("seed"),
                "ari": ari,
                "seconds": seconds,
                "family": row.get("family")
                or ("sota" if method != "banksy_py" else "spatial_aware"),
                "status": status or "success",
            }
        )

    by_method: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in accepted}):
        method_rows = [row for row in accepted if row["method"] == method]
        aris = [float(row["ari"]) for row in method_rows]
        secs = [float(row["seconds"]) for row in method_rows if row["seconds"] is not None]
        by_method[method] = {
            "n_cells": len(method_rows),
            "n_datasets": len({row["dataset"] for row in method_rows}),
            "mean_ari": float(np.mean(aris)) if aris else None,
            "median_ari": float(np.median(aris)) if aris else None,
            "mean_seconds": float(np.mean(secs)) if secs else None,
            "throughput_ari_per_minute": (
                float(np.mean(aris) / (np.mean(secs) / 60.0))
                if aris and secs and np.mean(secs) > 0
                else None
            ),
            "family": method_rows[0]["family"],
        }

    # Per-dataset ranking under the resource filter.
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({row["dataset"] for row in accepted}):
        dataset_rows = [row for row in accepted if row["dataset"] == dataset]
        # Average seeds per method.
        method_means: dict[str, float] = {}
        for method in {row["method"] for row in dataset_rows}:
            vals = [float(row["ari"]) for row in dataset_rows if row["method"] == method]
            method_means[method] = float(np.mean(vals))
        ranking = sorted(method_means.items(), key=lambda item: (-item[1], item[0]))
        per_dataset[str(dataset)] = {
            "ranking": [{"method": method, "mean_ari": score} for method, score in ranking],
            "winner": ranking[0][0] if ranking else None,
            "winner_ari": ranking[0][1] if ranking else None,
        }

    ranked_methods = sorted(
        ((method, stats) for method, stats in by_method.items() if stats["mean_ari"] is not None),
        key=lambda item: (-float(item[1]["mean_ari"]), item[0]),
    )
    return {
        "protocol": PROTOCOL_SOTA_RESOURCE,
        "resource_label": resource_label,
        "max_seconds": budget,
        "n_accepted_cells": len(accepted),
        "n_rejected_cells": len(rejected),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "by_method": by_method,
        "method_ranking": [{"method": method, **stats} for method, stats in ranked_methods],
        "per_dataset": per_dataset,
        "notes": [
            "Only cells that finish within the shared time budget and succeed contribute.",
            "Missing backends / OOMs / timeouts are rejected, not replaced by toy substitutes.",
            f"Resource profile label: {resource_label}.",
        ],
    }


# ---------------------------------------------------------------------------
# Endpoint 5 — Oracle-K leakage / non-oracle K ARI drop
# ---------------------------------------------------------------------------


def oracle_k_leakage_impact(
    path: str | Path,
    *,
    estimate_modes: tuple[str, ...] = (
        "estimate:silhouette",
        "estimate:spatial_silhouette",
        "estimate:ensemble",
    ),
    primary_estimate_mode: str = "estimate:silhouette",
    score_col: str = "ari",
    status_col: str = "status",
) -> dict[str, Any]:
    """Quantify ARI drop when Oracle-K protection is removed.

    Expects a dual-track long CSV such as ``non_oracle_k_sota/benchmark_long.csv``
    with columns ``dataset``, ``method``, ``mode`` (or ``k_policy``+``estimator``),
    ``k_used``, ``oracle_k``, and ``ari``.

    Primary endpoint (decision-protocol claim):

        mean_ari(mode=oracle) − mean_ari(mode=primary_estimate_mode)

    per method and overall.  Recovery of alternative estimate modes relative to
    the primary estimate mode is reported but is **not** required to be positive.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "dataset" not in fields or "method" not in fields:
            raise ValueError(f"{path} must contain dataset and method columns")
        if score_col not in fields:
            raise ValueError(f"{path} must contain {score_col!r}")
        for raw in reader:
            status = str(raw.get(status_col) or "success").strip().lower()
            if status and status not in {"success", "ok", "completed"}:
                continue
            ari = _f(raw.get(score_col))
            if ari is None:
                continue
            mode = str(raw.get("mode") or "").strip()
            if not mode:
                k_policy = str(raw.get("k_policy") or "").strip().lower()
                estimator = str(raw.get("estimator") or "").strip()
                if k_policy == "oracle":
                    mode = "oracle"
                elif k_policy == "estimate" and estimator:
                    mode = f"estimate:{estimator}"
                elif k_policy:
                    mode = k_policy
            if not mode:
                continue
            rows.append(
                {
                    "dataset": str(raw.get("dataset") or "").strip(),
                    "method": str(raw.get("method") or "").strip(),
                    "mode": mode,
                    "ari": ari,
                    "k_used": _f(raw.get("k_used")),
                    "oracle_k": _f(raw.get("oracle_k")),
                    "k_match": str(raw.get("k_match") or "").lower() in {"1", "true", "yes"},
                    "seed": raw.get("seed"),
                }
            )

    if not rows:
        raise ValueError(f"no successful ARI rows in {path}")

    methods = sorted({row["method"] for row in rows if row["method"]})
    modes_present = sorted({row["mode"] for row in rows})
    by_method: dict[str, Any] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        mode_stats: dict[str, Any] = {}
        for mode in modes_present:
            mode_rows = [row for row in method_rows if row["mode"] == mode]
            if not mode_rows:
                continue
            aris = [float(row["ari"]) for row in mode_rows]
            k_used = [row["k_used"] for row in mode_rows if row["k_used"] is not None]
            mode_stats[mode] = {
                "n": len(aris),
                "mean_ari": float(np.mean(aris)),
                "std_ari": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
                "mean_k_used": float(np.mean(k_used)) if k_used else None,
                "k_match_rate": float(
                    np.mean([1.0 if row["k_match"] else 0.0 for row in mode_rows])
                ),
            }
        oracle_mean = (mode_stats.get("oracle") or {}).get("mean_ari")
        primary_mean = (mode_stats.get(primary_estimate_mode) or {}).get("mean_ari")
        drop = (
            float(oracle_mean - primary_mean)
            if oracle_mean is not None and primary_mean is not None
            else None
        )
        recovery: dict[str, Any] = {}
        if primary_mean is not None:
            for mode in estimate_modes:
                if mode == primary_estimate_mode:
                    continue
                alt = (mode_stats.get(mode) or {}).get("mean_ari")
                if alt is None:
                    continue
                recovered = float(alt - primary_mean)
                recovery[mode] = {
                    "mean_ari": alt,
                    "ari_recovered_vs_primary_estimate": recovered,
                    "fraction_of_drop_recovered": (
                        float(max(0.0, recovered) / drop)
                        if drop is not None and drop > 1e-12
                        else None
                    ),
                }
        # Per-slice oracle vs primary estimate (seed-averaged).
        per_dataset: list[dict[str, Any]] = []
        datasets = sorted({row["dataset"] for row in method_rows})
        for dataset in datasets:
            o_vals = [
                float(row["ari"])
                for row in method_rows
                if row["dataset"] == dataset and row["mode"] == "oracle"
            ]
            e_vals = [
                float(row["ari"])
                for row in method_rows
                if row["dataset"] == dataset and row["mode"] == primary_estimate_mode
            ]
            if not o_vals or not e_vals:
                continue
            o_m = float(np.mean(o_vals))
            e_m = float(np.mean(e_vals))
            per_dataset.append(
                {
                    "dataset": dataset,
                    "oracle_mean_ari": o_m,
                    "estimate_mean_ari": e_m,
                    "ari_drop": float(o_m - e_m),
                }
            )
        by_method[method] = {
            "mode_stats": mode_stats,
            "oracle_mean_ari": oracle_mean,
            "primary_estimate_mode": primary_estimate_mode,
            "primary_estimate_mean_ari": primary_mean,
            "mean_ari_drop_oracle_minus_estimate": drop,
            "recovery_vs_primary_estimate": recovery,
            "per_dataset": per_dataset,
            "max_slice_drop": (
                float(max(item["ari_drop"] for item in per_dataset)) if per_dataset else None
            ),
        }

    overall_drops = [
        stats["mean_ari_drop_oracle_minus_estimate"]
        for stats in by_method.values()
        if stats.get("mean_ari_drop_oracle_minus_estimate") is not None
    ]
    return {
        "protocol": PROTOCOL_ORACLE_K_LEAK,
        "source": str(path).replace("\\", "/"),
        "n_rows": len(rows),
        "methods": methods,
        "modes_present": modes_present,
        "primary_estimate_mode": primary_estimate_mode,
        "estimate_modes": list(estimate_modes),
        "by_method": by_method,
        "mean_ari_drop_across_methods": (float(np.mean(overall_drops)) if overall_drops else None),
        "notes": [
            "Oracle-K injects domain_truth.nunique() into n_domains; estimate modes do not.",
            "Positive mean_ari_drop means removing Oracle-K lowers ARI (leakage inflation).",
            "Recovery fractions may be zero or negative; that is a valid scientific outcome.",
            "This endpoint audits leakage sensitivity; it does not validate a K estimator.",
        ],
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def format_study_grouped_report(
    summary: StudyGroupedSummary | dict[str, Any],
    queries: list[StudyGroupedQuery] | list[dict[str, Any]],
) -> str:
    summary_dict = summary.to_dict() if isinstance(summary, StudyGroupedSummary) else dict(summary)
    lines = [
        "# Study-Grouped Recommendation Validation",
        "",
        f"- **Protocol**: `{summary_dict.get('protocol', PROTOCOL_STUDY)}`",
        f"- **Queries**: {summary_dict.get('n_queries')}",
        f"- **Training pool**: {summary_dict.get('n_training_pool')}",
        f"- **Meets ≥{summary_dict.get('min_queries_target', 20)} target**: "
        f"{'yes' if summary_dict.get('meets_query_target') else 'no'}",
        f"- **Top-1 accuracy**: {float(summary_dict.get('top1_accuracy', 0.0)):.2%}",
        f"- **Top-3 accuracy**: {float(summary_dict.get('top3_accuracy', 0.0)):.2%}",
        f"- **Mean selection regret**: {float(summary_dict.get('mean_selection_regret', float('nan'))):.4f}",
        f"- **Mean global-best regret**: {float(summary_dict.get('mean_global_best_regret', float('nan'))):.4f}",
        f"- **Beats global-best (mean regret)**: {summary_dict.get('beats_global_best')}",
        f"- **k-NN beats global rate**: {float(summary_dict.get('knn_beats_global_rate', 0.0)):.0%}",
        f"- **k-NN beats random rate**: {float(summary_dict.get('knn_beats_random_rate', 0.0)):.0%}",
        "",
        "| Held-out | Platform | Oracle | Selected | Conf | Top1 | Regret | Δ vs global |",
        "|----------|----------|--------|----------|------|------|--------|-------------|",
    ]
    for item in queries:
        row = item.to_dict() if isinstance(item, StudyGroupedQuery) else dict(item)
        conf = row.get("confidence")
        conf_s = f"{float(conf):.3f}" if conf is not None else "·"
        sel_r = row.get("selection_regret")
        glb_r = row.get("global_best_regret")
        delta = (
            f"{float(sel_r) - float(glb_r):+.4f}"
            if sel_r is not None and glb_r is not None
            else "·"
        )
        lines.append(
            f"| {row.get('held_out')} "
            f"| {row.get('platform') or '·'} "
            f"| {', '.join((row.get('oracle_methods') or [])[:2]) or '·'} "
            f"| {row.get('selected_method') or '·'} "
            f"| {conf_s} "
            f"| {'Y' if row.get('top1_hit') else '·'} "
            f"| {(float(sel_r) if sel_r is not None else float('nan')):.4f} "
            f"| {delta} |"
        )
    lines += ["", "## Caveats"] + [f"- {c}" for c in summary_dict.get("caveats") or []]
    return "\n".join(lines)


def write_protocol_bundle(
    output_dir: str | Path,
    *,
    study_queries: list[StudyGroupedQuery],
    study_summary: StudyGroupedSummary,
    selective: dict[str, Any],
    pareto_stability: dict[str, Any] | None = None,
    sota_resource: dict[str, Any] | None = None,
    oracle_k_leakage: dict[str, Any] | None = None,
    landscape_meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the full endpoint artefact bundle."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    study_path = output / "study_grouped_20_recommendation.json"
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL_STUDY,
        "queries": [q.to_dict() for q in study_queries],
        "summary": study_summary.to_dict(),
        "landscape": landscape_meta or {},
    }
    _atomic_json(study_path, payload)
    paths["study_grouped"] = study_path

    report_path = output / "study_grouped_20_report.md"
    report_path.write_text(
        format_study_grouped_report(study_summary, study_queries), encoding="utf-8"
    )
    paths["study_report"] = report_path

    selective_path = output / "selective_regret_coverage.json"
    _atomic_json(selective_path, selective)
    paths["selective"] = selective_path

    if pareto_stability is not None:
        pareto_path = output / "pareto_stability.json"
        _atomic_json(pareto_path, pareto_stability)
        paths["pareto_stability"] = pareto_path

    if sota_resource is not None:
        sota_path = output / "sota_unified_resource.json"
        _atomic_json(sota_path, sota_resource)
        paths["sota_resource"] = sota_path

    if oracle_k_leakage is not None:
        leak_path = output / "oracle_k_leakage.json"
        _atomic_json(leak_path, oracle_k_leakage)
        paths["oracle_k_leakage"] = leak_path

    master = {
        "schema_version": 1,
        "endpoints": {
            "study_grouped_personalisation": {
                "n_queries": study_summary.n_queries,
                "meets_target": study_summary.meets_query_target,
                "beats_global_best": study_summary.beats_global_best,
                "mean_selection_regret": study_summary.mean_selection_regret,
                "mean_global_best_regret": study_summary.mean_global_best_regret,
            },
            "selective_regret_coverage": {
                "recommended_threshold": selective.get("recommended_threshold"),
                "recommended_coverage": selective.get("recommended_coverage"),
                "recommended_hybrid_regret": selective.get("recommended_hybrid_regret"),
                "recommended_policy": selective.get("recommended_policy"),
            },
            "pareto_stability": {
                "n_datasets": (pareto_stability or {}).get("n_datasets"),
                "n_boot": (pareto_stability or {}).get("n_boot"),
            }
            if pareto_stability
            else None,
            "sota_unified_resource": {
                "n_accepted_cells": (sota_resource or {}).get("n_accepted_cells"),
                "top_method": ((sota_resource or {}).get("method_ranking") or [{}])[0].get("method")
                if sota_resource
                else None,
            }
            if sota_resource
            else None,
            "oracle_k_leakage": {
                "mean_ari_drop_across_methods": (oracle_k_leakage or {}).get(
                    "mean_ari_drop_across_methods"
                ),
                "primary_estimate_mode": (oracle_k_leakage or {}).get("primary_estimate_mode"),
                "methods": (oracle_k_leakage or {}).get("methods"),
                "source": (oracle_k_leakage or {}).get("source"),
            }
            if oracle_k_leakage
            else None,
        },
        "artifacts": {key: str(path.name) for key, path in paths.items()},
    }
    master_path = output / "protocol_endpoints_summary.json"
    _atomic_json(master_path, master)
    paths["summary"] = master_path

    md_lines = [
        "# Protocol endpoints summary",
        "",
        "Falsifiable evaluation bundle aligned with `docs/decision-protocol.md`.",
        "",
        "## Personalisation (study-grouped holdout)",
        "",
        f"- Queries: **{study_summary.n_queries}** "
        f"(target ≥{study_summary.min_queries_target}: "
        f"{'met' if study_summary.meets_query_target else 'not met'})",
        f"- Mean selection regret: **{study_summary.mean_selection_regret:.4f}**",
        f"- Mean global-best regret: **{study_summary.mean_global_best_regret:.4f}**",
        f"- Beats global-best: **{study_summary.beats_global_best}**",
        f"- Top-1 / Top-3: **{study_summary.top1_accuracy:.1%}** / "
        f"**{study_summary.top3_accuracy:.1%}**",
        "",
        "## Selective regret–coverage",
        "",
        f"- Recommended policy: **{selective.get('recommended_policy')}**",
        f"- Recommended confidence threshold: **{selective.get('recommended_threshold')}**",
        f"- Coverage at threshold: **{selective.get('recommended_coverage')}**",
        f"- Hybrid mean regret: **{selective.get('recommended_hybrid_regret')}**",
        "",
    ]
    if pareto_stability is not None:
        md_lines += [
            "## Pareto stability",
            "",
            f"- Datasets: **{pareto_stability.get('n_datasets')}**",
            f"- Bootstrap resamples: **{pareto_stability.get('n_boot')}**",
            "",
        ]
    if sota_resource is not None:
        ranking = sota_resource.get("method_ranking") or []
        top = ranking[0] if ranking else {}
        md_lines += [
            "## SOTA under unified resources",
            "",
            f"- Accepted cells: **{sota_resource.get('n_accepted_cells')}**",
            f"- Rejected cells: **{sota_resource.get('n_rejected_cells')}**",
            f"- Top method (mean ARI): **{top.get('method')}** "
            f"(ARI={top.get('mean_ari')}, s={top.get('mean_seconds')})",
            "",
        ]
    if oracle_k_leakage is not None:
        md_lines += [
            "## Oracle-K leakage (non-oracle K ARI drop)",
            "",
            f"- Protocol: `{oracle_k_leakage.get('protocol')}`",
            f"- Source: `{oracle_k_leakage.get('source')}`",
            f"- Primary estimate mode: **{oracle_k_leakage.get('primary_estimate_mode')}**",
            f"- Mean ARI drop (oracle − estimate) across methods: "
            f"**{oracle_k_leakage.get('mean_ari_drop_across_methods')}**",
            "",
            "| Method | Oracle ARI | Estimate ARI | Drop | Max slice drop |",
            "|--------|-----------:|-------------:|-----:|---------------:|",
        ]
        for method, stats in (oracle_k_leakage.get("by_method") or {}).items():
            md_lines.append(
                f"| {method} "
                f"| {stats.get('oracle_mean_ari')} "
                f"| {stats.get('primary_estimate_mean_ari')} "
                f"| {stats.get('mean_ari_drop_oracle_minus_estimate')} "
                f"| {stats.get('max_slice_drop')} |"
            )
        md_lines += [
            "",
            "Positive drop = Oracle-K inflated ARI relative to blind `k_policy=estimate`.",
            "Recovery of spatial estimators may be zero; dual-track still required.",
            "",
        ]
    for caveat in study_summary.caveats:
        md_lines.append(f"- {caveat}")
    md_path = output / "protocol_endpoints_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    paths["report"] = md_path
    return paths
