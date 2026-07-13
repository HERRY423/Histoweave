"""Study-grouped holdout validation for real-data recommendation.

The 3-query synthetic LOOCV cannot support a generalisation claim.  This module
implements a study-grouped protocol: hold out one *study* (e.g. an entire DLPFC
slice or all slices from one tissue type), train on the rest, and evaluate
whether the recommendation engine:

1. Beats the global-best baseline (train-set mean) — proving k-NN adds value.
2. Beats random selection — proving the feature space is informative.
3. Maintains low regret compared to the oracle on the held-out group.

A valid result requires **≥5 training datasets** with heterogeneous data types
(synthetic + brain + tumour + ...) so that neighbour selection exercises real
discrimination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..data import SpatialTable
from .figure3 import FIGURE3_DATASETS, _select_datasets, _validate_methods
from .landscape import LandscapeResult, run_task_landscape
from .real_data import _DLPFC_SLICES, dlpfc_benchmark_suite
from .recommend import MethodRecommender

_BENCHMARK_SLICES: tuple[str, ...] = tuple(_DLPFC_SLICES)

STUDY_GROUPED_METHODS = (
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "gaussian_mixture",
    "kmeans",
    "minibatch_kmeans",
    "spectral",
)
STUDY_GROUPED_SYNTHETIC = FIGURE3_DATASETS


@dataclass
class StudyGroupedResult:
    output_dir: Path
    landscape_path: Path
    recommendation_path: Path
    report_path: Path
    summary: dict[str, Any]
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def run_study_grouped_validation(
    output_dir: str | Path,
    *,
    seed: int = 42,
    k_neighbours: int = 3,
    include_dlpfc: bool = True,
    dlpfc_slices: tuple[str, ...] = _BENCHMARK_SLICES,
) -> StudyGroupedResult:
    """Run the study-grouped benchmark and hold-one-slice-out validation.

    Builds a multi-dataset landscape containing:

    * 3 synthetic datasets (clean_easy, noisy_hard, sparse_scattered)
    * 3 real DLPFC slices (151507, 151508, 151509)

    Then holds out each DLPFC slice in turn and runs recommendation from
    the remaining 5 datasets, evaluating:

    * **top-1 / top-3 accuracy** (oracle in top-k?)
    * **selection regret** (ARI loss vs oracle on held-out)
    * **k-NN vs global-best delta** (does the recommender beat the
      training-set mean recommendation?)

    Parameters
    ----------
    output_dir : Path
        Directory for output artefacts.
    seed : int
        Reproducibility seed.
    k_neighbours : int
        k for the k-NN recommender (default 3).
    include_dlpfc : bool
        Whether to include real DLPFC data.
    dlpfc_slices : tuple of str
        Which DLPFC slices to use.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ---- 1. Assemble datasets ----------------------------------------------
    datasets: dict[str, SpatialTable] = {}
    synthetic = _select_datasets(seed)
    datasets.update(synthetic)

    if include_dlpfc:
        dlpfc = dlpfc_benchmark_suite(dlpfc_slices, seed=seed)
        datasets.update(dlpfc)

    method_metadata = _validate_methods_study()

    # ---- 2. Build full landscape ------------------------------------------
    from ..plugins import MethodCategory

    landscape = run_task_landscape(
        datasets,
        category=MethodCategory.DOMAIN_DETECTION,
        methods=list(STUDY_GROUPED_METHODS),
        extra_params_factory=lambda _data: {"n_domains": 7, "random_state": seed},
    )
    landscape.task = "domain_detection"
    landscape.metric = "ARI"
    landscape.higher_is_better = True

    landscape_path = output / "study_grouped_landscape.json"
    MethodRecommender(
        landscape, k_neighbours=min(k_neighbours, len(datasets) - 1)
    ).save_knowledge_base(landscape_path)

    # ---- 3. Study-grouped leave-one-slice-out ------------------------------
    per_query: list[dict[str, Any]] = []
    dlpfc_keys = [k for k in datasets if k.startswith("dlpfc_")]

    for held_out_key in dlpfc_keys:
        training_keys = [k for k in datasets if k != held_out_key]
        training = _subset_landscape(landscape, training_keys)

        recommender = MethodRecommender(
            training,
            k_neighbours=min(k_neighbours, len(training_keys)),
        )
        recommendation = recommender.recommend(datasets[held_out_key], dataset_name=held_out_key)

        held_out_scores = {
            m: s for m, s in landscape.performance[held_out_key].items() if np.isfinite(s)
        }
        oracle_score = max(held_out_scores.values())
        oracle_methods = sorted(
            m
            for m, s in held_out_scores.items()
            if np.isclose(s, oracle_score, atol=1e-12, rtol=0.0)
        )
        recommended = [item.method for item in recommendation.ranked_methods]
        selected = recommended[0] if recommended else None
        selected_score = (
            held_out_scores.get(selected, float("nan")) if selected is not None else float("nan")
        )

        # Global-best baseline: method with highest mean score on training sets
        training_means = _training_means(landscape, training_keys)
        global_best = max(training_means, key=lambda m: (training_means[m], m))
        global_score = held_out_scores.get(global_best, float("nan"))

        # Random baseline: mean of all held-out scores
        random_score = float(np.mean(list(held_out_scores.values())))

        query = {
            "held_out": held_out_key,
            "training_sets": training_keys,
            "oracle_methods": oracle_methods,
            "oracle_score": oracle_score,
            "recommended_methods": recommended[:3],
            "selected_method": selected,
            "selected_score": _f(selected_score),
            "top1_hit": selected in oracle_methods,
            "top3_hit": bool(set(recommended[:3]) & set(oracle_methods)),
            "selection_regret": _f(oracle_score - selected_score),
            "global_best_method": global_best,
            "global_best_score": _f(global_score),
            "global_best_regret": _f(oracle_score - global_score),
            "knn_beats_global": (selected_score >= global_score - 1e-9)
            if selected is not None
            else False,
            "random_score": _f(random_score),
            "knn_beats_random": selected_score > random_score if selected else False,
            "neighbours": recommendation.neighbours,
        }
        per_query.append(query)

    # ---- 4. Summary -------------------------------------------------------
    regrets = [q["selection_regret"] for q in per_query]
    global_regrets = [q["global_best_regret"] for q in per_query]
    top1 = [q["top1_hit"] for q in per_query]
    top3 = [q["top3_hit"] for q in per_query]

    summary = {
        "n_queries": len(per_query),
        "n_training_datasets": len(datasets),
        "dlpfc_slices": list(dlpfc_slices),
        "top1_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "mean_selection_regret": float(np.mean(regrets)),
        "max_selection_regret": float(np.max(regrets)),
        "mean_global_best_regret": float(np.mean(global_regrets)),
        "knn_beats_global_rate": float(np.mean([q["knn_beats_global"] for q in per_query])),
        "knn_beats_random_rate": float(np.mean([q["knn_beats_random"] for q in per_query])),
        "caveats": [
            f"Only {len(dlpfc_keys)} real-data queries — insufficient for precision.",
            "DLPFC slices share donor, platform, and tissue — not cross-study.",
            "Ground truth is GMM-derived, not manual layer annotation.",
            "Requires {>=5} heterogeneous training datasets for k-NN discrimination.",
        ]
        if len(per_query) < 5
        else ["Multi-study validation supports generalisation claims."],
    }

    # ---- 5. Write outputs -------------------------------------------------
    recommendation_path = output / "study_grouped_recommendation.json"
    _atomic_write(
        recommendation_path,
        {
            "schema_version": 1,
            "protocol": "study_grouped_holdout",
            "queries": per_query,
            "summary": summary,
            "method_versions": method_metadata,
        },
    )

    report_path = output / "study_grouped_report.md"
    report_path.write_text(_format_report(summary, per_query), encoding="utf-8")

    return StudyGroupedResult(
        output_dir=output.resolve(),
        landscape_path=landscape_path.resolve(),
        recommendation_path=recommendation_path.resolve(),
        report_path=report_path.resolve(),
        summary=summary,
        per_query=per_query,
    )


def _validate_methods_study() -> dict[str, str]:
    registered = _validate_methods()  # reuse figure3 validation
    missing = sorted(set(STUDY_GROUPED_METHODS) - set(registered))
    if missing:
        raise RuntimeError(f"Study-grouped methods not registered: {missing}")
    return {name: str(registered.get(name, "0.1.0")) for name in STUDY_GROUPED_METHODS}


def _subset_landscape(landscape: LandscapeResult, names: list[str]) -> LandscapeResult:
    return LandscapeResult(
        performance={n: dict(landscape.performance[n]) for n in names},
        features={n: landscape.features[n].copy() for n in names},
        embedding={n: landscape.embedding.get(n, (0.0, 0.0)) for n in names},
        best_method={n: landscape.best_method.get(n, "?") for n in names},
        niches={m: [n for n in members if n in names] for m, members in landscape.niches.items()},
        timings={n: dict(landscape.timings.get(n, {})) for n in names},
        feature_order=list(landscape.feature_order),
        method_count=landscape.method_count,
        dataset_count=len(names),
        task=landscape.task,
        metric=landscape.metric,
        higher_is_better=landscape.higher_is_better,
    )


def _training_means(landscape: LandscapeResult, training_keys: list[str]) -> dict[str, float]:
    means: dict[str, float] = {}
    for method in STUDY_GROUPED_METHODS:
        values = [landscape.performance[k].get(method, float("nan")) for k in training_keys]
        finite = [v for v in values if np.isfinite(v)]
        means[method] = float(np.mean(finite)) if finite else float("nan")
    return means


def _f(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _atomic_write(path, value):
    import json as _json

    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        tmp.write_text(_json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _format_report(summary, queries) -> str:
    lines = [
        "# Study-Grouped Recommendation Validation",
        "",
        f"- **Queries**: {summary['n_queries']}",
        f"- **Training datasets**: {summary['n_training_datasets']}",
        f"- **Top-1 accuracy**: {summary['top1_accuracy']:.2%}",
        f"- **Top-3 accuracy**: {summary['top3_accuracy']:.2%}",
        f"- **Mean selection regret**: {summary['mean_selection_regret']:.4f}",
        f"- **k-NN beats global-best**: {summary['knn_beats_global_rate']:.0%}",
        f"- **k-NN beats random**: {summary['knn_beats_random_rate']:.0%}",
        "",
        "| Held-out | Oracle | Recommended | Score | Top1 | Regret |",
        "|----------|--------|-------------|-------|------|--------|",
    ]
    for q in queries:
        lines.append(
            f"| {q['held_out']} | {', '.join(q['oracle_methods'][:2])} "
            f"| {q['selected_method']} "
            f"| {q['selected_score']:.4f} "
            f"| {'Y' if q['top1_hit'] else '·'} "
            f"| {q['selection_regret']:.4f} |"
        )
    lines += [
        "",
        "## Caveats",
    ] + [f"- {c}" for c in summary["caveats"]]
    return "\n".join(lines)
