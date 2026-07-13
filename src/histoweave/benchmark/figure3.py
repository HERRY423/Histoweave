"""Reproducible 10-method x 3-dataset Figure 3 benchmark experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..datasets.synthetic import make_benchmark_suite
from ..plugins import MethodCategory, list_methods
from .landscape import LandscapeResult, run_task_landscape
from .recommend import MethodRecommender

FIGURE3_DATASETS = ("clean_easy", "noisy_hard", "sparse_scattered")
FIGURE3_METHODS = (
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "dbscan",
    "gaussian_mixture",
    "kmeans",
    "mean_shift",
    "minibatch_kmeans",
    "optics",
    "spectral",
)
PROTOCOL_VERSION = "histoweave.figure3.synthetic.v1"


@dataclass(frozen=True)
class Figure3Result:
    """Paths and headline metrics from one Figure 3 experiment."""

    output_dir: Path
    landscape_path: Path
    performance_matrix_path: Path
    benchmark_long_path: Path
    features_path: Path
    recommendation_path: Path
    recommendation_csv_path: Path
    figure3_data_path: Path
    validation_path: Path
    manifest_path: Path
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def run_figure3_experiment(
    output_dir: str | Path,
    *,
    seed: int = 42,
    k_neighbours: int = 2,
) -> Figure3Result:
    """Run the full performance landscape and dataset-level LOOCV evaluation.

    The three datasets come from one deterministic synthetic generator and all
    contain three domains. The protocol therefore passes a fixed n_domains=3
    to methods that require it instead of deriving cluster count from query truth.
    """
    if k_neighbours < 1:
        raise ValueError("k_neighbours must be at least 1")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = _select_datasets(seed)
    method_metadata = _validate_methods()

    landscape = run_task_landscape(
        datasets,
        category=MethodCategory.DOMAIN_DETECTION,
        methods=list(FIGURE3_METHODS),
        extra_params_factory=lambda _data: {"n_domains": 3},
    )
    landscape.task = "domain_detection"
    landscape.metric = "ARI"
    landscape.higher_is_better = True

    landscape_path = output / "landscape.json"
    MethodRecommender(
        landscape,
        k_neighbours=min(k_neighbours, len(datasets)),
    ).save_knowledge_base(landscape_path)

    performance_matrix_path = output / "performance_matrix.csv"
    _write_performance_matrix(performance_matrix_path, landscape)

    benchmark_long_path = output / "benchmark_long.csv"
    benchmark_rows = _benchmark_rows(landscape)
    _write_csv(
        benchmark_long_path,
        benchmark_rows,
        (
            "dataset",
            "method",
            "score",
            "seconds",
            "status",
            "rank",
            "is_best",
        ),
    )

    features_path = output / "dataset_features.csv"
    _write_feature_matrix(features_path, landscape)

    recommendation_rows = _leave_one_dataset_out(
        datasets,
        landscape,
        k_neighbours=min(k_neighbours, len(datasets) - 1),
    )
    summary = _recommendation_summary(recommendation_rows)
    recommendation_path = output / "recommendation_loocv.json"
    _write_json_atomic(
        recommendation_path,
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "rows": recommendation_rows,
            "summary": summary,
        },
    )
    recommendation_csv_path = output / "recommendation_loocv.csv"
    _write_csv(
        recommendation_csv_path,
        _flatten_recommendations(recommendation_rows),
        (
            "held_out_dataset",
            "training_datasets",
            "oracle_methods",
            "oracle_score",
            "recommended_methods",
            "recommended_method",
            "recommended_score",
            "top1_hit",
            "top3_hit",
            "selection_regret",
            "global_best_baseline",
            "global_best_score",
            "global_best_regret",
            "random_expected_score",
            "random_expected_regret",
        ),
    )

    validation = _validate_outputs(landscape, benchmark_rows, recommendation_rows)
    validation_path = output / "validation.json"
    _write_json_atomic(validation_path, validation)

    figure3_data_path = output / "figure3_data.json"
    _write_json_atomic(
        figure3_data_path,
        {
            "schema_version": 1,
            "protocol": {
                "version": PROTOCOL_VERSION,
                "seed": seed,
                "task": "domain_detection",
                "metric": "ARI",
                "higher_is_better": True,
                "datasets": list(FIGURE3_DATASETS),
                "methods": list(FIGURE3_METHODS),
                "method_versions": method_metadata,
                "fixed_n_domains": 3,
                "recommendation_validation": "dataset-level leave-one-out",
                "k_neighbours": min(k_neighbours, len(datasets) - 1),
            },
            "performance_matrix": _json_safe(landscape.performance),
            "best_method": landscape.best_method,
            "embedding": {
                name: list(coordinates) for name, coordinates in landscape.embedding.items()
            },
            "recommendation": {
                "rows": recommendation_rows,
                "summary": summary,
            },
            "validation": validation,
            "limitations": [
                "All three datasets are synthetic and generated by one family.",
                "There are only three held-out recommendation queries.",
                "This validates software behavior, not real-data generalization.",
            ],
        },
    )

    manifest_path = output / "manifest.json"
    artifacts = [
        landscape_path,
        performance_matrix_path,
        benchmark_long_path,
        features_path,
        recommendation_path,
        recommendation_csv_path,
        figure3_data_path,
        validation_path,
    ]
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifacts
            ],
        },
    )

    return Figure3Result(
        output_dir=output.resolve(),
        landscape_path=landscape_path.resolve(),
        performance_matrix_path=performance_matrix_path.resolve(),
        benchmark_long_path=benchmark_long_path.resolve(),
        features_path=features_path.resolve(),
        recommendation_path=recommendation_path.resolve(),
        recommendation_csv_path=recommendation_csv_path.resolve(),
        figure3_data_path=figure3_data_path.resolve(),
        validation_path=validation_path.resolve(),
        manifest_path=manifest_path.resolve(),
        summary=summary,
    )


def _select_datasets(seed: int) -> dict[str, Any]:
    suite = make_benchmark_suite(seed=seed)
    datasets = {name: suite.datasets[name] for name in FIGURE3_DATASETS}
    for name, data in datasets.items():
        n_domains = int(data.obs["domain_truth"].nunique())
        if n_domains != 3:
            raise RuntimeError(f"{name} has {n_domains} domains; protocol requires 3")
    return datasets


def _validate_methods() -> dict[str, str]:
    registered = {
        item["name"]: item
        for item in list_methods(MethodCategory.DOMAIN_DETECTION)
        if item["language"] == "python"
    }
    missing = sorted(set(FIGURE3_METHODS) - set(registered))
    if missing:
        raise RuntimeError(f"Figure 3 methods are not registered: {missing}")
    if len(FIGURE3_METHODS) != 10 or len(set(FIGURE3_METHODS)) != 10:
        raise RuntimeError("Figure 3 protocol must contain exactly 10 unique methods")
    return {name: str(registered[name]["version"]) for name in FIGURE3_METHODS}


def _benchmark_rows(landscape: LandscapeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in FIGURE3_DATASETS:
        scores = landscape.performance[dataset]
        finite = {method: score for method, score in scores.items() if np.isfinite(score)}
        ordered = sorted(finite, key=lambda method: (-finite[method], method))
        ranks = {method: rank for rank, method in enumerate(ordered, 1)}
        for method in FIGURE3_METHODS:
            score = scores.get(method, float("nan"))
            seconds = landscape.timings.get(dataset, {}).get(method)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "score": _finite_or_none(score),
                    "seconds": _finite_or_none(seconds),
                    "status": "success" if np.isfinite(score) else "failed",
                    "rank": ranks.get(method),
                    "is_best": landscape.best_method.get(dataset) == method,
                }
            )
    return rows


def _leave_one_dataset_out(
    datasets: dict[str, Any],
    landscape: LandscapeResult,
    *,
    k_neighbours: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held_out in FIGURE3_DATASETS:
        training_names = [name for name in FIGURE3_DATASETS if name != held_out]
        training = _subset_landscape(landscape, training_names)
        recommendation = MethodRecommender(
            training,
            k_neighbours=min(k_neighbours, len(training_names)),
        ).recommend(datasets[held_out], dataset_name=held_out)
        recommended = [item.method for item in recommendation.ranked_methods]

        held_out_scores = {
            method: score
            for method, score in landscape.performance[held_out].items()
            if np.isfinite(score)
        }
        oracle_score = max(held_out_scores.values())
        oracle_methods = sorted(
            method
            for method, score in held_out_scores.items()
            if np.isclose(score, oracle_score, atol=1e-12, rtol=0.0)
        )
        selected = recommended[0] if recommended else None
        selected_score = (
            held_out_scores.get(selected, float("nan")) if selected is not None else float("nan")
        )

        training_means = {
            method: float(
                np.mean(
                    [
                        landscape.performance[name][method]
                        for name in training_names
                        if np.isfinite(landscape.performance[name].get(method, np.nan))
                    ]
                )
            )
            for method in FIGURE3_METHODS
            if any(
                np.isfinite(landscape.performance[name].get(method, np.nan))
                for name in training_names
            )
        }
        global_best = min(
            training_means,
            key=lambda method: (-training_means[method], method),
        )
        global_score = held_out_scores.get(global_best, float("nan"))
        random_expected_score = float(np.mean(list(held_out_scores.values())))

        rows.append(
            {
                "held_out_dataset": held_out,
                "training_datasets": training_names,
                "oracle_methods": oracle_methods,
                "oracle_score": oracle_score,
                "recommended_methods": recommended[:3],
                "recommended_method": selected,
                "recommended_score": _finite_or_none(selected_score),
                "top1_hit": selected in oracle_methods,
                "top3_hit": bool(set(recommended[:3]) & set(oracle_methods)),
                "selection_regret": _finite_or_none(oracle_score - selected_score),
                "global_best_baseline": global_best,
                "global_best_score": _finite_or_none(global_score),
                "global_best_regret": _finite_or_none(oracle_score - global_score),
                "random_expected_score": random_expected_score,
                "random_expected_regret": oracle_score - random_expected_score,
                "neighbours": recommendation.neighbours,
                "feature_order": recommendation.feature_order,
            }
        )
    return rows


def _subset_landscape(
    landscape: LandscapeResult,
    names: list[str],
) -> LandscapeResult:
    return LandscapeResult(
        performance={name: dict(landscape.performance[name]) for name in names},
        features={name: landscape.features[name].copy() for name in names},
        embedding={name: landscape.embedding[name] for name in names},
        best_method={name: landscape.best_method[name] for name in names},
        niches={
            method: [name for name in members if name in names]
            for method, members in landscape.niches.items()
        },
        timings={name: dict(landscape.timings[name]) for name in names},
        feature_order=list(landscape.feature_order),
        method_count=landscape.method_count,
        dataset_count=len(names),
        task=landscape.task,
        metric=landscape.metric,
        higher_is_better=landscape.higher_is_better,
    )


def _flatten_recommendations(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    excluded = {
        "training_datasets",
        "oracle_methods",
        "recommended_methods",
        "neighbours",
        "feature_order",
    }
    for row in rows:
        flattened.append(
            {
                **{key: value for key, value in row.items() if key not in excluded},
                "training_datasets": "|".join(row["training_datasets"]),
                "oracle_methods": "|".join(row["oracle_methods"]),
                "recommended_methods": "|".join(row["recommended_methods"]),
            }
        )
    return flattened


def _recommendation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [float(row["selection_regret"]) for row in rows]
    global_regrets = [float(row["global_best_regret"]) for row in rows]
    random_regrets = [float(row["random_expected_regret"]) for row in rows]
    mean_regret = float(np.mean(regrets))
    global_mean = float(np.mean(global_regrets))
    random_mean = float(np.mean(random_regrets))
    reduction_vs_random = 1.0 - mean_regret / random_mean if random_mean > 0 else None
    reduction_vs_global = 1.0 - mean_regret / global_mean if global_mean > 0 else None
    return {
        "n_queries": len(rows),
        "top1_accuracy": float(np.mean([row["top1_hit"] for row in rows])),
        "top3_accuracy": float(np.mean([row["top3_hit"] for row in rows])),
        "mean_selection_regret": mean_regret,
        "median_selection_regret": float(np.median(regrets)),
        "max_selection_regret": float(np.max(regrets)),
        "global_best_mean_regret": global_mean,
        "random_expected_mean_regret": random_mean,
        "regret_reduction_vs_global_best": reduction_vs_global,
        "regret_reduction_vs_random": reduction_vs_random,
    }


def _validate_outputs(
    landscape: LandscapeResult,
    benchmark_rows: list[dict[str, Any]],
    recommendation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "dataset_count_is_3": landscape.dataset_count == 3,
        "method_count_is_10": landscape.method_count == 10,
        "matrix_has_30_cells": len(benchmark_rows) == 30,
        "all_methods_succeeded": all(row["status"] == "success" for row in benchmark_rows),
        "loocv_has_3_queries": len(recommendation_rows) == 3,
        "all_recommendations_nonempty": all(
            bool(row["recommended_method"]) for row in recommendation_rows
        ),
        "regret_is_nonnegative": all(
            float(row["selection_regret"]) >= -1e-12 for row in recommendation_rows
        ),
    }
    return {
        "status": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "successful_benchmark_cells": sum(row["status"] == "success" for row in benchmark_rows),
        "total_benchmark_cells": len(benchmark_rows),
        "scope": "synthetic software and algorithm validation",
        "required_caveats": [
            "Three queries are insufficient for a precise accuracy estimate.",
            "All datasets share one synthetic generator family.",
            "Equal regret versus global-best does not show incremental k-NN value.",
            "Real-data study-grouped validation is required for a paper claim.",
        ],
    }


def _write_performance_matrix(path: Path, landscape: LandscapeResult) -> None:
    rows = [
        {
            "dataset": dataset,
            **{
                method: _finite_or_none(landscape.performance[dataset].get(method))
                for method in FIGURE3_METHODS
            },
        }
        for dataset in FIGURE3_DATASETS
    ]
    _write_csv(path, rows, ("dataset", *FIGURE3_METHODS))


def _write_feature_matrix(path: Path, landscape: LandscapeResult) -> None:
    rows = [
        {
            "dataset": dataset,
            **{
                feature: _finite_or_none(landscape.features[dataset][index])
                for index, feature in enumerate(landscape.feature_order)
            },
        }
        for dataset in FIGURE3_DATASETS
    ]
    _write_csv(path, rows, ("dataset", *landscape.feature_order))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(_json_safe(value), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
