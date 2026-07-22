"""Unit tests for falsifiable protocol evaluation endpoints."""

from __future__ import annotations

import csv
import json

import numpy as np

from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER
from histoweave.benchmark.landscape import LandscapeResult
from histoweave.benchmark.protocol_endpoints import (
    leave_one_study_out,
    oracle_k_leakage_impact,
    pareto_stability_from_long_csv,
    selective_regret_coverage,
    sota_unified_resource_compare,
    summarise_study_grouped,
    write_protocol_bundle,
)
from histoweave.benchmark.task_contract import AnalysisTask, GroundTruthKind


def _toy_landscape(n: int = 6) -> LandscapeResult:
    methods = ["kmeans", "spectral", "agglomerative"]
    performance: dict[str, dict[str, float]] = {}
    features: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        name = f"study_{i:02d}"
        # Make kmeans best on even, spectral on odd — personalisation can win.
        if i % 2 == 0:
            performance[name] = {
                "kmeans": 0.80 + 0.01 * i,
                "spectral": 0.50,
                "agglomerative": 0.40,
            }
        else:
            performance[name] = {
                "kmeans": 0.45,
                "spectral": 0.75 + 0.01 * i,
                "agglomerative": 0.40,
            }
        vec = rng.normal(size=len(RECOMMENDATION_FEATURE_ORDER))
        # Cluster even/odd studies in feature space.
        vec[0] = float(i % 2)
        features[name] = vec
        meta[name] = {
            "platform": "visium" if i % 2 == 0 else "xenium",
            "task": AnalysisTask.SPATIAL_DOMAIN.value,
            "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
            "study_group": name,
        }
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding={name: (float(i), 0.0) for i, name in enumerate(performance)},
        best_method={name: max(row, key=lambda m: row[m]) for name, row in performance.items()},
        niches={},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=len(methods),
        dataset_count=n,
        task=AnalysisTask.SPATIAL_DOMAIN.value,
        metric="ARI",
        higher_is_better=True,
        dataset_meta=meta,
    )


def test_leave_one_study_out_produces_queries_and_summary() -> None:
    landscape = _toy_landscape(8)
    queries, summary = leave_one_study_out(landscape, k_neighbours=2, min_training=3)
    assert summary.n_queries == 8
    assert len(queries) == 8
    assert 0.0 <= summary.top1_accuracy <= 1.0
    assert all(q.selection_regret is not None for q in queries)
    assert all(q.confidence is not None for q in queries)
    payload = summary.to_dict()
    assert payload["protocol"] == "study_grouped_holdout"


def test_selective_regret_coverage_curve() -> None:
    landscape = _toy_landscape(6)
    queries, _ = leave_one_study_out(landscape, k_neighbours=2, min_training=2)
    curve = selective_regret_coverage(queries, thresholds=(0.0, 0.5, 0.99))
    assert curve["n_queries"] == len(queries)
    # Three requested thresholds + pure global-default sentinel.
    assert len(curve["curve"]) == 4
    # Full coverage at threshold 0.
    assert curve["curve"][0]["coverage"] == 1.0
    assert curve["curve"][0]["mean_regret_accepted"] is not None
    assert curve["curve"][-1]["label"] == "always_global_default"
    assert curve["curve"][-1]["coverage"] == 0.0


def test_pareto_stability_from_long_csv(tmp_path) -> None:
    path = tmp_path / "long.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "config", "seed", "ari", "seconds", "status"]
        )
        writer.writeheader()
        for seed in (1, 2, 3):
            writer.writerow(
                {
                    "dataset": "d1",
                    "config": "kmeans@sw0.0",
                    "seed": seed,
                    "ari": 0.7 + 0.01 * seed,
                    "seconds": 1.0,
                    "status": "success",
                }
            )
            writer.writerow(
                {
                    "dataset": "d1",
                    "config": "spectral@sw0.8",
                    "seed": seed,
                    "ari": 0.6,
                    "seconds": 5.0 + seed,
                    "status": "success",
                }
            )
    report = pareto_stability_from_long_csv(path, n_boot=50, seed=1)
    assert report["n_datasets"] == 1
    assert "kmeans@sw0.0" in report["datasets"]["d1"]["inclusion_probability"]
    assert report["datasets"]["d1"]["point_frontier"]


def test_sota_unified_resource_filter(tmp_path) -> None:
    path = tmp_path / "sota.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "method", "seed", "ari", "seconds", "status", "family"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "151673",
                "method": "spagcn",
                "seed": 1,
                "ari": 0.4,
                "seconds": 10,
                "status": "success",
                "family": "sota",
            }
        )
        writer.writerow(
            {
                "dataset": "151673",
                "method": "graphst",
                "seed": 1,
                "ari": 0.9,
                "seconds": 9999,
                "status": "success",
                "family": "sota",
            }
        )
        writer.writerow(
            {
                "dataset": "151673",
                "method": "stagate",
                "seed": 1,
                "ari": "",
                "seconds": 1,
                "status": "failed",
                "family": "sota",
            }
        )
    report = sota_unified_resource_compare(path, max_seconds=100.0)
    assert report["n_accepted_cells"] == 1
    assert report["n_rejected_cells"] == 2
    assert report["method_ranking"][0]["method"] == "spagcn"


def test_oracle_k_leakage_impact_from_dual_track_csv(tmp_path) -> None:
    path = tmp_path / "dual.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "seed",
                "mode",
                "k_used",
                "oracle_k",
                "k_match",
                "ari",
                "status",
            ],
        )
        writer.writeheader()
        # SpaGCN: oracle 0.4, silhouette 0.2 → drop 0.2; ensemble 0.25 recovers +0.05
        for mode, ari, k in (
            ("oracle", 0.40, 7),
            ("estimate:silhouette", 0.20, 2),
            ("estimate:ensemble", 0.25, 3),
        ):
            writer.writerow(
                {
                    "dataset": "151673",
                    "method": "spagcn",
                    "seed": 42,
                    "mode": mode,
                    "k_used": k,
                    "oracle_k": 7,
                    "k_match": mode == "oracle",
                    "ari": ari,
                    "status": "success",
                }
            )
        writer.writerow(
            {
                "dataset": "151674",
                "method": "spagcn",
                "seed": 42,
                "mode": "oracle",
                "k_used": 7,
                "oracle_k": 7,
                "k_match": True,
                "ari": 0.30,
                "status": "success",
            }
        )
        writer.writerow(
            {
                "dataset": "151674",
                "method": "spagcn",
                "seed": 42,
                "mode": "estimate:silhouette",
                "k_used": 2,
                "oracle_k": 7,
                "k_match": False,
                "ari": 0.18,
                "status": "success",
            }
        )
    report = oracle_k_leakage_impact(path)
    assert report["protocol"] == "histoweave.oracle_k_leakage.v1"
    sp = report["by_method"]["spagcn"]
    # Means: oracle (0.4+0.3)/2=0.35; silhouette (0.2+0.18)/2=0.19; drop=0.16
    assert abs(sp["mean_ari_drop_oracle_minus_estimate"] - 0.16) < 1e-9
    assert sp["max_slice_drop"] is not None and sp["max_slice_drop"] > 0.1
    ens = sp["recovery_vs_primary_estimate"]["estimate:ensemble"]
    assert ens["ari_recovered_vs_primary_estimate"] > 0


def test_write_protocol_bundle(tmp_path) -> None:
    landscape = _toy_landscape(5)
    queries, summary = leave_one_study_out(landscape, k_neighbours=2, min_training=2)
    # Force meets_query_target check path via summarise
    summary = summarise_study_grouped(
        queries,
        n_training_pool=5,
        methods=["kmeans", "spectral", "agglomerative"],
        min_queries_target=20,
    )
    selective = selective_regret_coverage(queries)
    leak = {
        "protocol": "histoweave.oracle_k_leakage.v1",
        "source": "non_oracle_k_sota/benchmark_long.csv",
        "primary_estimate_mode": "estimate:silhouette",
        "methods": ["spagcn"],
        "mean_ari_drop_across_methods": 0.06,
        "by_method": {
            "spagcn": {
                "oracle_mean_ari": 0.30,
                "primary_estimate_mean_ari": 0.24,
                "mean_ari_drop_oracle_minus_estimate": 0.06,
                "max_slice_drop": 0.23,
            }
        },
    }
    paths = write_protocol_bundle(
        tmp_path,
        study_queries=queries,
        study_summary=summary,
        selective=selective,
        pareto_stability={"protocol": "x", "n_datasets": 1, "n_boot": 10},
        sota_resource={
            "n_accepted_cells": 1,
            "n_rejected_cells": 0,
            "method_ranking": [{"method": "spagcn", "mean_ari": 0.3, "mean_seconds": 1.0}],
        },
        oracle_k_leakage=leak,
    )
    assert paths["study_grouped"].is_file()
    assert paths["oracle_k_leakage"].is_file()
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["endpoints"]["study_grouped_personalisation"]["n_queries"] == len(queries)
    assert payload["endpoints"]["oracle_k_leakage"]["mean_ari_drop_across_methods"] == 0.06
    report = paths["report"].read_text(encoding="utf-8")
    assert "Protocol endpoints summary" in report
    assert "Oracle-K leakage" in report
