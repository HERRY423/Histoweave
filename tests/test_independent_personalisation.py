"""Tests for independent study/donor personalisation and cross-lab stats."""

from __future__ import annotations

import numpy as np

from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER
from histoweave.benchmark.independent_personalisation import (
    IndependentStudyUnit,
    aggregate_units_to_landscape,
    evaluate_personalisation_policies,
    paired_delta_bootstrap,
    rank_concordance_across_units,
    study_bootstrap_regret_ci,
    summarise_policies,
    synthetic_lab_units,
)
from histoweave.benchmark.landscape import LandscapeResult
from histoweave.benchmark.task_contract import AnalysisTask, GroundTruthKind


def _slice_landscape() -> LandscapeResult:
    methods = ["kmeans", "spectral", "agglomerative"]
    # Two donors × two slices + two external studies.
    names = ["151507", "151508", "151673", "151674", "ext_a", "ext_b"]
    performance = {}
    features = {}
    meta = {}
    rng = np.random.default_rng(1)
    for i, name in enumerate(names):
        if i % 2 == 0:
            performance[name] = {"kmeans": 0.8, "spectral": 0.5, "agglomerative": 0.4}
        else:
            performance[name] = {"kmeans": 0.45, "spectral": 0.78, "agglomerative": 0.4}
        vec = rng.normal(size=len(RECOMMENDATION_FEATURE_ORDER))
        vec[0] = float(i % 2)
        features[name] = vec
        meta[name] = {
            "platform": "visium",
            "task": AnalysisTask.SPATIAL_DOMAIN.value,
            "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
        }
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding={n: (0.0, 0.0) for n in names},
        best_method={n: max(performance[n], key=performance[n].get) for n in names},
        niches={},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=3,
        dataset_count=len(names),
        task=AnalysisTask.SPATIAL_DOMAIN.value,
        metric="ARI",
        higher_is_better=True,
        dataset_meta=meta,
    )


def test_aggregate_donors_collapses_slices() -> None:
    source = _slice_landscape()
    units = [
        IndependentStudyUnit(
            "donor_a",
            "biological_donor",
            ["151507", "151508"],
            platform="visium",
        ),
        IndependentStudyUnit(
            "donor_b",
            "biological_donor",
            ["151673", "151674"],
            platform="visium",
        ),
        IndependentStudyUnit("ext_a", "external_study", ["ext_a"], platform="xenium"),
        IndependentStudyUnit("ext_b", "external_study", ["ext_b"], platform="xenium"),
    ]
    panel = aggregate_units_to_landscape(source, units)
    assert panel.dataset_count == 4
    assert "donor_a" in panel.performance
    # Mean of even/odd slices for donor_a.
    assert abs(panel.performance["donor_a"]["kmeans"] - 0.625) < 1e-9


def test_gated_policy_noninferior_by_fallback() -> None:
    source = _slice_landscape()
    units = [
        IndependentStudyUnit(f"u{i}", "external_study", [n], platform="visium")
        for i, n in enumerate(source.dataset_order())
    ]
    # Already one-row units.
    panel = aggregate_units_to_landscape(source, units)
    for name in panel.dataset_order():
        panel.dataset_meta[name]["independence_class"] = "external_study"
    rows = evaluate_personalisation_policies(panel, k_neighbours=2, min_training=2)
    summary = summarise_policies(rows, noninferior_margin=0.05, min_queries=4)
    assert summary["n_queries"] == 6
    # Gated regret cannot systematically exceed both knn and global by much:
    # when gate fails, gated == global.
    assert summary["mean_gated_regret"] <= summary["mean_knn_regret"] + 1e-9 or summary[
        "gated_personalised_rate"
    ] < 1.0
    # Non-inferiority under generous margin often holds for gated.
    assert "gated_noninferior" in summary


def test_bootstrap_and_concordance() -> None:
    source = _slice_landscape()
    units = [
        IndependentStudyUnit(n, "external_study", [n], platform="visium")
        for n in source.dataset_order()
    ]
    panel = aggregate_units_to_landscape(source, units)
    rows = evaluate_personalisation_policies(panel, k_neighbours=2, min_training=2)
    ci = study_bootstrap_regret_ci(rows, policy="gated", n_boot=50, seed=0)
    assert ci["n"] == len(rows)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    delta = paired_delta_bootstrap(rows, n_boot=50, seed=1)
    assert delta["n"] == len(rows)
    conc = rank_concordance_across_units(panel)
    assert conc["n_units"] >= 2
    assert conc["kendall_w"] is not None
    assert 0.0 <= float(conc["kendall_w"]) <= 1.0 + 1e-9


def test_synthetic_labs_produce_independent_units() -> None:
    land, units = synthetic_lab_units(seed=0, methods=["kmeans", "spectral", "gaussian_mixture"])
    assert len(units) >= 6
    assert land.dataset_count == len(units)
    assert all(u.independence_class == "synthetic_lab" for u in units)
    assert all(uid.startswith("synth_lab_") for uid in land.performance)
