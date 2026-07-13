"""Tests for study-grouped validation helpers that do not require network data."""

from __future__ import annotations

import json

import numpy as np

from histoweave.benchmark.landscape import LandscapeResult
from histoweave.benchmark.study_grouped import (
    STUDY_GROUPED_METHODS,
    _atomic_write,
    _f,
    _format_report,
    _subset_landscape,
    _training_means,
)


def _landscape() -> LandscapeResult:
    scores_a = {method: float(index) for index, method in enumerate(STUDY_GROUPED_METHODS)}
    scores_b = {method: float(index + 2) for index, method in enumerate(STUDY_GROUPED_METHODS)}
    scores_a[STUDY_GROUPED_METHODS[-1]] = float("nan")
    scores_b[STUDY_GROUPED_METHODS[-1]] = float("nan")
    return LandscapeResult(
        performance={"a": scores_a, "b": scores_b},
        features={"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
        embedding={"a": (0.1, 0.2)},
        best_method={"a": STUDY_GROUPED_METHODS[0]},
        niches={STUDY_GROUPED_METHODS[0]: ["a", "outside"]},
        timings={"a": {STUDY_GROUPED_METHODS[0]: 0.1}},
        feature_order=["f1", "f2"],
        method_count=len(STUDY_GROUPED_METHODS),
        dataset_count=2,
        task="domain_detection",
        metric="ari",
    )


def test_subset_and_training_means_are_stable() -> None:
    landscape = _landscape()
    subset = _subset_landscape(landscape, ["a", "b"])

    assert subset.dataset_count == 2
    assert subset.embedding["b"] == (0.0, 0.0)
    assert subset.best_method["b"] == "?"
    assert subset.niches[STUDY_GROUPED_METHODS[0]] == ["a"]
    assert subset.timings["b"] == {}
    assert subset.features["a"] is not landscape.features["a"]

    means = _training_means(subset, ["a", "b"])
    assert means[STUDY_GROUPED_METHODS[0]] == 1.0
    assert np.isnan(means[STUDY_GROUPED_METHODS[-1]])
    assert _f(None) is None
    assert _f(float("nan")) is None
    assert _f(np.float64(1.25)) == 1.25


def test_atomic_write_and_report_format(tmp_path) -> None:
    destination = tmp_path / "summary.json"
    _atomic_write(destination, {"status": "ok", "score": 0.9})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "ok",
        "score": 0.9,
    }
    assert list(tmp_path.iterdir()) == [destination]

    summary = {
        "n_queries": 1,
        "n_training_datasets": 4,
        "top1_accuracy": 1.0,
        "top3_accuracy": 1.0,
        "mean_selection_regret": 0.0,
        "knn_beats_global_rate": 1.0,
        "knn_beats_random_rate": 1.0,
        "caveats": ["External validation remains required."],
    }
    queries = [{
        "held_out": "a",
        "oracle_methods": ["m1", "m2", "m3"],
        "selected_method": "m1",
        "selected_score": 0.95,
        "top1_hit": True,
        "selection_regret": 0.0,
    }]
    report = _format_report(summary, queries)

    assert "# Study-Grouped Recommendation Validation" in report
    assert "| a | m1, m2 | m1 | 0.9500 | Y | 0.0000 |" in report
    assert "External validation remains required." in report
