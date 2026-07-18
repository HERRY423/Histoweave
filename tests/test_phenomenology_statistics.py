from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from histoweave.benchmark.phenomenology_statistics import (
    benjamini_hochberg,
    capability_index,
    coverage_summary,
    paired_bootstrap_ci,
    paired_method_comparisons,
)


def _tables():
    runs = []
    metrics = []
    run_number = 0
    for method, base in [("a", 0.8), ("b", 0.6)]:
        for replicate in range(5):
            for condition, penalty in [("clean", 0.0), ("low_signal_noise", 0.2)]:
                run_id = f"run-{run_number}"
                run_number += 1
                runs.append(
                    {
                        "run_id": run_id,
                        "category": "domain_detection",
                        "method": method,
                        "version": "1.0",
                        "role": "direct_inference",
                        "track": "locked",
                        "phenomenon": "compartment",
                        "condition": condition,
                        "replicate": replicate,
                        "applicability": True,
                        "status": "ok",
                        "seconds": 1.0 if method == "a" else 2.0,
                        "peak_rss_mb": 100.0 if method == "a" else 200.0,
                    }
                )
                metrics.append(
                    {
                        "run_id": run_id,
                        "metric": "phenomenon_recovery",
                        "value": base - penalty,
                        "direction": "maximize",
                        "primary": True,
                        "normalized_value": base - penalty,
                    }
                )
    return pd.DataFrame(runs), pd.DataFrame(metrics)


def test_coverage_separates_na_environment_and_scientific_failure() -> None:
    runs, _ = _tables()
    extras = pd.DataFrame(
        [
            {
                **runs.iloc[0].to_dict(),
                "run_id": "na",
                "applicability": False,
                "status": "not_applicable",
            },
            {**runs.iloc[0].to_dict(), "run_id": "env", "status": "backend_unavailable"},
            {**runs.iloc[0].to_dict(), "run_id": "fail", "status": "method_error"},
        ]
    )
    summary = coverage_summary(pd.concat([runs, extras], ignore_index=True))
    row = summary.loc[summary["method"] == "a"].iloc[0]
    assert row["environment_gap_units"] == 1
    assert row["scientific_failure_units"] == 1
    assert row["applicability_coverage"] < 1.0
    assert row["execution_coverage"] < 1.0


def test_capability_index_uses_science_first_weights_without_rank() -> None:
    runs, metrics = _tables()
    summary = capability_index(runs, metrics)
    assert "rank" not in summary.columns
    assert set(summary["method"]) == {"a", "b"}
    a = summary.loc[summary["method"] == "a"].iloc[0]
    assert a["recovery_score"] == pytest.approx(0.8)
    assert a["robustness_score"] == pytest.approx(0.8)
    assert a["reliability_score"] == pytest.approx(1.0)
    assert 0 <= a["capability_index"] <= 1


def test_scientific_failure_scores_zero_but_backend_gap_is_excluded() -> None:
    runs, metrics = _tables()
    a_clean = runs.index[(runs["method"] == "a") & (runs["condition"] == "clean")]
    failed_index = a_clean[0]
    backend_index = a_clean[1]
    runs.loc[failed_index, "status"] = "method_error"
    runs.loc[backend_index, "status"] = "backend_unavailable"
    metrics = metrics.loc[
        ~metrics["run_id"].isin(
            [runs.loc[failed_index, "run_id"], runs.loc[backend_index, "run_id"]]
        )
    ]
    summary = capability_index(runs, metrics)
    a = summary.loc[summary["method"] == "a"].iloc[0]
    assert a["recovery_score"] == pytest.approx(0.6)
    assert a["scientific_runs"] == 9


def test_paired_bootstrap_is_reproducible_and_keeps_replicates() -> None:
    runs, metrics = _tables()
    frame = runs.merge(metrics, on="run_id")
    first = paired_bootstrap_ci(frame, n_resamples=500, seed=5)
    second = paired_bootstrap_ci(frame, n_resamples=500, seed=5)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["n_replicates"]) == {5}
    assert np.all(first["ci_low"] <= first["mean"])
    assert np.all(first["mean"] <= first["ci_high"])


def test_bh_is_monotone_in_rank_and_comparison_stays_within_family() -> None:
    q_values = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])
    assert np.all((0 <= q_values) & (q_values <= 1))
    assert q_values[0] <= q_values[2] <= q_values[1] <= q_values[3]

    runs, metrics = _tables()
    frame = runs.merge(metrics, on="run_id")
    comparisons = paired_method_comparisons(frame)
    assert len(comparisons) == 1
    row = comparisons.iloc[0]
    assert {row["method_left"], row["method_right"]} == {"a", "b"}
    assert row["n_pairs"] == 10
    assert 0 <= row["q_value"] <= 1
