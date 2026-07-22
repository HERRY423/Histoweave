"""Contracts for the strict n=8 independent-unit validation artefacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_external_validation" / "n8_strict_region"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_n8_loocv_extends_query_count_without_overclaiming_superiority():
    summary = _load("loocv_summary.json")
    validation = _load("decision_validation.json")

    assert summary["n_queries"] >= 8
    assert summary["task_contract"]["ground_truth_kind"] == "spatial_domain"
    assert summary["task_contract"]["proxy_label_units_excluded"] is True
    assert summary["primary_noninferior"] is True
    assert summary["primary_superior"] is False
    assert summary["mean_gated_regret"] == summary["mean_global_best_regret"]
    assert validation["beats_global_best"] is False
    assert validation["noninferior_to_global_best"] is True


def test_tissue_flip_is_exploratory_and_fails_deployment_negative_control():
    result = _load("tissue_condition_flip.json")
    summaries = {row["condition"]: row for row in result["condition_summary"]}

    assert result["status"] == "exploratory_not_deployment_ready"
    assert summaries["human_cerebral_cortex"]["oracle_wins"] == {
        "gaussian_mixture": 2,
        "spectral": 1,
    }
    assert summaries["human_tumor"]["oracle_wins"] == {"spectral": 3}
    assert summaries["mouse_brain"]["oracle_wins"] == {"spectral": 2}
    assert result["within_condition_loocv_mean_regret"] > result["global_loocv_mean_regret"]
