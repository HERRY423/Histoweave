from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_external_validation" / "independent_test_wu2021"


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_protocol_was_frozen_as_a_one_shot_independent_test() -> None:
    protocol = _json("preregistered_protocol.json")
    assert protocol["status_at_lock"] == "outcomes_not_downloaded_or_inspected"
    assert protocol["frozen_policy"]["selected_method"] == "spectral"
    assert protocol["frozen_policy"]["personalisation_enabled"] is False
    assert protocol["confirmatory_endpoint"]["success_margin_ari"] == 0.02
    assert protocol["anti_leakage"]["test_data_may_enter_training_after_this_evaluation"] is False
    assert protocol["anti_leakage"]["test_outcomes_may_change_method_or_threshold"] is False


def test_independent_result_retains_the_negative_decision() -> None:
    summary = _json("independent_test_summary.json")
    assert summary["n_evaluable_sections"] == 6
    assert summary["frozen_policy"] == "spectral"
    assert summary["mean_frozen_policy_regret"] > summary["success_margin_ari"]
    assert summary["success"] is False
    assert summary["decision"] == "independent_test_fail"
    assert summary["top1_frequency"] == 2 / 6
    assert summary["bootstrap_ci"]["low"] > summary["success_margin_ari"]


def test_all_six_sections_and_independence_audit_are_present() -> None:
    with (RESULTS / "sample_regret.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["frozen_method"] for row in rows} == {"spectral"}
    assert sum(row["frozen_top1"] == "True" for row in rows) == 2

    audit = _json("independence_audit.json")
    assert audit["all_identifier_matches_empty"] is True
    assert audit["study_level_independence"] is True
    assert all(
        not matches for matches in audit["test_identifiers_found_in_training_sources"].values()
    )


def test_reproducible_outputs_are_complete() -> None:
    required = {
        "benchmark_long.csv",
        "dataset_manifest.json",
        "fig_independent_test_wu2021.png",
        "fig_independent_test_wu2021.svg",
        "independent_test_summary.json",
        "manifest.json",
        "REPORT_independent_test_wu2021.md",
        "run_independent_test.py",
    }
    assert all((RESULTS / name).is_file() for name in required)
