"""Contracts for the task-stratified strict external panel v2."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_external_validation" / "strict_external_panel_v2"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_strict_panel_v2_rebuilds_and_preserves_claim_boundary() -> None:
    subprocess.run(
        [sys.executable, str(RESULTS / "build_strict_external_panel_v2.py")],
        cwd=ROOT,
        check=True,
    )
    summary = _load("loocv_summary.json")
    assert summary["n_registry_units"] == 10
    assert summary["n_domain_eligible_units"] == 9
    assert summary["n_queries"] == 9
    assert summary["primary_noninferior"] is True
    assert summary["primary_superior"] is False
    assert summary["mean_gated_regret"] == summary["mean_global_best_regret"]
    assert summary["task_contract"]["sota_missing_cells_imputed"] is False


def test_tls_second_dataset_is_reported_as_negative_transport() -> None:
    tls = _load("tls_two_dataset_summary.json")
    assert tls["n_independent_datasets"] == 2
    assert tls["cross_dataset_decision"] == "not_replicated"
    lymph = next(row for row in tls["datasets"] if row["unit_id"] == "xenium_human_lymph_node")
    assert lymph["decision"] == "not_replicated"
    assert lymph["pathology_gc_f1"] == 0.0
    assert lymph["neighbourhood_auc"] < 0.5


def test_sota_coverage_is_aligned_without_imputation() -> None:
    sota = _load("sota_coverage_summary.json")
    assert sota["missing_cells_imputed"] is False
    assert sota["confirmatory_loocv_inclusion"] == []
    assert sota["methods"]["banksy_py"]["available_units"] == 9
    assert sota["methods"]["banksy_py"]["strict_complete_units"] == 6
    for method in ("spagcn", "stagate", "graphst", "bayesspace"):
        assert sota["methods"][method]["available_units"] == 3

    with (RESULTS / "strict_external_units.csv").open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    assert len(registry) == 10
    assert sum(row["domain_loocv_eligible"] == "True" for row in registry) == 9
    assert sum(row["tls_evidence_eligible"] == "True" for row in registry) == 2
