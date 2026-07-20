"""Lock the dry-lab intercept case study: unjustified promotions must fail closed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from histoweave.benchmark import DecisionAction

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "case_study_intercepted_recommendation.py"


def _load_example_module():
    import sys

    name = "case_study_intercepted_recommendation"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses (and other PEP 563 tooling) require the module in sys.modules
    # before exec_module when loaded from a file path.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_intercept_scenarios_match_protocol_contract(tmp_path):
    mod = _load_example_module()
    results = mod.run_all_scenarios(tmp_path)
    by_id = {item.scenario_id: item for item in results}

    assert by_id["A"].action == DecisionAction.EVIDENCE_REQUIRED.value
    assert by_id["A"].primary_set == []
    assert any(
        c["name"] == "heldout_validation" and c["status"] == "not_evaluated"
        for c in by_id["A"].intercept_checks
    )

    assert by_id["B"].action == DecisionAction.GLOBAL_DEFAULT.value
    assert by_id["B"].primary_set  # global comparator retained
    assert any(
        c["name"] == "heldout_validation" and c["status"] == "fail"
        for c in by_id["B"].intercept_checks
    )

    assert by_id["C"].action == DecisionAction.ABSTAIN.value
    assert by_id["C"].primary_set == []
    assert any(
        c["name"] == "task_compatibility" and c["status"] == "fail"
        for c in by_id["C"].intercept_checks
    )

    assert by_id["D"].action != DecisionAction.PERSONALISED_SET.value
    d_card = by_id["D"].card
    neighbour_names = {n["name"] for n in d_card["recommendation"]["neighbours"]}
    assert "proxy_celltype" not in neighbour_names


def test_example_main_writes_report_and_json(tmp_path):
    mod = _load_example_module()
    rc = mod.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    md = tmp_path / "intercept_case_report.md"
    js = tmp_path / "intercept_case_cards.json"
    assert md.is_file()
    assert js.is_file()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["protocol"] == "histoweave.case_study.intercepted_recommendation.v1"
    assert payload["n_scenarios"] == 4
    actions = {row["scenario_id"]: row["action"] for row in payload["scenarios"]}
    assert actions["A"] == "evidence_required"
    assert actions["B"] == "global_default"
    assert actions["C"] == "abstain"
    assert actions["D"] != "personalised_set"
    text = md.read_text(encoding="utf-8")
    assert "intercepting unjustified" in text.lower()
    assert "cluster_proxy" in text or "Circular" in text
