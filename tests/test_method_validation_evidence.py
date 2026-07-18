"""Gates for multi-dataset method validation evidence packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from histoweave.plugins import list_methods
from histoweave.plugins.builtin.release_manifest import (
    CONTRACT_VALIDATED_METHODS,
    MULTI_DATASET_EVIDENCE_METHODS,
    SCIENTIFIC_VALIDATED_METHODS,
    VALIDATED_METHODS,
    VALIDATION_EVIDENCE,
)

ROOT = Path(__file__).resolve().parents[1]
VAL_DOCS = ROOT / "docs" / "methods" / "validation"
EXPANSION_BATCH = {
    "agglomerative",
    "birch",
    "minibatch_kmeans",
    "banksy",
    "cell2location",
}

SOTA_BATCH = {
    "spagcn",
    "graphst",
    "stagate",
    "rctd",
    "spatialde",
}

ALL_EXPANSION = EXPANSION_BATCH | SOTA_BATCH
CONTRACT_BATCH = {"cell2location", "rctd", "spatialde"}
SCIENTIFIC_EXPANSION = ALL_EXPANSION - CONTRACT_BATCH


def test_validation_evidence_covers_expansion_batch():
    missing = ALL_EXPANSION - set(VALIDATION_EVIDENCE)
    assert not missing, f"VALIDATION_EVIDENCE missing expansion methods: {sorted(missing)}"


def test_ledger_unifies_10_scientific_3_contract_13_total():
    assert len(SCIENTIFIC_VALIDATED_METHODS) == 10
    assert len(CONTRACT_VALIDATED_METHODS) == 3
    assert len(MULTI_DATASET_EVIDENCE_METHODS) == 13
    assert VALIDATED_METHODS == SCIENTIFIC_VALIDATED_METHODS
    assert VALIDATED_METHODS | CONTRACT_VALIDATED_METHODS == MULTI_DATASET_EVIDENCE_METHODS
    assert set(VALIDATION_EVIDENCE) == MULTI_DATASET_EVIDENCE_METHODS
    for name in SCIENTIFIC_VALIDATED_METHODS:
        assert VALIDATION_EVIDENCE[name]["kind"] == "scientific"
    for name in CONTRACT_VALIDATED_METHODS:
        assert VALIDATION_EVIDENCE[name]["kind"] == "contract"


def test_scientific_methods_have_maturity_validated():
    registered = {m["name"]: m for m in list_methods()}
    for name in SCIENTIFIC_EXPANSION | (SCIENTIFIC_VALIDATED_METHODS - SCIENTIFIC_EXPANSION):
        assert name in registered, f"{name} not registered"
        assert registered[name]["maturity"] == "validated", (
            f"{name} maturity={registered[name]['maturity']!r}, expected validated"
        )


def test_contract_methods_have_maturity_contract_validated():
    registered = {m["name"]: m for m in list_methods()}
    for name in CONTRACT_BATCH:
        assert name in registered, f"{name} not registered"
        assert registered[name]["maturity"] == "contract_validated", (
            f"{name} maturity={registered[name]['maturity']!r}, expected contract_validated"
        )
        assert registered[name]["metadata"]["validation_kind"] == "contract"


def test_validation_reports_exist_for_all_evidence_methods():
    for name in MULTI_DATASET_EVIDENCE_METHODS:
        path = VAL_DOCS / f"{name}.md"
        assert path.is_file(), f"missing formal report {path}"
        text = path.read_text(encoding="utf-8")
        assert "Decision:" in text or "decision" in text.lower()
        assert "Limitations" in text or "limitations" in text.lower()


def test_validation_index_lists_all_evidence_and_ledger():
    index = VAL_DOCS / "index.md"
    assert index.is_file()
    text = index.read_text(encoding="utf-8")
    for name in MULTI_DATASET_EVIDENCE_METHODS:
        assert name in text
    assert "10" in text and "3" in text and "13" in text


def test_validation_summary_json_when_present():
    summary = ROOT / "research" / "method_validation" / "results" / "validation_summary.json"
    if not summary.is_file():
        pytest.skip("compile_validation_evidence.py has not been run")
    import json

    payload = json.loads(summary.read_text(encoding="utf-8"))
    # Historical compile may still label contract methods "validated"; ledger is source of truth.
    assert payload.get("n_validated", 0) >= 5


def test_sota_batch_json_when_present():
    path = ROOT / "research" / "method_validation" / "results" / "sota_batch_multidataset.json"
    if not path.is_file():
        pytest.skip("run_sota_batch_multidataset.py has not been run")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods", {})
    for name in SOTA_BATCH:
        assert name in methods
    # spagcn must carry real multi-slice ARI evidence
    sp = methods["spagcn"].get("sota_csv") or {}
    assert float(sp.get("mean_ari", 0)) >= 0.12
    assert int(sp.get("n_datasets", 0)) >= 3


def test_real_graphst_stagate_ari_when_present():
    path = ROOT / "research" / "method_validation" / "results" / "graphst_stagate_real_ari.json"
    if not path.is_file():
        pytest.skip("run_real_graphst_stagate_ari.py has not been run")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("backend_mode") == "official_real"
    for method in ("graphst", "stagate"):
        summary = (payload.get("summary") or {}).get(method) or {}
        assert int(summary.get("n_success", 0)) >= 9
        assert float(summary.get("mean_ari", 0)) >= 0.10
        assert len(summary.get("per_dataset_mean_ari") or {}) >= 3
