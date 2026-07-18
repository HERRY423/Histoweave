"""Gates for multi-dataset method validation evidence packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from histoweave.plugins import list_methods
from histoweave.plugins.builtin.release_manifest import (
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


def test_validation_evidence_covers_expansion_batch():
    missing = ALL_EXPANSION - set(VALIDATION_EVIDENCE)
    assert not missing, f"VALIDATION_EVIDENCE missing expansion methods: {sorted(missing)}"


def test_validated_methods_match_evidence_keys():
    assert VALIDATED_METHODS == set(VALIDATION_EVIDENCE)


def test_validated_methods_have_maturity_validated():
    registered = {m["name"]: m for m in list_methods()}
    for name in ALL_EXPANSION:
        assert name in registered, f"{name} not registered"
        assert registered[name]["maturity"] == "validated", (
            f"{name} maturity={registered[name]['maturity']!r}, expected validated"
        )


def test_validation_reports_exist_for_expansion_batch():
    for name in ALL_EXPANSION:
        path = VAL_DOCS / f"{name}.md"
        assert path.is_file(), f"missing formal report {path}"
        text = path.read_text(encoding="utf-8")
        assert "Decision:" in text or "decision" in text.lower()
        assert "Limitations" in text or "limitations" in text.lower()


def test_validation_index_lists_expansion_batch():
    index = VAL_DOCS / "index.md"
    assert index.is_file()
    text = index.read_text(encoding="utf-8")
    for name in ALL_EXPANSION:
        assert name in text


def test_validation_summary_json_when_present():
    summary = ROOT / "research" / "method_validation" / "results" / "validation_summary.json"
    if not summary.is_file():
        pytest.skip("compile_validation_evidence.py has not been run")
    import json

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload.get("n_validated", 0) >= 10
    for name in ALL_EXPANSION:
        assert payload["methods"][name]["decision"] == "validated"


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
