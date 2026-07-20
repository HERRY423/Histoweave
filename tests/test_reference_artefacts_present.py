"""Gate: citable reference summaries must be present and match MANIFEST."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reference_artefacts" / "MANIFEST.json"
BUILDER = ROOT / "scripts" / "build_reference_artefact_manifest.py"

REQUIRED = [
    "independent_personalisation_results/independent_personalisation_summary.json",
    "independent_personalisation_results/independent_personalisation_report.md",
    "protocol_endpoints_results/protocol_endpoints_summary.json",
    "protocol_endpoints_results/protocol_endpoints_report.md",
    "protocol_endpoints_results/oracle_k_leakage.json",
    "non_oracle_k_sota/summary.json",
    "pareto_isus_results/pareto_report.json",
    "benchmark_external_validation/decision_validation.json",
]


def test_required_reference_artefacts_exist_and_are_small():
    max_bytes = 5 * 1024 * 1024
    for rel in REQUIRED:
        path = ROOT / rel
        assert path.is_file(), f"missing reference artefact: {rel}"
        size = path.stat().st_size
        assert size > 0, f"empty artefact: {rel}"
        assert size <= max_bytes, f"artefact too large for git summary policy: {rel} ({size} B)"


def test_manifest_lists_required_and_check_passes():
    assert MANIFEST.is_file(), "run scripts/build_reference_artefact_manifest.py"
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload.get("schema") == "histoweave.reference_artefacts.manifest.v1"
    paths = {row["path"] for row in payload.get("files", [])}
    for rel in REQUIRED:
        assert rel in paths, f"{rel} missing from MANIFEST.json"

    proc = subprocess.run(
        [sys.executable, str(BUILDER), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_personalisation_and_protocol_summaries_are_json_parseable():
    for rel in (
        "independent_personalisation_results/independent_personalisation_summary.json",
        "protocol_endpoints_results/protocol_endpoints_summary.json",
        "benchmark_external_validation/decision_validation.json",
    ):
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
