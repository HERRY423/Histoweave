from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "submission_freeze_v1"


def test_submission_freeze_v1_reproduces_locked_artifacts() -> None:
    subprocess.run(
        [
            sys.executable,
            str(FREEZE / "reproduce_submission_freeze.py"),
        ],
        cwd=ROOT,
        check=True,
    )

    figure_lock = json.loads((FREEZE / "main_figures.lock.json").read_text(encoding="utf-8"))
    assert figure_lock["n_main_figures"] == 5
    assert [fig["figure"] for fig in figure_lock["figures"]] == [
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5",
    ]
    for fig in figure_lock["figures"]:
        assert fig["formats"]["svg"]["sha256"]
        assert fig["formats"]["png"]["sha256"]

    with (FREEZE / "supplement_benchmark_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[1]["endpoint"] == "Strict task-stratified external panel v2"
    assert rows[1]["n"] == "9"
    assert "TLS transport not replicated" in rows[1]["decision"]
    assert rows[2]["endpoint"] == "Frozen independent study test"
    assert rows[2]["n"] == "6"
    assert rows[2]["observed_value"] == "0.131261"
    assert "excluded from training" in rows[2]["decision"]
    assert rows[3]["endpoint"] == "Selective regret-coverage"
    assert rows[3]["decision"] == "full abstention/global default selected"

    manifest = json.loads((FREEZE / "submission_freeze_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "histoweave.submission_freeze.v1"
    assert manifest["supplement_table"]["rows"] == 6
    assert manifest["independent_test"]["decision"] == "independent_test_fail"
    assert manifest["independent_test"]["training_exclusion_locked"] is True
    assert len(manifest["independent_test"]["artifacts"]) == 7
    assert (
        "benchmark_external_validation/independent_test_wu2021/REPORT_independent_test_wu2021.md"
    ) in manifest["primary_source_reports"]
