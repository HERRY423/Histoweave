"""Regression tests for the selective regret-coverage paper figure."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_external_validation import make_figures

ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "protocol_endpoints_results" / "selective_regret_coverage.json"


def test_selective_regret_curve_and_figure(tmp_path, monkeypatch):
    payload = json.loads(CURVE.read_text(encoding="utf-8"))
    rows = payload["curve"]

    assert payload["n_queries"] == 20
    assert len(rows) == 13
    assert payload["recommended_policy"] == "always_global_default"
    assert payload["recommended_coverage"] == 0.0
    assert all(
        row["mean_regret_always_global"]
        < row["mean_regret_always_personalised"]
        for row in rows
    )

    monkeypatch.setattr(make_figures, "FIG", tmp_path)
    make_figures.selective_regret_coverage()

    assert (tmp_path / "selective_regret_coverage.svg").stat().st_size > 10_000
    assert (tmp_path / "selective_regret_coverage.png").stat().st_size > 10_000
