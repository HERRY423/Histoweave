"""CLI wiring for stats-review and discovery bootstrap-ci."""

from __future__ import annotations

import json

from histoweave.cli import main


def test_stats_review_cli(tmp_path, capsys):
    performance = {
        "d1": {"A": 0.9, "B": 0.4, "C": 0.3},
        "d2": {"A": 0.85, "B": 0.5, "C": 0.35},
        "d3": {"A": 0.88, "B": 0.45, "C": 0.4},
        "d4": {"A": 0.92, "B": 0.42, "C": 0.38},
        "d5": {"A": 0.87, "B": 0.48, "C": 0.33},
    }
    landscape = tmp_path / "landscape.json"
    landscape.write_text(
        json.dumps({"performance": performance, "schema_version": 1}),
        encoding="utf-8",
    )
    out = tmp_path / "stats.json"
    assert (
        main(
            [
                "stats-review",
                "--landscape",
                str(landscape),
                "--n-boot",
                "50",
                "--n-perm",
                "100",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Stats review" in captured.out
    assert "A" in captured.out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["protocol"] == "histoweave.stats_review.v1"
    assert payload["rank_summary"][0]["method"] == "A"


def test_discovery_bootstrap_ci_cli(tmp_path, capsys):
    # Minimal panel CSV
    csv = tmp_path / "panel.csv"
    csv.write_text(
        "label,slice_id,expected_class,direction_ok,l3_delta_rest,myelin_delta_rest,n\n"
        "a,dlpfc_151508,L3_program,True,0.4,-0.3,100\n"
        "b,dlpfc_151669,L3_program,True,0.3,-0.2,80\n"
        "c,dlpfc_151673,L3_program,True,0.5,-0.4,50\n",
        encoding="utf-8",
    )
    out = tmp_path / "boot.json"
    assert (
        main(
            [
                "discovery",
                "bootstrap-ci",
                "--panel-csv",
                str(csv),
                "--n-boot",
                "200",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Donor-stratified bootstrap" in text
    assert "L3" in text
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_donors"] == 3
    assert payload["point"]["l3_delta_rest"] > 0


def test_sota_show_contract(capsys):
    assert main(["sota", "--show-contract"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
