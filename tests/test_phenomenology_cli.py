import json

from histoweave.cli import main


def test_phenomenology_dry_run_reports_full_locked_design(tmp_path, capsys) -> None:
    rc = main(
        [
            "benchmark",
            "--suite",
            "phenomenology",
            "--dry-run",
            "--json",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["method_count"] == 54
    assert payload["scenario_count"] == 6 * 5 * 5
    assert payload["design_unit_count_per_track"] == 6 * 5 * 5 * 54
    assert payload["run_record_count"] == 6 * 5 * 5 * 54
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "method_manifest.json").exists()
    assert (tmp_path / "capability_matrix.csv").exists()


def test_phenomenology_tiny_executes_one_dependency_light_run(tmp_path, capsys) -> None:
    rc = main(
        [
            "benchmark",
            "--suite",
            "phenomenology",
            "--phenomena",
            "compartment",
            "--conditions",
            "clean",
            "--methods",
            "basic_qc",
            "--seeds",
            "1729,2718",
            "--tiny",
            "--json",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    assert payload["scenario_count"] == 1
    assert payload["run_record_count"] == 1
    assert payload["status_counts"] == {"ok": 1}
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "metrics.csv").exists()


def test_phenomenology_rejects_tuned_execution_without_calibration(tmp_path, capsys) -> None:
    rc = main(
        [
            "benchmark",
            "--suite",
            "phenomenology",
            "--phenomena",
            "compartment",
            "--conditions",
            "clean",
            "--methods",
            "basic_qc",
            "--track",
            "tuned",
            "--tiny",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 2
    assert "calibration manifest" in capsys.readouterr().err


def test_phenomenology_rejects_unknown_method(tmp_path, capsys) -> None:
    rc = main(
        [
            "benchmark",
            "--suite",
            "phenomenology",
            "--methods",
            "not_a_release_method",
            "--dry-run",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 2
    assert "frozen release manifest" in capsys.readouterr().err
