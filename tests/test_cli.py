import json

from histoweave.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "histoweave" in capsys.readouterr().out


def test_cli_output_emits_structured_event_without_changing_stdout(capsys):
    assert main(["--log-level", "DEBUG", "--log-format", "json", "version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("histoweave ")
    events = [json.loads(line) for line in captured.err.splitlines()]
    output = next(event for event in events if event.get("event") == "cli.output")
    assert output["channel"] == "stdout"
    assert output["message"].startswith("histoweave ")



def test_list_methods_json(capsys):
    assert main(["list-methods", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(m["name"] == "basic_qc" for m in data)


def test_list_methods_filtered(capsys):
    assert main(["list-methods", "--category", "domain_detection"]) == 0
    out = capsys.readouterr().out
    assert "kmeans" in out


def test_run_demo_writes_report(tmp_path, capsys):
    out = tmp_path / "r.html"
    manifest = tmp_path / "m.json"
    rc = main(["run", "--demo", "--out", str(out), "--manifest", str(manifest)])
    assert rc == 0
    assert out.exists()
    assert manifest.exists()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["steps"]


def test_benchmark_cli(capsys):
    assert main(["benchmark"]) == 0
    assert "Recommended" in capsys.readouterr().out


def test_ingest_step_report_chain(tmp_path):
    # Mirrors the Nextflow demo DAG: ingest -> qc -> normalize -> domains -> report,
    # threaded through bundle directories on disk.
    data = tmp_path / "data.ttab"
    qc = tmp_path / "qc.ttab"
    norm = tmp_path / "norm.ttab"
    domains = tmp_path / "domains.ttab"
    report = tmp_path / "report.html"

    assert main(["ingest", "--demo", "--seed", "0", "--out", str(data)]) == 0
    assert (data / "X.npy").exists()
    assert main(["step", "qc", "--method", "basic_qc", "--in", str(data), "--out", str(qc)]) == 0
    assert main(
        ["step", "normalization", "--method", "log1p_cp10k", "--in", str(qc), "--out", str(norm)]
    ) == 0
    assert main(
        ["step", "domain_detection", "--method", "kmeans",
         "--param", "n_domains=3", "--in", str(norm), "--out", str(domains)]
    ) == 0
    assert main(["report", "--in", str(domains), "--out", str(report)]) == 0
    assert report.exists()
    assert "<svg" in report.read_text(encoding="utf-8")


def test_step_reports_unknown_method(tmp_path, capsys):
    data = tmp_path / "data.ttab"
    assert main(["ingest", "--demo", "--out", str(data)]) == 0
    rc = main(["step", "qc", "--method", "nope", "--in", str(data), "--out", str(tmp_path / "o")])
    assert rc == 2
    assert "error" in capsys.readouterr().err.lower()


def test_ingest_from_visium_fixture(tmp_path):
    from histoweave.datasets import write_visium_fixture

    root = write_visium_fixture(tmp_path / "visium", n_spots=40, n_genes=12, seed=0)
    out = tmp_path / "data.ttab"
    assert main(["ingest", "--input", str(root), "--assay", "visium", "--out", str(out)]) == 0
    assert (out / "obs.parquet").exists()


def test_validate_bundle_and_doctor_commands(tmp_path, capsys):
    bundle = tmp_path / "data.ttab"
    assert main(["ingest", "--demo", "--out", str(bundle)]) == 0
    capsys.readouterr()

    assert main(["validate-bundle", str(bundle), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["verified"] is True
    assert validation["schema_version"] == 1

    assert main(["doctor", "--json"]) == 0
    diagnostic = json.loads(capsys.readouterr().out)
    assert diagnostic["status"] == "ok"
    assert diagnostic["registered_methods"] >= 4


def test_benchmark_can_persist_and_enforce_threshold(tmp_path, capsys):
    output = tmp_path / "benchmark.json"
    assert main(
        [
            "benchmark",
            "--task",
            "domain_detection",
            "--out",
            str(output),
            "--min-score",
            "0.9",
            "--fail-on-error",
        ]
    ) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["leaderboard"][0]["score"] > 0.9
