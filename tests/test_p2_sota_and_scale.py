"""P2 tests: SOTA reproduction pipeline, scale contracts, CLI."""

from __future__ import annotations

import json

from histoweave.benchmark.sota_pipeline import (
    SotaBenchmarkReport,
    SotaCellResult,
    env_contract,
    probe_all,
    probe_backend,
    run_sota_benchmark,
    write_sota_artifacts,
)
from histoweave.cli import main
from histoweave.datasets import get_dataset, registry_scale_table, scale_contract_for_assay
from histoweave.datasets.scale_contract import SCALE_CONTRACTS


def test_env_contract_lists_official_backends():
    contract = env_contract()
    assert contract["protocol"].startswith("histoweave.sota")
    assert "spagcn" in contract["methods"]
    assert "bayesspace" in contract["methods"]
    assert contract["task"] == "spatial_domain"


def test_probe_banksy_py_is_available():
    probe = probe_backend("banksy_py")
    assert probe.available is True
    assert probe.runtime == "python"


def test_probe_all_returns_one_row_per_method():
    probes = probe_all(["banksy_py", "spagcn"])
    assert {p.method for p in probes} == {"banksy_py", "spagcn"}


def test_dry_run_writes_long_csv_and_throughput(tmp_path):
    report = run_sota_benchmark(
        methods=["banksy_py", "spagcn"],
        slices=["151673"],
        seeds=[42],
        out_dir=tmp_path,
        dry_run=True,
    )
    assert report.dry_run is True
    assert len(report.cells) == 2
    csv_path = tmp_path / "sota_benchmark_long.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "status" in text
    assert "banksy_py" in text
    assert "skipped" in text
    throughput = json.loads((tmp_path / "sota_throughput.json").read_text(encoding="utf-8"))
    assert throughput["n_cells"] == 2
    probe = json.loads((tmp_path / "sota_probe.json").read_text(encoding="utf-8"))
    assert "probes" in probe


def test_write_sota_artifacts_schema(tmp_path):
    report = SotaBenchmarkReport(
        cells=[
            SotaCellResult(
                dataset="151673",
                method="spagcn",
                seed=42,
                ari=0.31,
                seconds=12.5,
                status="success",
                n_domains_truth=7,
                n_obs=3611,
            ),
            SotaCellResult(
                dataset="151673",
                method="graphst",
                seed=42,
                ari=None,
                seconds=0.0,
                status="skipped_missing_backend",
                error="not installed",
            ),
        ]
    )
    paths = write_sota_artifacts(report, tmp_path)
    rows = paths["csv"].read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3  # header + 2
    assert "0.31" in rows[1]


def test_scale_contracts_for_imaging_assays():
    xenium = scale_contract_for_assay("xenium", 50_000)
    assert xenium.sparse_required is True
    assert xenium.recommended_subsample is not None
    plan = xenium.plan_for(80_000)
    assert plan["subsample_to"] == xenium.recommended_subsample
    merfish = scale_contract_for_assay("merfish", 300_000)
    assert merfish.name in SCALE_CONTRACTS or merfish.name == "merfish_atlas"
    table = registry_scale_table()
    names = {row["dataset"] for row in table}
    assert "merfish_mouse_brain" in names
    assert "xenium_breast_cancer" in names
    entry = get_dataset("merfish_mouse_brain")
    assert entry.scale_contract_name == "merfish_100k"
    assert entry.scale_plan()["sparse_required"] is True


def test_cli_sota_dry_run_json(tmp_path, capsys):
    out = tmp_path / "sota_out"
    rc = main(
        [
            "sota",
            "--dry-run",
            "--methods",
            "banksy_py",
            "--slices",
            "151673",
            "--seeds",
            "42",
            "--out-dir",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert (out / "sota_benchmark_long.csv").exists()


def test_cli_sota_show_contract(capsys):
    rc = main(["sota", "--show-contract"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "methods" in payload
    assert "spagcn" in payload["methods"]
