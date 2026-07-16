"""Static contracts for release, documentation, workflow, and container automation."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pypi_release_uses_trusted_publishing_with_least_privilege() -> None:
    workflow = _read(WORKFLOWS / "publish.yml")
    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "needs: build" in workflow
    assert "workflow_dispatch:" not in workflow
    assert 'release_version="${GITHUB_REF_NAME#v}"' in workflow
    assert "does not match package version" in workflow
    assert "name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "PYPI_TOKEN" not in workflow
    assert "password:" not in workflow


def test_nextflow_smoke_runs_the_complete_local_demo_profile() -> None:
    workflow = _read(WORKFLOWS / "nextflow-smoke.yml")
    assert "nf-core/setup-nextflow@v3" in workflow
    assert 'version: "26.04.4"' in workflow
    assert (
        "nextflow run workflows/nextflow/main.nf -profile local,test --outdir nf-smoke"
        in workflow
    )
    assert 'grep -q "COMPLETED" nf-smoke/pipeline_trace.tsv' in workflow
    assert "if: always()" in workflow


def test_container_workflow_validates_prs_and_pushes_main_images() -> None:
    workflow = _read(WORKFLOWS / "containers.yml")
    assert "packages: write" in workflow
    assert "docker/login-action@v3" in workflow
    assert "docker/metadata-action@v5" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "pull_request:" in workflow
    assert "if: github.event_name != 'pull_request'" in workflow
    assert "push: ${{ github.event_name != 'pull_request' }}" in workflow
    assert "sbom: true" in workflow
    for name in ("histoweave-python", "histoweave-r"):
        assert f"name: {name}" in workflow
        assert (ROOT / "workflows" / "containers" / name / "Dockerfile").is_file()


def test_mkdocstrings_api_reference_is_enabled_and_built_in_ci() -> None:
    config = _read(ROOT / "mkdocs.yml")
    api = _read(ROOT / "docs" / "api.md")
    ci = _read(WORKFLOWS / "ci.yml")

    assert "API Reference: api.md" in config
    assert "- mkdocstrings:" in config
    assert "default_handler: python" in config
    assert "paths: [src]" in config
    assert "          import:" not in config
    assert "          inventories:" not in config
    assert 'python -m pip install -e ".[dev,docs]"' in ci
    assert "mkdocs build --strict" in ci
    for module in (
        "histoweave",
        "histoweave.data",
        "histoweave.workflow",
        "histoweave.plugins",
        "histoweave.io",
        "histoweave.benchmark",
        "histoweave.report",
    ):
        assert f"::: {module}" in api


def test_leaderboard_pages_workflow_packages_explicit_tracked_assets() -> None:
    workflow = _read(WORKFLOWS / "pages.yml")
    gitignore = _read(ROOT / ".gitignore")
    required_assets = (
        "leaderboard/index.html",
        "leaderboard/styles.css",
        "leaderboard/main.js",
        "leaderboard/data.json",
        "leaderboard/README.md",
    )

    assert "!leaderboard/index.html" in gitignore
    assert "leaderboard/*.html" not in workflow
    assert "Required GitHub Pages asset is missing" in workflow
    for asset in required_assets:
        assert (ROOT / asset).is_file()
        assert asset in workflow


def test_spatialtable_runtime_dependencies_are_core_dependencies() -> None:
    pyproject = _read(ROOT / "pyproject.toml")
    core_dependencies = pyproject.split("[project.optional-dependencies]", maxsplit=1)[0]

    for dependency in ("anndata>=0.10", "scipy>=1.10", "spatialdata>=0.1"):
        assert f'"{dependency}"' in core_dependencies


def test_method_adapter_and_r_script_names_are_canonical() -> None:
    adapters = ROOT / "5x15_spatial_aware" / "adapters"
    assert (adapters / "banksy_py_adapter.py").is_file()
    assert not (adapters / "banksy_python_adapter.py").exists()

    r_dir = ROOT / "workflows" / "containers" / "histoweave-r"
    scripts = {path.name for path in r_dir.glob("*.R")}
    assert scripts == {
        "histoweave-banksy.R",
        "histoweave-nnsvg.R",
        "histoweave-r-lognorm.R",
        "histoweave-sctransform.R",
    }
    all_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "src" / "histoweave" / "plugins" / "builtin" / "r_demo.py",
            r_dir / "Dockerfile",
        )
    )
    assert "histoweave-sc-transform.R" not in all_source
    assert "sc_transform.R" not in all_source


def test_cross_platform_topography_artifacts_match_manifest() -> None:
    study = ROOT / "7x15_cross_platform"
    manifest = json.loads((study / "topography_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["protocol"] == "histoweave.cross_platform_topography.v1"
    for record in [*manifest["inputs"], *manifest["artifacts"]]:
        artifact = study / record["path"]
        content = artifact.read_bytes()
        assert len(content) == record["bytes"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]

    payload = json.loads((study / "platform_topography.json").read_text(encoding="utf-8"))
    validation = payload["validation"]
    assert validation == {
        "dataset_count": 8,
        "platform_count": 4,
        "method_configuration_count": 15,
        "target_derived_features": [],
        "finite_coordinates": True,
    }

    with (study / "platform_topography.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == validation["dataset_count"]
    assert {row["platform"] for row in rows} == {
        "Visium",
        "MERFISH",
        "Slide-seqV2",
        "Xenium",
    }
    for row in rows:
        for field in ("pc1", "pc2", "best_score", "top2_margin", "selection_ambiguity"):
            assert math.isfinite(float(row[field]))
        assert 0.0 <= float(row["selection_ambiguity"]) <= 1.0
