"""Static contracts for release, documentation, workflow, and container automation."""

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


def test_container_workflow_pushes_both_images_to_ghcr() -> None:
    workflow = _read(WORKFLOWS / "containers.yml")
    assert "packages: write" in workflow
    assert "docker/login-action@v3" in workflow
    assert "docker/metadata-action@v5" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "push: true" in workflow
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


def test_spatialtable_runtime_dependencies_are_core_dependencies() -> None:
    pyproject = _read(ROOT / "pyproject.toml")
    core_dependencies = pyproject.split("[project.optional-dependencies]", maxsplit=1)[0]

    for dependency in ("anndata>=0.10", "scipy>=1.10", "spatialdata>=0.1"):
        assert f'"{dependency}"' in core_dependencies
