"""Static contracts for the portable Nextflow entry point and its CI gate."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "workflows" / "nextflow" / "main.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "workflows" / "nextflow" / "nextflow.config").read_text(
    encoding="utf-8"
)
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_real_vendor_input_is_ingested_before_analysis() -> None:
    assert "process INGEST_VENDOR" in MAIN
    assert re.search(r"INGEST_VENDOR\(ch_input,\s*assay,\s*engine\)", MAIN)
    assert "ch_bundle = INGEST_VENDOR.out.bundle" in MAIN
    assert 'histoweave ingest --input "${vendor_input}"' in MAIN
    assert "--assay ${assay}" in MAIN
    assert "--engine ${engine}" in MAIN
    assert not re.search(r"ch_bundle\s*=\s*Channel\.value\(file\(params\.input", MAIN)


def test_compiler_bundle_handoff_bypasses_vendor_reingestion() -> None:
    assert "params.bundle" in MAIN
    assert "--demo, --input, and --bundle are mutually exclusive" in MAIN
    assert "Channel.value(file(params.bundle, checkIfExists: true))" in MAIN
    assert "params.deconvolution_params" in MAIN
    assert "params.demo || params.bundle != null" in MAIN
    assert "raw_params instanceof Collection" in MAIN


def test_analysis_processes_consume_pinned_method_versions() -> None:
    workflow = MAIN.split("process INGEST_DEMO", maxsplit=1)[0]
    stages = {
        "QC": "qc",
        "NORMALIZE": "normalize",
        "DOMAIN_DETECTION": "domain",
        "ANNOTATION": "annotation",
        "DECONVOLUTION": "deconvolution",
    }

    for stage, prefix in stages.items():
        assert re.search(
            rf"{stage}\(.*?params\.{prefix}_version",
            workflow,
            flags=re.DOTALL,
        )
        process = re.search(
            rf"process {stage} \{{(?P<body>.*?)(?=\nprocess [A-Z_]+ \{{|\ndef _param_args)",
            MAIN,
            flags=re.DOTALL,
        )
        assert process is not None
        body = process.group("body")
        assert "val  method_version" in body
        assert '"--method-version ${_shell_quote(method_version.toString())}"' in body
        assert "${version_arg}" in body


def test_workflow_parameters_are_validated_and_domain_count_is_forwarded() -> None:
    for declaration in ("params.assay", "params.engine", "params.n_domains"):
        assert declaration in MAIN
    for validator in ("_validate_assay", "_validate_engine", "_validate_n_domains"):
        assert validator in MAIN
    for value in ("visium", "xenium", "stereo_seq", "native", "spatialdata"):
        assert value in MAIN
    assert "val  n_domains" in MAIN
    assert "--param n_domains=${n_domains}" in MAIN


def test_steps_are_exact_tokens_and_real_safe_by_default() -> None:
    match = re.search(r"params\.steps\s*=\s*'([^']+)'", MAIN)
    assert match is not None
    assert match.group(1).split(",") == ["qc", "normalize", "domain_detection", "report"]
    assert "_parse_steps(params.steps)" in MAIN
    assert " in params.steps" not in MAIN
    for step in ("qc", "normalize", "domain_detection", "annotation", "report"):
        assert f"selected_steps.contains('{step}')" in MAIN


def test_processes_have_resource_labels_and_images_are_versioned() -> None:
    assert "label 'domain_detection'" in MAIN
    assert "label 'deconvolution'" in MAIN
    assert "params.container_version" in MAIN
    assert ":latest" not in MAIN


def test_documented_execution_profiles_exist() -> None:
    for profile in ("local", "test", "docker", "singularity", "kubernetes", "slurm"):
        assert re.search(rf"(?m)^\s{{4}}{profile}\s*\{{", CONFIG)
    assert "params.demo        = true" in CONFIG
    assert "params.n_domains   = 3" in CONFIG
    assert "process.executor   = 'k8s'" in CONFIG
    assert "process.executor   = 'slurm'" in CONFIG


def test_ci_separates_quality_build_from_fresh_install_tests() -> None:
    assert re.search(r"(?m)^  quality_build:$", CI)
    assert re.search(r"(?m)^  test:$", CI)
    assert "needs: quality_build" in CI
    assert "python -m build" in CI
    assert "python -m pip check" in CI
    assert 'python -m pip install -e ".[dev]"' in CI
    assert (
        "histoweave benchmark --min-score 0.90 --fail-on-error --out benchmark.json" in CI
    )

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    dependencies = project["project"]["optional-dependencies"]["dev"]
    names = {re.match(r"[A-Za-z0-9_.-]+", item).group(0).lower() for item in dependencies}
    required = {"build", "h5py", "networkx", "pyarrow", "scikit-learn", "scipy"}
    assert required <= names
