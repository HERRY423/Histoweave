import pytest

from histoweave import (
    PipelineExecutionError,
    PipelineStep,
    build_report,
    default_pipeline,
    run_pipeline,
)
from histoweave.datasets import make_synthetic
from histoweave.plugins import Method, MethodCategory, MethodSpec, register


def test_default_pipeline_runs_end_to_end():
    data = make_synthetic(seed=0)
    result = run_pipeline(data)
    # Each stage left its mark.
    assert "domain" in result.obs
    assert "cell_type" in result.obs
    assert "run_manifest" in result.uns
    steps = result.uns["run_manifest"]["steps"]
    assert [s["category"] for s in steps] == [
        "qc",
        "normalization",
        "domain_detection",
        "annotation",
    ]


def test_provenance_chain_is_complete():
    data = make_synthetic(seed=0)
    result = run_pipeline(data)
    methods = [p["method"] for p in result.provenance]
    # ingestion (make_synthetic) + 4 pipeline steps
    assert "make_synthetic" in methods
    assert "basic_qc" in methods
    assert "kmeans" in methods


def test_custom_pipeline_subset():
    data = make_synthetic(seed=0)
    steps = default_pipeline()[:2]  # qc + normalize only
    result = run_pipeline(data, steps)
    assert "domain" not in result.obs
    assert len(result.uns["run_manifest"]["steps"]) == 2


def test_build_report_writes_html(tmp_path):
    data = make_synthetic(seed=0)
    result = run_pipeline(data)
    out = build_report(result, tmp_path / "report.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "HistoWeave Analysis Report" in text
    assert "<svg" in text  # spatial maps embedded
    assert "Pipeline" in text


def test_pipeline_failure_has_machine_readable_partial_receipt():
    @register
    class AlwaysFails(Method):
        spec = MethodSpec(
            name="always_fails_receipt_test",
            category=MethodCategory.QC,
            version="1.0.0",
        )

        def run(self, data):
            raise RuntimeError("deliberate failure")

    data = make_synthetic(seed=0)
    with pytest.raises(PipelineExecutionError) as caught:
        run_pipeline(data, [PipelineStep("qc", "always_fails_receipt_test")])

    error = caught.value
    assert error.manifest.status == "failed"
    assert error.manifest.finished
    assert error.manifest.steps[0]["status"] == "failed"
    assert error.manifest.steps[0]["error"] == {
        "type": "RuntimeError",
        "message": "deliberate failure",
    }
    assert error.partial_result.uns["run_manifest"]["status"] == "failed"


def test_pipeline_continue_policy_records_partial_run():
    @register
    class ContinueFailure(Method):
        spec = MethodSpec(
            name="continue_failure_test",
            category=MethodCategory.QC,
            version="1.0.0",
        )

        def run(self, data):
            raise RuntimeError("skip me")

    result = run_pipeline(
        make_synthetic(seed=0),
        [
            PipelineStep("qc", "continue_failure_test"),
            PipelineStep("normalization", "log1p_cp10k"),
        ],
        on_error="continue",
    )

    assert result.uns["run_manifest"]["status"] == "partial"
    assert [step["status"] for step in result.uns["run_manifest"]["steps"]] == [
        "failed",
        "success",
    ]
    assert "counts" in result.layers


def test_pipeline_rejects_plugin_that_omits_provenance():
    @register
    class MissingProvenance(Method):
        spec = MethodSpec(
            name="missing_provenance_test",
            category=MethodCategory.QC,
            version="1.0.0",
        )

        def run(self, data):
            return data.copy()

    with pytest.raises(PipelineExecutionError, match="did not append provenance") as caught:
        run_pipeline(
            make_synthetic(seed=0),
            [PipelineStep("qc", "missing_provenance_test")],
        )
    assert caught.value.manifest.steps[0]["error"]["type"] == "TypeError"
