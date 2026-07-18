from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from histoweave.benchmark.phenomenology_contracts import (
    build_evaluation_contracts,
    freeze_release_manifest,
)
from histoweave.benchmark.phenomenology_runner import (
    BenchmarkExecutionConfig,
    ParameterTrack,
    PhenomenologyRunSpec,
    RunStatus,
    _prepare_method_input,
    execute_run,
    write_long_tables,
)
from histoweave.datasets import (
    ObservationCondition,
    SpatialPhenomenon,
    default_scenario_manifest,
    make_phenomenology_scenario,
)
from histoweave.plugins import MethodReference


def _components(method_name: str):
    manifest = freeze_release_manifest()
    method = next(method for method in manifest.methods if method.name == method_name)
    reference = MethodReference(method.category, method.name, method.version)
    return method, build_evaluation_contracts(manifest)[reference]


def _scenario(phenomenon: SpatialPhenomenon):
    return default_scenario_manifest(
        phenomenon,
        ObservationCondition.CLEAN,
        seed=44,
        n_obs=60,
        n_genes=64,
        image_size=40,
    )


def _config(tmp_path):
    return BenchmarkExecutionConfig(
        standard_budget_seconds=30,
        heavy_budget_seconds=30,
        memory_limit_gb=4,
        checkpoint_dir=str(tmp_path),
    )


def test_not_applicable_is_a_status_without_metrics(tmp_path) -> None:
    method, contract = _components("marker_deconv")
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.GRADIENT),
        method,
        contract,
    )
    outcome = execute_run(spec, _config(tmp_path))
    assert outcome.run.status == RunStatus.NOT_APPLICABLE
    assert outcome.run.applicability is False
    assert outcome.metrics == ()


def test_ingestion_without_legal_fixture_is_not_faked(tmp_path) -> None:
    method, contract = _components("merfish_reader")
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.COMPARTMENT),
        method,
        contract,
    )
    outcome = execute_run(spec, _config(tmp_path))
    assert outcome.run.status == RunStatus.FIXTURE_UNAVAILABLE
    assert outcome.run.error_fingerprint
    assert outcome.metrics == ()


@pytest.mark.parametrize("method_name", ["visium_reader", "xenium_reader"])
def test_native_vendor_fixture_runs_roundtrip_fidelity(tmp_path, method_name) -> None:
    method, contract = _components(method_name)
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.COMPARTMENT),
        method,
        contract,
    )

    outcome = execute_run(spec, _config(tmp_path))

    assert outcome.run.status == RunStatus.OK
    assert {metric.metric for metric in outcome.metrics} == {
        "roundtrip_fidelity",
        "coordinate_fidelity",
        "metadata_fidelity",
    }
    metric_values = {metric.metric: metric.normalized_value for metric in outcome.metrics}
    assert metric_values["roundtrip_fidelity"] == 1.0
    assert metric_values["metadata_fidelity"] == 1.0
    assert metric_values["coordinate_fidelity"] > 0.99


def test_missing_backend_is_preserved_separately_from_scientific_failure(tmp_path) -> None:
    method, contract = _components("marker_deconv")
    unavailable = replace(
        method,
        implementation="external",
        backends=(("definitely-not-a-real-histoweave-backend", ">=99", "python", None),),
    )
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.MIXTURE),
        unavailable,
        contract,
    )
    outcome = execute_run(spec, _config(tmp_path))
    assert outcome.run.status == RunStatus.BACKEND_UNAVAILABLE
    assert "not installed" in (outcome.run.error_message or "")
    assert outcome.metrics == ()


def test_locked_native_method_runs_in_isolated_process_and_emits_metrics(tmp_path) -> None:
    method, contract = _components("marker_deconv")
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.MIXTURE),
        method,
        contract,
        method_seed=19,
    )
    outcome = execute_run(spec, _config(tmp_path))
    assert outcome.run.status == RunStatus.OK
    assert outcome.run.seconds is not None and outcome.run.seconds > 0
    assert outcome.run.peak_rss_mb is not None and outcome.run.peak_rss_mb > 0
    assert {metric.metric for metric in outcome.metrics} == {
        "proportion_rmse",
        "jensen_shannon_similarity",
        "cell_type_correlation",
    }
    assert all(0 <= metric.normalized_value <= 1 for metric in outcome.metrics)


def test_checkpoint_resume_returns_same_run_without_duplicate(tmp_path) -> None:
    method, contract = _components("spatial_graph")
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.HOTSPOT),
        method,
        contract,
    )
    first = execute_run(spec, _config(tmp_path))
    second = execute_run(spec, _config(tmp_path))
    assert first == second
    checkpoints = list((tmp_path / "runs").glob("*.json"))
    assert len(checkpoints) == 1
    assert checkpoints[0].stem == spec.run_id


def test_registered_image_key_is_injected_for_multimodal_methods() -> None:
    method, contract = _components("image_expression_attention")
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.COMPARTMENT),
        method,
        contract,
    )
    reference = make_phenomenology_scenario(spec.scenario)

    prepared, adapter_params = _prepare_method_input(reference, spec)

    assert "synthetic_tissue" in prepared.images
    assert adapter_params["image_key"] == "synthetic_tissue"


def test_tuned_track_without_preregistered_space_is_not_copied_from_locked(tmp_path) -> None:
    method, contract = _components("marker_deconv")
    spec = PhenomenologyRunSpec(
        _scenario(SpatialPhenomenon.MIXTURE),
        method,
        contract,
        track=ParameterTrack.TUNED,
    )
    outcome = execute_run(spec, _config(tmp_path))
    assert outcome.run.status == RunStatus.NOT_TUNABLE
    assert outcome.metrics == ()


def test_long_tables_keep_success_failure_and_na_rows(tmp_path) -> None:
    deconv, deconv_contract = _components("marker_deconv")
    reader, reader_contract = _components("merfish_reader")
    outcomes = [
        execute_run(
            PhenomenologyRunSpec(
                _scenario(SpatialPhenomenon.MIXTURE),
                deconv,
                deconv_contract,
            ),
            _config(tmp_path / "checkpoints"),
        ),
        execute_run(
            PhenomenologyRunSpec(
                _scenario(SpatialPhenomenon.GRADIENT),
                deconv,
                deconv_contract,
            ),
            _config(tmp_path / "checkpoints"),
        ),
        execute_run(
            PhenomenologyRunSpec(
                _scenario(SpatialPhenomenon.GRADIENT),
                reader,
                reader_contract,
            ),
            _config(tmp_path / "checkpoints"),
        ),
    ]
    runs_path, metrics_path = write_long_tables(outcomes, tmp_path / "tables")
    runs = pd.read_csv(runs_path)
    metrics = pd.read_csv(metrics_path)
    assert set(runs["status"]) == {
        RunStatus.OK,
        RunStatus.NOT_APPLICABLE,
        RunStatus.FIXTURE_UNAVAILABLE,
    }
    assert set(metrics["run_id"]) == {outcomes[0].run.run_id}
    assert len(runs) == 3
