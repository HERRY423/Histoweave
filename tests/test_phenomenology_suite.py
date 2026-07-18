import json

import pandas as pd

from histoweave.benchmark import (
    BenchmarkExecutionConfig,
    ParameterTrack,
    RunStatus,
    build_suite_plan,
    execute_suite,
    write_suite_plan,
)


def test_full_locked_plan_has_frozen_factorial_dimensions() -> None:
    plan = build_suite_plan()
    payload = plan.to_dict()

    assert payload["method_count"] == 54
    assert payload["scenario_count"] == 6 * 5 * 5
    assert payload["design_unit_count_per_track"] == 6 * 5 * 5 * 54
    assert payload["run_record_count"] == 6 * 5 * 5 * 54
    assert len(payload["method_manifest_hash"]) == 64


def test_both_tracks_double_run_records_not_design_units() -> None:
    plan = build_suite_plan(
        phenomena=("compartment",),
        conditions=("clean",),
        methods=("basic_qc",),
        seeds=(1729,),
        tracks=(ParameterTrack.LOCKED, ParameterTrack.TUNED),
        n_obs=60,
        n_genes=64,
        image_size=64,
    )
    payload = plan.to_dict()

    assert payload["scenario_count"] == 1
    assert payload["design_unit_count_per_track"] == 1
    assert payload["run_record_count"] == 2


def test_write_suite_plan_emits_auditable_manifests(tmp_path) -> None:
    plan = build_suite_plan(
        phenomena=("gradient",),
        conditions=("clean",),
        methods=("morans_i",),
        seeds=(11,),
        n_obs=60,
        n_genes=64,
        image_size=64,
    )
    paths = write_suite_plan(plan, tmp_path)

    assert set(paths) == {"experiment_manifest", "method_manifest", "capability_matrix"}
    manifest = json.loads(paths["experiment_manifest"].read_text(encoding="utf-8"))
    capability = pd.read_csv(paths["capability_matrix"])
    assert manifest["run_record_count"] == 1
    assert len(capability) == 54 * 6


def test_execute_tiny_suite_writes_long_tables_and_summaries(tmp_path) -> None:
    plan = build_suite_plan(
        phenomena=("compartment",),
        conditions=("clean",),
        methods=("basic_qc",),
        seeds=(1729,),
        n_obs=60,
        n_genes=64,
        image_size=64,
    )
    config = BenchmarkExecutionConfig(
        standard_budget_seconds=30,
        heavy_budget_seconds=30,
        memory_limit_gb=4,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    outcomes, paths = execute_suite(plan, config=config, output_dir=tmp_path)

    assert len(outcomes) == 1
    assert outcomes[0].run.status == RunStatus.OK.value
    assert paths["runs"].exists()
    assert paths["metrics"].exists()
    assert paths["coverage_summary"].exists()
    assert paths["capability_index"].exists()
