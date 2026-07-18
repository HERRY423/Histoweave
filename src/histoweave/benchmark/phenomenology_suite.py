"""Planning and orchestration for the full phenomenology factorial suite."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..datasets import (
    ObservationCondition,
    SpatialPhenomenon,
    default_scenario_manifest,
)
from ..plugins import MethodReference
from .phenomenology_contracts import (
    FrozenMethodManifest,
    build_evaluation_contracts,
    capability_matrix_rows,
    freeze_release_manifest,
)
from .phenomenology_runner import (
    BenchmarkExecutionConfig,
    ParameterTrack,
    PhenomenologyRunSpec,
    RunOutcome,
    execute_run,
    write_long_tables,
)
from .phenomenology_statistics import capability_index, coverage_summary

EXPERIMENT_SCHEMA_VERSION = "1.0.0"
DEFAULT_EVALUATION_SEEDS = (1729, 2718, 3141, 5772, 8111)


@dataclass(frozen=True)
class PhenomenologySuitePlan:
    """Frozen method/scenario/run plan constructed without generating data."""

    method_manifest: FrozenMethodManifest
    runs: tuple[PhenomenologyRunSpec, ...]
    phenomena: tuple[SpatialPhenomenon, ...]
    conditions: tuple[ObservationCondition, ...]
    seeds: tuple[int, ...]
    tracks: tuple[ParameterTrack, ...]
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        methods = sorted({run.method.name for run in self.runs})
        scenarios = sorted(
            {
                (
                    run.scenario.manifest_hash,
                    SpatialPhenomenon(run.scenario.phenomenon.name).value,
                    ObservationCondition(run.scenario.condition.name).value,
                    run.scenario.replicate,
                    run.scenario.seed,
                )
                for run in self.runs
            }
        )
        return {
            "schema_version": self.schema_version,
            "method_manifest_hash": self.method_manifest.manifest_hash,
            "method_count": len(methods),
            "methods": methods,
            "phenomena": [item.value for item in self.phenomena],
            "conditions": [item.value for item in self.conditions],
            "seeds": list(self.seeds),
            "tracks": [item.value for item in self.tracks],
            "scenario_count": len(scenarios),
            "design_unit_count_per_track": len(scenarios) * len(methods),
            "run_record_count": len(self.runs),
            "scenarios": [
                {
                    "manifest_hash": manifest_hash,
                    "phenomenon": phenomenon,
                    "condition": condition,
                    "replicate": replicate,
                    "seed": seed,
                }
                for manifest_hash, phenomenon, condition, replicate, seed in scenarios
            ],
        }


def build_suite_plan(
    *,
    phenomena: tuple[SpatialPhenomenon | str, ...] = tuple(SpatialPhenomenon),
    conditions: tuple[ObservationCondition | str, ...] = tuple(ObservationCondition),
    methods: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = DEFAULT_EVALUATION_SEEDS,
    tracks: tuple[ParameterTrack | str, ...] = (ParameterTrack.LOCKED,),
    n_obs: int = 600,
    n_genes: int = 256,
    image_size: int = 256,
) -> PhenomenologySuitePlan:
    """Build a complete factorial run plan from the frozen release registry."""

    parsed_phenomena = tuple(SpatialPhenomenon(item) for item in phenomena)
    parsed_conditions = tuple(ObservationCondition(item) for item in conditions)
    parsed_tracks = tuple(ParameterTrack(item) for item in tracks)
    if not parsed_phenomena or not parsed_conditions or not seeds or not parsed_tracks:
        raise ValueError("phenomena, conditions, seeds and tracks must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")

    manifest = freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)
    selected = tuple(methods) if methods else tuple(method.name for method in manifest.methods)
    unknown = set(selected) - {method.name for method in manifest.methods}
    if unknown:
        raise ValueError(f"methods are not in the frozen release manifest: {sorted(unknown)}")
    selected_methods = [method for method in manifest.methods if method.name in set(selected)]

    runs: list[PhenomenologyRunSpec] = []
    for replicate, seed in enumerate(seeds):
        for phenomenon in parsed_phenomena:
            for condition in parsed_conditions:
                scenario = default_scenario_manifest(
                    phenomenon,
                    condition,
                    replicate=replicate,
                    seed=seed,
                    n_obs=n_obs,
                    n_genes=n_genes,
                    image_size=image_size,
                )
                for method in selected_methods:
                    reference = MethodReference(method.category, method.name, method.version)
                    for track in parsed_tracks:
                        runs.append(
                            PhenomenologyRunSpec(
                                scenario=scenario,
                                method=method,
                                contract=contracts[reference],
                                track=track,
                                method_seed=seed,
                            )
                        )
    return PhenomenologySuitePlan(
        method_manifest=manifest,
        runs=tuple(runs),
        phenomena=parsed_phenomena,
        conditions=parsed_conditions,
        seeds=seeds,
        tracks=parsed_tracks,
    )


def write_suite_plan(plan: PhenomenologySuitePlan, output_dir: str | Path) -> dict[str, Path]:
    """Persist the experiment, method and applicability manifests atomically."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = output_dir / "experiment_manifest.json"
    methods_path = output_dir / "method_manifest.json"
    capability_path = output_dir / "capability_matrix.csv"
    _write_json_atomic(experiment_path, plan.to_dict())
    method_payload = plan.method_manifest.to_dict()
    method_payload["manifest_hash"] = plan.method_manifest.manifest_hash
    _write_json_atomic(methods_path, method_payload)
    capability = pd.DataFrame(capability_matrix_rows(plan.method_manifest))
    _write_csv_atomic(capability, capability_path)
    return {
        "experiment_manifest": experiment_path,
        "method_manifest": methods_path,
        "capability_matrix": capability_path,
    }


def execute_suite(
    plan: PhenomenologySuitePlan,
    *,
    config: BenchmarkExecutionConfig,
    output_dir: str | Path,
    workers: int = 1,
) -> tuple[list[RunOutcome], dict[str, Path]]:
    """Execute a frozen plan with bounded process fan-out and write summaries."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    if any(run.track is ParameterTrack.TUNED for run in plan.runs):
        raise ValueError(
            "tuned execution requires a separately frozen calibration manifest; "
            "use --track locked until calibration parameters are supplied"
        )
    output_dir = Path(output_dir)
    paths = write_suite_plan(plan, output_dir)
    outcomes: list[RunOutcome] = []
    if workers == 1:
        outcomes = [execute_run(spec, config) for spec in plan.runs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(execute_run, spec, config): spec.run_id for spec in plan.runs
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
        outcomes.sort(key=lambda outcome: outcome.run.run_id)

    runs_path, metrics_path = write_long_tables(outcomes, output_dir)
    runs = pd.read_csv(runs_path)
    metrics = pd.read_csv(metrics_path)
    coverage_path = output_dir / "coverage_summary.csv"
    index_path = output_dir / "capability_index.csv"
    _write_csv_atomic(coverage_summary(runs), coverage_path)
    if not metrics.empty:
        _write_csv_atomic(capability_index(runs, metrics), index_path)
    else:
        _write_csv_atomic(pd.DataFrame(), index_path)
    paths.update(
        {
            "runs": runs_path,
            "metrics": metrics_path,
            "coverage_summary": coverage_path,
            "capability_index": index_path,
        }
    )
    return outcomes, paths


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
