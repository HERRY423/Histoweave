"""Isolated, resumable execution for phenomenon-centred benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from packaging.requirements import Requirement
from packaging.version import Version

from ..datasets import (
    ObservationCondition,
    ScenarioManifest,
    SpatialPhenomenon,
    make_phenomenology_scenario,
    write_visium_fixture,
    write_xenium_fixture,
)
from ..plugins import MethodCategory, MethodReference, create_method
from .phenomenology_contracts import (
    FrozenMethod,
    MethodEvaluationContract,
    ResourceClass,
)
from .phenomenology_metrics import evaluate_method_output

RUN_SCHEMA_VERSION = "1.0.0"


class ParameterTrack(StrEnum):
    """Preregistered parameter-selection tracks."""

    LOCKED = "locked"
    TUNED = "tuned"


class RunStatus(StrEnum):
    """Mutually exclusive execution outcomes; no state is silently discarded."""

    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    NOT_TUNABLE = "not_tunable"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    FIXTURE_UNAVAILABLE = "fixture_unavailable"
    INVALID_INPUT = "invalid_input"
    METHOD_ERROR = "method_error"
    TIMEOUT = "timeout"
    OOM = "oom"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True)
class BenchmarkExecutionConfig:
    """Resource and persistence policy for isolated runs."""

    standard_budget_seconds: float = 600.0
    heavy_budget_seconds: float = 1800.0
    memory_limit_gb: float = 16.0
    poll_interval_seconds: float = 0.05
    checkpoint_dir: str | None = None
    resume: bool = True

    def __post_init__(self) -> None:
        if self.standard_budget_seconds <= 0 or self.heavy_budget_seconds <= 0:
            raise ValueError("execution budgets must be positive")
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")


@dataclass(frozen=True)
class PhenomenologyRunSpec:
    """Complete definition of one method × scenario × track attempt."""

    scenario: ScenarioManifest
    method: FrozenMethod
    contract: MethodEvaluationContract
    track: ParameterTrack = ParameterTrack.LOCKED
    params: dict[str, Any] = field(default_factory=dict)
    method_seed: int = 0

    def __post_init__(self) -> None:
        expected = MethodReference(self.method.category, self.method.name, self.method.version)
        if self.contract.reference != expected:
            raise ValueError("method snapshot and evaluation contract reference do not match")
        object.__setattr__(self, "track", ParameterTrack(self.track))

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.params, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        payload = ":".join(
            [
                self.scenario.manifest_hash,
                self.method.category,
                self.method.name,
                self.method.version,
                self.track.value,
                self.config_hash,
                str(self.method_seed),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    """One row in the run-status long table."""

    run_id: str
    manifest_hash: str
    phenomenon: str
    condition: str
    replicate: int
    track: str
    category: str
    method: str
    version: str
    role: str
    applicability: bool
    status: str
    config_hash: str
    data_seed: int
    method_seed: int
    seconds: float | None
    peak_rss_mb: float | None
    backend_versions: dict[str, str]
    error_fingerprint: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    schema_version: str = RUN_SCHEMA_VERSION


@dataclass(frozen=True)
class MetricRecord:
    """One row in the metric long table."""

    run_id: str
    metric: str
    value: float
    direction: str
    primary: bool
    normalized_value: float


@dataclass(frozen=True)
class RunOutcome:
    """Status row plus zero or more metric rows."""

    run: RunRecord
    metrics: tuple[MetricRecord, ...] = ()


def execute_run(
    spec: PhenomenologyRunSpec,
    config: BenchmarkExecutionConfig | None = None,
) -> RunOutcome:
    """Execute one attempt in a child process with budget and RSS monitoring."""

    config = config or BenchmarkExecutionConfig()
    checkpoint = _checkpoint_path(spec, config)
    if config.resume and checkpoint is not None and checkpoint.exists():
        return _load_checkpoint(checkpoint)

    applicable = spec.contract.is_applicable(spec.scenario.phenomenon.name)
    if not applicable:
        outcome = _status_only(spec, RunStatus.NOT_APPLICABLE)
        _save_checkpoint(checkpoint, outcome)
        return outcome
    if spec.method.category == MethodCategory.INGESTION.value and spec.method.name not in {
        "visium_reader",
        "xenium_reader",
    }:
        outcome = _status_only(
            spec,
            RunStatus.FIXTURE_UNAVAILABLE,
            error_type="FixtureUnavailable",
            error_message="no legal vendor fixture adapter is registered for this reader",
        )
        _save_checkpoint(checkpoint, outcome)
        return outcome
    if spec.track is ParameterTrack.TUNED and not spec.contract.tuning_space:
        outcome = _status_only(spec, RunStatus.NOT_TUNABLE)
        _save_checkpoint(checkpoint, outcome)
        return outcome

    backend_versions, backend_error = _backend_versions(spec.method)
    if backend_error is not None:
        outcome = _status_only(
            spec,
            RunStatus.BACKEND_UNAVAILABLE,
            backend_versions=backend_versions,
            error_type="BackendUnavailable",
            error_message=backend_error,
        )
        _save_checkpoint(checkpoint, outcome)
        return outcome

    budget = (
        config.heavy_budget_seconds
        if spec.contract.resource_class is ResourceClass.HEAVY
        else config.standard_budget_seconds
    )
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_child_run, args=(spec, result_queue))
    started = time.perf_counter()
    process.start()
    peak_rss_mb = 0.0
    status_override: RunStatus | None = None
    while process.is_alive():
        elapsed = time.perf_counter() - started
        peak_rss_mb = max(peak_rss_mb, _process_tree_rss_mb(process.pid))
        if peak_rss_mb > config.memory_limit_gb * 1024.0:
            status_override = RunStatus.OOM
            process.terminate()
            break
        if elapsed > budget:
            status_override = RunStatus.BUDGET_EXCEEDED
            process.terminate()
            break
        time.sleep(config.poll_interval_seconds)
    process.join(timeout=2.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
        status_override = status_override or RunStatus.TIMEOUT

    seconds = time.perf_counter() - started
    if status_override is not None:
        outcome = _status_only(
            spec,
            status_override,
            seconds=seconds,
            peak_rss_mb=peak_rss_mb,
            backend_versions=backend_versions,
            error_type=status_override.value,
            error_message=f"isolated process exceeded {status_override.value} limit",
        )
        _save_checkpoint(checkpoint, outcome)
        return outcome

    try:
        payload = result_queue.get(timeout=1.0)
    except queue.Empty:
        outcome = _status_only(
            spec,
            RunStatus.METHOD_ERROR,
            seconds=seconds,
            peak_rss_mb=peak_rss_mb,
            backend_versions=backend_versions,
            error_type="ChildProcessError",
            error_message=f"child exited with code {process.exitcode} without a result",
        )
        _save_checkpoint(checkpoint, outcome)
        return outcome
    finally:
        result_queue.close()

    peak_rss_mb = max(peak_rss_mb, float(payload.get("peak_rss_mb", 0.0)))
    status = RunStatus(payload["status"])
    if status is not RunStatus.OK:
        outcome = _status_only(
            spec,
            status,
            seconds=seconds,
            peak_rss_mb=peak_rss_mb,
            backend_versions=backend_versions,
            error_type=payload.get("error_type"),
            error_message=payload.get("error_message"),
        )
    else:
        run = _run_record(
            spec,
            RunStatus.OK,
            seconds=seconds,
            peak_rss_mb=peak_rss_mb,
            backend_versions=backend_versions,
        )
        metrics = tuple(
            MetricRecord(
                run_id=spec.run_id,
                metric=metric["name"],
                value=metric["value"],
                direction=metric["direction"],
                primary=metric["primary"],
                normalized_value=metric["normalized_value"],
            )
            for metric in payload.get("metrics", [])
        )
        outcome = RunOutcome(run=run, metrics=metrics)
    _save_checkpoint(checkpoint, outcome)
    return outcome


def write_long_tables(
    outcomes: list[RunOutcome],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Atomically write status and metric long tables without dropping failures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.csv"
    metrics_path = output_dir / "metrics.csv"
    runs = pd.DataFrame([asdict(outcome.run) for outcome in outcomes])
    metrics = pd.DataFrame(
        [asdict(metric) for outcome in outcomes for metric in outcome.metrics],
        columns=["run_id", "metric", "value", "direction", "primary", "normalized_value"],
    )
    _write_csv_atomic(runs, runs_path)
    _write_csv_atomic(metrics, metrics_path)
    return runs_path, metrics_path


def _child_run(spec: PhenomenologyRunSpec, result_queue) -> None:
    started_rss = _self_peak_rss_mb()
    try:
        reference = make_phenomenology_scenario(spec.scenario)
        prepared, adapter_params = _prepare_method_input(reference, spec)
        if spec.method.category == MethodCategory.INGESTION.value:
            with tempfile.TemporaryDirectory(prefix="histoweave-vendor-fixture-") as temporary:
                fixture_writer = (
                    write_visium_fixture
                    if spec.method.name == "visium_reader"
                    else write_xenium_fixture
                )
                fixture_path = fixture_writer(temporary, table=reference)
                method = create_method(
                    spec.method.category,
                    spec.method.name,
                    version=spec.method.version,
                    **{
                        **spec.params,
                        **adapter_params,
                        "path": str(fixture_path),
                        "engine": "native",
                    },
                )
                result = method.run(prepared)
        else:
            method = create_method(
                spec.method.category,
                spec.method.name,
                version=spec.method.version,
                **{**spec.params, **adapter_params},
            )
            result = method.run(prepared)
        metrics = evaluate_method_output(
            spec.method.category,
            result,
            reference,
            spec.scenario.phenomenon.name,
        )
        payload: dict[str, Any] = {
            "status": RunStatus.OK.value,
            "metrics": [asdict(metric) for metric in metrics],
        }
    except ModuleNotFoundError as exc:
        payload = _error_payload(RunStatus.BACKEND_UNAVAILABLE, exc)
    except (KeyError, TypeError, ValueError) as exc:
        payload = _error_payload(RunStatus.INVALID_INPUT, exc)
    except MemoryError as exc:
        payload = _error_payload(RunStatus.OOM, exc)
    except Exception as exc:  # noqa: BLE001 - child boundary must preserve all failures
        payload = _error_payload(RunStatus.METHOD_ERROR, exc)
    payload["peak_rss_mb"] = max(started_rss, _self_peak_rss_mb())
    result_queue.put(payload)


def _prepare_method_input(
    reference,
    spec: PhenomenologyRunSpec,
):
    data = reference.copy()
    category = MethodCategory(spec.method.category)
    if (
        category
        in {
            MethodCategory.ANNOTATION,
            MethodCategory.DOMAIN_DETECTION,
            MethodCategory.DECONVOLUTION,
            MethodCategory.SPATIALLY_VARIABLE_GENES,
            MethodCategory.CELL_CELL_COMMUNICATION,
            MethodCategory.INTEGRATION,
        }
        and spec.method.name != "cell2location"
    ):
        data = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(data)
    if spec.method.name == "cell2location":
        profiles = np.asarray(data.uns["reference_profiles"], dtype=float)
        data.uns["cell2location_reference"] = pd.DataFrame(
            profiles.T,
            index=data.var_names,
            columns=data.uns["reference_cell_types"],
        )
    adapter_params: dict[str, Any] = {}
    if "image" in spec.method.modalities:
        # Input-key injection is an adapter concern, not hyperparameter tuning.
        adapter_params["image_key"] = "synthetic_tissue"
    if spec.method.name == "cellpose2":
        adapter_params["channel_axis"] = 2
    if spec.method.name == "liana_plus":
        adapter_params["groupby"] = "cell_type_truth"
    return data, adapter_params


def _backend_versions(method: FrozenMethod) -> tuple[dict[str, str], str | None]:
    versions: dict[str, str] = {}
    if method.language == "container" and not (shutil.which("docker") or shutil.which("podman")):
        return versions, "container runtime docker/podman is unavailable"
    for name, requirement, runtime, _ in method.backends:
        if runtime == "r":
            if method.language != "container" and shutil.which("Rscript") is None:
                return versions, f"Rscript is unavailable for {name}"
            versions[name] = "container-managed" if method.language == "container" else "available"
            continue
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return versions, f"Python backend {name}{requirement} is not installed"
        versions[name] = installed
        parsed = Requirement(f"{name}{requirement}")
        if Version(installed) not in parsed.specifier:
            return versions, f"Python backend {name}=={installed} does not satisfy {requirement}"
    return versions, None


def _run_record(
    spec: PhenomenologyRunSpec,
    status: RunStatus,
    *,
    seconds: float | None = None,
    peak_rss_mb: float | None = None,
    backend_versions: dict[str, str] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> RunRecord:
    fingerprint = None
    if error_type or error_message:
        fingerprint = hashlib.sha256(f"{error_type}:{error_message}".encode()).hexdigest()[:16]
    return RunRecord(
        run_id=spec.run_id,
        manifest_hash=spec.scenario.manifest_hash,
        phenomenon=SpatialPhenomenon(spec.scenario.phenomenon.name).value,
        condition=ObservationCondition(spec.scenario.condition.name).value,
        replicate=spec.scenario.replicate,
        track=spec.track.value,
        category=spec.method.category,
        method=spec.method.name,
        version=spec.method.version,
        role=spec.contract.role.value,
        applicability=spec.contract.is_applicable(spec.scenario.phenomenon.name),
        status=status.value,
        config_hash=spec.config_hash,
        data_seed=spec.scenario.seed,
        method_seed=spec.method_seed,
        seconds=seconds,
        peak_rss_mb=peak_rss_mb,
        backend_versions=dict(backend_versions or {}),
        error_fingerprint=fingerprint,
        error_type=error_type,
        error_message=error_message,
    )


def _status_only(
    spec: PhenomenologyRunSpec,
    status: RunStatus,
    **kwargs: Any,
) -> RunOutcome:
    return RunOutcome(run=_run_record(spec, status, **kwargs))


def _checkpoint_path(spec: PhenomenologyRunSpec, config: BenchmarkExecutionConfig) -> Path | None:
    if config.checkpoint_dir is None:
        return None
    return Path(config.checkpoint_dir) / "runs" / f"{spec.run_id}.json"


def _save_checkpoint(path: Path | None, outcome: RunOutcome) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": asdict(outcome.run),
        "metrics": [asdict(metric) for metric in outcome.metrics],
    }
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> RunOutcome:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunOutcome(
        run=RunRecord(**payload["run"]),
        metrics=tuple(MetricRecord(**metric) for metric in payload.get("metrics", [])),
    )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _error_payload(status: RunStatus, exc: Exception) -> dict[str, Any]:
    return {
        "status": status.value,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(limit=20),
    }


def _process_tree_rss_mb(pid: int | None) -> float:
    if pid is None:
        return 0.0
    total_kb = _read_proc_rss_kb(pid)
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        child_pids = [int(value) for value in children_path.read_text().split()]
    except (FileNotFoundError, PermissionError, ValueError):
        child_pids = []
    for child_pid in child_pids:
        total_kb += _read_proc_rss_kb(child_pid)
    return total_kb / 1024.0


def _read_proc_rss_kb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return 0.0
    return 0.0


def _self_peak_rss_mb() -> float:
    try:
        import resource
    except ModuleNotFoundError:
        # Windows does not provide ``resource``. Prefer the peak working set,
        # falling back to current RSS when the platform omits that field.
        try:
            import psutil

            memory = psutil.Process().memory_info()
            peak_bytes = float(getattr(memory, "peak_wset", memory.rss))
            return peak_bytes / (1024.0**2)
        except (ImportError, OSError):
            return 0.0
    else:
        getrusage = getattr(resource, "getrusage", None)
        rusage_self = getattr(resource, "RUSAGE_SELF", None)
        if getrusage is None or rusage_self is None:
            return 0.0
        value = float(getrusage(rusage_self).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value / 1024.0 if value < 10**9 else value / (1024.0**2)
