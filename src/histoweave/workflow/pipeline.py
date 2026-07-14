"""Workflow execution with machine-readable run and failure receipts.

Nextflow remains HistoWeave's portable executor (see ``workflows/nextflow``).  This
module is the in-process SDK/CLI executor.  Both surfaces use the same step records so
that a successful run, a partial run, and a failed run have identical audit semantics.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from ..data import SpatialTable
from ..logging import log_context, log_event
from ..plugins import MethodCategory, create_method

_logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
ErrorPolicy = Literal["stop", "continue"]


@dataclass(frozen=True)
class PipelineStep:
    """One declarative step: a method (by category + name) with parameters."""

    category: MethodCategory | str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    method_version: str | None = None


@dataclass
class RunManifest:
    """Versioned reproducibility receipt for an executed pipeline."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    run_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "running"
    steps: list[dict[str, Any]] = field(default_factory=list)
    started: str = field(default_factory=lambda: _now())
    finished: str = ""
    histoweave_version: str = ""
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    code_revision: str | None = field(default_factory=lambda: os.getenv("HISTOWEAVE_GIT_COMMIT"))
    container_digest: str | None = field(
        default_factory=lambda: os.getenv("HISTOWEAVE_CONTAINER_DIGEST")
    )
    executor: str = field(default_factory=lambda: os.getenv("HISTOWEAVE_EXECUTOR", "in-process"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineStepError(RuntimeError):
    """A single method failed; ``record`` is safe to persist as a run receipt."""

    def __init__(self, record: dict[str, Any]) -> None:
        error = record.get("error", {})
        super().__init__(
            f"pipeline step {record['name']!r} failed: "
            f"{error.get('type', 'Error')}: {error.get('message', '')}"
        )
        self.record = record


class PipelineExecutionError(RuntimeError):
    """Pipeline stopped after a failed step.

    ``manifest`` and ``partial_result`` let callers persist diagnostics without
    re-running successful work.  The original exception remains available through
    normal exception chaining.
    """

    def __init__(
        self,
        step_error: PipelineStepError,
        manifest: RunManifest,
        partial_result: SpatialTable,
    ) -> None:
        super().__init__(str(step_error))
        self.step_error = step_error
        self.manifest = manifest
        self.partial_result = partial_result


def default_pipeline() -> list[PipelineStep]:
    """Reference demo pipeline: QC -> normalize -> domains -> annotate.

    Real datasets must provide a domain count and marker genes (or choose methods that
    do not require them).  The CLI performs that preflight before invoking this runner.
    """

    return [
        PipelineStep(MethodCategory.QC, "basic_qc"),
        PipelineStep(MethodCategory.NORMALIZATION, "log1p_cp10k"),
        PipelineStep(MethodCategory.DOMAIN_DETECTION, "kmeans"),
        PipelineStep(MethodCategory.ANNOTATION, "marker_score"),
    ]


def execute_step(
    data: SpatialTable,
    step: PipelineStep,
) -> tuple[SpatialTable, dict[str, Any]]:
    """Execute one step and return its result plus a versioned status record.

    Failures raise :class:`PipelineStepError`; its record contains timings, input
    dimensions, resolved parameters where available, and a bounded error description.
    """

    category = _category_value(step.category)
    started = _now()
    t0 = time.perf_counter()
    record: dict[str, Any] = {
        "name": step.name or f"{category}:{step.method}",
        "category": category,
        "method": step.method,
        "params": dict(step.params),
        "requested_version": step.method_version,
        "status": "running",
        "started": started,
        "n_obs_before": data.n_obs,
        "n_vars_before": data.n_vars,
    }
    provenance_before = len(data.provenance)

    log_event(
        _logger,
        20,
        "step_start",
        "pipeline step starting",
        category=category,
        method=step.method,
        params=dict(step.params),
    )

    try:
        method = create_method(
            step.category,
            step.method,
            version=step.method_version,
            **step.params,
        )
        record["version"] = method.spec.version
        record["params"] = dict(method.params)
        result = method.run(data)
        if not isinstance(result, SpatialTable):
            raise TypeError(
                f"{category}:{step.method} returned {type(result).__name__}; "
                "plugins must return SpatialTable"
            )
        if len(result.provenance) <= provenance_before:
            raise TypeError(
                f"{category}:{step.method} did not append provenance; "
                "plugins must return self.finalize(result)"
            )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "finished": _now(),
                "seconds": round(time.perf_counter() - t0, 4),
                "error": {
                    "type": type(exc).__name__,
                    "message": _bounded_message(str(exc)),
                },
            }
        )
        log_event(
            _logger,
            40,
            "step_failed",
            "pipeline step failed",
            category=category,
            method=step.method,
            error_type=type(exc).__name__,
            seconds=record["seconds"],
        )
        raise PipelineStepError(record) from exc

    record.update(
        {
            "status": "success",
            "finished": _now(),
            "seconds": round(time.perf_counter() - t0, 4),
            "n_obs_after": result.n_obs,
            "n_vars_after": result.n_vars,
            "provenance_index": len(result.provenance) - 1,
        }
    )
    log_event(
        _logger,
        20,
        "step_ok",
        "pipeline step completed",
        category=category,
        method=step.method,
        seconds=record["seconds"],
        n_obs=result.n_obs,
    )
    return result, record


def run_pipeline(
    data: SpatialTable,
    steps: list[PipelineStep] | None = None,
    *,
    verbose: bool = False,
    on_error: ErrorPolicy = "stop",
) -> SpatialTable:
    """Run an ordered plugin pipeline and attach a complete run manifest.

    Parameters
    ----------
    on_error
        ``"stop"`` (default) raises :class:`PipelineExecutionError` with the partial
        result and failure receipt.  ``"continue"`` keeps the last successful object,
        records the failed step, and attempts later independent steps.
    """

    from .. import __version__

    if on_error not in ("stop", "continue"):
        raise ValueError("on_error must be 'stop' or 'continue'")

    selected_steps = steps if steps is not None else default_pipeline()
    manifest = RunManifest(histoweave_version=__version__)
    current = data
    failures = 0

    with log_context(run_id=manifest.run_id):
        log_event(
            _logger,
            20,
            "pipeline_start",
            "pipeline run starting",
            steps=[f"{_category_value(s.category)}:{s.method}" for s in selected_steps],
            on_error=on_error,
        )
        for step in selected_steps:
            with log_context(step_id=f"{_category_value(step.category)}:{step.method}"):
                try:
                    current, record = execute_step(current, step)
                except PipelineStepError as exc:
                    failures += 1
                    manifest.steps.append(exc.record)
                    if verbose:
                        _logger.info(
                            "  [failed] %s: %s",
                            exc.record["name"],
                            exc.record["error"]["message"],
                        )
                    if on_error == "stop":
                        manifest.status = "failed"
                        manifest.finished = _now()
                        current.uns["run_manifest"] = manifest.to_dict()
                        log_event(
                            _logger,
                            40,
                            "pipeline_aborted",
                            "pipeline stopped after step failure",
                            failed_step=exc.record["name"],
                        )
                        raise PipelineExecutionError(exc, manifest, current) from exc.__cause__
                    continue

                manifest.steps.append(record)
                if verbose:
                    _logger.info(
                        "  [ok] %-30s %7d obs  (%.3fs)",
                        record["name"],
                        current.n_obs,
                        record["seconds"],
                    )

    manifest.status = "partial" if failures else "success"
    manifest.finished = _now()
    current.uns["run_manifest"] = manifest.to_dict()
    log_event(
        _logger,
        20,
        "pipeline_done",
        "pipeline run finished",
        status=manifest.status,
        steps_completed=len(manifest.steps),
    )
    return current


def _category_value(category: MethodCategory | str) -> str:
    return category.value if isinstance(category, MethodCategory) else str(category)


def _bounded_message(message: str, limit: int = 2000) -> str:
    """Keep receipts useful without allowing arbitrary exception payload growth."""

    return message if len(message) <= limit else f"{message[: limit - 3]}..."


def _now() -> str:
    return datetime.now(UTC).isoformat()
