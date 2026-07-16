"""Natural language to executable spatial analysis pipelines."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ..data import SpatialTable
from ..plugins import get_method
from .catalog import build_catalog
from .executor import run_compiled
from .gaps import append_gaps
from .llm import CompilerProviderError, request_plan
from .prompts import build_messages
from .schema import CapabilityGap, CompiledPlan, CompiledStep, CompilerSchemaError
from .serialization import (
    COMPILER_SCHEMA_VERSION,
    catalog_digest,
    load_plan,
    save_plan,
    seal_plan,
    verify_plan_identity,
)
from .validate import CompilerValidationError, validate_plan


def _context_scalar(value: Any, field_name: str) -> bool | int | float | str | None:
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise CompilerSchemaError(f"data context {field_name} must be finite")
    raise CompilerSchemaError(
        f"data context {field_name} must be a JSON scalar, got {type(value).__name__}"
    )


def _context(data: SpatialTable | None) -> dict[str, Any]:
    if data is None:
        return {}
    assay = str(data.uns.get("assay", "unknown")).strip() or "unknown"
    if len(assay) > 128:
        raise CompilerSchemaError("data context assay must be at most 128 characters")
    marker_genes = data.uns.get("marker_genes")
    try:
        has_marker_genes = marker_genes is not None and len(marker_genes) > 0
    except TypeError:
        has_marker_genes = marker_genes is not None and bool(marker_genes)
    return {
        "assay": assay,
        "n_obs": int(data.n_obs),
        "n_vars": int(data.n_vars),
        "n_domains": _context_scalar(data.uns.get("n_domains"), "n_domains"),
        "has_spatial": data.spatial is not None,
        "obs_columns": list(map(str, data.obs.columns[:50])),
        "has_marker_genes": has_marker_genes,
    }


def _materialize_steps(plan: CompiledPlan) -> None:
    """Pin registry releases and resolved defaults before sealing a plan."""
    resolved: list[CompiledStep] = []
    for step in plan.steps:
        method_cls = get_method(
            step.category,
            step.method,
            version=step.method_version,
        )
        params = dict(method_cls(**step.params).params)
        resolved.append(
            CompiledStep.from_dict(
                {
                    "category": step.category,
                    "method": step.method,
                    "method_version": method_cls.spec.version,
                    "params": params,
                    "purpose": step.purpose,
                }
            )
        )
    plan.steps = resolved


def compile(
    question: str,
    *,
    data: SpatialTable | None = None,
    provider: str | None = None,
    executor: str = "in-process",
    dry_run: bool = True,
    gaps_path: str | Path | None = None,
    timeout: float | None = None,
    max_repair_attempts: int = 1,
) -> CompiledPlan:
    """Compile a question into a validated, registry-backed pipeline plan."""
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise ValueError("question must be a non-empty string of at most 4000 characters")
    question = question.strip()
    if executor not in {"in-process", "nextflow"}:
        raise ValueError("executor must be 'in-process' or 'nextflow'")
    if (
        not isinstance(max_repair_attempts, int)
        or isinstance(max_repair_attempts, bool)
        or not 0 <= max_repair_attempts <= 3
    ):
        raise ValueError("max_repair_attempts must be an integer between 0 and 3")
    if timeout is None:
        raw_timeout = os.getenv("HISTOWEAVE_COMPILER_TIMEOUT")
        if raw_timeout:
            try:
                timeout = float(raw_timeout)
            except ValueError as exc:
                raise ValueError("HISTOWEAVE_COMPILER_TIMEOUT must be numeric") from exc
    model = provider or os.getenv("HISTOWEAVE_COMPILER_MODEL") or "mock"
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise ValueError("model must be a non-empty string of at most 256 characters")
    if timeout is not None:
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be numeric") from exc
        if not math.isfinite(timeout) or not 1.0 <= timeout <= 600.0:
            raise ValueError("timeout must be between 1 and 600 seconds")
    context = _context(data)
    catalog_assay = None if context.get("assay") == "unknown" else context.get("assay")
    catalog = build_catalog(assay=catalog_assay)
    catalog_fingerprint = catalog_digest(catalog)
    validation_error = None
    last_error: Exception | None = None
    total_attempts = max_repair_attempts + 1
    for attempt in range(1, total_attempts + 1):
        messages = build_messages(
            question,
            catalog,
            context=context,
            validation_error=validation_error,
        )
        try:
            raw = request_plan(
                model=model,
                messages=messages,
                question=question,
                context=context,
                timeout=timeout,
            )
            plan = CompiledPlan.from_dict(
                raw,
                question=question,
                executor=executor,
                dry_run=dry_run,
                model=model,
            )
            validate_plan(plan)
            _materialize_steps(plan)
            validate_plan(plan)
        except (
            CompilerSchemaError,
            CompilerValidationError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            validation_error = str(exc)
            continue
        seal_plan(
            plan,
            catalog_fingerprint=catalog_fingerprint,
            catalog_assay=catalog_assay,
            attempt_count=attempt,
        )
        if gaps_path is not None:
            append_gaps(plan, gaps_path)
        return plan
    raise CompilerValidationError(
        f"compiler failed after {total_attempts} attempt(s): {last_error}"
    ) from last_error


compile_pipeline = compile

__all__ = [
    "CapabilityGap",
    "CompiledPlan",
    "CompiledStep",
    "CompilerProviderError",
    "CompilerSchemaError",
    "CompilerValidationError",
    "COMPILER_SCHEMA_VERSION",
    "build_catalog",
    "catalog_digest",
    "compile",
    "compile_pipeline",
    "load_plan",
    "run_compiled",
    "save_plan",
    "seal_plan",
    "validate_plan",
    "verify_plan_identity",
]
