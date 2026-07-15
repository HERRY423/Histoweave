"""Natural language to executable spatial analysis pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..data import SpatialTable
from .catalog import build_catalog
from .executor import run_compiled
from .gaps import append_gaps
from .llm import request_plan
from .prompts import build_messages
from .schema import CapabilityGap, CompiledPlan, CompiledStep, CompilerSchemaError
from .validate import CompilerValidationError, validate_plan


def _context(data: SpatialTable | None) -> dict[str, Any]:
    if data is None:
        return {}
    return {
        "assay": data.uns.get("assay", "unknown"),
        "n_obs": data.n_obs,
        "n_vars": data.n_vars,
        "n_domains": data.uns.get("n_domains"),
        "has_spatial": data.spatial is not None,
        "obs_columns": list(map(str, data.obs.columns[:50])),
        "has_marker_genes": bool(data.uns.get("marker_genes")),
    }


def compile(
    question: str,
    *,
    data: SpatialTable | None = None,
    provider: str | None = None,
    executor: str = "in-process",
    dry_run: bool = True,
    gaps_path: str | Path | None = None,
) -> CompiledPlan:
    """Compile a question into a validated, registry-backed pipeline plan."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if executor not in {"in-process", "nextflow"}:
        raise ValueError("executor must be 'in-process' or 'nextflow'")
    model = provider or os.getenv("HISTOWEAVE_COMPILER_MODEL") or "mock"
    context = _context(data)
    catalog = build_catalog(
        assay=None if context.get("assay") == "unknown" else context.get("assay")
    )
    validation_error = None
    last_error: Exception | None = None
    for _attempt in range(2):
        messages = build_messages(
            question,
            catalog,
            context=context,
            validation_error=validation_error,
        )
        try:
            raw = request_plan(model=model, messages=messages, question=question, context=context)
            plan = CompiledPlan.from_dict(
                raw,
                question=question,
                executor=executor,
                dry_run=dry_run,
                model=model,
            )
            validate_plan(plan)
            if gaps_path is not None:
                append_gaps(plan, gaps_path)
            return plan
        except (CompilerSchemaError, CompilerValidationError, ValueError) as exc:
            last_error = exc
            validation_error = str(exc)
    raise CompilerValidationError(f"compiler failed after one retry: {last_error}") from last_error


compile_pipeline = compile

__all__ = [
    "CapabilityGap",
    "CompiledPlan",
    "CompiledStep",
    "CompilerSchemaError",
    "CompilerValidationError",
    "build_catalog",
    "compile",
    "compile_pipeline",
    "run_compiled",
    "validate_plan",
]
