"""Validate compiler output against the live method registry."""

from __future__ import annotations

import json

from ..plugins import MethodCategory, get_method
from .schema import MAX_PLAN_STEPS, CompiledPlan


class CompilerValidationError(ValueError):
    """A plan is valid JSON but is not executable by the current registry."""


_ORDER = {
    MethodCategory.INGESTION: 0,
    MethodCategory.QC: 1,
    MethodCategory.NORMALIZATION: 2,
    MethodCategory.SEGMENTATION: 3,
    MethodCategory.INTEGRATION: 3,
    MethodCategory.DOMAIN_DETECTION: 4,
    MethodCategory.ANNOTATION: 5,
    MethodCategory.DECONVOLUTION: 5,
    MethodCategory.SPATIALLY_VARIABLE_GENES: 6,
    MethodCategory.NEIGHBORHOOD: 6,
    MethodCategory.CELL_CELL_COMMUNICATION: 7,
}


def validate_plan(plan: CompiledPlan) -> CompiledPlan:
    """Reject invented methods, invalid params, duplicates, and backward ordering."""
    if plan.executor not in {"in-process", "nextflow"}:
        raise CompilerValidationError("executor must be 'in-process' or 'nextflow'")
    if not plan.steps:
        raise CompilerValidationError("plan must contain at least one step")
    if len(plan.steps) > MAX_PLAN_STEPS:
        raise CompilerValidationError(f"plan must contain at most {MAX_PLAN_STEPS} steps")
    last_rank = -1
    seen: set[tuple[str, str, str]] = set()
    for index, step in enumerate(plan.steps, start=1):
        try:
            category = MethodCategory(step.category)
        except ValueError as exc:
            raise CompilerValidationError(
                f"step {index}: unknown category {step.category!r}"
            ) from exc
        rank = _ORDER.get(category)
        if rank is None:
            raise CompilerValidationError(
                f"step {index}: category {category.value!r} is not compiler-executable"
            )
        if rank < last_rank:
            raise CompilerValidationError(
                f"step {index}: {category.value} appears after a later-stage category"
            )
        last_rank = rank
        try:
            method_cls = get_method(category, step.method, version=step.method_version)
        except (KeyError, ValueError) as exc:
            raise CompilerValidationError(f"step {index}: {exc}") from exc
        specs = {param.name: param for param in method_cls.spec.params}
        unknown = set(step.params) - set(specs)
        if unknown:
            raise CompilerValidationError(
                f"step {index}: {step.method} has unknown params {sorted(unknown)}"
            )
        for name, value in step.params.items():
            try:
                specs[name].validate(value, method=step.method)
            except (TypeError, ValueError) as exc:
                raise CompilerValidationError(f"step {index}: {exc}") from exc
        identity = (
            category.value,
            step.method,
            json.dumps(
                step.params,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        if identity in seen:
            raise CompilerValidationError(
                f"step {index}: duplicate {category.value}:{step.method} step"
            )
        seen.add(identity)
    return plan
