"""Validate compiler output against the live method registry."""

from __future__ import annotations

from ..plugins import MethodCategory, get_method
from .schema import CompiledPlan


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
    """Reject invented methods, invalid params, and backward stage ordering."""
    last_rank = -1
    for index, step in enumerate(plan.steps, start=1):
        try:
            category = MethodCategory(step.category)
        except ValueError as exc:
            raise CompilerValidationError(
                f"step {index}: unknown category {step.category!r}"
            ) from exc
        rank = _ORDER[category]
        if rank < last_rank:
            raise CompilerValidationError(
                f"step {index}: {category.value} appears after a later-stage category"
            )
        last_rank = rank
        try:
            method_cls = get_method(category, step.method)
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
    return plan
