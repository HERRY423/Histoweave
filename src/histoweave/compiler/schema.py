"""Serializable intermediate representation for natural-language pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..workflow import PipelineStep


class CompilerSchemaError(ValueError):
    """Raised when an LLM response does not match the compiler schema."""


def _expect_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompilerSchemaError(f"{field_name} must be a JSON object")
    return value


@dataclass(frozen=True)
class CapabilityGap:
    concept: str
    reason: str
    degraded_to: str

    @classmethod
    def from_dict(cls, value: Any) -> CapabilityGap:
        row = _expect_dict(value, "gap")
        allowed = {"concept", "reason", "degraded_to"}
        unknown = set(row) - allowed
        if unknown:
            raise CompilerSchemaError(f"gap has unknown fields: {sorted(unknown)}")
        concept = row.get("concept")
        reason = row.get("reason")
        degraded_to = row.get("degraded_to")
        fields = (concept, reason, degraded_to)
        if not all(isinstance(item, str) and item.strip() for item in fields):
            raise CompilerSchemaError("gap fields must be non-empty strings")
        assert isinstance(concept, str)
        assert isinstance(reason, str)
        assert isinstance(degraded_to, str)
        return cls(concept=concept, reason=reason, degraded_to=degraded_to)

    def to_dict(self) -> dict[str, str]:
        return {
            "concept": self.concept,
            "reason": self.reason,
            "degraded_to": self.degraded_to,
        }


@dataclass(frozen=True)
class CompiledStep:
    category: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    @classmethod
    def from_dict(cls, value: Any) -> CompiledStep:
        row = _expect_dict(value, "step")
        allowed = {"category", "method", "params", "purpose"}
        unknown = set(row) - allowed
        if unknown:
            raise CompilerSchemaError(f"step has unknown fields: {sorted(unknown)}")
        category = row.get("category")
        method = row.get("method")
        params = row.get("params", {})
        purpose = row.get("purpose", "")
        if not isinstance(category, str) or not category:
            raise CompilerSchemaError("step.category must be a non-empty string")
        if not isinstance(method, str) or not method:
            raise CompilerSchemaError("step.method must be a non-empty string")
        if not isinstance(params, dict):
            raise CompilerSchemaError("step.params must be a JSON object")
        if not isinstance(purpose, str):
            raise CompilerSchemaError("step.purpose must be a string")
        return cls(category=category, method=method, params=params, purpose=purpose)

    def to_pipeline_step(self) -> PipelineStep:
        return PipelineStep(self.category, self.method, dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "method": self.method,
            "params": dict(self.params),
            "purpose": self.purpose,
        }


@dataclass
class CompiledPlan:
    question: str
    rationale: str
    steps: list[CompiledStep]
    gaps: list[CapabilityGap] = field(default_factory=list)
    assay_assumed: str = "unknown"
    executor: str = "in-process"
    dry_run: bool = True
    model: str = "unknown"

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        question: str,
        executor: str = "in-process",
        dry_run: bool = True,
        model: str = "unknown",
    ) -> CompiledPlan:
        row = _expect_dict(value, "plan")
        allowed = {"rationale", "steps", "gaps", "assay_assumed"}
        unknown = set(row) - allowed
        if unknown:
            raise CompilerSchemaError(f"plan has unknown fields: {sorted(unknown)}")
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise CompilerSchemaError("plan.rationale must be a non-empty string")
        if len(rationale) > 800:
            raise CompilerSchemaError("plan.rationale must be at most 800 characters")
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise CompilerSchemaError("plan.steps must be a non-empty array")
        raw_gaps = row.get("gaps", [])
        if not isinstance(raw_gaps, list):
            raise CompilerSchemaError("plan.gaps must be an array")
        assay = row.get("assay_assumed", "unknown")
        if not isinstance(assay, str):
            raise CompilerSchemaError("plan.assay_assumed must be a string")
        return cls(
            question=question,
            rationale=rationale,
            steps=[CompiledStep.from_dict(step) for step in raw_steps],
            gaps=[CapabilityGap.from_dict(gap) for gap in raw_gaps],
            assay_assumed=assay,
            executor=executor,
            dry_run=dry_run,
            model=model,
        )

    @property
    def pipeline_steps(self) -> list[PipelineStep]:
        return [step.to_pipeline_step() for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "rationale": self.rationale,
            "steps": [step.to_dict() for step in self.steps],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "assay_assumed": self.assay_assumed,
            "executor": self.executor,
            "dry_run": self.dry_run,
            "model": self.model,
        }
