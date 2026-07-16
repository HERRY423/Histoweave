"""Serializable intermediate representation for natural-language pipelines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..workflow import PipelineStep


class CompilerSchemaError(ValueError):
    """Raised when an LLM response does not match the compiler schema."""


MAX_PLAN_STEPS = 32
MAX_PLAN_GAPS = 16
MAX_PARAM_BYTES = 32_768
MAX_JSON_DEPTH = 8


def _validate_json_value(value: Any, field_name: str, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise CompilerSchemaError(f"{field_name} exceeds maximum JSON depth")
    if value is None or isinstance(value, bool | int | str):
        if isinstance(value, str) and len(value) > 10_000:
            raise CompilerSchemaError(f"{field_name} string is too long")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CompilerSchemaError(f"{field_name} must contain only finite numbers")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise CompilerSchemaError(f"{field_name} array is too long")
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 256:
            raise CompilerSchemaError(f"{field_name} object has too many fields")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CompilerSchemaError(f"{field_name} keys must be non-empty strings")
            _validate_json_value(item, f"{field_name}.{key}", depth=depth + 1)
        return
    raise CompilerSchemaError(f"{field_name} contains non-JSON value {type(value).__name__}")


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
    method_version: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> CompiledStep:
        row = _expect_dict(value, "step")
        allowed = {"category", "method", "params", "purpose", "method_version"}
        unknown = set(row) - allowed
        if unknown:
            raise CompilerSchemaError(f"step has unknown fields: {sorted(unknown)}")
        category = row.get("category")
        method = row.get("method")
        params = row.get("params", {})
        purpose = row.get("purpose", "")
        method_version = row.get("method_version")
        if (
            not isinstance(category, str)
            or not category
            or category != category.strip()
            or len(category) > 128
        ):
            raise CompilerSchemaError("step.category must be a trimmed non-empty string")
        if (
            not isinstance(method, str)
            or not method
            or method != method.strip()
            or len(method) > 128
        ):
            raise CompilerSchemaError("step.method must be a trimmed non-empty string")
        if not isinstance(params, dict):
            raise CompilerSchemaError("step.params must be a JSON object")
        _validate_json_value(params, "step.params")
        encoded_params = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
        if len(encoded_params.encode("utf-8")) > MAX_PARAM_BYTES:
            raise CompilerSchemaError(f"step.params must be at most {MAX_PARAM_BYTES} bytes")
        if not isinstance(purpose, str) or len(purpose) > 500:
            raise CompilerSchemaError("step.purpose must be a string of at most 500 characters")
        if method_version is not None and (
            not isinstance(method_version, str)
            or not method_version
            or method_version != method_version.strip()
            or len(method_version) > 64
        ):
            raise CompilerSchemaError(
                "step.method_version must be null or a trimmed string of at most 64 characters"
            )
        return cls(
            category=category,
            method=method,
            params=params,
            purpose=purpose,
            method_version=method_version,
        )

    def to_pipeline_step(self) -> PipelineStep:
        return PipelineStep(
            self.category,
            self.method,
            dict(self.params),
            method_version=self.method_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "method": self.method,
            "params": dict(self.params),
            "purpose": self.purpose,
            "method_version": self.method_version,
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
    schema_version: int = 1
    plan_id: str = ""
    catalog_digest: str = ""
    catalog_assay: str | None = None
    attempt_count: int = 1

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
        if len(raw_steps) > MAX_PLAN_STEPS:
            raise CompilerSchemaError(f"plan.steps must contain at most {MAX_PLAN_STEPS} steps")
        raw_gaps = row.get("gaps", [])
        if not isinstance(raw_gaps, list):
            raise CompilerSchemaError("plan.gaps must be an array")
        if len(raw_gaps) > MAX_PLAN_GAPS:
            raise CompilerSchemaError(f"plan.gaps must contain at most {MAX_PLAN_GAPS} gaps")
        assay = row.get("assay_assumed", "unknown")
        if not isinstance(assay, str) or len(assay) > 128:
            raise CompilerSchemaError(
                "plan.assay_assumed must be a string of at most 128 characters"
            )
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

    @classmethod
    def from_serialized_dict(cls, value: Any) -> CompiledPlan:
        """Reconstruct a complete v1 plan artifact before integrity validation."""
        row = _expect_dict(value, "serialized plan")
        allowed = {
            "schema_version",
            "plan_id",
            "catalog_digest",
            "catalog_assay",
            "attempt_count",
            "question",
            "rationale",
            "steps",
            "gaps",
            "assay_assumed",
            "executor",
            "dry_run",
            "model",
        }
        unknown = set(row) - allowed
        if unknown:
            raise CompilerSchemaError(f"serialized plan has unknown fields: {sorted(unknown)}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
            raise CompilerSchemaError("serialized plan.question must be 1-4000 characters")
        executor = row.get("executor")
        dry_run = row.get("dry_run")
        model = row.get("model")
        schema_version = row.get("schema_version")
        plan_id = row.get("plan_id")
        digest = row.get("catalog_digest")
        catalog_assay = row.get("catalog_assay")
        attempts = row.get("attempt_count")
        if executor not in {"in-process", "nextflow"}:
            raise CompilerSchemaError("serialized plan.executor is invalid")
        if not isinstance(dry_run, bool):
            raise CompilerSchemaError("serialized plan.dry_run must be boolean")
        if not isinstance(model, str) or not model or len(model) > 256:
            raise CompilerSchemaError("serialized plan.model must be 1-256 characters")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise CompilerSchemaError("serialized plan.schema_version must be an integer")
        if not isinstance(plan_id, str) or not plan_id:
            raise CompilerSchemaError("serialized plan.plan_id must be a non-empty string")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise CompilerSchemaError("serialized plan.catalog_digest must be a sha256 digest")
        if catalog_assay is not None and (
            not isinstance(catalog_assay, str)
            or not catalog_assay
            or catalog_assay != catalog_assay.strip()
            or len(catalog_assay) > 128
        ):
            raise CompilerSchemaError(
                "serialized plan.catalog_assay must be null or a trimmed string"
            )
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
            raise CompilerSchemaError("serialized plan.attempt_count must be positive")
        model_payload = {
            key: row[key] for key in ("rationale", "steps", "gaps", "assay_assumed") if key in row
        }
        plan = cls.from_dict(
            model_payload,
            question=question,
            executor=executor,
            dry_run=dry_run,
            model=model,
        )
        plan.schema_version = schema_version
        plan.plan_id = plan_id
        plan.catalog_digest = digest
        plan.catalog_assay = catalog_assay
        plan.attempt_count = attempts
        return plan

    @property
    def pipeline_steps(self) -> list[PipelineStep]:
        return [step.to_pipeline_step() for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "catalog_digest": self.catalog_digest,
            "catalog_assay": self.catalog_assay,
            "attempt_count": self.attempt_count,
            "question": self.question,
            "rationale": self.rationale,
            "steps": [step.to_dict() for step in self.steps],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "assay_assumed": self.assay_assumed,
            "executor": self.executor,
            "dry_run": self.dry_run,
            "model": self.model,
        }
