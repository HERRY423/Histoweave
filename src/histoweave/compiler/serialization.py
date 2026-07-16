"""Versioned serialization and integrity checks for compiled pipeline plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .catalog import build_catalog
from .schema import CompiledPlan, CompilerSchemaError

COMPILER_SCHEMA_VERSION = 1
_PLAN_ID_PREFIX = "hwc1_"


def catalog_digest(catalog: list[dict[str, Any]]) -> str:
    """Return a stable digest of the exact live method catalog shown to the model."""

    ordered = sorted(
        catalog,
        key=lambda row: (
            str(row.get("category", "")),
            str(row.get("name", "")),
            str(row.get("version", "")),
        ),
    )
    encoded = json.dumps(
        ordered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _identity_payload(plan: CompiledPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "question": plan.question,
        "rationale": plan.rationale,
        "steps": [step.to_dict() for step in plan.steps],
        "gaps": [gap.to_dict() for gap in plan.gaps],
        "assay_assumed": plan.assay_assumed,
        "executor": plan.executor,
        "dry_run": plan.dry_run,
        "model": plan.model,
        "catalog_digest": plan.catalog_digest,
        "catalog_assay": plan.catalog_assay,
        "attempt_count": plan.attempt_count,
    }


def plan_fingerprint(plan: CompiledPlan) -> str:
    """Compute the content-addressed v1 plan identifier."""

    encoded = json.dumps(
        _identity_payload(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _PLAN_ID_PREFIX + hashlib.sha256(encoded).hexdigest()[:24]


def seal_plan(
    plan: CompiledPlan,
    *,
    catalog_fingerprint: str,
    catalog_assay: str | None = None,
    attempt_count: int,
) -> CompiledPlan:
    """Attach compiler provenance and a deterministic integrity identifier."""

    if not catalog_fingerprint.startswith("sha256:"):
        raise ValueError("catalog_fingerprint must be a sha256 digest")
    if catalog_assay is not None and (
        not isinstance(catalog_assay, str)
        or not catalog_assay.strip()
        or len(catalog_assay) > 128
    ):
        raise ValueError("catalog_assay must be null or a non-empty string")
    if attempt_count < 1:
        raise ValueError("attempt_count must be positive")
    plan.schema_version = COMPILER_SCHEMA_VERSION
    plan.catalog_digest = catalog_fingerprint
    plan.catalog_assay = catalog_assay
    plan.attempt_count = int(attempt_count)
    plan.plan_id = plan_fingerprint(plan)
    return plan


def verify_plan_identity(plan: CompiledPlan) -> CompiledPlan:
    """Reject unsupported schema versions or tampered serialized plans."""

    if plan.schema_version != COMPILER_SCHEMA_VERSION:
        raise CompilerSchemaError(
            f"unsupported compiler schema_version {plan.schema_version}; "
            f"expected {COMPILER_SCHEMA_VERSION}"
        )
    if not plan.plan_id:
        raise CompilerSchemaError("serialized plan is missing plan_id")
    expected = plan_fingerprint(plan)
    if plan.plan_id != expected:
        raise CompilerSchemaError(
            f"plan_id integrity check failed: stored {plan.plan_id!r}, expected {expected!r}"
        )
    return plan


def save_plan(plan: CompiledPlan, path: str | Path) -> Path:
    """Atomically persist a sealed plan as strict finite JSON."""

    verify_plan_identity(plan)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(
                plan.to_dict(),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_plan(
    path: str | Path,
    *,
    require_catalog_match: bool = False,
) -> CompiledPlan:
    """Load, integrity-check, and registry-validate a serialized v1 plan.

    Catalog drift is reported through an opt-in strict check because adding an unrelated
    method changes the full catalog digest without invalidating a registry-backed plan.
    Every referenced method and parameter is always validated against the live registry.
    """

    from .validate import CompilerValidationError, validate_plan

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompilerSchemaError(f"serialized plan is not valid JSON: {exc}") from exc
    plan = CompiledPlan.from_serialized_dict(payload)
    verify_plan_identity(plan)
    validate_plan(plan)
    if require_catalog_match:
        current = catalog_digest(build_catalog(assay=plan.catalog_assay))
        if current != plan.catalog_digest:
            raise CompilerValidationError(
                "compiled plan catalog digest differs from the live registry; "
                "recompile the question or load without require_catalog_match"
            )
    return plan
