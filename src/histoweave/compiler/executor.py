"""Dispatch validated compiled plans to HistoWeave executors."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from ..data import SpatialTable
from ..io import write_bundle
from ..plugins import MethodCategory, get_method
from ..report import build_report
from ..workflow import run_pipeline
from .schema import CompiledPlan


def run_compiled(
    plan: CompiledPlan,
    *,
    data: SpatialTable,
    out: str | Path = "histoweave_compiled_report.html",
) -> SpatialTable | dict[str, Any]:
    """Run in process, or emit a Nextflow hand-off without spawning a shell."""
    for gap in plan.gaps:
        warnings.warn(
            f"Compiler approximated {gap.concept}: {gap.degraded_to}",
            UserWarning,
            stacklevel=2,
        )
    if plan.executor == "nextflow":
        out_path = Path(out)
        params_path = out_path.with_suffix(".params.json")
        params_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path = out_path.with_suffix(".input.ttab")
        handoff_data = data.copy()
        handoff_data.uns.setdefault("run_manifest", {})["compiler"] = _compiler_metadata(plan)
        write_bundle(handoff_data, bundle_path)
        payload = _nextflow_params(plan, bundle_path=bundle_path, outdir=out_path.parent)
        params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "params_path": str(params_path),
            "command": f"nextflow run workflows/nextflow/main.nf -params-file {params_path}",
        }
    if plan.executor != "in-process":
        raise ValueError("executor must be 'in-process' or 'nextflow'")
    result = run_pipeline(data, plan.pipeline_steps)
    manifest = result.uns.setdefault("run_manifest", {})
    manifest["compiler"] = _compiler_metadata(plan)
    build_report(result, out)
    return result


_NEXTFLOW_STEPS = {
    "qc": ("qc", "qc_method", "qc_params"),
    "normalization": ("normalize", "normalize_method", "normalize_params"),
    "domain_detection": ("domain_detection", "domain_method", "domain_params"),
    "annotation": ("annotation", "annotation_method", "annotation_params"),
    "deconvolution": ("deconvolution", "deconvolution_method", "deconvolution_params"),
}


def _nextflow_params(
    plan: CompiledPlan,
    *,
    bundle_path: Path,
    outdir: Path,
) -> dict[str, Any]:
    """Translate compiler IR to the flat params schema consumed by ``main.nf``."""
    unsupported = sorted({step.category for step in plan.steps} - set(_NEXTFLOW_STEPS))
    if unsupported:
        raise ValueError(
            "Nextflow executor does not support compiler categories "
            f"{unsupported}; use the in-process executor"
        )

    payload: dict[str, Any] = {
        "bundle": str(bundle_path.resolve()),
        "outdir": str(outdir.resolve()),
    }
    tokens: list[str] = []
    for step in plan.steps:
        token, method_key, params_key = _NEXTFLOW_STEPS[step.category]
        if token in tokens:
            raise ValueError(f"Nextflow executor supports only one {step.category!r} step")
        tokens.append(token)
        payload[method_key] = step.method
        params = dict(step.params)
        if step.category == "domain_detection":
            n_domains = params.pop("n_domains", None)
            if n_domains is None:
                method = get_method(MethodCategory.DOMAIN_DETECTION, step.method)
                n_domains = next(
                    (spec.default for spec in method.spec.params if spec.name == "n_domains"),
                    None,
                )
            if n_domains is None:
                raise ValueError(f"{step.method} must define n_domains for Nextflow execution")
            payload["n_domains"] = n_domains
        payload[params_key] = _encode_nextflow_params(params)
    payload["steps"] = ",".join([*tokens, "report"])
    return payload


def _encode_nextflow_params(params: dict[str, Any]) -> list[str]:
    return [
        f"{name}={json.dumps(value, separators=(',', ':'))}" for name, value in params.items()
    ]


def _compiler_metadata(plan: CompiledPlan) -> dict[str, Any]:
    return {
        "question": plan.question,
        "rationale": plan.rationale,
        "assay_assumed": plan.assay_assumed,
        "model": plan.model,
        "executor": plan.executor,
        "gaps": [gap.to_dict() for gap in plan.gaps],
    }
