"""Public facade for HistoWeave's evidence-governed decision protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .benchmark.decision import (
    CLAIM_BOUNDARY,
    CORE_RESEARCH_QUESTION,
    DECISION_SCHEMA_VERSION,
    DecisionAction,
    DecisionCard,
    DecisionEngine,
    DecisionPolicy,
    EvidenceCheck,
    EvidenceStatus,
    build_decision_card,
    load_decision_evidence,
)
from .benchmark.failure_fingerprint import FailureFingerprintAtlas
from .benchmark.isus import ISUSResult
from .benchmark.pareto import ObjectiveTable, ParetoDatasetResult
from .data import SpatialTable
from .io import read_bundle


def decide(
    data: SpatialTable,
    *,
    knowledge_base: Any,
    k_neighbours: int = 3,
    policy: DecisionPolicy | None = None,
    dataset_name: str = "user_dataset",
    task: str | None = None,
    platform: str | None = None,
    spatial_context_policy: str | None = None,
    objective_table: ObjectiveTable | None = None,
    pareto: ParetoDatasetResult | dict[str, Any] | None = None,
    isus: ISUSResult | None = None,
    isus_domain_key: str | None = None,
    failure_atlas: FailureFingerprintAtlas | dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> DecisionCard:
    """Return the method set justified by the supplied evidence.

    This is the single in-memory decision entry point. Its explicit arguments
    mirror the histoweave decide command so Python callers receive the same
    evidence contract and validation as CLI callers.
    """
    engine = DecisionEngine(
        knowledge_base,
        k_neighbours=k_neighbours,
        policy=policy,
        failure_atlas=failure_atlas,
        validation=validation,
    )
    return engine.decide(
        data,
        dataset_name=dataset_name,
        task=task,
        platform=platform,
        spatial_context_policy=spatial_context_policy,
        objective_table=objective_table,
        pareto=pareto,
        isus=isus,
        isus_domain_key=isus_domain_key,
        failure_atlas=failure_atlas,
        validation=validation,
    )


def decide_from_bundle(
    input_path: str | Path,
    *,
    knowledge_base: Any,
    task: str,
    dataset_name: str = "user_dataset",
    platform: str | None = None,
    spatial_context_policy: str | None = None,
    k_neighbours: int = 3,
    policy: DecisionPolicy | None = None,
    pareto_report: str | Path | None = None,
    isus_domain_key: str | None = None,
    failure_atlas_report: str | Path | None = None,
    validation_report: str | Path | None = None,
    out: str | Path | None = None,
) -> DecisionCard:
    """Run the file-backed decision workflow used by the CLI.

    Evidence report paths are loaded with the same schema-aware loader as the
    command line. When out is supplied, the decision card is written atomically
    as strict JSON.
    """
    data = read_bundle(input_path)
    pareto = load_decision_evidence(pareto_report) if pareto_report else None
    failure_atlas = load_decision_evidence(failure_atlas_report) if failure_atlas_report else None
    validation = load_decision_evidence(validation_report) if validation_report else None
    card = decide(
        data,
        knowledge_base=knowledge_base,
        k_neighbours=k_neighbours,
        policy=policy,
        dataset_name=dataset_name,
        task=task,
        platform=platform,
        spatial_context_policy=spatial_context_policy,
        pareto=pareto,
        isus_domain_key=isus_domain_key,
        failure_atlas=failure_atlas,
        validation=validation,
    )
    if out is not None:
        _write_card_atomic(Path(out), card)
    return card


def _write_card_atomic(path: Path, card: DecisionCard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(card.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CLAIM_BOUNDARY",
    "CORE_RESEARCH_QUESTION",
    "DECISION_SCHEMA_VERSION",
    "DecisionAction",
    "DecisionCard",
    "DecisionEngine",
    "DecisionPolicy",
    "EvidenceCheck",
    "EvidenceStatus",
    "build_decision_card",
    "decide",
    "decide_from_bundle",
]
