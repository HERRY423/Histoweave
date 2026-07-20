"""Public facade for HistoWeave's evidence-governed decision protocol."""

from __future__ import annotations

from typing import Any

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
)
from .data import SpatialTable


def decide(
    data: SpatialTable,
    *,
    knowledge_base: Any,
    k_neighbours: int = 3,
    policy: DecisionPolicy | None = None,
    **kwargs: Any,
) -> DecisionCard:
    """Return the method set justified by the supplied evidence.

    This is the single high-level decision entry point. Optional Pareto, ISUS,
    failure-atlas, and held-out validation evidence is passed through ``kwargs``
    to :meth:`DecisionEngine.decide` and retains its declared evidence role.
    """
    engine = DecisionEngine(
        knowledge_base,
        k_neighbours=k_neighbours,
        policy=policy,
    )
    return engine.decide(data, **kwargs)


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
]
