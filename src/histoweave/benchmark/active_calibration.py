"""Active-learning calibration for the method recommender.

When the recommender fails to beat the global-best baseline
(``beats_global_best_baseline=False``), personalisation is not yet justified.
This module produces an **evidence-acquisition todo list**: specific
``(dataset, method)`` pairs to run next, ranked by a practical **expected
information gain (EIG)** heuristic that prioritises experiments most likely to
reduce recommendation uncertainty.

EIG for pair ``(d, m)``
-----------------------
::

    EIG(d, m) = similarity(query, d)
                × method_importance(m)
                × novelty(d, m)
                × decision_relevance(d, m)

* **novelty** — 1 if the performance entry is missing/NaN, else a small residual
  for high-variance re-checks (default 0 for complete cells).
* **method_importance** — higher for methods near the recommendation frontier
  (top-k, global-best, or high-uncertainty configs).
* **decision_relevance** — boost when filling the cell could change the
  top-1 vs global-best comparison on that neighbour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .landscape import LandscapeResult
from .recommend import MethodRecommender, Recommendation

CALIBRATION_SCHEMA_VERSION = 1


@dataclass
class EvidenceTask:
    """One recommended experiment: run *method* on *dataset* and record the score."""

    dataset: str
    method: str
    expected_information_gain: float
    reason: str
    priority: int
    novelty: float
    method_importance: float
    neighbour_similarity: float
    currently_missing: bool
    decision_relevance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        return f"{self.dataset} × {self.method}"


@dataclass
class CalibrationPlan:
    """Full evidence-acquisition plan for one recommendation."""

    needed: bool
    beats_global_best_baseline: bool | None
    global_best_method: str | None
    recommended_method: str | None
    selection_regret_vs_global_best: float | None
    todo: list[EvidenceTask] = field(default_factory=list)
    summary_message: str = ""
    schema_version: int = CALIBRATION_SCHEMA_VERSION
    protocol: str = "histoweave.active_calibration.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "needed": self.needed,
            "beats_global_best_baseline": self.beats_global_best_baseline,
            "global_best_method": self.global_best_method,
            "recommended_method": self.recommended_method,
            "selection_regret_vs_global_best": self.selection_regret_vs_global_best,
            "summary_message": self.summary_message,
            "todo": [t.to_dict() for t in self.todo],
            "n_tasks": len(self.todo),
        }

    def summary(self) -> str:
        if not self.needed:
            return self.summary_message or "No evidence acquisition needed."
        lines = [self.summary_message or "Evidence acquisition todo:"]
        for task in self.todo:
            lines.append(
                f"  {task.priority:>2}. {task.label():<40} "
                f"EIG={task.expected_information_gain:.4f}  — {task.reason}"
            )
        return "\n".join(lines)


def propose_evidence_acquisition(
    recommender: MethodRecommender,
    recommendation: Recommendation,
    *,
    top_n: int = 10,
    top_methods: int = 5,
    min_eig: float = 1e-6,
) -> CalibrationPlan:
    """Build an evidence-acquisition todo list for *recommendation*.

    Always computable; ``needed=True`` when the recommender does not beat the
    global-best baseline (or when baseline diagnostics are unavailable but the
    knowledge base has coverage holes near the query).
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    beats = recommendation.beats_global_best_baseline
    best = recommendation.best()
    chosen = best.method if best is not None else None
    global_best = recommendation.global_best_method
    regret = recommendation.selection_regret_vs_global_best

    kb: LandscapeResult = recommender._kb  # intentional: calibrated against same KB
    higher = bool(getattr(kb, "higher_is_better", True))
    methods = list(kb.method_order())
    if not methods:
        return CalibrationPlan(
            needed=False,
            beats_global_best_baseline=beats,
            global_best_method=global_best,
            recommended_method=chosen,
            selection_regret_vs_global_best=regret,
            summary_message="Knowledge base has no methods; nothing to acquire.",
        )

    # Methods near the decision frontier.
    ranked_names = [m.method for m in recommendation.ranked_methods[:top_methods]]
    frontier: set[str] = set(ranked_names)
    if global_best:
        frontier.add(global_best)
    if chosen:
        frontier.add(chosen)
    # Also include methods with high uncertainty on the recommendation.
    for m in recommendation.ranked_methods[:top_methods]:
        if m.uncertainty >= 0.05:
            frontier.add(m.method)

    importance = _method_importance(recommendation, methods, frontier, global_best)

    # Neighbour similarities (query-local). Fall back to uniform over KB datasets.
    neighbour_sim = {
        str(n["name"]): float(n.get("similarity") or 0.0) for n in recommendation.neighbours
    }
    if not neighbour_sim:
        # Use all reference datasets with tiny equal weight.
        for name in recommender._ref_names:
            neighbour_sim[str(name)] = 0.1

    # Candidate datasets: neighbours first, then other KB datasets with lower sim.
    candidate_datasets = list(neighbour_sim.keys())
    for name in recommender._ref_names:
        if name not in neighbour_sim:
            candidate_datasets.append(name)
            neighbour_sim[name] = 0.05  # far from query → low priority

    tasks: list[EvidenceTask] = []
    for dataset in candidate_datasets:
        row = kb.performance.get(dataset, {})
        sim = float(neighbour_sim.get(dataset, 0.05))
        for method in methods:
            raw = row.get(method, float("nan"))
            missing = True
            if raw is not None:
                try:
                    missing = not np.isfinite(float(raw))
                except (TypeError, ValueError):
                    missing = True

            novelty = 1.0 if missing else 0.0
            # Only propose missing landscape cells (re-runs belong to multi-seed protocols).
            if not missing:
                continue

            imp = float(importance.get(method, 0.05))
            decision = _decision_relevance(
                dataset=dataset,
                method=method,
                row=row,
                chosen=chosen,
                global_best=global_best,
                higher=higher,
                missing=missing,
            )
            eig = sim * imp * novelty * (1.0 + decision)
            if eig < min_eig:
                continue
            reason = _reason(
                dataset=dataset,
                method=method,
                missing=missing,
                sim=sim,
                imp=imp,
                decision=decision,
                chosen=chosen,
                global_best=global_best,
                in_frontier=method in frontier,
            )
            tasks.append(
                EvidenceTask(
                    dataset=dataset,
                    method=method,
                    expected_information_gain=round(float(eig), 6),
                    reason=reason,
                    priority=0,
                    novelty=novelty,
                    method_importance=round(imp, 4),
                    neighbour_similarity=round(sim, 4),
                    currently_missing=missing,
                    decision_relevance=round(decision, 4),
                )
            )

    tasks.sort(
        key=lambda t: (-t.expected_information_gain, t.dataset, t.method),
    )
    tasks = tasks[:top_n]
    for i, task in enumerate(tasks, start=1):
        task.priority = i

    needed = beats is False or (beats is None and len(tasks) > 0)
    # If we beat global best and have no missing frontier cells, not needed.
    if beats is True and not tasks:
        needed = False

    if beats is False:
        msg = (
            "Recommender does not beat the global-best baseline "
            f"({global_best!r}). Run the following dataset×method pairs to "
            "maximise expected information gain and re-calibrate."
        )
    elif beats is None:
        msg = (
            "Baseline diagnostics unavailable. Filling missing performance "
            "cells near the query will improve recommendation confidence."
        )
    elif tasks:
        msg = (
            "Recommender beats global-best, but coverage holes remain near the "
            "query. Optional evidence tasks (lower priority):"
        )
        needed = False  # optional — still return todo but mark not required
    else:
        msg = (
            "Recommender beats the global-best baseline; no high-EIG missing "
            "cells near the query."
        )
        needed = False

    # When beats is False we force needed=True even if todo is empty (explain why).
    if beats is False and not tasks:
        msg = (
            "Recommender does not beat the global-best baseline "
            f"({global_best!r}), but the knowledge base has no missing "
            "performance cells near the query. Expand the landscape with new "
            "reference datasets close to this sample."
        )
        needed = True

    return CalibrationPlan(
        needed=needed,
        beats_global_best_baseline=beats,
        global_best_method=global_best,
        recommended_method=chosen,
        selection_regret_vs_global_best=regret,
        todo=tasks,
        summary_message=msg,
    )


def attach_calibration(
    recommender: MethodRecommender,
    recommendation: Recommendation,
    *,
    top_n: int = 10,
    always: bool = False,
) -> CalibrationPlan:
    """Compute calibration plan and attach it onto *recommendation* in-place.

    Parameters
    ----------
    always
        When *False* (default), only attach a non-empty plan when
        ``beats_global_best_baseline is False`` (or diagnostics missing with
        holes).  When *True*, always attach the plan for inspection.
    """
    plan = propose_evidence_acquisition(recommender, recommendation, top_n=top_n)
    if always or plan.needed or plan.todo:
        # Store on recommendation for to_dict / CLI.
        recommendation.evidence_todo = [t.to_dict() for t in plan.todo]
        recommendation.calibration = plan.to_dict()
        if plan.needed:
            recommendation.warnings.append(
                "Active calibration: recommender does not beat global-best; "
                f"{len(plan.todo)} evidence task(s) proposed "
                "(see calibration.todo / evidence_todo)."
            )
    return plan


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _method_importance(
    recommendation: Recommendation,
    methods: list[str],
    frontier: set[str],
    global_best: str | None,
) -> dict[str, float]:
    imp: dict[str, float] = {m: 0.05 for m in methods}
    n = max(len(recommendation.ranked_methods), 1)
    for rank, m in enumerate(recommendation.ranked_methods):
        # Higher rank (lower index) → higher importance.
        base = 1.0 - rank / n
        unc_boost = 0.25 * float(m.uncertainty or 0.0)
        imp[m.method] = max(imp.get(m.method, 0.05), 0.15 + 0.7 * base + unc_boost)
    for name in frontier:
        imp[name] = max(imp.get(name, 0.05), 0.55)
    if global_best:
        imp[global_best] = max(imp.get(global_best, 0.05), 0.75)
    return imp


def _decision_relevance(
    *,
    dataset: str,
    method: str,
    row: dict[str, Any],
    chosen: str | None,
    global_best: str | None,
    higher: bool,
    missing: bool,
) -> float:
    """Boost when this cell could flip top-1 vs global-best on this neighbour."""
    _ = dataset
    if not missing:
        return 0.0
    boost = 0.0
    if method == chosen or method == global_best:
        boost += 0.5
    # If we have the other of {chosen, global_best} on this dataset, filling the
    # missing one is highly decision-relevant.
    if chosen and global_best and chosen != global_best:
        other = global_best if method == chosen else chosen if method == global_best else None
        if other is not None:
            other_val = row.get(other)
            if other_val is not None:
                try:
                    if np.isfinite(float(other_val)):
                        boost += 0.5
                except (TypeError, ValueError):
                    pass
    # Mild boost for any missing frontier cell.
    if higher:
        pass
    return float(np.clip(boost, 0.0, 1.5))


def _reason(
    *,
    dataset: str,
    method: str,
    missing: bool,
    sim: float,
    imp: float,
    decision: float,
    chosen: str | None,
    global_best: str | None,
    in_frontier: bool,
) -> str:
    parts: list[str] = []
    if missing:
        parts.append("missing score")
    if sim >= 0.5:
        parts.append(f"near query (sim={sim:.2f})")
    elif sim >= 0.15:
        parts.append(f"neighbour-adjacent (sim={sim:.2f})")
    if method == global_best:
        parts.append("global-best baseline")
    if method == chosen:
        parts.append("current recommendation")
    elif in_frontier:
        parts.append("recommendation frontier")
    if decision >= 0.5:
        parts.append("could revise top-1 vs baseline")
    if imp >= 0.7:
        parts.append("high method importance")
    if not parts:
        parts.append("coverage expansion")
    return "; ".join(parts)
