"""Evidence-governed, set-valued method decisions.

This module is the scientific decision layer of HistoWeave.  The surrounding
modules produce evidence; this module defines what that evidence is allowed to
justify.  In particular, it prevents a nearest-neighbour ranking from being
presented as a personalised recommendation when it does not beat a strong
global default or when its task/coverage contract is inadequate.

The protocol is deliberately set-valued:

1. admit only task-compatible benchmark evidence;
2. compare personalisation with the global-best baseline;
3. preserve non-dominated trade-offs when a Pareto analysis is available;
4. attach failure phenotypes and spatial-information diagnostics without
   treating either as proof of biological correctness; and
5. fall back or abstain when the evidence cannot support personalisation.

ISUS is a post-hoc descriptor because it requires domain labels.  It is never
used here as a target-free gate or as a predictor of method improvement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..data import SpatialTable
from .failure_fingerprint import FailureFingerprintAtlas
from .isus import ISUSResult, compute_isus_from_table
from .pareto import ObjectiveTable, ParetoDatasetResult, analyze_dataset
from .recommend import MethodRecommender, Recommendation
from .task_contract import AnalysisTask, is_domain_partition_task, tasks_admissible

DECISION_SCHEMA_VERSION = 1
CORE_RESEARCH_QUESTION = (
    "Given an explicit spatial-analysis task and incomplete benchmark evidence, "
    "which method set is justified, and when should the workflow fall back or abstain?"
)
CLAIM_BOUNDARY = (
    "The decision card prioritises methods for comparative execution. It does not establish "
    "biological validity, universal superiority, or a causal benefit from spatial modelling."
)


class DecisionAction(StrEnum):
    """Action justified by the currently admitted evidence."""

    PERSONALISED_SET = "personalised_set"
    GLOBAL_DEFAULT = "global_default"
    EVIDENCE_REQUIRED = "evidence_required"
    ABSTAIN = "abstain"


class EvidenceStatus(StrEnum):
    """Outcome of one auditable evidence check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class DecisionPolicy:
    """Predeclared thresholds for turning evidence into an action."""

    shortlist_size: int = 3
    min_support: int = 2
    min_rank_support_score: float = 0.25
    severe_failure_threshold: float = 0.65
    require_baseline_advantage: bool = True
    require_heldout_validation: bool = True

    def __post_init__(self) -> None:
        if self.shortlist_size < 1:
            raise ValueError("shortlist_size must be at least 1")
        if self.min_support < 1:
            raise ValueError("min_support must be at least 1")
        if not 0.0 <= self.min_rank_support_score <= 1.0:
            raise ValueError("min_rank_support_score must be between 0 and 1")
        if not 0.0 <= self.severe_failure_threshold <= 1.0:
            raise ValueError("severe_failure_threshold must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCheck:
    """One machine-readable check in the decision trace."""

    name: str
    status: EvidenceStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
        }


@dataclass
class DecisionCard:
    """Auditable output of the HistoWeave decision protocol."""

    dataset_name: str
    task: str
    action: DecisionAction
    primary_set: list[str]
    comparison_set: list[str]
    checks: list[EvidenceCheck]
    rationale: list[str]
    warnings: list[str]
    required_controls: list[str]
    recommendation: dict[str, Any]
    pareto: dict[str, Any] | None = None
    spatial_utility: dict[str, Any] | None = None
    failure_evidence: dict[str, Any] = field(default_factory=dict)
    validation_evidence: dict[str, Any] = field(default_factory=dict)
    evidence_roles: dict[str, str] = field(
        default_factory=lambda: {
            "recommendation": "pre_execution_reference_proxy",
            "pareto": "matched_multiobjective_decision_set",
            "spatial_utility": "posthoc_label_conditioned_descriptor",
            "failure_evidence": "synthetic_stress_test_warning",
            "validation_evidence": "grouped_heldout_generalisation_test",
        }
    )
    evidence_todo: list[dict[str, Any]] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)
    research_question: str = CORE_RESEARCH_QUESTION
    claim_boundary: str = CLAIM_BOUNDARY
    schema_version: int = DECISION_SCHEMA_VERSION

    @property
    def can_personalise(self) -> bool:
        return self.action is DecisionAction.PERSONALISED_SET

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": "histoweave.evidence_decision.v1",
            "research_question": self.research_question,
            "dataset_name": self.dataset_name,
            "task": self.task,
            "action": self.action.value,
            "can_personalise": self.can_personalise,
            "primary_set": list(self.primary_set),
            "comparison_set": list(self.comparison_set),
            "checks": [check.to_dict() for check in self.checks],
            "rationale": list(self.rationale),
            "warnings": list(self.warnings),
            "required_controls": list(self.required_controls),
            "recommendation": self.recommendation,
            "pareto": self.pareto,
            "spatial_utility": self.spatial_utility,
            "failure_evidence": self.failure_evidence,
            "validation_evidence": self.validation_evidence,
            "evidence_roles": dict(self.evidence_roles),
            "evidence_todo": list(self.evidence_todo),
            "policy": dict(self.policy),
            "claim_boundary": self.claim_boundary,
        }

    def summary(self) -> str:
        primary = ", ".join(self.primary_set) if self.primary_set else "none"
        comparison = ", ".join(self.comparison_set) if self.comparison_set else "none"
        lines = [
            f"Decision card for {self.dataset_name!r} [{self.task}]",
            f"  action: {self.action.value}",
            f"  primary set: {primary}",
            f"  comparison set: {comparison}",
            "  evidence checks:",
        ]
        lines.extend(
            f"    - {check.name}: {check.status.value} -- {check.detail}"
            for check in self.checks
        )
        if self.required_controls:
            lines.append("  required controls:")
            lines.extend(f"    - {control}" for control in self.required_controls)
        lines.append(f"  claim boundary: {self.claim_boundary}")
        return "\n".join(lines)


class DecisionEngine:
    """Run recommendation and convert it into an evidence-governed decision card."""

    def __init__(
        self,
        knowledge_base: Any,
        *,
        k_neighbours: int = 3,
        policy: DecisionPolicy | None = None,
        failure_atlas: FailureFingerprintAtlas | dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> None:
        self.recommender = MethodRecommender(knowledge_base, k_neighbours=k_neighbours)
        self.policy = policy or DecisionPolicy()
        self.failure_atlas = failure_atlas
        self.validation = validation

    def decide(
        self,
        data: SpatialTable,
        *,
        dataset_name: str = "user_dataset",
        task: str | AnalysisTask | None = None,
        platform: str | None = None,
        spatial_context_policy: str | None = None,
        objective_table: ObjectiveTable | None = None,
        pareto: ParetoDatasetResult | dict[str, Any] | None = None,
        isus: ISUSResult | None = None,
        isus_domain_key: str | None = None,
        failure_atlas: FailureFingerprintAtlas | dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> DecisionCard:
        """Create a decision card without silently inventing missing evidence."""
        if objective_table is not None and pareto is not None:
            raise ValueError("provide objective_table or pareto, not both")
        if isus is not None and isus_domain_key is not None:
            raise ValueError("provide isus or isus_domain_key, not both")

        recommendation = self.recommender.recommend(
            data,
            dataset_name=dataset_name,
            task=task,
            platform=platform,
            spatial_context_policy=spatial_context_policy,
        )
        resolved_pareto: ParetoDatasetResult | dict[str, Any] | None = pareto
        if objective_table is not None:
            resolved_pareto = analyze_dataset(objective_table)
        resolved_isus = isus
        if isus_domain_key is not None:
            resolved_isus = compute_isus_from_table(
                data,
                domain_key=isus_domain_key,
                dataset=dataset_name,
            )
        return build_decision_card(
            recommendation,
            pareto=resolved_pareto,
            isus=resolved_isus,
            failure_atlas=failure_atlas or self.failure_atlas,
            validation=self.validation if validation is None else validation,
            policy=self.policy,
        )


def build_decision_card(
    recommendation: Recommendation,
    *,
    pareto: ParetoDatasetResult | dict[str, Any] | None = None,
    isus: ISUSResult | None = None,
    failure_atlas: FailureFingerprintAtlas | dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    policy: DecisionPolicy | None = None,
) -> DecisionCard:
    """Apply the predeclared decision rule to already-computed evidence."""
    resolved_policy = policy or DecisionPolicy()
    checks: list[EvidenceCheck] = []
    rationale: list[str] = []
    warnings = list(recommendation.warnings)
    required_controls = [
        "Run the global-best method as a fixed comparator whenever personalisation is tested.",
        "Include an expression-only or coordinate-shuffled control for spatial claims.",
    ]

    ranked = list(recommendation.ranked_methods)
    ranked_names = [item.method for item in ranked]
    shortlist = ranked_names[: resolved_policy.shortlist_size]
    top = ranked[0] if ranked else None

    task_failure, task_detail = _task_evidence_status(recommendation)
    checks.append(EvidenceCheck("task_compatibility", task_failure, task_detail))

    if top is None or not recommendation.neighbours:
        coverage_status = EvidenceStatus.FAIL
        coverage_detail = "No ranked method with query-local benchmark evidence is available."
    elif top.support < resolved_policy.min_support:
        coverage_status = EvidenceStatus.WARN
        coverage_detail = (
            f"Top method support={top.support}, below the predeclared minimum "
            f"{resolved_policy.min_support}."
        )
    else:
        coverage_status = EvidenceStatus.PASS
        coverage_detail = (
            f"Top method has support={top.support} across "
            f"{len(recommendation.neighbours)} query-local references."
        )
    checks.append(EvidenceCheck("local_support", coverage_status, coverage_detail))

    if top is None:
        confidence_status = EvidenceStatus.FAIL
        confidence_detail = "No finite top-method confidence is available."
    elif top.confidence < resolved_policy.min_rank_support_score:
        confidence_status = EvidenceStatus.WARN
        confidence_detail = (
            f"Top rank-support heuristic={top.confidence:.3f}, below the predeclared minimum "
            f"{resolved_policy.min_rank_support_score:.3f}; it is not a calibrated probability."
        )
    else:
        confidence_status = EvidenceStatus.PASS
        confidence_detail = (
            f"Top rank-support heuristic={top.confidence:.3f}; it is not a calibrated probability."
        )
    checks.append(EvidenceCheck("rank_support_heuristic", confidence_status, confidence_detail))

    beats = recommendation.beats_global_best_baseline
    if beats is True:
        baseline_status = EvidenceStatus.PASS
        baseline_detail = (
            "The query-local proxy is strictly better on its reference neighbours; "
            "this is not held-out performance."
        )
    elif beats is False:
        baseline_status = EvidenceStatus.FAIL
        baseline_detail = (
            "The query-local reference-neighbour proxy does not beat the global-best comparator; "
            "personalisation is not justified."
        )
    else:
        baseline_status = EvidenceStatus.WARN
        baseline_detail = (
            "Reference-neighbour baseline advantage is unavailable; personalisation is unverified."
        )
    checks.append(
        EvidenceCheck("reference_neighbour_baseline_proxy", baseline_status, baseline_detail)
    )

    validation_status, validation_detail = _validation_status(validation)
    checks.append(EvidenceCheck("heldout_validation", validation_status, validation_detail))

    pareto_payload, pareto_frontier, pareto_matches = _pareto_payload(
        pareto,
        recommendation.dataset_name,
    )
    pareto_shortlist = _ordered_frontier_intersection(ranked_names, pareto_frontier)
    if pareto is None:
        pareto_status = EvidenceStatus.NOT_EVALUATED
        pareto_detail = "No matched multi-objective table was supplied."
    elif not pareto_matches:
        pareto_status = EvidenceStatus.WARN
        pareto_detail = "Pareto evidence belongs to a different dataset and was not used."
        warnings.append(pareto_detail)
    elif not pareto_frontier:
        pareto_status = EvidenceStatus.WARN
        pareto_detail = "Matched Pareto evidence contains no non-dominated configurations."
    elif not pareto_shortlist:
        pareto_status = EvidenceStatus.WARN
        pareto_detail = "The ranked shortlist and matched Pareto frontier do not overlap."
        warnings.append(pareto_detail)
    else:
        pareto_status = EvidenceStatus.PASS
        pareto_detail = (
            f"{len(pareto_shortlist)} ranked configuration(s) remain non-dominated."
        )
    checks.append(EvidenceCheck("pareto_tradeoffs", pareto_status, pareto_detail))

    spatial_payload: dict[str, Any] | None = None
    if isus is None:
        spatial_status = EvidenceStatus.NOT_EVALUATED
        spatial_detail = "ISUS was not computed; no label-dependent spatial-utility claim is made."
    elif str(isus.dataset) != str(recommendation.dataset_name):
        spatial_status = EvidenceStatus.WARN
        spatial_detail = "ISUS belongs to a different dataset and was not used."
        warnings.append(spatial_detail)
    else:
        spatial_payload = isus.to_dict()
        spatial_status = EvidenceStatus.WARN if isus.flags else EvidenceStatus.PASS
        null_bit = ""
        if isus.n_null > 0 and isus.significant is not None:
            z = isus.z_score_i_d_s_given_e
            z_s = (
                "NA"
                if z is None
                else ("inf" if abs(float(z)) == float("inf") else f"{float(z):.3g}")
            )
            null_bit = (
                f" permutation significant={isus.significant} Z={z_s}"
                f" (n_null={isus.n_null}, band_source={isus.band_source});"
            )
        gain_bit = ""
        if isus.expected_spatial_ari_gain is not None:
            gain_bit = (
                f" expected spatial ARI gain≈{isus.expected_spatial_ari_gain:.3f}"
                f" (reliability={isus.gain_prediction_reliability});"
            )
        spatial_detail = (
            f"Post-hoc ISUS band={isus.band};{null_bit}{gain_bit} this descriptor is not a "
            "pre-execution predictor of method-level improvement."
        )
        if isus.band == "expression-sufficient":
            required_controls.append(
                "Treat the expression-only analysis as a co-primary comparator; low ISUS does "
                "not by itself veto a spatial method."
            )
    checks.append(EvidenceCheck("spatial_information_audit", spatial_status, spatial_detail))

    failure_payload, severe_methods = _failure_evidence(
        failure_atlas,
        shortlist,
        threshold=resolved_policy.severe_failure_threshold,
    )
    if failure_atlas is None:
        failure_status = EvidenceStatus.NOT_EVALUATED
        failure_detail = "No failure-fingerprint atlas was supplied."
    elif not failure_payload:
        failure_status = EvidenceStatus.WARN
        failure_detail = "The supplied atlas does not cover the ranked shortlist."
    elif severe_methods:
        failure_status = EvidenceStatus.WARN
        failure_detail = (
            "Synthetic failure evidence flags severe modes for: " + ", ".join(severe_methods)
        )
        required_controls.append(
            "Inspect fragmentation, merge, noise, and structural failure modes for every "
            "flagged method; fingerprints are warnings, not automatic exclusions."
        )
    else:
        failure_status = EvidenceStatus.PASS
        failure_detail = "Covered candidates are below the predeclared severe-mode threshold."
    checks.append(EvidenceCheck("failure_phenotypes", failure_status, failure_detail))

    hard_failure = task_failure is EvidenceStatus.FAIL or coverage_status is EvidenceStatus.FAIL
    weak_local_evidence = (
        task_failure is EvidenceStatus.WARN
        or coverage_status is EvidenceStatus.WARN
        or confidence_status is EvidenceStatus.WARN
    )
    weak_tradeoff_evidence = pareto is not None and pareto_status is not EvidenceStatus.PASS
    global_method = recommendation.global_best_method
    if hard_failure:
        action = DecisionAction.ABSTAIN
        primary: list[str] = []
        comparison = shortlist
        rationale.append("Task-valid query-local evidence is absent, so the protocol abstains.")
    elif beats is False or validation_status is EvidenceStatus.FAIL:
        action = DecisionAction.GLOBAL_DEFAULT
        primary = [global_method] if global_method else []
        comparison = _without(shortlist, primary)
        rationale.append(
            "The personalised ranking failed its fixed baseline gate; use the global default "
            "and retain the local shortlist only as a comparison panel."
        )
    elif (
        weak_local_evidence
        or weak_tradeoff_evidence
        or (beats is None and resolved_policy.require_baseline_advantage)
        or (
            resolved_policy.require_heldout_validation
            and validation_status is not EvidenceStatus.PASS
        )
    ):
        action = DecisionAction.EVIDENCE_REQUIRED
        primary = []
        comparison = shortlist
        rationale.append(
            "Coverage, confidence, or baseline evidence is insufficient for a deployment choice."
        )
    else:
        action = DecisionAction.PERSONALISED_SET
        primary = pareto_shortlist or shortlist
        comparison = _without([global_method] if global_method else [], primary)
        rationale.append(
            "Personalisation passed the baseline gate; the output remains a method set rather "
            "than an asserted universal winner."
        )
        if pareto_shortlist:
            rationale.append("The primary set is restricted to matched non-dominated candidates.")

    return DecisionCard(
        dataset_name=recommendation.dataset_name,
        task=recommendation.task,
        action=action,
        primary_set=_unique(primary),
        comparison_set=_unique(comparison),
        checks=checks,
        rationale=rationale,
        warnings=_unique(warnings),
        required_controls=_unique(required_controls),
        recommendation=recommendation.to_dict(),
        pareto=pareto_payload if pareto_matches else None,
        spatial_utility=spatial_payload,
        failure_evidence=failure_payload,
        validation_evidence=dict(validation or {}),
        evidence_todo=list(recommendation.evidence_todo),
        policy=resolved_policy.to_dict(),
    )


def _task_evidence_status(recommendation: Recommendation) -> tuple[EvidenceStatus, str]:
    invalid_kinds = {"self_supervised", "leiden", "louvain"}
    if is_domain_partition_task(recommendation.task):
        invalid_kinds.add("cluster_proxy")
    if recommendation.task == AnalysisTask.VIRTUAL_ST.value:
        # Virtual ST may not borrow domain-partition or cell-type ground truth.
        invalid_kinds.update(
            {
                "spatial_domain",
                "spatial_protein_domain",
                "spatial_chromatin_domain",
                "cluster_proxy",
                "cell_type",
            }
        )
    invalid = [
        str(item.get("name"))
        for item in recommendation.neighbours
        if str(item.get("ground_truth_kind") or "").lower() in invalid_kinds
    ]
    if invalid:
        return (
            EvidenceStatus.FAIL,
            "Circular/self-supervised ground truth occurs in query-local evidence: "
            + ", ".join(invalid),
        )
    missing_kind = [
        str(item.get("name"))
        for item in recommendation.neighbours
        if not item.get("ground_truth_kind")
    ]
    if missing_kind:
        return (
            EvidenceStatus.WARN,
            "Ground-truth semantics are undeclared for query-local evidence: "
            + ", ".join(missing_kind),
        )
    declared = [item for item in recommendation.neighbours if item.get("task")]
    compatible = [
        item
        for item in declared
        if tasks_admissible(recommendation.task, item.get("task"))
    ]
    if declared and not compatible:
        return EvidenceStatus.FAIL, "No declared query-local reference matches the analysis task."
    if declared and len(compatible) < len(declared):
        return (
            EvidenceStatus.WARN,
            "Only part of the query-local evidence matches the task "
            "(cross-modal domain evidence is never admitted).",
        )
    return EvidenceStatus.PASS, f"Evidence is compatible with task={recommendation.task!r}."


def _validation_status(validation: dict[str, Any] | None) -> tuple[EvidenceStatus, str]:
    if validation is None:
        return (
            EvidenceStatus.NOT_EVALUATED,
            "No study- or dataset-grouped held-out validation was supplied.",
        )
    protocol = str(validation.get("protocol") or "").lower()
    accepted = {"study_grouped_holdout", "dataset_grouped_holdout", "external_holdout"}
    if protocol not in accepted:
        return EvidenceStatus.WARN, "Validation protocol is not a recognised grouped holdout."
    try:
        n_queries = int(validation.get("n_queries") or 0)
    except (TypeError, ValueError):
        n_queries = 0
    beats = validation.get("beats_global_best")
    if beats is False:
        return (
            EvidenceStatus.FAIL,
            f"Grouped holdout (n={n_queries}) does not beat the global-best comparator.",
        )
    if beats is not True:
        return EvidenceStatus.WARN, "Grouped holdout does not report a baseline conclusion."
    if n_queries < 5:
        return EvidenceStatus.WARN, f"Grouped holdout has only n={n_queries} queries."
    return EvidenceStatus.PASS, f"Grouped holdout supports baseline advantage across n={n_queries}."


def _pareto_payload(
    pareto: ParetoDatasetResult | dict[str, Any] | None,
    dataset_name: str,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    if pareto is None:
        return None, [], False
    if isinstance(pareto, ParetoDatasetResult):
        payload = pareto.to_dict()
    else:
        payload = dict(pareto)
        datasets = payload.get("datasets")
        if isinstance(datasets, dict):
            if dataset_name in datasets:
                payload = dict(datasets[dataset_name])
                payload.setdefault("dataset", dataset_name)
            elif len(datasets) == 1:
                only_name, only_payload = next(iter(datasets.items()))
                payload = dict(only_payload)
                payload.setdefault("dataset", str(only_name))
    dataset = str(payload.get("dataset") or "")
    matches = dataset == str(dataset_name)
    frontier = [str(value) for value in payload.get("frontier", [])]
    return payload, frontier, matches


def _ordered_frontier_intersection(ranked: list[str], frontier: list[str]) -> list[str]:
    exact = set(frontier)
    base = {_base_method(name) for name in frontier}
    return [
        name
        for name in ranked
        if name in exact or ("@" not in name and _base_method(name) in base)
    ]


def _failure_evidence(
    atlas: FailureFingerprintAtlas | dict[str, Any] | None,
    candidates: list[str],
    *,
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    if atlas is None:
        return {}, []
    if isinstance(atlas, FailureFingerprintAtlas):
        rows = {name: fp.to_dict() for name, fp in atlas.by_method().items()}
    else:
        raw = atlas.get("fingerprints", [])
        rows = {
            str(item.get("method")): dict(item)
            for item in raw
            if isinstance(item, dict) and item.get("method")
        }
    selected: dict[str, Any] = {}
    severe: list[str] = []
    for candidate in candidates:
        row = rows.get(candidate) or rows.get(_base_method(candidate))
        if not row:
            continue
        selected[candidate] = row
        vector = row.get("vector", {})
        values = [float(value) for value in vector.values() if _is_number(value)]
        if values and max(values) >= threshold:
            severe.append(candidate)
    return selected, severe


def load_decision_evidence(path: str | Path) -> dict[str, Any]:
    """Load a JSON evidence artifact for CLI and workflow composition."""
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("decision evidence JSON must contain an object")
    return payload


def _base_method(name: str) -> str:
    return str(name).split("@", 1)[0].strip()


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _without(values: list[str], excluded: list[str]) -> list[str]:
    blocked = set(excluded)
    return [value for value in values if value not in blocked]
