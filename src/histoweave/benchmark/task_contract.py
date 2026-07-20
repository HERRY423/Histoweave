"""Task contracts for scientifically valid benchmarking.

Nature-Methods-grade evaluation requires that the *question*, the *ground-truth
kind*, and the *metric* are declared and checked before scores are aggregated.

Key hard rules
--------------
* Spatial-domain recovery may only score against expert / histology-derived
  spatial partitions (or other declared ``spatial_domain`` labels).
* Protein- and chromatin-domain recovery are **separate analysis questions**
  from RNA spatial domains; their rankings are never mixed.
* Cell-type recovery may score against cell-type or state annotations.
* Virtual ST (H&E → predicted expression) is scored against *measured*
  expression, never against self-predicted labels.
* **Self-supervised cluster labels (e.g. Leiden computed on the same expression
  matrix) are forbidden as domain ground truth** — they create circular advantage
  for expression-only methods and invalidate cross-platform ARI comparisons.
* Cross-modal evidence is either exact-match (admissible) or explicitly rejected;
  related domain-family pairs are auditable but never scored into a ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ..data import SpatialTable


class AnalysisTask(StrEnum):
    """Benchmark / recommendation analysis targets."""

    # Domain-partition recovery on measured molecular modalities.
    SPATIAL_DOMAIN = "spatial_domain"
    SPATIAL_PROTEIN_DOMAIN = "spatial_protein_domain"
    SPATIAL_CHROMATIN_DOMAIN = "spatial_chromatin_domain"
    # Classical ST tasks.
    CELL_TYPE = "cell_type"
    SVG = "svg"
    DECONVOLUTION = "deconvolution"
    # H&E (or other histology) → predicted spatial transcriptomics.
    VIRTUAL_ST = "virtual_st"


class GroundTruthKind(StrEnum):
    """What the evaluation labels / targets actually represent."""

    SPATIAL_DOMAIN = "spatial_domain"
    SPATIAL_PROTEIN_DOMAIN = "spatial_protein_domain"
    SPATIAL_CHROMATIN_DOMAIN = "spatial_chromatin_domain"
    CELL_TYPE = "cell_type"
    CLUSTER_PROXY = "cluster_proxy"
    SELF_SUPERVISED = "self_supervised"
    # Paired measured ST used to score H&E→expression predictors.
    MEASURED_EXPRESSION = "measured_expression"
    NONE = "none"


class CrossModalRelation(StrEnum):
    """Audit label for how two analysis tasks relate across modalities.

    Only :attr:`SAME` is admissible for recommendation / decision evidence.
    :attr:`SAME_FAMILY` means both tasks recover spatial partitions but on
    different molecular modalities — scientifically related, **not** transferable
    for method ranking. :attr:`INCOMPATIBLE` means distinct scientific questions.
    """

    SAME = "same"
    SAME_FAMILY = "same_family"
    INCOMPATIBLE = "incompatible"


# Labels that are never acceptable as spatial-domain ground truth.
_FORBIDDEN_DOMAIN_LABEL_TOKENS = (
    "leiden",
    "louvain",
    "self_cluster",
    "self-supervised",
    "proxy_leiden",
)

# Domain-partition recovery tasks share a question *shape* (recover spatial
# partitions) but operate on different molecular matrices. Cross-modal transfer
# of method rankings is forbidden.
DOMAIN_PARTITION_TASKS: frozenset[AnalysisTask] = frozenset(
    {
        AnalysisTask.SPATIAL_DOMAIN,
        AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
        AnalysisTask.SPATIAL_CHROMATIN_DOMAIN,
    }
)

# Canonical ground-truth kinds allowed per task (hard contract).
_TASK_ALLOWED_GROUND_TRUTH: dict[AnalysisTask, frozenset[GroundTruthKind]] = {
    AnalysisTask.SPATIAL_DOMAIN: frozenset({GroundTruthKind.SPATIAL_DOMAIN}),
    AnalysisTask.SPATIAL_PROTEIN_DOMAIN: frozenset({GroundTruthKind.SPATIAL_PROTEIN_DOMAIN}),
    AnalysisTask.SPATIAL_CHROMATIN_DOMAIN: frozenset({GroundTruthKind.SPATIAL_CHROMATIN_DOMAIN}),
    AnalysisTask.CELL_TYPE: frozenset(
        {GroundTruthKind.CELL_TYPE, GroundTruthKind.CLUSTER_PROXY}
    ),
    AnalysisTask.SVG: frozenset({GroundTruthKind.NONE, GroundTruthKind.CELL_TYPE}),
    AnalysisTask.DECONVOLUTION: frozenset(
        {GroundTruthKind.CELL_TYPE, GroundTruthKind.NONE}
    ),
    AnalysisTask.VIRTUAL_ST: frozenset(
        {GroundTruthKind.MEASURED_EXPRESSION, GroundTruthKind.NONE}
    ),
}

# Default evaluation metrics (documentation + contract validation).
_TASK_DEFAULT_METRICS: dict[AnalysisTask, str] = {
    AnalysisTask.SPATIAL_DOMAIN: "ARI",
    AnalysisTask.SPATIAL_PROTEIN_DOMAIN: "ARI",
    AnalysisTask.SPATIAL_CHROMATIN_DOMAIN: "ARI",
    AnalysisTask.CELL_TYPE: "ARI",
    AnalysisTask.SVG: "precision_at_k",
    AnalysisTask.DECONVOLUTION: "one_minus_rmsd",
    AnalysisTask.VIRTUAL_ST: "mean_gene_pearson",
}

# Alias normalisation for legacy landscape strings.
_TASK_ALIASES: dict[str, str] = {
    "domain_detection": AnalysisTask.SPATIAL_DOMAIN.value,
    "domain": AnalysisTask.SPATIAL_DOMAIN.value,
    "protein_domain": AnalysisTask.SPATIAL_PROTEIN_DOMAIN.value,
    "chromatin_domain": AnalysisTask.SPATIAL_CHROMATIN_DOMAIN.value,
    "virtual_stain": AnalysisTask.VIRTUAL_ST.value,
    "he2st": AnalysisTask.VIRTUAL_ST.value,
    "h&e2st": AnalysisTask.VIRTUAL_ST.value,
}


def normalize_task(task: str | AnalysisTask | None) -> str | None:
    """Normalise task strings / enums to canonical :class:`AnalysisTask` values."""
    if task is None:
        return None
    if isinstance(task, AnalysisTask):
        return task.value
    key = str(task).strip().lower()
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    try:
        return AnalysisTask(key).value
    except ValueError:
        return key or None


def coerce_analysis_task(task: str | AnalysisTask) -> AnalysisTask:
    """Coerce a string or enum to :class:`AnalysisTask`, raising on unknowns."""
    normalised = normalize_task(task)
    if normalised is None:
        raise ValueError("analysis task must be non-empty")
    return AnalysisTask(normalised)


def cross_modal_relation(
    query_task: str | AnalysisTask | None,
    reference_task: str | AnalysisTask | None,
) -> CrossModalRelation:
    """Classify how a reference task relates to a query task across modalities."""
    query = normalize_task(query_task)
    reference = normalize_task(reference_task)
    if query is None or reference is None:
        return CrossModalRelation.INCOMPATIBLE
    if query == reference:
        return CrossModalRelation.SAME
    try:
        q = AnalysisTask(query)
        r = AnalysisTask(reference)
    except ValueError:
        return CrossModalRelation.INCOMPATIBLE
    if q in DOMAIN_PARTITION_TASKS and r in DOMAIN_PARTITION_TASKS:
        return CrossModalRelation.SAME_FAMILY
    return CrossModalRelation.INCOMPATIBLE


def tasks_admissible(
    query_task: str | AnalysisTask | None,
    reference_task: str | AnalysisTask | None,
) -> bool:
    """Hard admissibility gate for recommendation / decision evidence.

    Only exact task matches are admitted. Cross-modal domain-family pairs
    (RNA / protein / chromatin partitions) and virtual-ST vs measured-domain
    pairs are **never** admitted into a ranking score.
    """
    return cross_modal_relation(query_task, reference_task) is CrossModalRelation.SAME


def ground_truth_admissible(
    task: str | AnalysisTask,
    ground_truth_kind: str | GroundTruthKind | None,
) -> bool:
    """Return whether a ground-truth kind is allowed for the analysis task."""
    task_enum = coerce_analysis_task(task)
    if ground_truth_kind is None:
        return False
    kind = (
        ground_truth_kind
        if isinstance(ground_truth_kind, GroundTruthKind)
        else GroundTruthKind(str(ground_truth_kind).strip().lower())
    )
    allowed = _TASK_ALLOWED_GROUND_TRUTH.get(task_enum)
    if allowed is None:
        return False
    return kind in allowed


def default_metric_for_task(task: str | AnalysisTask) -> str:
    """Return the default evaluation metric name for an analysis task."""
    return _TASK_DEFAULT_METRICS[coerce_analysis_task(task)]


def is_domain_partition_task(task: str | AnalysisTask | None) -> bool:
    """True when the task recovers spatial partitions on a measured modality."""
    normalised = normalize_task(task)
    if normalised is None:
        return False
    try:
        return AnalysisTask(normalised) in DOMAIN_PARTITION_TASKS
    except ValueError:
        return False


@dataclass(frozen=True)
class TaskContract:
    """Machine-checked description of one benchmark cell."""

    task: AnalysisTask
    ground_truth_kind: GroundTruthKind
    label_key: str
    metric: str = "ARI"
    higher_is_better: bool = True
    platform: str | None = None
    study: str | None = None
    notes: str = ""
    # When False (default), landscape runners must not inject the true domain
    # count as ``n_domains`` — use estimate_n_domains / k_policy='estimate'.
    # Set True only for controlled oracle-K ablations.
    allow_oracle_k: bool = False
    spatial_context_policies: tuple[str, ...] = ("default",)
    # Optional modality tag for audit (expression / protein / chromatin / image).
    modality: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task"] = self.task.value
        payload["ground_truth_kind"] = self.ground_truth_kind.value
        return payload

    def validate(self) -> None:
        """Raise ``ValueError`` when the contract is scientifically invalid."""
        if not isinstance(self.task, AnalysisTask):
            raise ValueError(f"task_contract: unknown task {self.task!r}")
        if not isinstance(self.ground_truth_kind, GroundTruthKind):
            raise ValueError(
                f"task_contract: unknown ground_truth_kind {self.ground_truth_kind!r}"
            )

        # Domain-partition tasks: specific diagnostics first (preserve legacy
        # error substrings used by CI adversarial fixtures).
        if self.task in DOMAIN_PARTITION_TASKS:
            if self.ground_truth_kind is GroundTruthKind.SELF_SUPERVISED:
                raise ValueError(
                    "task_contract: self-supervised labels (e.g. Leiden on the same "
                    "expression matrix) cannot serve as spatial-domain ground truth"
                )
            if self.ground_truth_kind is GroundTruthKind.CLUSTER_PROXY:
                raise ValueError(
                    "task_contract: cluster_proxy labels are not valid spatial-domain "
                    "ground truth; score them under AnalysisTask.CELL_TYPE instead, "
                    "or obtain expert spatial partitions"
                )
            expected_kind = {
                AnalysisTask.SPATIAL_DOMAIN: GroundTruthKind.SPATIAL_DOMAIN,
                AnalysisTask.SPATIAL_PROTEIN_DOMAIN: GroundTruthKind.SPATIAL_PROTEIN_DOMAIN,
                AnalysisTask.SPATIAL_CHROMATIN_DOMAIN: GroundTruthKind.SPATIAL_CHROMATIN_DOMAIN,
            }[self.task]
            if self.ground_truth_kind is not expected_kind:
                raise ValueError(
                    f"task_contract: {self.task.value} requires "
                    f"ground_truth_kind={expected_kind.value!r} "
                    f"(cross-modal ground truth is not admissible)"
                )
            lowered = self.label_key.lower()
            if any(token in lowered for token in _FORBIDDEN_DOMAIN_LABEL_TOKENS):
                raise ValueError(
                    f"task_contract: label_key {self.label_key!r} looks like a "
                    "self-supervised clustering column and is forbidden for "
                    f"{self.task.value} evaluation"
                )

        if self.task is AnalysisTask.CELL_TYPE:
            if self.ground_truth_kind is GroundTruthKind.SELF_SUPERVISED:
                raise ValueError(
                    "task_contract: self-supervised labels cannot serve as cell-type "
                    "ground truth either (circular evaluation)"
                )

        if self.task is AnalysisTask.VIRTUAL_ST:
            if self.ground_truth_kind is GroundTruthKind.SELF_SUPERVISED:
                raise ValueError(
                    "task_contract: self-supervised labels cannot score virtual_st "
                    "(requires measured expression or none for inference-only)"
                )
            if self.ground_truth_kind is GroundTruthKind.MEASURED_EXPRESSION:
                if not self.label_key:
                    raise ValueError(
                        "task_contract: virtual_st with measured_expression requires "
                        "label_key naming the measured layer or matrix slot "
                        "(e.g. 'X' or 'counts')"
                    )
            # Virtual ST must not claim domain-partition metrics as primary.
            if self.metric.upper() == "ARI" and not (
                self.notes and "domain" in self.notes.lower()
            ):
                raise ValueError(
                    "task_contract: virtual_st default metric is continuous expression "
                    "agreement (e.g. mean_gene_pearson); ARI requires notes explaining "
                    "a secondary domain-style evaluation"
                )

        allowed = _TASK_ALLOWED_GROUND_TRUTH[self.task]
        if self.ground_truth_kind not in allowed:
            raise ValueError(
                f"task_contract: task={self.task.value!r} requires ground_truth_kind in "
                f"{sorted(k.value for k in allowed)}, got {self.ground_truth_kind.value!r}"
            )

        if not self.label_key and self.ground_truth_kind is not GroundTruthKind.NONE:
            raise ValueError("task_contract: label_key is required when ground truth exists")

        if self.allow_oracle_k and self.task in DOMAIN_PARTITION_TASKS:
            # Not an error — oracle K is an explicit ablation — but require a note
            # so published contracts cannot silently leak K without documentation.
            if not (self.notes and "oracle" in self.notes.lower()):
                raise ValueError(
                    "task_contract: allow_oracle_k=True requires notes mentioning "
                    "'oracle' (document why true domain count is injected)"
                )


@dataclass
class DatasetBenchmarkRecord:
    """One dataset entry ready for landscape / recommendation use."""

    name: str
    contract: TaskContract
    n_obs: int = 0
    n_vars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract": self.contract.to_dict(),
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "metadata": dict(self.metadata),
        }


def assert_labels_usable(
    data: SpatialTable,
    contract: TaskContract,
) -> np.ndarray:
    """Validate contract + extract label vector for scoring.

    For :attr:`AnalysisTask.VIRTUAL_ST` with measured expression the contract
    ``label_key`` names a matrix slot (``X`` / layer name); continuous targets
    are not returned here — callers should read the matrix directly after
    :meth:`TaskContract.validate`.
    """
    contract.validate()
    if contract.task is AnalysisTask.VIRTUAL_ST:
        if contract.ground_truth_kind is GroundTruthKind.NONE:
            return np.array([], dtype=str)
        # Continuous expression target — validate presence only.
        key = contract.label_key
        if key in {"X", "x", "expression"}:
            if data.X is None:
                raise KeyError("task_contract: measured expression matrix X is missing")
            return np.array(["X"], dtype=str)
        if key in data.layers:
            return np.array([key], dtype=str)
        if key in data.obs.columns:
            # Allow obs-level continuous targets only as an escape hatch.
            series = data.obs[key]
            if series.isna().all():
                raise ValueError(f"task_contract: obs[{key!r}] is entirely missing")
            return series.astype(str).to_numpy()
        raise KeyError(
            f"task_contract: measured expression target {key!r} not found in X, "
            "layers, or obs"
        )

    if contract.label_key not in data.obs.columns:
        raise KeyError(f"task_contract: obs[{contract.label_key!r}] is missing on the dataset")
    series = data.obs[contract.label_key]
    if series.isna().all():
        raise ValueError(f"task_contract: obs[{contract.label_key!r}] is entirely missing")
    labels = series.astype(str).to_numpy()
    if contract.task in DOMAIN_PARTITION_TASKS:
        # Guard against silently scoring domain recovery with a Leiden column
        # that happens to be renamed to domain_truth.
        unique = {value.lower() for value in pd.unique(series.dropna().astype(str))}
        if unique and unique <= {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            # Numeric-only labels are fine (manual layers often use integers);
            # only block when the *key* looks self-supervised (already checked)
            # or metadata declares self_supervised.
            pass
        if contract.ground_truth_kind is GroundTruthKind.SELF_SUPERVISED:
            raise ValueError("self-supervised labels rejected for domain scoring")
    return labels


def classify_platform(value: str | None) -> str | None:
    """Normalise common platform strings for recommendation priors."""
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "visium": "visium",
        "10xvisium": "visium",
        "visiumhd": "visium_hd",
        "xenium": "xenium",
        "10xxenium": "xenium",
        "merfish": "merfish",
        "vizgen": "merfish",
        "merscope": "merscope",
        "cosmx": "cosmx",
        "slideseqv2": "slideseq",
        "slideseq": "slideseq",
        "stereoseq": "stereoseq",
        # Imaging / multi-omics platforms often paired with virtual ST.
        "he": "histology",
        "hne": "histology",
        "histology": "histology",
        "codex": "codex",
        "mibi": "mibi",
        "imc": "imc",
        "cutntag": "chromatin",
        "scatac": "chromatin",
        "spatialatac": "chromatin",
    }
    return aliases.get(key, key or None)


def default_spatial_context_policy(task: AnalysisTask | str) -> str:
    """Task-aware default for the spatial-context knob."""
    task = coerce_analysis_task(task)
    if task in DOMAIN_PARTITION_TASKS:
        return "high"  # prefer neighbourhood-aware configurations
    if task is AnalysisTask.VIRTUAL_ST:
        return "high"  # morphology is spatially structured
    if task is AnalysisTask.CELL_TYPE:
        return "off"  # expression-first
    return "default"


def split_method_policy(name: str) -> tuple[str, str | None]:
    """Split ``method@sw0.8`` style configuration keys into method + policy."""
    if "@" not in name:
        return name, None
    method, policy = name.split("@", 1)
    return method, policy


def evidence_compatibility_report(
    query_task: str | AnalysisTask,
    reference_task: str | AnalysisTask,
    *,
    reference_ground_truth_kind: str | GroundTruthKind | None = None,
) -> dict[str, Any]:
    """Machine-readable compatibility audit for one reference → query pair.

    Used by recommenders and decision cards to explain why evidence was admitted
    or zeroed out without soft-weighting incompatible modalities.
    """
    relation = cross_modal_relation(query_task, reference_task)
    admissible = relation is CrossModalRelation.SAME
    reasons: list[str] = []
    if relation is CrossModalRelation.SAME:
        reasons.append("exact task match")
    elif relation is CrossModalRelation.SAME_FAMILY:
        reasons.append(
            "both tasks recover spatial partitions but on different molecular "
            "modalities; method rankings are not transferable"
        )
        admissible = False
    else:
        reasons.append("tasks ask different scientific questions")
        admissible = False

    gt_ok: bool | None = None
    if reference_ground_truth_kind is not None and admissible:
        try:
            gt_ok = ground_truth_admissible(query_task, reference_ground_truth_kind)
        except ValueError:
            gt_ok = False
        if not gt_ok:
            admissible = False
            reasons.append(
                f"ground_truth_kind={reference_ground_truth_kind!r} is not valid "
                f"for task={normalize_task(query_task)!r}"
            )

    return {
        "query_task": normalize_task(query_task),
        "reference_task": normalize_task(reference_task),
        "relation": relation.value,
        "admissible": admissible,
        "ground_truth_admissible": gt_ok,
        "reasons": reasons,
    }
