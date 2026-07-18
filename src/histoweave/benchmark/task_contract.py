"""Task contracts for scientifically valid benchmarking.

Nature-Methods-grade evaluation requires that the *question*, the *ground-truth
kind*, and the *metric* are declared and checked before scores are aggregated.

Key hard rules
--------------
* Spatial-domain recovery may only score against expert / histology-derived
  spatial partitions (or other declared ``spatial_domain`` labels).
* Cell-type recovery may score against cell-type or state annotations.
* **Self-supervised cluster labels (e.g. Leiden computed on the same expression
  matrix) are forbidden as domain ground truth** — they create circular advantage
  for expression-only methods and invalidate cross-platform ARI comparisons.
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

    SPATIAL_DOMAIN = "spatial_domain"
    CELL_TYPE = "cell_type"
    SVG = "svg"
    DECONVOLUTION = "deconvolution"


class GroundTruthKind(StrEnum):
    """What the evaluation labels actually represent."""

    SPATIAL_DOMAIN = "spatial_domain"
    CELL_TYPE = "cell_type"
    CLUSTER_PROXY = "cluster_proxy"
    SELF_SUPERVISED = "self_supervised"
    NONE = "none"


# Labels that are never acceptable as spatial-domain ground truth.
_FORBIDDEN_DOMAIN_LABEL_TOKENS = (
    "leiden",
    "louvain",
    "self_cluster",
    "self-supervised",
    "proxy_leiden",
)


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task"] = self.task.value
        payload["ground_truth_kind"] = self.ground_truth_kind.value
        return payload

    def validate(self) -> None:
        """Raise ``ValueError`` when the contract is scientifically invalid."""
        if self.task is AnalysisTask.SPATIAL_DOMAIN:
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
            if self.ground_truth_kind not in {
                GroundTruthKind.SPATIAL_DOMAIN,
            }:
                raise ValueError(
                    "task_contract: spatial_domain task requires ground_truth_kind='spatial_domain'"
                )
            lowered = self.label_key.lower()
            if any(token in lowered for token in _FORBIDDEN_DOMAIN_LABEL_TOKENS):
                raise ValueError(
                    f"task_contract: label_key {self.label_key!r} looks like a "
                    "self-supervised clustering column and is forbidden for "
                    "spatial_domain evaluation"
                )
        if self.task is AnalysisTask.CELL_TYPE:
            if self.ground_truth_kind is GroundTruthKind.SELF_SUPERVISED:
                raise ValueError(
                    "task_contract: self-supervised labels cannot serve as cell-type "
                    "ground truth either (circular evaluation)"
                )
            if self.ground_truth_kind not in {
                GroundTruthKind.CELL_TYPE,
                GroundTruthKind.CLUSTER_PROXY,
            }:
                raise ValueError(
                    "task_contract: cell_type task requires cell_type or "
                    "cluster_proxy ground truth (not self_supervised)"
                )
        if not self.label_key and self.ground_truth_kind is not GroundTruthKind.NONE:
            raise ValueError("task_contract: label_key is required when ground truth exists")
        if self.allow_oracle_k and self.task is AnalysisTask.SPATIAL_DOMAIN:
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
    """Validate contract + extract label vector for scoring."""
    contract.validate()
    if contract.label_key not in data.obs.columns:
        raise KeyError(f"task_contract: obs[{contract.label_key!r}] is missing on the dataset")
    series = data.obs[contract.label_key]
    if series.isna().all():
        raise ValueError(f"task_contract: obs[{contract.label_key!r}] is entirely missing")
    labels = series.astype(str).to_numpy()
    if contract.task is AnalysisTask.SPATIAL_DOMAIN:
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
    }
    return aliases.get(key, key or None)


def default_spatial_context_policy(task: AnalysisTask | str) -> str:
    """Task-aware default for the spatial-context knob."""
    task = AnalysisTask(task) if not isinstance(task, AnalysisTask) else task
    if task is AnalysisTask.SPATIAL_DOMAIN:
        return "high"  # prefer neighbourhood-aware configurations
    if task is AnalysisTask.CELL_TYPE:
        return "off"  # expression-first
    return "default"


def split_method_policy(name: str) -> tuple[str, str | None]:
    """Split ``method@sw0.8`` style configuration keys into method + policy."""
    if "@" not in name:
        return name, None
    method, policy = name.split("@", 1)
    return method, policy
