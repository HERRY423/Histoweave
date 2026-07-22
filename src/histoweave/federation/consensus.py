"""Derived cross-lab consensus view for the federated evidence network.

This is the **consensus layer** of the dual-layer design. It reads the
append-only :class:`~histoweave.federation.store.EvidenceStore` and computes,
per ``(task, dataset, method)`` cell, a robust cross-lab summary that
deliberately **preserves disagreement and reports uncertainty** (the project's
scientific stance) rather than fabricating a single global "best":

* ``n_labs`` / ``n_records`` / ``n_seeds`` — how much independent support exists.
* ``consensus_score`` — **median of per-lab means** (robust to one bad lab).
* ``mean`` — mean of per-lab means (for comparison, not used for ranking).
* ``mad`` / ``spread`` — median absolute deviation across lab means.
* ``cross_lab_ci`` — bootstrap CI over the *lab means* (not over spots), so the
  interval reflects between-lab reproducibility, matching the philosophy of
  ``stats_review``/``donor_bootstrap`` (resample the independent unit).
* ``reproducibility`` — fraction of lab means within ``tolerance`` of the
  consensus; outlier labs are flagged, not dropped.
* ``verification_status`` — tiered-trust rollup: ``verified`` once >= 2
  independent nodes agree within tolerance, ``disputed`` when independent nodes
  irreconcilably disagree, else ``unverified``.

Everything here is a pure function of the store, so ``consensus.json`` is fully
regenerable and never a source of truth on its own.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import now_iso
from .store import DEFAULT_TOLERANCE, EvidenceStore, StoredRecord, latest_records

CONSENSUS_SCHEMA_VERSION = "histoweave.consensus.v1"

#: Statuses that mean "this record has no usable score".
_NON_SCORING = frozenset({"failed", "error", "timeout", "oom"})


@dataclass
class LabAggregate:
    """Per-lab summary of one cell (one node's mean over its own seeds)."""

    node_id: str
    mean: float
    n_seeds: int
    verification_status: str
    within_tolerance: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "mean": _round(self.mean),
            "n_seeds": self.n_seeds,
            "verification_status": self.verification_status,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class ConsensusCell:
    """Cross-lab consensus for one ``(task, dataset, method)`` cell."""

    task: str
    dataset: str
    method: str
    metric: str
    higher_is_better: bool
    n_labs: int
    n_records: int
    n_seeds: int
    consensus_score: float
    mean: float
    mad: float
    spread: float
    cross_lab_ci: tuple[float, float]
    reproducibility: float
    verification_status: str
    node_ids: list[str]
    labs: list[LabAggregate]
    outlier_node_ids: list[str] = field(default_factory=list)
    dataset_meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset": self.dataset,
            "method": self.method,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "n_labs": self.n_labs,
            "n_records": self.n_records,
            "n_seeds": self.n_seeds,
            "consensus_score": _round(self.consensus_score),
            "mean": _round(self.mean),
            "mad": _round(self.mad),
            "spread": _round(self.spread),
            "cross_lab_ci": [_round(self.cross_lab_ci[0]), _round(self.cross_lab_ci[1])],
            "reproducibility": _round(self.reproducibility),
            "verification_status": self.verification_status,
            "node_ids": list(self.node_ids),
            "outlier_node_ids": list(self.outlier_node_ids),
            "labs": [a.to_json() for a in self.labs],
            "dataset_meta": self.dataset_meta,
        }


@dataclass
class NodeTrackRecord:
    """Per-node reproducibility track-record (for future reputation weighting)."""

    node_id: str
    contributed_cells: int = 0
    reproduced_others: int = 0
    was_reproduced: int = 0
    disputed: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "contributed_cells": self.contributed_cells,
            "reproduced_others": self.reproduced_others,
            "was_reproduced": self.was_reproduced,
            "disputed": self.disputed,
        }


@dataclass
class ConsensusView:
    """The full derived consensus document."""

    cells: list[ConsensusCell]
    track_records: list[NodeTrackRecord]
    tolerance: float
    schema_version: str = CONSENSUS_SCHEMA_VERSION
    generated_at: str = field(default_factory=now_iso)

    def to_json(self) -> dict[str, Any]:
        n_verified = sum(1 for c in self.cells if c.verification_status == "verified")
        n_disputed = sum(1 for c in self.cells if c.verification_status == "disputed")
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "tolerance": self.tolerance,
            "summary": {
                "n_cells": len(self.cells),
                "n_verified": n_verified,
                "n_disputed": n_disputed,
                "n_unverified": len(self.cells) - n_verified - n_disputed,
                "n_nodes": len(self.track_records),
                "n_datasets": len({c.dataset for c in self.cells}),
                "n_methods": len({c.method for c in self.cells}),
            },
            "cells": [c.to_json() for c in self.cells],
            "track_records": [t.to_json() for t in self.track_records],
        }


def _round(x: float, ndigits: int = 6) -> float:
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return float("nan")


def _tolerance_for(task: str, tolerance: float | None, per_task: dict[str, float] | None) -> float:
    if per_task and task in per_task:
        return float(per_task[task])
    if tolerance is not None:
        return float(tolerance)
    return DEFAULT_TOLERANCE


def _bootstrap_ci_over_means(
    lab_means: Sequence[float], *, n_boot: int = 2000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI over the per-lab means (the independent unit).

    With a single lab the CI collapses to the point estimate (no between-lab
    information yet) — honest about the fact that one lab is not reproduced.
    """
    arr = np.asarray([m for m in lab_means if m == m], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_medians = np.median(arr[idx], axis=1)
    lo = float(np.quantile(boot_medians, (1 - ci) / 2))
    hi = float(np.quantile(boot_medians, 1 - (1 - ci) / 2))
    return (lo, hi)


def _scoring(record: StoredRecord) -> bool:
    return (
        record.status.lower() not in _NON_SCORING
        and record.score is not None
        and record.score == record.score
    )


def build_consensus(
    store: EvidenceStore | Iterable[StoredRecord],
    *,
    tolerance: float | None = None,
    per_task_tolerance: dict[str, float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> ConsensusView:
    """Build the derived consensus view from the append-only store.

    Parameters
    ----------
    tolerance:
        Absolute metric tolerance for "two labs agree" (default 0.05 for ARI).
    per_task_tolerance:
        Optional per-task override, e.g. ``{"cell_type": 0.03}``.
    """
    records = list(store) if not isinstance(store, EvidenceStore) else store.read()
    records = latest_records(records)

    # Group scoring records by cell, then by lab.
    by_cell: dict[tuple[str, str, str], list[StoredRecord]] = defaultdict(list)
    for r in records:
        if not _scoring(r):
            continue
        by_cell[(r.task, r.dataset, r.method)].append(r)

    default_tol = tolerance if tolerance is not None else DEFAULT_TOLERANCE
    cells: list[ConsensusCell] = []
    track: dict[str, NodeTrackRecord] = defaultdict(lambda: NodeTrackRecord(node_id=""))

    for (task, dataset, method), recs in sorted(by_cell.items()):
        tol = _tolerance_for(task, tolerance, per_task_tolerance)
        higher_is_better = recs[0].higher_is_better
        metric = recs[0].metric

        # Per-lab mean over that lab's seeds.
        by_lab: dict[str, list[StoredRecord]] = defaultdict(list)
        for r in recs:
            by_lab[r.node_id].append(r)

        lab_aggs: list[LabAggregate] = []
        for node_id, lab_recs in sorted(by_lab.items()):
            scores = [r.score for r in lab_recs if r.score is not None]
            lab_mean = float(statistics.fmean(scores)) if scores else float("nan")
            # A lab's status for this cell = strongest status it declared.
            statuses = {r.verification_status for r in lab_recs}
            lab_status = (
                "verified"
                if "verified" in statuses
                else ("disputed" if "disputed" in statuses else "unverified")
            )
            lab_aggs.append(
                LabAggregate(
                    node_id=node_id,
                    mean=lab_mean,
                    n_seeds=len({r.seed for r in lab_recs}),
                    verification_status=lab_status,
                )
            )

        lab_means = [a.mean for a in lab_aggs if a.mean == a.mean]
        consensus_score = float(statistics.median(lab_means)) if lab_means else float("nan")
        mean_val = float(statistics.fmean(lab_means)) if lab_means else float("nan")
        mad = (
            float(statistics.median([abs(m - consensus_score) for m in lab_means]))
            if lab_means
            else float("nan")
        )
        spread = float(max(lab_means) - min(lab_means)) if len(lab_means) > 1 else 0.0
        cross_lab_ci = _bootstrap_ci_over_means(lab_means, n_boot=n_boot, seed=seed, ci=ci)

        # Reproducibility: fraction of lab means within tolerance of consensus.
        within_flags = [abs(a.mean - consensus_score) <= tol for a in lab_aggs]
        for agg, ok in zip(lab_aggs, within_flags, strict=True):
            agg.within_tolerance = bool(ok)
        reproducibility = float(np.mean(within_flags)) if within_flags else 0.0
        outliers = [a.node_id for a, ok in zip(lab_aggs, within_flags, strict=True) if not ok]

        n_labs = len(lab_aggs)
        n_agree = sum(within_flags)
        # Tiered-trust rollup at the cell level.
        if n_labs >= 2 and n_agree >= 2:
            verification = "verified"
        elif n_labs >= 2 and n_agree < 2:
            verification = "disputed"
        else:
            verification = "unverified"

        node_ids = sorted(a.node_id for a in lab_aggs)
        # Merge dataset_meta (prefer any non-empty).
        ds_meta: dict[str, Any] = {}
        for r in recs:
            if r.dataset_meta:
                ds_meta = dict(r.dataset_meta)
                break

        cells.append(
            ConsensusCell(
                task=task,
                dataset=dataset,
                method=method,
                metric=metric,
                higher_is_better=higher_is_better,
                n_labs=n_labs,
                n_records=len(recs),
                n_seeds=sum(a.n_seeds for a in lab_aggs),
                consensus_score=consensus_score,
                mean=mean_val,
                mad=mad,
                spread=spread,
                cross_lab_ci=cross_lab_ci,
                reproducibility=reproducibility,
                verification_status=verification,
                node_ids=node_ids,
                labs=lab_aggs,
                outlier_node_ids=outliers,
                dataset_meta=ds_meta,
            )
        )

        # Track-record accrual.
        for agg in lab_aggs:
            tr = track[agg.node_id]
            tr.node_id = agg.node_id
            tr.contributed_cells += 1
            if verification == "verified":
                if agg.within_tolerance and n_labs >= 2:
                    tr.was_reproduced += 1
                    if n_agree >= 2 and agg.within_tolerance:
                        tr.reproduced_others += 1
            if verification == "disputed" and not agg.within_tolerance:
                tr.disputed += 1

    track_records = [track[k] for k in sorted(track)]
    return ConsensusView(
        cells=cells,
        track_records=track_records,
        tolerance=default_tol if not per_task_tolerance else default_tol,
    )
