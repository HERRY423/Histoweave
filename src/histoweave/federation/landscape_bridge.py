"""Bridge federated consensus into the existing recommender landscape.

The whole point of this module is that **the recommender does not change**. We
convert the derived :class:`~histoweave.federation.consensus.ConsensusView` into
the exact long-format rows that
:func:`histoweave.benchmark.landscape_io.landscape_from_long_csv` already
consumes, then reuse that function verbatim to produce a
:class:`~histoweave.benchmark.landscape.LandscapeResult`. The federated evidence
therefore drives ``MethodRecommender`` and passes
``validate_landscape_contracts`` with zero recommender code touched.

Two policies control which evidence is admitted:

* ``include_unverified`` — if ``False``, only cells that reached ``verified`` (>=2
  independent nodes agreeing within tolerance) feed the recommender; if ``True``
  (default) unverified cells are included too, matching today's behaviour where
  a single-source result still appears (just now labelled).
* ``drop_disputed`` — disputed cells are excluded by default from the point
  estimate (they are still surfaced in the leaderboard as disputed).

Each consensus cell contributes **one row** whose score is the robust
``consensus_score`` (median of lab means), so the landscape reflects the
cross-lab estimate rather than any single lab.
"""

from __future__ import annotations

import csv
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..benchmark.landscape import LandscapeResult
from ..benchmark.landscape_io import (
    attach_dataset_meta,
    landscape_from_long_csv,
    validate_landscape_contracts,
)
from ..benchmark.task_contract import AnalysisTask
from .consensus import ConsensusView

_LONG_FIELDS = (
    "dataset",
    "method",
    "config",
    "seed",
    "ari",
    "seconds",
    "status",
    "n_labs",
    "verification_status",
    "reproducibility",
)


def consensus_to_long_rows(
    view: ConsensusView,
    *,
    task: str | AnalysisTask = AnalysisTask.SPATIAL_DOMAIN,
    include_unverified: bool = True,
    drop_disputed: bool = True,
) -> list[dict[str, Any]]:
    """Flatten consensus cells to landscape long rows (one row per cell).

    ``method`` and ``config`` are both set to the cell's method key so
    ``landscape_from_long_csv`` (which prefers ``config``) ranks the same key
    the federation aggregated on.
    """
    task_value = task.value if isinstance(task, AnalysisTask) else str(task)
    rows: list[dict[str, Any]] = []
    for cell in view.cells:
        if cell.task != task_value:
            continue
        if cell.verification_status == "disputed" and drop_disputed:
            continue
        if cell.verification_status == "unverified" and not include_unverified:
            continue
        if cell.consensus_score != cell.consensus_score:  # NaN guard
            continue
        rows.append(
            {
                "dataset": cell.dataset,
                "method": cell.method,
                "config": cell.method,
                "seed": 0,
                "ari": cell.consensus_score,
                "seconds": "",
                "status": "success",
                "n_labs": cell.n_labs,
                "verification_status": cell.verification_status,
                "reproducibility": cell.reproducibility,
            }
        )
    return rows


def _dataset_meta_from_view(view: ConsensusView) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for cell in view.cells:
        if cell.dataset not in meta and cell.dataset_meta:
            meta[cell.dataset] = dict(cell.dataset_meta)
    return meta


def landscape_from_consensus(
    view: ConsensusView,
    *,
    task: str | AnalysisTask = AnalysisTask.SPATIAL_DOMAIN,
    metric: str = "ARI",
    include_unverified: bool = True,
    drop_disputed: bool = True,
    higher_is_better: bool = True,
    dataset_meta: Mapping[str, dict[str, Any]] | None = None,
) -> LandscapeResult:
    """Build a :class:`LandscapeResult` from a consensus view.

    Internally writes the long rows to a temporary CSV and reuses
    :func:`landscape_from_long_csv`, so the landscape is produced by exactly the
    same code path as a local ``benchmark_long.csv``.
    """
    rows = consensus_to_long_rows(
        view,
        task=task,
        include_unverified=include_unverified,
        drop_disputed=drop_disputed,
    )
    if not rows:
        raise ValueError(
            "no admissible consensus cells for task "
            f"{task!r} (include_unverified={include_unverified}, "
            f"drop_disputed={drop_disputed})"
        )

    meta = dict(dataset_meta) if dataset_meta is not None else _dataset_meta_from_view(view)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "consensus_long.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_LONG_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        landscape = landscape_from_long_csv(
            csv_path,
            task=task,
            metric=metric,
            score_col="ari",
            config_col="config",
            prefer_config_as_method=True,
            dataset_meta=meta or None,
            higher_is_better=higher_is_better,
        )
    if meta:
        landscape = attach_dataset_meta(landscape, meta, overwrite=False)
    return landscape


def validate_consensus_landscape(view: ConsensusView, **kwargs: Any) -> list[str]:
    """Convenience: build the landscape then run the recommender contract check."""
    landscape = landscape_from_consensus(view, **kwargs)
    return validate_landscape_contracts(landscape)
