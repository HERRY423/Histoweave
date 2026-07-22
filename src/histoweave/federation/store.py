"""Append-only evidence store for the federated evidence network.

The store is the **raw layer** of the dual-layer design: one JSON object per
line (JSONL), each line a single benchmark *record* carrying full provenance
(node, bundle, content hash, task/metric, dataset descriptor) plus a
``verification_status`` in the tiered-trust lifecycle
(``unverified`` -> ``verified`` / ``disputed``).

Invariants:

* **Append-only.** Corrections are new lines, never edits of prior lines. The
  file is an audit log; the consensus layer (:mod:`.consensus`) is the derived,
  regenerable view.
* **Deduplicated by content.** A ``(content_hash, dataset, method, seed)`` key
  is used so re-pulling the same bundle is idempotent, while an independent
  lab's reproduction of the same cell is a distinct row.
* **Contract- and privacy-safe on ingest.** Only records from a bundle that
  passed :meth:`EvidenceBundle.validate` (and, in CI, signature verification)
  should be appended; :func:`append_bundle` re-validates as a safety net.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import (
    VERIFICATION_STATES,
    EvidenceBundle,
    SchemaError,
    now_iso,
    resolve_method_name,
)

DEFAULT_STORE_PATH = "federation/evidence_store.jsonl"

#: Absolute floor tolerance for treating two lab means as "the same result".
#: Overridable per task via :func:`consensus.build_consensus`.
DEFAULT_TOLERANCE = 0.05


@dataclass
class StoredRecord:
    """One line of the append-only store."""

    node_id: str
    dataset: str
    method: str
    task: str
    metric: str
    score: float | None
    seed: int
    seconds: float | None
    status: str
    verification_status: str
    bundle_id: str
    content_hash: str | None
    higher_is_better: bool = True
    config: str | None = None
    ingested_at: str = field(default_factory=now_iso)
    dataset_meta: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> tuple[str, str, str, int]:
        """Identity used to suppress exact re-ingests of the same record."""
        return (self.content_hash or "", self.dataset, self.method, self.seed)

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dataset": self.dataset,
            "method": self.method,
            "config": self.config,
            "task": self.task,
            "metric": self.metric,
            "score": self.score,
            "seed": self.seed,
            "seconds": self.seconds,
            "status": self.status,
            "verification_status": self.verification_status,
            "bundle_id": self.bundle_id,
            "content_hash": self.content_hash,
            "higher_is_better": self.higher_is_better,
            "ingested_at": self.ingested_at,
            "dataset_meta": self.dataset_meta,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> StoredRecord:
        return cls(
            node_id=str(obj.get("node_id", "")),
            dataset=str(obj.get("dataset", "")),
            method=str(obj.get("method", "")),
            task=str(obj.get("task", "")),
            metric=str(obj.get("metric", "ARI")),
            score=_opt_float(obj.get("score")),
            seed=int(obj.get("seed") or 0),
            seconds=_opt_float(obj.get("seconds")),
            status=str(obj.get("status", "success")),
            verification_status=str(obj.get("verification_status", "unverified")),
            bundle_id=str(obj.get("bundle_id", "")),
            content_hash=obj.get("content_hash"),
            higher_is_better=bool(obj.get("higher_is_better", True)),
            config=obj.get("config"),
            ingested_at=str(obj.get("ingested_at", now_iso())),
            dataset_meta=dict(obj.get("dataset_meta", {})),
            provenance=dict(obj.get("provenance", {})),
        )


def _opt_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


class EvidenceStore:
    """Reader/writer for the append-only JSONL evidence store."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)

    # -- reading ---------------------------------------------------------- #
    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> list[StoredRecord]:
        if not self.path.exists():
            return []
        records: list[StoredRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(StoredRecord.from_json(json.loads(line)))
        return records

    def __iter__(self) -> Iterator[StoredRecord]:
        return iter(self.read())

    def dedup_keys(self) -> set[tuple[str, str, str, int]]:
        return {r.dedup_key() for r in self.read()}

    # -- writing ---------------------------------------------------------- #
    def _append_records(self, records: Iterable[StoredRecord]) -> int:
        rows = list(records)
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append atomically-ish: open in append mode, one line per record.
        with self.path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_json(), ensure_ascii=False, sort_keys=True))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        return len(rows)

    def append_bundle(
        self,
        bundle: EvidenceBundle,
        *,
        verification_status: str = "unverified",
        revalidate: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> int:
        """Append every record of *bundle*, skipping exact duplicates.

        Returns the number of new lines written. Re-ingesting the same signed
        bundle is a no-op (0). Independent reproductions are new rows.
        """
        if verification_status not in VERIFICATION_STATES:
            raise SchemaError(
                f"unknown verification_status {verification_status!r}; "
                f"expected one of {VERIFICATION_STATES}"
            )
        if revalidate:
            bundle.validate()
        if bundle.content_hash is None:
            bundle.compute_hash()

        existing = self.dedup_keys()
        new_rows: list[StoredRecord] = []
        seen_this_call: set[tuple[str, str, str, int]] = set()
        for rec in bundle.records:
            method = resolve_method_name(rec) or str(rec.get("method") or "")
            score_col = bundle.score_col()
            ds = str(rec.get("dataset", "")).strip()
            stored = StoredRecord(
                node_id=bundle.node_id,
                dataset=ds,
                method=str(rec.get("method") or method).strip(),
                config=rec.get("config"),
                task=bundle.task,
                metric=bundle.metric,
                score=_opt_float(rec.get(score_col)),
                seed=int(rec.get("seed") or 0),
                seconds=_opt_float(rec.get("seconds")),
                status=str(rec.get("status", "success") or "success"),
                verification_status=verification_status,
                bundle_id=bundle.bundle_id,
                content_hash=bundle.content_hash,
                higher_is_better=bundle.higher_is_better,
                dataset_meta=dict(bundle.dataset_meta.get(ds, {})),
                provenance=dict(provenance or {}),
            )
            key = stored.dedup_key()
            if key in existing or key in seen_this_call:
                continue
            seen_this_call.add(key)
            new_rows.append(stored)
        return self._append_records(new_rows)

    def append_status_correction(
        self,
        record: StoredRecord,
        new_status: str,
        *,
        reason: str = "",
    ) -> int:
        """Append a *new* line reflecting a verification-status change.

        The store is append-only, so a status transition is recorded as a fresh
        line (same content_hash, updated ``verification_status``) rather than an
        in-place edit. The latest line for a given content wins when the
        consensus view is rebuilt.
        """
        if new_status not in VERIFICATION_STATES:
            raise SchemaError(
                f"unknown verification_status {new_status!r}; expected {VERIFICATION_STATES}"
            )
        updated = StoredRecord.from_json(record.to_json())
        updated.verification_status = new_status
        updated.ingested_at = now_iso()
        prov = dict(updated.provenance)
        prov["status_change"] = {"to": new_status, "reason": reason, "at": updated.ingested_at}
        updated.provenance = prov
        return self._append_records([updated])


def latest_records(records: Iterable[StoredRecord]) -> list[StoredRecord]:
    """Collapse append-only history to the latest line per logical record.

    Two lines are the *same logical record* when they share
    ``(content_hash, dataset, method, seed)``. The most recently ingested line
    (by ``ingested_at``) is kept so status corrections supersede originals.
    """
    latest: dict[tuple[str, str, str, int], StoredRecord] = {}
    for r in records:
        key = r.dedup_key()
        prev = latest.get(key)
        if prev is None or r.ingested_at >= prev.ingested_at:
            latest[key] = r
    return list(latest.values())
