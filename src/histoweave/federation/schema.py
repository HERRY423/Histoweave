"""Schemas for the HistoWeave federated evidence network.

This module defines the two on-the-wire documents of the federation protocol and
the validation rules that make a contribution *scientifically* and *privacy*
safe before it is allowed into the shared, append-only evidence store:

* :class:`EvidenceBundle` (``histoweave.evidence.v1``) — the atomic unit a lab
  signs and publishes. A strict **superset** of the legacy
  ``histoweave.external_submission.v1`` CSV so existing submitters upgrade with
  no loss of meaning, and every field maps onto what
  :func:`histoweave.benchmark.landscape_from_long_csv` and
  ``leaderboard/generate.py`` already consume.
* :class:`NodeRegistryEntry` (``histoweave.node_registry.v1``) — the record a
  lab registers (via reviewed PR) so its public key / identity is trusted and
  its self-hosted evidence feed can be pulled.

Design commitments realised here (see ``federation/PROTOCOL.md``):

* **Raw data never leaves the lab.** :func:`enforce_privacy_gate` rejects any
  payload that looks like per-spot/per-cell values, coordinates, or expression
  matrices. Only scalar metrics + registry-style dataset descriptors travel.
* **Task contracts are re-used verbatim.** Domain recovery scored against
  Leiden/Louvain/self-supervised labels is rejected by delegating to
  :class:`histoweave.benchmark.task_contract.TaskContract`.
* **Deterministic hashing.** :func:`canonical_json` / :func:`content_hash`
  produce a stable digest so signatures are reproducible and duplicates are
  detectable across nodes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ..benchmark.task_contract import (
    AnalysisTask,
    GroundTruthKind,
    TaskContract,
    classify_platform,
)

EVIDENCE_SCHEMA_VERSION = "histoweave.evidence.v1"
NODE_REGISTRY_SCHEMA_VERSION = "histoweave.node_registry.v1"

#: Verification lifecycle of a stored evidence cell (tiered trust).
VERIFICATION_STATES = ("unverified", "verified", "disputed")

#: Record fields the validator understands. Anything else is rejected so a lab
#: cannot smuggle raw-data-shaped payloads through an unknown column.
_ALLOWED_RECORD_KEYS = frozenset(
    {
        "dataset",
        "method",
        "config",
        "seed",
        "score",
        "ari",
        "seconds",
        "status",
        "error",
        "n_domains",
        "oracle_k",
    }
)

#: Keys whose names strongly imply raw data (per-cell/per-spot). Presence
#: anywhere in a bundle is a hard rejection — the whole point of federation is
#: that these never travel.
_PRIVACY_DENYLIST_TOKENS = (
    "counts",
    "coords",
    "coordinate",
    "spatial",
    "expression",
    "matrix",
    "adata",
    "obsm",
    "layers",
    "barcode",
    "cell_id",
    "spot_id",
    "raw_x",
)

#: Any embedded sequence longer than this is treated as smuggled per-observation
#: data and rejected. Legitimate metadata values are scalars or short lists.
_MAX_EMBEDDED_SEQUENCE_LEN = 8

#: Dataset descriptor keys we allow to travel (all already public in the
#: dataset registry / leaderboard feed).
_ALLOWED_DATASET_META_KEYS = frozenset(
    {
        "platform",
        "tissue",
        "task",
        "ground_truth_kind",
        "label_key",
        "study",
        "registry_name",
        "license",
        "n_obs",
        "n_domains",
        "sparsity",
        "dataset_visibility",
        "track",
    }
)


class SchemaError(ValueError):
    """Raised when a federation document violates its schema or a hard rule."""


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return a canonical JSON string for hashing/signing.

    Keys are sorted, separators are tight, and non-ASCII is preserved so the
    digest is stable across platforms and Python builds.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 (``sha256:`` prefixed) over the canonicalized payload.

    The ``content_hash`` and ``signature`` keys are excluded so the digest
    covers only the meaningful content and is reproducible by any verifier.
    """
    clean = {k: v for k, v in payload.items() if k not in {"content_hash", "signature"}}
    digest = hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# --------------------------------------------------------------------------- #
# Privacy gate
# --------------------------------------------------------------------------- #
def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    """Yield ``(path, value)`` for every node in a nested JSON-like structure."""
    yield path, value
    if isinstance(value, Mapping):
        for key, sub in value.items():
            yield from _walk(sub, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for i, sub in enumerate(value):
            yield from _walk(sub, f"{path}[{i}]")


def enforce_privacy_gate(payload: Mapping[str, Any]) -> None:
    """Raise :class:`SchemaError` if *payload* looks like it carries raw data.

    This is the core mechanism behind "raw data never leaves the lab": we
    actively reject key names and value shapes associated with per-observation
    data, rather than trusting the contributor to have stripped them.
    """
    for path, value in _walk(payload):
        leaf = path.split(".")[-1].split("[")[0].lower()
        if any(tok in leaf for tok in _PRIVACY_DENYLIST_TOKENS):
            raise SchemaError(
                f"privacy gate: key {path!r} matches a raw-data denylist token; "
                "only scalar metrics and public dataset descriptors may be shared"
            )
        if isinstance(value, (list, tuple)) and len(value) > _MAX_EMBEDDED_SEQUENCE_LEN:
            # Records/keys arrays are handled structurally elsewhere; a long
            # numeric list at any leaf is treated as smuggled per-cell data.
            if all(isinstance(x, (int, float, bool)) for x in value):
                raise SchemaError(
                    f"privacy gate: numeric array at {path!r} has length "
                    f"{len(value)} (> {_MAX_EMBEDDED_SEQUENCE_LEN}); this looks "
                    "like per-observation data and is not allowed"
                )


# --------------------------------------------------------------------------- #
# Evidence bundle
# --------------------------------------------------------------------------- #
@dataclass
class MethodInfo:
    name: str
    version: str | None = None
    wraps: str | None = None
    commit: str | None = None
    container_digest: str | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class Signature:
    scheme: str = "ed25519"
    value: str | None = None
    public_key_id: str | None = None
    certificate: str | None = None  # sigstore bundle (opt-in)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class EvidenceBundle:
    """One signed contribution of benchmark evidence from a node.

    ``records`` is a list of dataset×method×seed rows using the same field
    names as ``benchmark_long.csv`` so the bundle can be flattened straight
    into the existing landscape/leaderboard machinery.
    """

    node_id: str
    task: str
    records: list[dict[str, Any]]
    dataset_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    method: MethodInfo | None = None
    metric: str = "ARI"
    higher_is_better: bool = True
    environment: dict[str, Any] = field(default_factory=dict)
    histoweave_version: str | None = None
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utcnow_iso)
    content_hash: str | None = None
    signature: Signature | None = None

    # -- serialization ---------------------------------------------------- #
    def to_payload(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return the JSON-serializable dict (optionally without signature)."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "node_id": self.node_id,
            "created_at": self.created_at,
            "histoweave_version": self.histoweave_version,
            "task": self.task,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "method": self.method.to_dict() if self.method else None,
            "environment": dict(self.environment),
            "records": [dict(r) for r in self.records],
            "dataset_meta": {k: dict(v) for k, v in self.dataset_meta.items()},
        }
        if self.content_hash is not None:
            payload["content_hash"] = self.content_hash
        if include_signature and self.signature is not None:
            payload["signature"] = self.signature.to_dict()
        return payload

    def compute_hash(self) -> str:
        """Compute and store ``content_hash`` over the unsigned payload."""
        self.content_hash = content_hash(self.to_payload(include_signature=False))
        return self.content_hash

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvidenceBundle:
        if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported evidence schema_version {payload.get('schema_version')!r}; "
                f"expected {EVIDENCE_SCHEMA_VERSION!r}"
            )
        method = payload.get("method")
        sig = payload.get("signature")
        return cls(
            node_id=str(payload.get("node_id", "")),
            task=str(payload.get("task", "")),
            records=list(payload.get("records", [])),
            dataset_meta=dict(payload.get("dataset_meta", {})),
            method=MethodInfo(**method) if isinstance(method, Mapping) else None,
            metric=str(payload.get("metric", "ARI")),
            higher_is_better=bool(payload.get("higher_is_better", True)),
            environment=dict(payload.get("environment", {})),
            histoweave_version=payload.get("histoweave_version"),
            schema_version=str(payload.get("schema_version")),
            bundle_id=str(payload.get("bundle_id", uuid.uuid4())),
            created_at=str(payload.get("created_at", _utcnow_iso())),
            content_hash=payload.get("content_hash"),
            signature=Signature(**sig) if isinstance(sig, Mapping) else None,
        )

    # -- validation ------------------------------------------------------- #
    def score_col(self) -> str:
        """Return the metric column name used inside records."""
        return "ari" if self.metric.upper() == "ARI" else "score"

    def validate(self, *, require_signature: bool = False) -> None:
        """Validate schema, task contracts, and privacy. Raise on any problem.

        Signature *verification* is a separate concern (see
        :mod:`histoweave.federation.signing`); this only checks structure and
        that a signature object is present when required.
        """
        if not self.node_id:
            raise SchemaError("evidence: node_id is required")
        if not self.records:
            raise SchemaError("evidence: at least one record is required")

        # Privacy gate first — before we even look at semantics.
        enforce_privacy_gate(self.to_payload(include_signature=False))

        task_value = self.task
        if task_value in {"domain_detection", "domain"}:
            task_value = AnalysisTask.SPATIAL_DOMAIN.value
        try:
            analysis_task = AnalysisTask(task_value)
        except ValueError as exc:
            raise SchemaError(f"evidence: unknown task {self.task!r}") from exc

        score_col = self.score_col()
        for i, rec in enumerate(self.records):
            unknown = set(rec) - _ALLOWED_RECORD_KEYS
            if unknown:
                raise SchemaError(
                    f"evidence: record[{i}] has unsupported keys {sorted(unknown)}; "
                    f"allowed keys are {sorted(_ALLOWED_RECORD_KEYS)}"
                )
            if not rec.get("dataset"):
                raise SchemaError(f"evidence: record[{i}] missing 'dataset'")
            if not (rec.get("method") or rec.get("config")):
                raise SchemaError(f"evidence: record[{i}] missing 'method'/'config'")
            status = str(rec.get("status", "success") or "success").lower()
            has_score = rec.get(score_col) not in (None, "")
            if status in {"failed", "error", "timeout", "oom"}:
                if has_score:
                    raise SchemaError(
                        f"evidence: record[{i}] status={status!r} must not carry a "
                        f"{score_col!r} value (no silent fallback)"
                    )
            elif not has_score:
                raise SchemaError(
                    f"evidence: record[{i}] is successful but has no {score_col!r} score"
                )
            if has_score:
                try:
                    float(rec[score_col])
                except (TypeError, ValueError) as exc:
                    raise SchemaError(
                        f"evidence: record[{i}] {score_col!r} is not numeric"
                    ) from exc

        # Dataset descriptors: whitelist keys + task-contract validation.
        for ds, meta in self.dataset_meta.items():
            unknown = set(meta) - _ALLOWED_DATASET_META_KEYS
            if unknown:
                raise SchemaError(
                    f"evidence: dataset_meta[{ds!r}] has unsupported keys {sorted(unknown)}"
                )
            visibility = str(meta.get("dataset_visibility", "public")).lower()
            if visibility != "public":
                raise SchemaError(
                    f"evidence: dataset_meta[{ds!r}] dataset_visibility={visibility!r}; "
                    "only 'public' is accepted in v1 (private-aggregate is a future extension)"
                )
            gt = meta.get("ground_truth_kind")
            if gt is None:
                raise SchemaError(f"evidence: dataset_meta[{ds!r}] missing 'ground_truth_kind'")
            label_key = str(meta.get("label_key") or "domain_truth")
            meta_task = meta.get("task") or task_value
            try:
                contract = TaskContract(
                    task=AnalysisTask(meta_task)
                    if not isinstance(meta_task, AnalysisTask)
                    else meta_task,
                    ground_truth_kind=GroundTruthKind(gt),
                    label_key=label_key,
                    platform=classify_platform(meta.get("platform")),
                )
                contract.validate()
            except ValueError as exc:
                raise SchemaError(f"evidence: dataset_meta[{ds!r}] contract: {exc}") from exc

        # Oracle-K documentation rule (mirrors TaskContract.allow_oracle_k).
        if analysis_task is AnalysisTask.SPATIAL_DOMAIN:
            for i, rec in enumerate(self.records):
                if bool(rec.get("oracle_k")):
                    note = " ".join(
                        str(self.environment.get(k, "")) for k in ("notes", "note")
                    ).lower()
                    if "oracle" not in note:
                        raise SchemaError(
                            f"evidence: record[{i}] oracle_k=True requires environment "
                            "note mentioning 'oracle' (document why true K was injected)"
                        )

    def to_long_rows(self) -> list[dict[str, Any]]:
        """Flatten records to landscape/leaderboard long rows (with provenance)."""
        score_col = self.score_col()
        rows: list[dict[str, Any]] = []
        for rec in self.records:
            method = str(rec.get("config") or rec.get("method") or "").strip()
            rows.append(
                {
                    "dataset": str(rec["dataset"]).strip(),
                    "method": str(rec.get("method") or method).strip(),
                    "config": rec.get("config"),
                    "seed": int(rec.get("seed") or 0),
                    "score": _safe_float(rec.get(score_col)),
                    "metric": self.metric,
                    "seconds": _safe_float(rec.get("seconds")),
                    "status": str(rec.get("status", "success") or "success"),
                    "node_id": self.node_id,
                    "task": self.task,
                    "bundle_id": self.bundle_id,
                    "content_hash": self.content_hash,
                }
            )
        return rows


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


# --------------------------------------------------------------------------- #
# Node registry
# --------------------------------------------------------------------------- #
_NODE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


@dataclass
class PublicKeyEntry:
    scheme: str  # "ed25519"
    id: str  # e.g. "ed25519:<b64>"
    value: str  # base64 raw public key
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class NodeRegistryEntry:
    node_id: str
    display_name: str
    evidence_feed: list[str] = field(default_factory=list)
    public_keys: list[PublicKeyEntry] = field(default_factory=list)
    contact: str | None = None
    sigstore_identity: str | None = None
    added_at: str = field(default_factory=lambda: _dt.date.today().isoformat())
    status: str = "active"
    schema_version: str = NODE_REGISTRY_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "display_name": self.display_name,
            "contact": self.contact,
            "evidence_feed": list(self.evidence_feed),
            "public_keys": [k.to_dict() for k in self.public_keys],
            "sigstore_identity": self.sigstore_identity,
            "added_at": self.added_at,
            "status": self.status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NodeRegistryEntry:
        if payload.get("schema_version") != NODE_REGISTRY_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported node registry schema_version "
                f"{payload.get('schema_version')!r}; expected {NODE_REGISTRY_SCHEMA_VERSION!r}"
            )
        feeds = payload.get("evidence_feed", [])
        if isinstance(feeds, str):
            feeds = [feeds]
        keys = [
            PublicKeyEntry(**k) if isinstance(k, Mapping) else k
            for k in payload.get("public_keys", [])
        ]
        return cls(
            node_id=str(payload.get("node_id", "")),
            display_name=str(payload.get("display_name", "")),
            evidence_feed=list(feeds),
            public_keys=keys,
            contact=payload.get("contact"),
            sigstore_identity=payload.get("sigstore_identity"),
            added_at=str(payload.get("added_at", _dt.date.today().isoformat())),
            status=str(payload.get("status", "active")),
            schema_version=str(payload.get("schema_version")),
        )

    def validate(self) -> None:
        if not _NODE_ID_RE.match(self.node_id or ""):
            raise SchemaError(
                f"node registry: node_id {self.node_id!r} must match {_NODE_ID_RE.pattern}"
            )
        if not self.display_name:
            raise SchemaError("node registry: display_name is required")
        if self.status not in {"active", "suspended"}:
            raise SchemaError(f"node registry: unknown status {self.status!r}")
        has_key = any(k.status == "active" for k in self.public_keys)
        if not has_key and not self.sigstore_identity:
            raise SchemaError(
                "node registry: at least one active public key or a sigstore_identity "
                "is required to verify this node's evidence"
            )
        for k in self.public_keys:
            if k.scheme != "ed25519":
                raise SchemaError(
                    f"node registry: unsupported key scheme {k.scheme!r} (v1 supports ed25519)"
                )

    def active_key(self, key_id: str | None = None) -> PublicKeyEntry | None:
        for k in self.public_keys:
            if k.status != "active":
                continue
            if key_id is None or k.id == key_id:
                return k
        return None


def resolve_method_name(record: Mapping[str, Any]) -> str:
    """Prefer ``config`` over ``method`` (matches leaderboard/landscape convention)."""
    return str(record.get("config") or record.get("method") or "").strip()


def coerce_task(task: str | AnalysisTask) -> str:
    value = task.value if isinstance(task, AnalysisTask) else str(task)
    if value in {"domain_detection", "domain"}:
        return AnalysisTask.SPATIAL_DOMAIN.value
    return value


def now_iso() -> str:
    return _utcnow_iso()
