"""Node registry loading and evidence-feed pulling.

This module is the **decentralization seam**. The federation stores only
*pointers + public keys* (the node registry), never a lab's raw data:

* ``federation/nodes/<node_id>.json`` — one maintainer-reviewed file per lab
  (:class:`~histoweave.federation.schema.NodeRegistryEntry`).
* each entry's ``evidence_feed`` is one or more URLs the **lab itself hosts**;
  :func:`pull_node` fetches those signed bundles and verifies them before
  anything enters the shared store.

Transport is intentionally small and swappable so a future REST or
content-addressed backend is a drop-in:

* ``file://`` and bare local paths (with ``*`` globs) — used by the test-suite's
  in-process multi-node simulation, no network required.
* ``http(s)://`` — a lab's ``raw.githubusercontent.com`` (or any host) feed.
  A feed URL may point at a single JSON bundle, a JSON array of bundles, or a
  small JSON "index" object listing bundle URLs.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema import EvidenceBundle, NodeRegistryEntry, SchemaError
from .signing import Verifier, default_verifiers, verify_bundle
from .store import EvidenceStore

DEFAULT_NODES_DIR = "federation/nodes"

#: Cap on bytes fetched from a single feed URL (defensive; feeds are tiny JSON).
_MAX_FEED_BYTES = 25 * 1024 * 1024


class RegistryError(RuntimeError):
    """Raised when the node registry cannot be loaded or a feed cannot be pulled."""


@dataclass
class PullResult:
    """Outcome of pulling one node's evidence feed(s)."""

    node_id: str
    fetched_bundles: int = 0
    accepted_bundles: int = 0
    rejected_bundles: int = 0
    appended_records: int = 0
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "fetched_bundles": self.fetched_bundles,
            "accepted_bundles": self.accepted_bundles,
            "rejected_bundles": self.rejected_bundles,
            "appended_records": self.appended_records,
            "errors": list(self.errors),
        }


# --------------------------------------------------------------------------- #
# Registry loading
# --------------------------------------------------------------------------- #
def load_node(path: str | Path) -> NodeRegistryEntry:
    """Load and validate a single node registry file."""
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read node registry {p}: {exc}") from exc
    entry = NodeRegistryEntry.from_payload(payload)
    entry.validate()
    return entry


def _is_node_file(fp: Path) -> bool:
    """Return ``True`` if *fp* is a real node entry (not a generated/template file).

    ``index.json`` is a generated convenience listing; ``TEMPLATE.*`` /
    ``EXAMPLE.*`` files are documentation skeletons contributors copy from. Neither
    is a registered node, so both are skipped by loaders and sweeps.
    """
    name = fp.name
    if name == "index.json":
        return False
    stem_upper = name.split(".", 1)[0].upper()
    return stem_upper not in {"TEMPLATE", "EXAMPLE"}


def load_registry(nodes_dir: str | Path = DEFAULT_NODES_DIR) -> list[NodeRegistryEntry]:
    """Load and validate every real ``*.json`` node file in *nodes_dir*.

    ``index.json`` and ``TEMPLATE.*`` / ``EXAMPLE.*`` skeletons are skipped.
    """
    d = Path(nodes_dir)
    if not d.exists():
        return []
    entries: list[NodeRegistryEntry] = []
    for fp in sorted(d.glob("*.json")):
        if not _is_node_file(fp):
            continue
        entries.append(load_node(fp))
    return entries


def build_index(entries: Iterable[NodeRegistryEntry]) -> dict[str, Any]:
    """Build the generated ``index.json`` payload summarizing all nodes."""
    items = [
        {
            "node_id": e.node_id,
            "display_name": e.display_name,
            "status": e.status,
            "n_public_keys": len([k for k in e.public_keys if k.status == "active"]),
            "sigstore_identity": e.sigstore_identity,
            "evidence_feed": list(e.evidence_feed),
        }
        for e in entries
    ]
    return {"schema_version": "histoweave.node_index.v1", "nodes": items}


# --------------------------------------------------------------------------- #
# Feed fetching (transport abstraction)
# --------------------------------------------------------------------------- #
def _fetch_text(url: str) -> str:
    """Fetch raw text from an http(s):// URL (size-capped)."""
    req = urllib.request.Request(url, headers={"User-Agent": "histoweave-fed/1"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - trusted feeds
        data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise RegistryError(f"feed {url} exceeds size cap ({_MAX_FEED_BYTES} bytes)")
    return data.decode("utf-8")


def _local_paths(feed: str) -> list[Path]:
    """Resolve a ``file://`` URL or bare path (supports ``*`` globs)."""
    if feed.startswith("file://"):
        parsed = urlparse(feed)
        raw = urllib.request.url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            raw = f"//{parsed.netloc}{raw}"
        elif os.name == "nt" and len(raw) >= 3 and raw[0] in ("/", chr(92)) and raw[2] == ":":
            raw = raw[1:]
    else:
        raw = feed
    matches = sorted(glob.glob(raw))
    if matches:
        return [Path(m) for m in matches]
    p = Path(raw)
    return [p] if p.exists() else []


def _payloads_from_json(text: str, *, source: str) -> list[dict[str, Any]]:
    """Interpret feed JSON as a single bundle, a list, or an index of URLs."""
    obj = json.loads(text)
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        # An index object listing bundle URLs.
        if "bundles" in obj and isinstance(obj["bundles"], list):
            out: list[dict[str, Any]] = []
            for ref in obj["bundles"]:
                if isinstance(ref, dict):
                    out.append(ref)
                elif isinstance(ref, str):
                    out.extend(fetch_feed(ref))
            return out
        return [obj]
    raise RegistryError(f"feed {source} did not contain JSON object(s)")


def fetch_feed(feed: str) -> list[dict[str, Any]]:
    """Fetch and parse one feed URL into a list of bundle payload dicts."""
    scheme = urlparse(feed).scheme
    if scheme in ("http", "https"):
        text = _fetch_text(feed)
        return _payloads_from_json(text, source=feed)
    # local / file://
    payloads: list[dict[str, Any]] = []
    for p in _local_paths(feed):
        payloads.extend(_payloads_from_json(p.read_text(encoding="utf-8"), source=str(p)))
    return payloads


# --------------------------------------------------------------------------- #
# Pull + verify + ingest
# --------------------------------------------------------------------------- #
def pull_node(
    node: NodeRegistryEntry,
    store: EvidenceStore,
    *,
    verifiers: dict[str, Verifier] | None = None,
    require_signature: bool = True,
    require_verifier_available: bool = False,
) -> PullResult:
    """Pull a node's feed(s), verify each bundle, and append passing records.

    A bundle is accepted only if (in order): schema+contract+privacy validation
    passes, its ``content_hash`` is consistent, its ``node_id`` matches the
    registry entry, and its signature verifies against a trusted key. Anything
    that fails is counted and reported, never appended.
    """
    verifiers = verifiers or default_verifiers()
    result = PullResult(node_id=node.node_id)
    if node.status != "active":
        result.errors.append(f"node {node.node_id} status={node.status}; skipped")
        return result

    payloads: list[dict[str, Any]] = []
    for feed in node.evidence_feed:
        try:
            payloads.extend(fetch_feed(feed))
        except (OSError, RegistryError, json.JSONDecodeError) as exc:
            result.errors.append(f"feed {feed}: {exc}")

    for payload in payloads:
        result.fetched_bundles += 1
        try:
            bundle = EvidenceBundle.from_payload(payload)
        except SchemaError as exc:
            result.rejected_bundles += 1
            result.errors.append(f"parse: {exc}")
            continue

        if bundle.node_id != node.node_id:
            result.rejected_bundles += 1
            result.errors.append(
                f"bundle {bundle.bundle_id} node_id {bundle.node_id!r} != registry {node.node_id!r}"
            )
            continue
        try:
            bundle.validate(require_signature=require_signature)
        except SchemaError as exc:
            result.rejected_bundles += 1
            result.errors.append(f"validate {bundle.bundle_id}: {exc}")
            continue

        if require_signature:
            ok = verify_bundle(
                bundle,
                node,
                verifiers=verifiers,
                require_available=require_verifier_available,
            )
            if not ok:
                result.rejected_bundles += 1
                result.errors.append(f"signature failed for bundle {bundle.bundle_id}")
                continue

        appended = store.append_bundle(
            bundle,
            verification_status="unverified",
            revalidate=False,
            provenance={"source": "pull", "node_id": node.node_id},
        )
        result.appended_records += appended
        result.accepted_bundles += 1
    return result


def pull_registry(
    nodes_dir: str | Path,
    store: EvidenceStore,
    *,
    require_signature: bool = True,
    require_verifier_available: bool = False,
) -> list[PullResult]:
    """Pull every active node in *nodes_dir* into *store*."""
    verifiers = default_verifiers()
    results: list[PullResult] = []
    for node in load_registry(nodes_dir):
        results.append(
            pull_node(
                node,
                store,
                verifiers=verifiers,
                require_signature=require_signature,
                require_verifier_available=require_verifier_available,
            )
        )
    return results
