"""``histoweave fed`` command group — the federation CLI surface.

Verbs (see ``federation/PROTOCOL.md`` for the protocol they implement):

* ``fed init-node``  — scaffold a node registry entry + generate an Ed25519
  keypair (private key written locally, **never** committed).
* ``fed sign``       — build + sign an evidence bundle from an existing
  ``benchmark_long.csv`` (+ a dataset-meta sidecar), bridging today's output.
* ``fed verify``     — offline verify signature + schema + task contracts +
  privacy gate for one bundle.
* ``fed pull``       — ingest every registered node's feed into the append-only
  store (used by CI; supports ``--offline`` / ``file://`` feeds for tests).
* ``fed consensus``  — (re)build the derived ``consensus.json`` view.
* ``fed status``     — human-readable summary of the current evidence landscape.

The functions here take an ``argparse.Namespace`` and return a process exit
code, matching the ``_cmd_*`` convention in :mod:`histoweave.cli`. Output goes
through the injected ``emit`` callable so it participates in the CLI's structured
logging.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..benchmark.task_contract import AnalysisTask
from .consensus import build_consensus
from .landscape_bridge import landscape_from_consensus
from .registry import (
    DEFAULT_NODES_DIR,
    build_index,
    load_registry,
    pull_registry,
)
from .schema import (
    EvidenceBundle,
    MethodInfo,
    NodeRegistryEntry,
    SchemaError,
    coerce_task,
)
from .signing import (
    Ed25519KeyPair,
    Ed25519Signer,
    SigningError,
    cryptography_available,
    default_verifiers,
    sigstore_available,
    verify_bundle,
)
from .store import DEFAULT_STORE_PATH, EvidenceStore

EmitFn = Callable[..., None]
DEFAULT_CONSENSUS_PATH = "federation/consensus.json"


def _default_emit(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
) -> None:  # pragma: no cover - fallback only
    """Fallback emitter used only when ``cmd_fed`` is called without an ``emit``.

    Mirrors ``histoweave.cli._emit``: writes to the stream directly (never the
    builtin ``print``) so the repo-wide logging contract holds for this module.
    """
    stream = file if file is not None else sys.stdout
    stream.write(sep.join(str(value) for value in values) + end)


# --------------------------------------------------------------------------- #
# argument wiring (called from histoweave.cli.main)
# --------------------------------------------------------------------------- #
def add_fed_subparser(sub: argparse._SubParsersAction) -> None:
    """Register the ``fed`` command group on the top-level subparsers object."""
    p_fed = sub.add_parser(
        "fed",
        help="Federated evidence network: sign/verify/pull/consensus across labs.",
    )
    fed_sub = p_fed.add_subparsers(dest="fed_command")

    p_init = fed_sub.add_parser(
        "init-node", help="Generate an Ed25519 keypair + a node registry entry."
    )
    p_init.add_argument("--node-id", required=True, help="Lab node id (e.g. lab-cambridge).")
    p_init.add_argument("--display-name", required=True, help="Human-readable lab name.")
    p_init.add_argument("--contact", default=None, help="Contact email/URL.")
    p_init.add_argument(
        "--evidence-feed",
        action="append",
        default=None,
        help="URL(s) where this lab hosts its signed bundles (repeatable).",
    )
    p_init.add_argument(
        "--out-dir",
        default="federation",
        help="Directory for the node file (nodes/<id>.json) and key (default: federation).",
    )
    p_init.add_argument(
        "--key-out",
        default=None,
        help="Path for the PRIVATE key (default: <out-dir>/<node-id>.key.json). Never commit it.",
    )

    p_sign = fed_sub.add_parser(
        "sign", help="Build + sign an evidence bundle from a benchmark_long.csv."
    )
    p_sign.add_argument("--in", dest="in_csv", required=True, help="benchmark_long.csv path.")
    p_sign.add_argument(
        "--sidecar",
        required=True,
        help="JSON with node_id, task, method, dataset_meta, environment.",
    )
    p_sign.add_argument("--key", required=True, help="Private key JSON from `fed init-node`.")
    p_sign.add_argument("--out", required=True, help="Output signed bundle JSON path.")
    p_sign.add_argument("--score-col", default="ari", help="Score column in the CSV (default ari).")

    p_verify = fed_sub.add_parser("verify", help="Offline-verify a signed evidence bundle.")
    p_verify.add_argument("bundle", help="Path to the bundle JSON.")
    p_verify.add_argument(
        "--nodes-dir",
        default=DEFAULT_NODES_DIR,
        help=f"Node registry directory (default: {DEFAULT_NODES_DIR}).",
    )
    p_verify.add_argument(
        "--node",
        default=None,
        help="Path to a single node registry file (overrides --nodes-dir lookup).",
    )

    p_pull = fed_sub.add_parser(
        "pull", help="Ingest all registered nodes' feeds into the append-only store."
    )
    p_pull.add_argument(
        "--nodes-dir",
        default=DEFAULT_NODES_DIR,
        help=f"Node registry directory (default: {DEFAULT_NODES_DIR}).",
    )
    p_pull.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"Append-only store path (default: {DEFAULT_STORE_PATH}).",
    )
    p_pull.add_argument(
        "--no-signature",
        action="store_true",
        help="Skip signature verification (schema+contract+privacy still enforced).",
    )
    p_pull.add_argument(
        "--json", action="store_true", help="Emit the per-node pull report as JSON."
    )

    p_cons = fed_sub.add_parser("consensus", help="(Re)build the derived consensus.json view.")
    p_cons.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"Append-only store path (default: {DEFAULT_STORE_PATH}).",
    )
    p_cons.add_argument(
        "--out",
        default=DEFAULT_CONSENSUS_PATH,
        help=f"Output consensus JSON (default: {DEFAULT_CONSENSUS_PATH}).",
    )
    p_cons.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Absolute agreement tolerance (default 0.05 for ARI).",
    )
    p_cons.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for CI.")
    p_cons.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    p_cons.add_argument(
        "--check-landscape",
        action="store_true",
        help="Also build the recommender landscape and report contract problems.",
    )

    p_status = fed_sub.add_parser("status", help="Summarize the current evidence landscape.")
    p_status.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"Append-only store path (default: {DEFAULT_STORE_PATH}).",
    )
    p_status.add_argument(
        "--nodes-dir",
        default=DEFAULT_NODES_DIR,
        help=f"Node registry directory (default: {DEFAULT_NODES_DIR}).",
    )
    p_status.add_argument("--json", action="store_true", help="Emit the summary as JSON.")


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def cmd_fed(args: argparse.Namespace, *, emit: EmitFn | None = None) -> int:
    """Dispatch a ``fed`` subcommand. Returns a process exit code."""
    say = emit or _default_emit
    sub = getattr(args, "fed_command", None)
    if not sub:
        say(
            "usage: histoweave fed {init-node,sign,verify,pull,consensus,status} …",
            file=sys.stderr,
        )
        return 2
    dispatch = {
        "init-node": _fed_init_node,
        "sign": _fed_sign,
        "verify": _fed_verify,
        "pull": _fed_pull,
        "consensus": _fed_consensus,
        "status": _fed_status,
    }
    handler = dispatch.get(sub)
    if handler is None:  # pragma: no cover - argparse guards this
        say(f"unknown fed subcommand {sub!r}", file=sys.stderr)
        return 2
    return handler(args, say)


# --------------------------------------------------------------------------- #
# verbs
# --------------------------------------------------------------------------- #
def _fed_init_node(args: argparse.Namespace, say: EmitFn) -> int:
    if not cryptography_available():
        say(
            "error: the 'federation' extra (cryptography) is required for init-node; "
            "install with `pip install histoweave-spatial[federation]`",
            file=sys.stderr,
        )
        return 2
    out_dir = Path(args.out_dir)
    nodes_dir = out_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    keypair = Ed25519KeyPair.generate()
    entry = NodeRegistryEntry(
        node_id=args.node_id,
        display_name=args.display_name,
        contact=args.contact,
        evidence_feed=list(args.evidence_feed or []),
        public_keys=[keypair.public_key_entry()],
    )
    try:
        entry.validate()
    except SchemaError as exc:
        say(f"error: {exc}", file=sys.stderr)
        return 2

    node_path = nodes_dir / f"{args.node_id}.json"
    node_path.write_text(json.dumps(entry.to_payload(), indent=2) + "\n", encoding="utf-8")

    key_path = Path(args.key_out) if args.key_out else out_dir / f"{args.node_id}.key.json"
    key_payload = {
        "node_id": args.node_id,
        "scheme": "ed25519",
        "public_key_id": keypair.key_id,
        "private_key_b64": keypair.private_key_b64,
        "public_key_b64": keypair.public_key_b64,
        "WARNING": "PRIVATE KEY — do not commit or share. Add to .gitignore.",
    }
    key_path.write_text(json.dumps(key_payload, indent=2) + "\n", encoding="utf-8")
    try:
        key_path.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass

    say(f"node registry entry written: {node_path}")
    say(f"PRIVATE key written: {key_path} (keep secret; never commit)")
    say(f"public key id: {keypair.key_id}")
    return 0


def _load_key(path: str | Path) -> Ed25519KeyPair:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    priv = payload.get("private_key_b64")
    pub = payload.get("public_key_b64")
    if not priv or not pub:
        raise SigningError(f"key file {path} missing private_key_b64/public_key_b64")
    return Ed25519KeyPair(private_key_b64=priv, public_key_b64=pub)


def _records_from_csv(path: str | Path, score_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SchemaError(f"{path} has no header")
        for raw in reader:
            rec: dict[str, Any] = {}
            if raw.get("dataset"):
                rec["dataset"] = raw["dataset"].strip()
            if raw.get("method"):
                rec["method"] = raw["method"].strip()
            if raw.get("config"):
                rec["config"] = raw["config"].strip()
            if raw.get("seed") not in (None, ""):
                rec["seed"] = int(float(raw["seed"]))
            if raw.get(score_col) not in (None, ""):
                rec["ari"] = float(raw[score_col])
            if raw.get("seconds") not in (None, ""):
                rec["seconds"] = float(raw["seconds"])
            status = (raw.get("status") or "success").strip() or "success"
            rec["status"] = status
            if raw.get("n_domains") not in (None, ""):
                rec["n_domains"] = int(float(raw["n_domains"]))
            elif raw.get("n_domains_truth") not in (None, ""):
                rec["n_domains"] = int(float(raw["n_domains_truth"]))
            if str(raw.get("oracle_k", "")).strip().lower() in {"1", "true", "yes"}:
                rec["oracle_k"] = True
            rows.append(rec)
    if not rows:
        raise SchemaError(f"{path} contained no records")
    return rows


def _fed_sign(args: argparse.Namespace, say: EmitFn) -> int:
    try:
        sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        say(f"error: cannot read sidecar: {exc}", file=sys.stderr)
        return 2
    node_id = sidecar.get("node_id")
    if not node_id:
        say("error: sidecar must include 'node_id'", file=sys.stderr)
        return 2
    task = coerce_task(sidecar.get("task", AnalysisTask.SPATIAL_DOMAIN.value))
    method_meta = sidecar.get("method")
    method = None
    if isinstance(method_meta, dict):
        method = MethodInfo(
            **{k: method_meta[k] for k in method_meta if k in MethodInfo.__dataclass_fields__}
        )
    elif isinstance(method_meta, str):
        method = MethodInfo(name=method_meta)

    try:
        records = _records_from_csv(args.in_csv, args.score_col)
    except (OSError, SchemaError, ValueError) as exc:
        say(f"error: {exc}", file=sys.stderr)
        return 2

    bundle = EvidenceBundle(
        node_id=node_id,
        task=task,
        records=records,
        dataset_meta=dict(sidecar.get("dataset_meta", {})),
        method=method,
        metric=sidecar.get("metric", "ARI"),
        higher_is_better=bool(sidecar.get("higher_is_better", True)),
        environment=dict(sidecar.get("environment", {})),
        histoweave_version=sidecar.get("histoweave_version"),
    )
    try:
        bundle.validate()
    except SchemaError as exc:
        say(f"error: bundle failed validation: {exc}", file=sys.stderr)
        return 2

    try:
        keypair = _load_key(args.key)
        Ed25519Signer.from_keypair(keypair).sign_bundle(bundle)
    except (OSError, SigningError, json.JSONDecodeError) as exc:
        say(f"error: signing failed: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle.to_payload(), indent=2) + "\n", encoding="utf-8")
    say(f"signed bundle written: {out}")
    say(f"  node_id={bundle.node_id} task={bundle.task} records={len(bundle.records)}")
    say(f"  content_hash={bundle.content_hash}")
    return 0


def _find_node_entry(
    node_id: str, nodes_dir: str, node_path: str | None
) -> NodeRegistryEntry | None:
    if node_path:
        from .registry import load_node

        return load_node(node_path)
    for entry in load_registry(nodes_dir):
        if entry.node_id == node_id:
            return entry
    return None


def _fed_verify(args: argparse.Namespace, say: EmitFn) -> int:
    try:
        payload = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
        bundle = EvidenceBundle.from_payload(payload)
    except (OSError, SchemaError, json.JSONDecodeError) as exc:
        say(f"INVALID: cannot parse bundle: {exc}", file=sys.stderr)
        return 1

    try:
        bundle.validate(require_signature=True)
    except SchemaError as exc:
        say(f"INVALID: {exc}", file=sys.stderr)
        return 1

    try:
        node = _find_node_entry(bundle.node_id, args.nodes_dir, args.node)
    except (OSError, SchemaError, json.JSONDecodeError) as exc:
        say(f"INVALID: cannot load node registry: {exc}", file=sys.stderr)
        return 1
    if node is None:
        say(
            f"INVALID: node_id {bundle.node_id!r} is not in the registry "
            f"({args.node or args.nodes_dir}); register it first",
            file=sys.stderr,
        )
        return 1

    scheme = bundle.signature.scheme if bundle.signature else "?"
    if scheme == "sigstore" and not sigstore_available():
        say(
            "SKIP: bundle is sigstore-signed but the 'sigstore' extra is not installed; "
            "cannot verify keyless signature in this environment"
        )
        return 3
    ok = verify_bundle(bundle, node, verifiers=default_verifiers())
    if ok:
        say(
            f"VALID: bundle {bundle.bundle_id} verified for node {bundle.node_id} (scheme={scheme})"
        )
        return 0
    say(f"INVALID: signature verification failed for bundle {bundle.bundle_id}", file=sys.stderr)
    return 1


def _fed_pull(args: argparse.Namespace, say: EmitFn) -> int:
    store = EvidenceStore(args.store)
    results = pull_registry(
        args.nodes_dir,
        store,
        require_signature=not args.no_signature,
    )
    # Refresh the generated node index alongside the registry.
    try:
        entries = load_registry(args.nodes_dir)
        index_path = Path(args.nodes_dir) / "index.json"
        index_path.write_text(json.dumps(build_index(entries), indent=2) + "\n", encoding="utf-8")
    except (OSError, SchemaError) as exc:  # pragma: no cover - defensive
        say(f"warning: could not refresh node index: {exc}", file=sys.stderr)

    if args.json:
        say(json.dumps([r.to_json() for r in results], indent=2))
        return 0
    total_new = sum(r.appended_records for r in results)
    total_rej = sum(r.rejected_bundles for r in results)
    say(f"pulled {len(results)} node(s); {total_new} new record(s), {total_rej} rejected bundle(s)")
    for r in results:
        say(
            f"  {r.node_id}: fetched={r.fetched_bundles} accepted={r.accepted_bundles} "
            f"rejected={r.rejected_bundles} new_records={r.appended_records}"
        )
        for err in r.errors:
            say(f"      ! {err}", file=sys.stderr)
    return 0 if total_rej == 0 else 1


def _fed_consensus(args: argparse.Namespace, say: EmitFn) -> int:
    store = EvidenceStore(args.store)
    if not store.exists():
        say(
            f"error: no evidence store at {args.store}; run `histoweave fed pull` first",
            file=sys.stderr,
        )
        return 2
    view = build_consensus(
        store,
        tolerance=args.tolerance,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(view.to_json(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = view.to_json()["summary"]
    say(f"consensus written: {out}")
    say(
        f"  cells={summary['n_cells']} verified={summary['n_verified']} "
        f"disputed={summary['n_disputed']} unverified={summary['n_unverified']} "
        f"nodes={summary['n_nodes']}"
    )
    if args.check_landscape:
        from ..benchmark.landscape_io import validate_landscape_contracts

        try:
            landscape = landscape_from_consensus(view)
            problems = validate_landscape_contracts(landscape)
        except ValueError as exc:
            say(f"  landscape: {exc}", file=sys.stderr)
            return 1
        if problems:
            for p in problems:
                say(f"  landscape contract problem: {p}", file=sys.stderr)
            return 1
        say(
            f"  landscape OK: {landscape.dataset_count} datasets x {landscape.method_count} methods"
        )
    return 0


def _fed_status(args: argparse.Namespace, say: EmitFn) -> int:
    store = EvidenceStore(args.store)
    records = store.read()
    try:
        nodes = load_registry(args.nodes_dir)
    except (OSError, SchemaError, json.JSONDecodeError):
        nodes = []
    view = build_consensus(store) if records else None
    summary: dict[str, Any] = {
        "registered_nodes": len(nodes),
        "store_records": len(records),
        "contributing_nodes": sorted({r.node_id for r in records}),
    }
    if view is not None:
        summary.update(view.to_json()["summary"])
    if args.json:
        say(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    say("HistoWeave federated evidence — status")
    say(f"  registered nodes:   {summary['registered_nodes']}")
    say(f"  contributing nodes: {len(summary['contributing_nodes'])}")
    say(f"  stored records:     {summary['store_records']}")
    if view is not None:
        say(
            f"  consensus cells:    {summary['n_cells']} "
            f"(verified={summary['n_verified']}, disputed={summary['n_disputed']}, "
            f"unverified={summary['n_unverified']})"
        )
        say(f"  datasets x methods: {summary['n_datasets']} x {summary['n_methods']}")
    else:
        say("  consensus cells:    0 (store empty; run `histoweave fed pull`)")
    return 0
