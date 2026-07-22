"""Shared builders for the federation test-suite.

Keeping the (repetitive) signed-bundle construction in one place lets each test
file focus on the behaviour under test rather than on schema plumbing.
"""

from __future__ import annotations

from typing import Any

from histoweave.federation import (
    Ed25519KeyPair,
    Ed25519Signer,
    EvidenceBundle,
    MethodInfo,
    NodeRegistryEntry,
    sign_bundle,
)
from histoweave.federation.schema import now_iso


#: A complete, contract-valid ``dataset_meta`` block for one public dataset.
def dataset_meta(
    dataset: str = "151673",
    *,
    n_domains: int = 7,
    visibility: str = "public",
) -> dict[str, dict[str, Any]]:
    return {
        dataset: {
            "platform": "Visium",
            "tissue": "DLPFC",
            "task": "spatial_domain",
            "ground_truth_kind": "spatial_domain",
            "label_key": "domain_truth",
            "dataset_visibility": visibility,
            "n_domains": n_domains,
        }
    }


def make_record(
    *,
    dataset: str = "151673",
    method: str = "kmeans",
    seed: int = 42,
    ari: float = 0.42,
    seconds: float = 8.0,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "config": method,
        "seed": seed,
        "ari": ari,
        "seconds": seconds,
        "status": status,
    }


def make_bundle(
    node_id: str,
    *,
    records: list[dict[str, Any]] | None = None,
    dataset: str = "151673",
    method: str = "kmeans",
    ari: float = 0.42,
    seeds: tuple[int, ...] = (42,),
    ds_meta: dict[str, dict[str, Any]] | None = None,
    sign: bool = True,
    keypair: Ed25519KeyPair | None = None,
) -> tuple[EvidenceBundle, Ed25519KeyPair]:
    """Build (and by default sign) an :class:`EvidenceBundle`.

    Returns the bundle together with the keypair used, so callers can register
    the matching public key in a node entry.
    """
    if records is None:
        records = [
            make_record(dataset=dataset, method=method, seed=s, ari=ari) for s in seeds
        ]
    bundle = EvidenceBundle(
        node_id=node_id,
        task="spatial_domain",
        records=records,
        dataset_meta=ds_meta or dataset_meta(dataset),
        method=MethodInfo(name=method),
        metric="ARI",
        higher_is_better=True,
        environment={"os": "linux", "python": "3.11"},
        histoweave_version="0.1.0b1",
    )
    kp = keypair or Ed25519KeyPair.generate()
    if sign:
        sign_bundle(bundle, Ed25519Signer.from_keypair(kp))
    return bundle, kp


def make_node(
    node_id: str,
    keypair: Ed25519KeyPair,
    *,
    evidence_feed: list[str] | None = None,
    display_name: str | None = None,
) -> NodeRegistryEntry:
    """Build a validated node-registry entry trusting *keypair*'s public key."""
    entry = NodeRegistryEntry(
        node_id=node_id,
        display_name=display_name or node_id.replace("-", " ").title(),
        evidence_feed=evidence_feed or [],
        public_keys=[keypair.public_key_entry()],
        contact=f"{node_id}@example.org",
        sigstore_identity=None,
        added_at=now_iso(),
    )
    entry.validate()
    return entry
