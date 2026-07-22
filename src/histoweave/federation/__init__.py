"""HistoWeave federated evidence network.

A decentralized, Git/GitHub-native protocol that lets many labs contribute
benchmark evidence **without sharing raw data**, forming a living evidence
landscape whose records *harden over time* as independent nodes reproduce them.

Public surface (stable):

* :mod:`~histoweave.federation.schema` — evidence & node-registry documents,
  canonical hashing, and the privacy gate.
* :mod:`~histoweave.federation.signing` — pluggable Ed25519 (default) and
  Sigstore (opt-in) signer/verifier backends.
* :mod:`~histoweave.federation.store` — append-only evidence store with
  tiered-trust verification-status transitions.
* :mod:`~histoweave.federation.consensus` — derived cross-lab consensus view.
* :mod:`~histoweave.federation.registry` — node registry loading + feed pulling.
* :mod:`~histoweave.federation.landscape_bridge` — consensus -> recommender
  landscape, reusing the existing landscape machinery unchanged.
"""

from __future__ import annotations

from .consensus import (
    CONSENSUS_SCHEMA_VERSION,
    ConsensusCell,
    ConsensusView,
    NodeTrackRecord,
    build_consensus,
)
from .landscape_bridge import (
    consensus_to_long_rows,
    landscape_from_consensus,
    validate_consensus_landscape,
)
from .registry import (
    DEFAULT_NODES_DIR,
    PullResult,
    RegistryError,
    build_index,
    fetch_feed,
    load_node,
    load_registry,
    pull_node,
    pull_registry,
)
from .schema import (
    EVIDENCE_SCHEMA_VERSION,
    NODE_REGISTRY_SCHEMA_VERSION,
    VERIFICATION_STATES,
    EvidenceBundle,
    MethodInfo,
    NodeRegistryEntry,
    PublicKeyEntry,
    SchemaError,
    Signature,
    canonical_json,
    content_hash,
    enforce_privacy_gate,
)
from .signing import (
    ED25519_SCHEME,
    SIGSTORE_SCHEME,
    Ed25519KeyPair,
    Ed25519Signer,
    Ed25519Verifier,
    SigningError,
    SigstoreVerifier,
    cryptography_available,
    default_verifiers,
    sign_bundle,
    sigstore_available,
    verify_bundle,
)
from .store import (
    DEFAULT_STORE_PATH,
    DEFAULT_TOLERANCE,
    EvidenceStore,
    StoredRecord,
    latest_records,
)

__all__ = [
    # schema
    "EVIDENCE_SCHEMA_VERSION",
    "NODE_REGISTRY_SCHEMA_VERSION",
    "VERIFICATION_STATES",
    "EvidenceBundle",
    "MethodInfo",
    "NodeRegistryEntry",
    "PublicKeyEntry",
    "SchemaError",
    "Signature",
    "canonical_json",
    "content_hash",
    "enforce_privacy_gate",
    # signing
    "ED25519_SCHEME",
    "SIGSTORE_SCHEME",
    "Ed25519KeyPair",
    "Ed25519Signer",
    "Ed25519Verifier",
    "SigningError",
    "SigstoreVerifier",
    "cryptography_available",
    "default_verifiers",
    "sign_bundle",
    "sigstore_available",
    "verify_bundle",
    # store
    "DEFAULT_STORE_PATH",
    "DEFAULT_TOLERANCE",
    "EvidenceStore",
    "StoredRecord",
    "latest_records",
    # consensus
    "CONSENSUS_SCHEMA_VERSION",
    "ConsensusCell",
    "ConsensusView",
    "NodeTrackRecord",
    "build_consensus",
    # registry
    "DEFAULT_NODES_DIR",
    "PullResult",
    "RegistryError",
    "build_index",
    "fetch_feed",
    "load_node",
    "load_registry",
    "pull_node",
    "pull_registry",
    # landscape bridge
    "consensus_to_long_rows",
    "landscape_from_consensus",
    "validate_consensus_landscape",
]
