"""Signing and verification for the HistoWeave federated evidence network.

Two identity backends are supported behind a small pluggable interface so the
core stays offline-testable while a keyless GitHub-native option is available to
labs that want it:

* **Ed25519** (default) — a registered public key. Pure in-process crypto via
  the ``cryptography`` package, deterministic, needs no network. This is what
  the reference implementation and the test-suite exercise.
* **Sigstore / OIDC** (opt-in) — keyless, GitHub-identity-tied. Import is
  *guarded*: if the optional ``sigstore`` extra is not installed the verifier
  degrades gracefully (``available() == False``) rather than crashing, matching
  the project's "fail-closed optional backend" convention.

Only the *signature* lives here. What is signed is the bundle's
:func:`histoweave.federation.schema.content_hash` over the canonicalized,
signature-excluded payload, so a signature is reproducible by any verifier and
is invalidated by any mutation of the evidence.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .schema import (
    EvidenceBundle,
    NodeRegistryEntry,
    PublicKeyEntry,
    SchemaError,
    Signature,
    content_hash,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

ED25519_SCHEME = "ed25519"
SIGSTORE_SCHEME = "sigstore"

# --------------------------------------------------------------------------- #
# Guarded optional backends
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised indirectly; availability depends on env
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )

    _CRYPTOGRAPHY_AVAILABLE = True
except Exception:  # pragma: no cover - only when cryptography missing
    _CRYPTOGRAPHY_AVAILABLE = False


def cryptography_available() -> bool:
    """Return ``True`` if the Ed25519 backend can be used."""
    return _CRYPTOGRAPHY_AVAILABLE


def sigstore_available() -> bool:
    """Return ``True`` if the optional ``sigstore`` verifier backend is importable."""
    try:  # pragma: no cover - depends on optional extra
        import sigstore  # type: ignore[import-not-found] # noqa: F401

        return True
    except Exception:  # pragma: no cover - the common case in CI/tests
        return False


class SigningError(RuntimeError):
    """Raised when signing or verification cannot be performed."""


# --------------------------------------------------------------------------- #
# base64 helpers (url-safe not used; registry stores standard base64)
# --------------------------------------------------------------------------- #
def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


# --------------------------------------------------------------------------- #
# Pluggable interfaces
# --------------------------------------------------------------------------- #
class Signer(ABC):
    """Produces a :class:`~histoweave.federation.schema.Signature` for a bundle."""

    scheme: str

    @abstractmethod
    def sign_bundle(self, bundle: EvidenceBundle) -> Signature:
        """Compute the bundle hash (if needed) and return a signature object."""

    @property
    @abstractmethod
    def public_key_id(self) -> str:
        """Stable identifier for the signing key (goes into the node registry)."""


class Verifier(ABC):
    """Verifies a bundle's signature against a trusted node registry entry."""

    scheme: str

    @abstractmethod
    def available(self) -> bool:
        """Whether this verifier can actually run in the current environment."""

    @abstractmethod
    def verify_bundle(self, bundle: EvidenceBundle, node: NodeRegistryEntry) -> bool:
        """Return ``True`` iff *bundle*'s signature is valid and trusted."""


# --------------------------------------------------------------------------- #
# Ed25519 backend
# --------------------------------------------------------------------------- #
@dataclass
class Ed25519KeyPair:
    """An Ed25519 keypair with base64 (de)serialization of the raw 32-byte keys.

    Private keys are **never** written into the repo; ``fed init-node`` writes
    them to a local file the contributor keeps. Only :attr:`public_key_b64`
    goes into the node registry.
    """

    private_key_b64: str
    public_key_b64: str

    @property
    def key_id(self) -> str:
        return f"{ED25519_SCHEME}:{self.public_key_b64}"

    def public_key_entry(self, status: str = "active") -> PublicKeyEntry:
        return PublicKeyEntry(
            scheme=ED25519_SCHEME,
            id=self.key_id,
            value=self.public_key_b64,
            status=status,
        )

    @classmethod
    def generate(cls) -> Ed25519KeyPair:
        if not _CRYPTOGRAPHY_AVAILABLE:  # pragma: no cover - env-dependent
            raise SigningError(
                "cryptography is required to generate Ed25519 keys; install the "
                "'federation' extra (pip install histoweave-spatial[federation])"
            )
        priv = _Ed25519PrivateKey.generate()
        raw_priv = priv.private_bytes_raw()
        raw_pub = priv.public_key().public_bytes_raw()
        return cls(private_key_b64=_b64e(raw_priv), public_key_b64=_b64e(raw_pub))


class Ed25519Signer(Signer):
    """Sign a bundle's ``content_hash`` with a raw Ed25519 private key."""

    scheme = ED25519_SCHEME

    def __init__(self, private_key_b64: str, public_key_b64: str | None = None) -> None:
        if not _CRYPTOGRAPHY_AVAILABLE:  # pragma: no cover - env-dependent
            raise SigningError(
                "cryptography is required for Ed25519 signing; install the "
                "'federation' extra (pip install histoweave-spatial[federation])"
            )
        self._priv = _Ed25519PrivateKey.from_private_bytes(_b64d(private_key_b64))
        if public_key_b64 is None:
            public_key_b64 = _b64e(self._priv.public_key().public_bytes_raw())
        self._public_key_b64 = public_key_b64

    @classmethod
    def from_keypair(cls, keypair: Ed25519KeyPair) -> Ed25519Signer:
        return cls(keypair.private_key_b64, keypair.public_key_b64)

    @property
    def public_key_id(self) -> str:
        return f"{ED25519_SCHEME}:{self._public_key_b64}"

    def sign_bundle(self, bundle: EvidenceBundle) -> Signature:
        digest = bundle.compute_hash()  # canonical hash over unsigned payload
        sig_bytes = self._priv.sign(digest.encode("utf-8"))
        signature = Signature(
            scheme=ED25519_SCHEME,
            value=_b64e(sig_bytes),
            public_key_id=self.public_key_id,
            certificate=None,
        )
        bundle.signature = signature
        return signature


class Ed25519Verifier(Verifier):
    """Verify an Ed25519 signature against an active key in the node registry."""

    scheme = ED25519_SCHEME

    def available(self) -> bool:
        return _CRYPTOGRAPHY_AVAILABLE

    def verify_bundle(self, bundle: EvidenceBundle, node: NodeRegistryEntry) -> bool:
        if not _CRYPTOGRAPHY_AVAILABLE:  # pragma: no cover - env-dependent
            raise SigningError("cryptography is required to verify Ed25519 signatures")
        sig = bundle.signature
        if sig is None or sig.scheme != ED25519_SCHEME:
            return False
        if not sig.value:
            return False

        # 1) content_hash must be internally consistent with the payload.
        recomputed = content_hash(bundle.to_payload(include_signature=False))
        if bundle.content_hash is not None and bundle.content_hash != recomputed:
            return False
        signed_message = (bundle.content_hash or recomputed).encode("utf-8")

        # 2) the signing key must be an *active, trusted* key for this node.
        key = _select_registry_key(node, sig.public_key_id)
        if key is None:
            return False
        try:
            pub = _Ed25519PublicKey.from_public_bytes(_b64d(key.value))
            pub.verify(_b64d(sig.value), signed_message)
        except (_InvalidSignature, ValueError, TypeError):
            return False
        return True


def _select_registry_key(
    node: NodeRegistryEntry, public_key_id: str | None
) -> PublicKeyEntry | None:
    """Pick the trusted Ed25519 key matching the signature's key id (if given)."""
    candidates = [
        k for k in node.public_keys if k.scheme == ED25519_SCHEME and k.status == "active"
    ]
    if public_key_id:
        for k in candidates:
            if k.id == public_key_id:
                return k
        return None
    # No key id on the signature: only unambiguous if the node has exactly one.
    return candidates[0] if len(candidates) == 1 else None


# --------------------------------------------------------------------------- #
# Sigstore backend (opt-in, guarded)
# --------------------------------------------------------------------------- #
class SigstoreVerifier(Verifier):
    """Keyless verifier for sigstore-signed bundles.

    The implementation is intentionally minimal and *guarded*: when the
    optional ``sigstore`` dependency is unavailable, :meth:`available` returns
    ``False`` and :meth:`verify_bundle` raises a clear, catchable error so
    callers/tests skip rather than fail. Full keyless verification is a v2
    surface; v1 wires the seam and the identity binding.
    """

    scheme = SIGSTORE_SCHEME

    def available(self) -> bool:
        return sigstore_available()

    def verify_bundle(self, bundle: EvidenceBundle, node: NodeRegistryEntry) -> bool:
        if not self.available():
            raise SigningError(
                "sigstore backend not installed; install the 'sigstore' extra to "
                "verify keyless (OIDC) signatures, or use Ed25519"
            )
        sig = bundle.signature
        if sig is None or sig.scheme != SIGSTORE_SCHEME:
            return False
        if not node.sigstore_identity:
            raise SchemaError(
                "node registry entry declares no sigstore_identity but bundle is sigstore-signed"
            )
        # only reachable when the optional extra is present
        return _verify_sigstore_bundle(bundle, node)  # pragma: no cover


def _verify_sigstore_bundle(
    bundle: EvidenceBundle, node: NodeRegistryEntry
) -> bool:  # pragma: no cover - requires optional extra + network
    """Best-effort keyless verification (only runs when sigstore is installed)."""
    from sigstore.models import Bundle  # type: ignore[import-not-found]
    from sigstore.verify import Verifier as _SSVerifier  # type: ignore[import-not-found]
    from sigstore.verify.policy import Identity  # type: ignore[import-not-found]

    sig = bundle.signature
    if sig is None or not sig.certificate:
        return False
    recomputed = content_hash(bundle.to_payload(include_signature=False))
    if bundle.content_hash is not None and bundle.content_hash != recomputed:
        return False
    verifier = _SSVerifier.production()
    ss_bundle = Bundle.from_json(sig.certificate)
    identity = node.sigstore_identity or ""
    try:
        verifier.verify_artifact(
            (bundle.content_hash or recomputed).encode("utf-8"),
            ss_bundle,
            Identity(identity=identity),
        )
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- #
# Dispatch helpers
# --------------------------------------------------------------------------- #
def default_verifiers() -> dict[str, Verifier]:
    """Return the built-in verifiers keyed by signature scheme."""
    return {ED25519_SCHEME: Ed25519Verifier(), SIGSTORE_SCHEME: SigstoreVerifier()}


def verify_bundle(
    bundle: EvidenceBundle,
    node: NodeRegistryEntry,
    *,
    verifiers: Mapping[str, Verifier] | None = None,
    require_available: bool = False,
) -> bool:
    """Verify *bundle* against *node* using the verifier for its scheme.

    Parameters
    ----------
    require_available:
        If ``True``, an unavailable backend (e.g. sigstore not installed) raises
        :class:`SigningError`. If ``False`` (default), an unavailable backend
        returns ``False`` so callers can skip gracefully.
    """
    sig = bundle.signature
    if sig is None:
        return False
    table = dict(default_verifiers()) if verifiers is None else dict(verifiers)
    verifier = table.get(sig.scheme)
    if verifier is None:
        raise SigningError(f"no verifier registered for signature scheme {sig.scheme!r}")
    if not verifier.available():
        if require_available:
            raise SigningError(
                f"verifier for scheme {sig.scheme!r} is not available in this environment"
            )
        return False
    return verifier.verify_bundle(bundle, node)


def sign_bundle(bundle: EvidenceBundle, signer: Signer) -> Signature:
    """Sign *bundle* in place with *signer* and return the signature."""
    return signer.sign_bundle(bundle)
