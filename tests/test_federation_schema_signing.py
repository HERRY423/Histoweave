"""Federation: schema round-trip, hashing, signing, privacy gate, and store.

Covers acceptance cases 1-5 and 11 from the federated-evidence-network plan:

1.  canonical round-trip + content hashing (order-independent, tamper-evident)
2.  Ed25519 sign / verify, tamper detection, unknown-key rejection
3.  contract enforcement (self-supervised labels, oracle-k)
4.  privacy gate (raw-data-shaped payloads are hard-rejected)
5.  append-only store dedup + status corrections
11. Sigstore backend degrades gracefully (skips, never crashes) when absent
"""

from __future__ import annotations

import json

import pytest
from federation_helpers import dataset_meta, make_bundle, make_node, make_record

from histoweave.federation import (
    Ed25519KeyPair,
    Ed25519Verifier,
    EvidenceBundle,
    EvidenceStore,
    MethodInfo,
    SchemaError,
    SigstoreVerifier,
    canonical_json,
    content_hash,
    enforce_privacy_gate,
    latest_records,
    sigstore_available,
    verify_bundle,
)

# --------------------------------------------------------------------------- #
# 1. Canonical serialization + content hashing
# --------------------------------------------------------------------------- #


def test_canonical_json_is_key_order_independent() -> None:
    a = {"b": 1, "a": 2, "c": [3, 2, 1]}
    b = {"a": 2, "c": [3, 2, 1], "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_content_hash_excludes_hash_and_signature_fields() -> None:
    bundle, _ = make_bundle("lab-alpha")
    payload = bundle.to_payload(include_signature=True)
    # Same payload with a mutated content_hash / signature must hash identically,
    # because those two fields are excluded from the digest by construction.
    payload_mut = json.loads(json.dumps(payload))
    payload_mut["content_hash"] = "deadbeef"
    payload_mut["signature"] = {"scheme": "ed25519", "value": "x", "public_key_id": "y"}
    assert content_hash(payload) == content_hash(payload_mut)


def test_bundle_payload_roundtrip_preserves_hash() -> None:
    bundle, _ = make_bundle("lab-alpha", seeds=(1, 2, 3))
    restored = EvidenceBundle.from_payload(bundle.to_payload())
    assert restored.content_hash == bundle.content_hash
    assert restored.node_id == bundle.node_id
    assert len(restored.records) == 3
    # Re-hashing the restored bundle reproduces the same digest.
    assert restored.compute_hash() == bundle.content_hash


# --------------------------------------------------------------------------- #
# 2. Ed25519 sign / verify / tamper / unknown key
# --------------------------------------------------------------------------- #


def test_sign_and_verify_roundtrip() -> None:
    bundle, kp = make_bundle("lab-alpha")
    node = make_node("lab-alpha", kp)
    assert verify_bundle(bundle, node) is True


def test_tampering_after_signing_is_detected() -> None:
    bundle, kp = make_bundle("lab-alpha", ari=0.42)
    node = make_node("lab-alpha", kp)
    assert verify_bundle(bundle, node) is True

    # Mutate a score after signing -> signature must no longer verify.
    bundle.records[0]["ari"] = 0.99
    assert verify_bundle(bundle, node) is False

    # Restoring the value and re-hashing makes it verify again (crypto is sound).
    bundle.records[0]["ari"] = 0.42
    bundle.compute_hash()
    assert verify_bundle(bundle, node) is True


def test_unknown_key_in_registry_is_rejected() -> None:
    bundle, _signing_kp = make_bundle("lab-alpha")
    # Node trusts a *different* key than the one that signed the bundle.
    other_kp = Ed25519KeyPair.generate()
    node = make_node("lab-alpha", other_kp)
    assert verify_bundle(bundle, node) is False


def test_revoked_key_is_not_trusted() -> None:
    bundle, kp = make_bundle("lab-alpha")
    node = make_node("lab-alpha", kp)
    # Revoke the only key.
    node.public_keys[0].status = "revoked"
    assert verify_bundle(bundle, node) is False


def test_verifier_available_reports_true_with_cryptography() -> None:
    assert Ed25519Verifier().available() is True


# --------------------------------------------------------------------------- #
# 3. Task-contract enforcement
# --------------------------------------------------------------------------- #


def test_self_supervised_label_key_rejected_as_ground_truth() -> None:
    # A "leiden" (self-clustering) label cannot masquerade as spatial-domain GT.
    ds = dataset_meta("151673")
    ds["151673"]["label_key"] = "leiden"
    with pytest.raises(SchemaError):
        EvidenceBundle(
            node_id="lab-alpha",
            task="spatial_domain",
            records=[make_record()],
            dataset_meta=ds,
            method=MethodInfo(name="kmeans"),
            metric="ARI",
            higher_is_better=True,
            environment={"os": "linux"},
            histoweave_version="0.1.0b1",
        ).validate()


def test_oracle_k_without_note_is_rejected() -> None:
    rec = make_record()
    rec["oracle_k"] = 7  # using the ground-truth cluster count must be disclosed
    with pytest.raises(SchemaError):
        EvidenceBundle(
            node_id="lab-alpha",
            task="spatial_domain",
            records=[rec],
            dataset_meta=dataset_meta(),
            method=MethodInfo(name="kmeans"),
            metric="ARI",
            higher_is_better=True,
            environment={"os": "linux"},
            histoweave_version="0.1.0b1",
        ).validate()


def test_unknown_record_key_is_rejected() -> None:
    rec = make_record()
    rec["leaked_field"] = 123
    with pytest.raises(SchemaError):
        EvidenceBundle(
            node_id="lab-alpha",
            task="spatial_domain",
            records=[rec],
            dataset_meta=dataset_meta(),
            method=MethodInfo(name="kmeans"),
            metric="ARI",
            higher_is_better=True,
            environment={"os": "linux"},
            histoweave_version="0.1.0b1",
        ).validate()


def test_failed_status_with_score_is_rejected() -> None:
    rec = make_record(status="failed")
    # A failed run must not also carry a score.
    with pytest.raises(SchemaError):
        EvidenceBundle(
            node_id="lab-alpha",
            task="spatial_domain",
            records=[rec],
            dataset_meta=dataset_meta(),
            method=MethodInfo(name="kmeans"),
            metric="ARI",
            higher_is_better=True,
            environment={"os": "linux"},
            histoweave_version="0.1.0b1",
        ).validate()


# --------------------------------------------------------------------------- #
# 4. Privacy gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_key",
    ["counts", "coords", "coordinate", "expression", "adata", "obsm", "barcode", "cell_id"],
)
def test_privacy_gate_rejects_raw_data_shaped_keys(bad_key: str) -> None:
    payload = {"records": [{**make_record(), bad_key: [1, 2, 3]}]}
    with pytest.raises(SchemaError):
        enforce_privacy_gate(payload)


def test_privacy_gate_rejects_long_numeric_arrays() -> None:
    # A short array is fine; a long one looks like a leaked vector.
    ok = {"note": [1, 2, 3]}
    enforce_privacy_gate(ok)  # no raise
    bad = {"note": list(range(50))}
    with pytest.raises(SchemaError):
        enforce_privacy_gate(bad)


def test_private_dataset_visibility_rejected_in_v1() -> None:
    with pytest.raises(SchemaError):
        EvidenceBundle(
            node_id="lab-alpha",
            task="spatial_domain",
            records=[make_record()],
            dataset_meta=dataset_meta("151673", visibility="private"),
            method=MethodInfo(name="kmeans"),
            metric="ARI",
            higher_is_better=True,
            environment={"os": "linux"},
            histoweave_version="0.1.0b1",
        ).validate()


def test_signed_bundle_passes_privacy_gate() -> None:
    bundle, _ = make_bundle("lab-alpha")
    # A clean, contract-valid bundle must sail through the gate.
    enforce_privacy_gate(bundle.to_payload())


# --------------------------------------------------------------------------- #
# 5. Append-only store: dedup + status corrections
# --------------------------------------------------------------------------- #


def test_store_appends_and_dedups(tmp_path) -> None:
    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    bundle, _ = make_bundle("lab-alpha", seeds=(1, 2, 3))
    n_first = store.append_bundle(bundle)
    assert n_first == 3
    # Re-appending the identical bundle adds nothing (idempotent ingest).
    n_second = store.append_bundle(bundle)
    assert n_second == 0
    assert len(store.read()) == 3


def test_store_is_append_only_on_disk(tmp_path) -> None:
    path = tmp_path / "store.jsonl"
    store = EvidenceStore(str(path))
    b1, _ = make_bundle("lab-alpha", ari=0.40)
    b2, _ = make_bundle("lab-beta", ari=0.41)
    store.append_bundle(b1)
    lines_after_1 = path.read_text(encoding="utf-8").count("\n")
    store.append_bundle(b2)
    lines_after_2 = path.read_text(encoding="utf-8").count("\n")
    # The file only ever grows; the first line is never rewritten.
    assert lines_after_2 > lines_after_1


def test_status_correction_is_appended_not_mutated(tmp_path) -> None:
    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    bundle, _ = make_bundle("lab-alpha")
    store.append_bundle(bundle, verification_status="unverified")
    before = len(store.read())
    store.append_status_correction(store.read()[0], "verified", reason="reproduced")
    after = store.read()
    assert len(after) == before + 1
    # latest_records collapses to the corrected status.
    latest = latest_records(after)
    assert latest[0].verification_status == "verified"


# --------------------------------------------------------------------------- #
# 11. Sigstore backend degrades gracefully when the extra is absent
# --------------------------------------------------------------------------- #


def test_sigstore_verifier_reports_availability_consistently() -> None:
    v = SigstoreVerifier()
    assert v.available() == sigstore_available()


@pytest.mark.skipif(sigstore_available(), reason="sigstore extra is installed")
def test_sigstore_absent_does_not_crash_verification() -> None:
    # With sigstore unavailable, an ed25519-signed bundle still verifies through
    # the ed25519 backend; the sigstore verifier simply reports unavailable.
    bundle, kp = make_bundle("lab-alpha")
    node = make_node("lab-alpha", kp)
    assert verify_bundle(bundle, node) is True
    # And a require_available check surfaces the gap rather than raising.
    v = SigstoreVerifier()
    assert v.available() is False
