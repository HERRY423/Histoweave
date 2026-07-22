"""Federation: registry pull, multi-node simulation, recommender + leaderboard.

Covers acceptance cases 8, 9, 10:

8.  the consensus-derived landscape feeds the existing recommender unchanged
9.  the leaderboard generator is byte-for-byte back-compatible when no
    federation files are present, and additively enriched when they are
10. an in-process multi-node simulation across two ingest rounds: signed
    bundles are pulled (never raw data), and evidence hardens as labs agree
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from federation_helpers import make_bundle, make_node

from histoweave.benchmark.recommend import MethodRecommender
from histoweave.datasets import make_synthetic
from histoweave.federation import (
    EvidenceStore,
    build_consensus,
    landscape_from_consensus,
    pull_node,
    validate_consensus_landscape,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_feed(path: Path, bundle) -> None:
    """A lab hosts its own signed bundle as a JSON file (the feed)."""
    path.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 10. Multi-node simulation (pull + verify + harden across two rounds)
# --------------------------------------------------------------------------- #


def test_multi_node_simulation_two_rounds(tmp_path) -> None:
    store = EvidenceStore(str(tmp_path / "shared_store.jsonl"))

    # --- Round 1: only Alpha has published. ---
    alpha_feed = tmp_path / "alpha.json"
    alpha_bundle, alpha_kp = make_bundle("lab-alpha", ari=0.42, seeds=(1, 2, 3))
    _write_feed(alpha_feed, alpha_bundle)
    alpha_node = make_node("lab-alpha", alpha_kp, evidence_feed=[str(alpha_feed)])

    r1 = pull_node(alpha_node, store)
    assert r1.accepted_bundles == 1
    assert r1.rejected_bundles == 0
    assert r1.appended_records == 3

    view1 = build_consensus(store, tolerance=0.05)
    assert view1.cells[0].verification_status == "unverified"
    assert view1.cells[0].n_labs == 1

    # --- Round 2: Beta reproduces Alpha (agrees); Gamma is an outlier. ---
    beta_feed = tmp_path / "beta.json"
    beta_bundle, beta_kp = make_bundle("lab-beta", ari=0.44, seeds=(1, 2, 3))
    _write_feed(beta_feed, beta_bundle)
    beta_node = make_node("lab-beta", beta_kp, evidence_feed=[str(beta_feed)])

    gamma_feed = tmp_path / "gamma.json"
    gamma_bundle, gamma_kp = make_bundle("lab-gamma", ari=0.10, seeds=(1, 2, 3))
    _write_feed(gamma_feed, gamma_bundle)
    gamma_node = make_node("lab-gamma", gamma_kp, evidence_feed=[str(gamma_feed)])

    # Re-pulling Alpha is idempotent (append-only dedup).
    r1b = pull_node(alpha_node, store)
    assert r1b.appended_records == 0

    pull_node(beta_node, store)
    pull_node(gamma_node, store)

    view2 = build_consensus(store, tolerance=0.05)
    cell = view2.cells[0]
    assert cell.n_labs == 3
    # Alpha+Beta agree (2/3 within tolerance) -> hardened to verified.
    assert cell.verification_status == "verified"
    # The robust median resisted Gamma's outlier, and Gamma is flagged.
    assert abs(cell.consensus_score - 0.42) < 1e-9
    assert "lab-gamma" in cell.outlier_node_ids


def test_pull_rejects_node_id_mismatch(tmp_path) -> None:
    # A bundle signed as lab-alpha offered under a lab-evil registry entry is
    # rejected (identity binding), and nothing enters the store.
    feed = tmp_path / "spoof.json"
    bundle, kp = make_bundle("lab-alpha", ari=0.42)
    _write_feed(feed, bundle)
    evil_node = make_node("lab-evil", kp, evidence_feed=[str(feed)])

    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    result = pull_node(evil_node, store)
    assert result.accepted_bundles == 0
    assert result.rejected_bundles == 1
    assert store.exists() is False or len(store.read()) == 0


def test_pull_rejects_tampered_bundle(tmp_path) -> None:
    # Tamper with a score after signing but before hosting -> signature fails,
    # bundle is rejected during pull.
    feed = tmp_path / "tampered.json"
    bundle, kp = make_bundle("lab-alpha", ari=0.42)
    payload = bundle.to_payload()
    payload["records"][0]["ari"] = 0.99  # tamper
    feed.write_text(json.dumps(payload), encoding="utf-8")
    node = make_node("lab-alpha", kp, evidence_feed=[str(feed)])

    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    result = pull_node(node, store)
    assert result.accepted_bundles == 0
    assert result.rejected_bundles == 1


# --------------------------------------------------------------------------- #
# 8. Consensus landscape -> existing recommender (unchanged)
# --------------------------------------------------------------------------- #


def test_consensus_landscape_feeds_recommender(tmp_path) -> None:
    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    # Two datasets, two labs each, so the landscape has >1 reference point.
    for ds, best_ari in [("151673", 0.42), ("151674", 0.30)]:
        for node, delta in [("lab-alpha", 0.0), ("lab-beta", 0.01)]:
            b, _ = make_bundle(node, dataset=ds, ari=best_ari + delta)
            store.append_bundle(b)

    view = build_consensus(store, tolerance=0.05)
    assert validate_consensus_landscape(view) == []

    landscape = landscape_from_consensus(view)
    recommender = MethodRecommender(landscape, k_neighbours=1)
    query = make_synthetic(n_cells=80, n_genes=20, noise=0.2, seed=7)
    rec = recommender.recommend(query, dataset_name="query")
    # The recommender returns a real ranking drawn from federated evidence.
    assert rec.ranked_methods
    assert rec.global_best_method in {"kmeans"}


# --------------------------------------------------------------------------- #
# 9. Leaderboard generator back-compat + additive enrichment
# --------------------------------------------------------------------------- #


def _import_generate():
    """Import the leaderboard generator module (it lives outside the package)."""
    gen_dir = str((REPO_ROOT / "leaderboard").resolve())
    if gen_dir not in sys.path:
        sys.path.insert(0, gen_dir)
    import generate  # noqa: PLC0415 - intentional local import of a script module

    return generate


def test_leaderboard_build_is_v2_without_federation(monkeypatch) -> None:
    # With the federation consensus pointed at a non-existent path, build() must
    # produce the pre-federation v2 feed with no federation block. This is the
    # guardrail for the additive/back-compatible contract.
    generate = _import_generate()
    monkeypatch.setattr(generate, "FEDERATION_CONSENSUS", REPO_ROOT / "does_not_exist.json")
    data = generate.build()
    assert data["protocol"] == "histoweave.leaderboard.v2"
    assert "federation" not in data
    assert all("verification_status" not in r for r in data["records"])


def test_leaderboard_build_is_v3_with_federation(monkeypatch, tmp_path) -> None:
    # Build a real consensus over two agreeing labs on a dataset/method that the
    # committed benchmark CSVs already contain, then confirm build() enriches
    # exactly the matching records and adds a top-level federation block.
    generate = _import_generate()

    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    for node, ari in [("lab-alpha", 0.181), ("lab-beta", 0.190)]:
        b, _ = make_bundle(node, dataset="151673", method="kmeans", ari=ari)
        store.append_bundle(b)
    view = build_consensus(store, tolerance=0.05)
    consensus_path = tmp_path / "consensus.json"
    consensus_path.write_text(json.dumps(view.to_json()), encoding="utf-8")
    monkeypatch.setattr(generate, "FEDERATION_CONSENSUS", consensus_path)

    data = generate.build()
    assert data["protocol"] == "histoweave.leaderboard.v3"
    assert data["federation"]["enabled"] is True
    assert data["federation"]["summary"]["n_verified"] == 1

    enriched = [
        r
        for r in data["records"]
        if r["dataset"] == "151673" and r["method"] == "kmeans" and "verification_status" in r
    ]
    plain = [r for r in data["records"] if "verification_status" not in r]
    # Only the matching cell's records are enriched; everything else stays plain.
    assert len(enriched) >= 1
    assert all(r["verification_status"] == "verified" for r in enriched)
    assert len(plain) > 0
