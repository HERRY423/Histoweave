"""Federation: node-registry loading, feed-format variants, and pull_registry.

Exercises the decentralization seam's local/file transport paths (no network):
directory loading, ``index.json`` generation, the three feed JSON shapes
(single bundle / list / index-of-bundles), glob feeds, inactive-node skipping,
and the ``pull_registry`` sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

from federation_helpers import make_bundle, make_node

from histoweave.federation import (
    EvidenceStore,
    build_index,
    fetch_feed,
    load_node,
    load_registry,
    pull_node,
    pull_registry,
)
from histoweave.federation.registry import RegistryError


def _write_node(nodes_dir: Path, node) -> Path:
    nodes_dir.mkdir(parents=True, exist_ok=True)
    p = nodes_dir / f"{node.node_id}.json"
    p.write_text(json.dumps(node.to_payload()), encoding="utf-8")
    return p


def test_load_node_and_registry(tmp_path: Path) -> None:
    nodes_dir = tmp_path / "nodes"
    _, kp_a = make_bundle("lab-alpha")
    _, kp_b = make_bundle("lab-beta")
    _write_node(nodes_dir, make_node("lab-alpha", kp_a))
    _write_node(nodes_dir, make_node("lab-beta", kp_b))

    entries = load_registry(nodes_dir)
    assert {e.node_id for e in entries} == {"lab-alpha", "lab-beta"}

    single = load_node(nodes_dir / "lab-alpha.json")
    assert single.node_id == "lab-alpha"


def test_load_registry_skips_index_json(tmp_path: Path) -> None:
    nodes_dir = tmp_path / "nodes"
    _, kp = make_bundle("lab-alpha")
    _write_node(nodes_dir, make_node("lab-alpha", kp))
    # An index.json must be ignored as a node file.
    (nodes_dir / "index.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    entries = load_registry(nodes_dir)
    assert [e.node_id for e in entries] == ["lab-alpha"]


def test_load_registry_skips_template_and_example(tmp_path: Path) -> None:
    nodes_dir = tmp_path / "nodes"
    _, kp = make_bundle("lab-alpha")
    _write_node(nodes_dir, make_node("lab-alpha", kp))
    # Documentation skeletons contributors copy from must never load as nodes,
    # even though they carry a placeholder public key that would fail validation.
    skeleton = {"schema_version": "histoweave.node_registry.v1", "node_id": "PLACEHOLDER"}
    (nodes_dir / "TEMPLATE.node.json").write_text(json.dumps(skeleton), encoding="utf-8")
    (nodes_dir / "EXAMPLE.json").write_text(json.dumps(skeleton), encoding="utf-8")
    entries = load_registry(nodes_dir)
    assert [e.node_id for e in entries] == ["lab-alpha"]


def test_build_index_summarizes_nodes(tmp_path: Path) -> None:
    _, kp_a = make_bundle("lab-alpha")
    _, kp_b = make_bundle("lab-beta")
    entries = [make_node("lab-alpha", kp_a), make_node("lab-beta", kp_b)]
    index = build_index(entries)
    assert index["schema_version"] == "histoweave.node_index.v1"
    ids = {n["node_id"] for n in index["nodes"]}
    assert ids == {"lab-alpha", "lab-beta"}
    assert all(n["n_public_keys"] == 1 for n in index["nodes"])


def test_load_registry_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "nope") == []


def test_fetch_feed_single_bundle(tmp_path: Path) -> None:
    feed = tmp_path / "feed.json"
    bundle, _ = make_bundle("lab-alpha")
    feed.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")
    payloads = fetch_feed(str(feed))
    assert len(payloads) == 1
    assert payloads[0]["node_id"] == "lab-alpha"


def test_fetch_feed_list_of_bundles(tmp_path: Path) -> None:
    feed = tmp_path / "feed.json"
    b1, _ = make_bundle("lab-alpha", ari=0.40)
    b2, _ = make_bundle("lab-alpha", dataset="151674", ari=0.41)
    feed.write_text(json.dumps([b1.to_payload(), b2.to_payload()]), encoding="utf-8")
    payloads = fetch_feed(str(feed))
    assert len(payloads) == 2


def test_fetch_feed_index_of_bundle_paths(tmp_path: Path) -> None:
    # An index object whose "bundles" lists file paths is expanded transitively.
    b1, _ = make_bundle("lab-alpha", ari=0.40)
    (tmp_path / "b1.json").write_text(json.dumps(b1.to_payload()), encoding="utf-8")
    index = {"bundles": [str(tmp_path / "b1.json"), b1.to_payload()]}
    feed = tmp_path / "index.json"
    feed.write_text(json.dumps(index), encoding="utf-8")
    payloads = fetch_feed(str(feed))
    assert len(payloads) == 2


def test_fetch_feed_glob(tmp_path: Path) -> None:
    b1, _ = make_bundle("lab-alpha", ari=0.40)
    b2, _ = make_bundle("lab-alpha", dataset="151674", ari=0.41)
    (tmp_path / "bundle_1.json").write_text(json.dumps(b1.to_payload()), encoding="utf-8")
    (tmp_path / "bundle_2.json").write_text(json.dumps(b2.to_payload()), encoding="utf-8")
    payloads = fetch_feed(str(tmp_path / "bundle_*.json"))
    assert len(payloads) == 2


def test_fetch_feed_file_uri(tmp_path: Path) -> None:
    feed = tmp_path / "feed.json"
    bundle, _ = make_bundle("lab-alpha")
    feed.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")
    payloads = fetch_feed(feed.as_uri())
    assert len(payloads) == 1


def test_fetch_feed_bad_json_shape_raises(tmp_path: Path) -> None:
    feed = tmp_path / "feed.json"
    feed.write_text(json.dumps("not a bundle"), encoding="utf-8")
    try:
        fetch_feed(str(feed))
    except RegistryError:
        pass
    else:  # pragma: no cover - explicit failure
        raise AssertionError("expected RegistryError for scalar JSON feed")


def test_pull_node_skips_inactive(tmp_path: Path) -> None:
    feed = tmp_path / "feed.json"
    bundle, kp = make_bundle("lab-alpha")
    feed.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")
    node = make_node("lab-alpha", kp, evidence_feed=[str(feed)])
    node.status = "inactive"
    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    result = pull_node(node, store)
    assert result.accepted_bundles == 0
    assert any("inactive" in e or "status" in e for e in result.errors)


def test_pull_node_without_signature_requirement(tmp_path: Path) -> None:
    # An unsigned bundle is accepted only when require_signature=False.
    feed = tmp_path / "feed.json"
    bundle, kp = make_bundle("lab-alpha", sign=False)
    feed.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")
    node = make_node("lab-alpha", kp, evidence_feed=[str(feed)])

    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    strict = pull_node(node, store, require_signature=True)
    assert strict.accepted_bundles == 0  # unsigned rejected under strict mode

    store2 = EvidenceStore(str(tmp_path / "store2.jsonl"))
    lax = pull_node(node, store2, require_signature=False)
    assert lax.accepted_bundles == 1


def test_pull_registry_sweeps_all_nodes(tmp_path: Path) -> None:
    nodes_dir = tmp_path / "nodes"
    # Two labs, each hosting one signed bundle on the same dataset/method.
    for node_id, ari in [("lab-alpha", 0.42), ("lab-beta", 0.43)]:
        feed = tmp_path / f"{node_id}.json"
        bundle, kp = make_bundle(node_id, ari=ari)
        feed.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")
        _write_node(nodes_dir, make_node(node_id, kp, evidence_feed=[str(feed)]))

    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    results = pull_registry(nodes_dir, store)
    assert len(results) == 2
    assert sum(r.accepted_bundles for r in results) == 2
    assert len(store.read()) == 2
