"""Federation: end-to-end ``histoweave fed`` CLI lifecycle.

Drives the real CLI entrypoint (``histoweave.cli.main``) through the full flow
a lab would use: init-node -> sign -> verify -> pull -> consensus -> status,
plus the tamper path (verify must exit non-zero for a mutated bundle). This
exercises ``cli_fed.py`` without any network or servers.
"""

from __future__ import annotations

import json
from pathlib import Path

from histoweave.cli import main


def _make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text(
        "dataset,method,config,seed,ari,seconds,status,n_domains_truth\n"
        "151673,kmeans,kmeans,42,0.42,8.0,success,7\n"
        "151673,kmeans,kmeans,1,0.43,7.5,success,7\n",
        encoding="utf-8",
    )
    sidecar = {
        "node_id": "lab-test",
        "task": "spatial_domain",
        "method": "kmeans",
        "metric": "ARI",
        "higher_is_better": True,
        "dataset_meta": {
            "151673": {
                "platform": "Visium",
                "tissue": "DLPFC",
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
                "label_key": "domain_truth",
                "dataset_visibility": "public",
                "n_domains": 7,
            }
        },
        "environment": {"os": "linux", "python": "3.11"},
    }
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return csv_path, sidecar_path


def test_fed_cli_full_lifecycle(tmp_path: Path, capsys) -> None:
    fed = tmp_path / "federation"
    csv_path, sidecar_path = _make_inputs(tmp_path)
    feed = tmp_path / "feed.json"

    # 1. init-node
    rc = main(
        [
            "fed", "init-node",
            "--node-id", "lab-test",
            "--display-name", "Test Lab",
            "--contact", "t@example.org",
            "--evidence-feed", str(feed),
            "--out-dir", str(fed),
        ]
    )
    assert rc == 0
    node_file = fed / "nodes" / "lab-test.json"
    key_file = fed / "lab-test.key.json"
    assert node_file.exists() and key_file.exists()
    capsys.readouterr()

    # 2. sign
    rc = main(
        [
            "fed", "sign",
            "--in", str(csv_path),
            "--sidecar", str(sidecar_path),
            "--key", str(key_file),
            "--out", str(feed),
        ]
    )
    assert rc == 0
    assert feed.exists()
    capsys.readouterr()

    # 3. verify -> VALID (exit 0)
    rc = main(["fed", "verify", str(feed), "--node", str(node_file)])
    assert rc == 0
    assert "VALID" in capsys.readouterr().out

    # 4. pull -> one accepted bundle, two records
    store = fed / "store.jsonl"
    rc = main(
        ["fed", "pull", "--nodes-dir", str(fed / "nodes"), "--store", str(store), "--json"]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report[0]["accepted_bundles"] == 1
    assert report[0]["appended_records"] == 2

    # 5. consensus
    consensus = fed / "consensus.json"
    rc = main(
        [
            "fed", "consensus",
            "--store", str(store),
            "--out", str(consensus),
            "--tolerance", "0.05",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    cdata = json.loads(consensus.read_text(encoding="utf-8"))
    assert cdata["summary"]["n_cells"] == 1

    # 6. status
    rc = main(
        ["fed", "status", "--store", str(store), "--nodes-dir", str(fed / "nodes"), "--json"]
    )
    assert rc == 0
    status = json.loads(capsys.readouterr().out)
    assert status["registered_nodes"] == 1
    assert status["store_records"] == 2


def test_fed_cli_verify_rejects_tampered_bundle(tmp_path: Path, capsys) -> None:
    fed = tmp_path / "federation"
    csv_path, sidecar_path = _make_inputs(tmp_path)
    feed = tmp_path / "feed.json"

    assert (
        main(
            [
                "fed", "init-node",
                "--node-id", "lab-test",
                "--display-name", "Test Lab",
                "--evidence-feed", str(feed),
                "--out-dir", str(fed),
            ]
        )
        == 0
    )
    key_file = fed / "lab-test.key.json"
    node_file = fed / "nodes" / "lab-test.json"
    capsys.readouterr()

    assert (
        main(
            [
                "fed", "sign",
                "--in", str(csv_path),
                "--sidecar", str(sidecar_path),
                "--key", str(key_file),
                "--out", str(feed),
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Tamper with the signed bundle on disk, then verify must fail (exit 1).
    payload = json.loads(feed.read_text(encoding="utf-8"))
    payload["records"][0]["ari"] = 0.99
    feed.write_text(json.dumps(payload), encoding="utf-8")

    rc = main(["fed", "verify", str(feed), "--node", str(node_file)])
    assert rc == 1
    captured = capsys.readouterr()
    # The INVALID diagnostic is written to stderr (stdout stays clean).
    assert "INVALID" in (captured.out + captured.err)


def test_fed_verify_bad_bundle_path_exits_one(tmp_path: Path, capsys) -> None:
    rc = main(["fed", "verify", str(tmp_path / "missing.json"), "--nodes-dir", str(tmp_path)])
    assert rc == 1
    assert "INVALID" in (capsys.readouterr().err)


def test_fed_verify_unregistered_node_exits_one(tmp_path: Path, capsys) -> None:
    fed = tmp_path / "federation"
    csv_path, sidecar_path = _make_inputs(tmp_path)
    feed = tmp_path / "feed.json"
    assert (
        main(
            [
                "fed", "init-node",
                "--node-id", "lab-test",
                "--display-name", "Test Lab",
                "--evidence-feed", str(feed),
                "--out-dir", str(fed),
            ]
        )
        == 0
    )
    key_file = fed / "lab-test.key.json"
    capsys.readouterr()
    assert (
        main(
            [
                "fed", "sign",
                "--in", str(csv_path),
                "--sidecar", str(sidecar_path),
                "--key", str(key_file),
                "--out", str(feed),
            ]
        )
        == 0
    )
    capsys.readouterr()
    # Point at an empty registry dir -> node_id not found -> INVALID exit 1.
    empty_nodes = tmp_path / "empty_nodes"
    empty_nodes.mkdir()
    rc = main(["fed", "verify", str(feed), "--nodes-dir", str(empty_nodes)])
    assert rc == 1
    assert "not in the registry" in capsys.readouterr().err


def test_fed_sign_missing_node_id_exits_two(tmp_path: Path, capsys) -> None:
    csv_path, _ = _make_inputs(tmp_path)
    bad_sidecar = tmp_path / "bad.json"
    bad_sidecar.write_text(json.dumps({"task": "spatial_domain"}), encoding="utf-8")
    # A dummy key file (never reached; sidecar validation fails first).
    key = tmp_path / "k.json"
    key.write_text(json.dumps({"private_key_b64": "x", "public_key_b64": "y"}), encoding="utf-8")
    rc = main(
        [
            "fed", "sign",
            "--in", str(csv_path),
            "--sidecar", str(bad_sidecar),
            "--key", str(key),
            "--out", str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    assert "node_id" in capsys.readouterr().err


def test_fed_consensus_empty_store_exits_two(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "fed", "consensus",
            "--store", str(tmp_path / "nope.jsonl"),
            "--out", str(tmp_path / "c.json"),
        ]
    )
    assert rc == 2
    assert "no evidence store" in capsys.readouterr().err


def _write_node_inputs(
    tmp_path: Path, node_id: str, aris: tuple[float, float]
) -> tuple[Path, Path]:
    """Per-node CSV + sidecar whose ``node_id`` matches the registry entry.

    Each lab reports the same (dataset, method) cell but with its own scores, so a
    two-lab consensus is meaningful (and reproducible within tolerance).
    """
    csv_path = tmp_path / f"{node_id}.bench.csv"
    csv_path.write_text(
        "dataset,method,config,seed,ari,seconds,status,n_domains_truth\n"
        f"151673,kmeans,kmeans,42,{aris[0]},8.0,success,7\n"
        f"151673,kmeans,kmeans,1,{aris[1]},7.5,success,7\n",
        encoding="utf-8",
    )
    sidecar = {
        "node_id": node_id,
        "task": "spatial_domain",
        "method": "kmeans",
        "metric": "ARI",
        "higher_is_better": True,
        "dataset_meta": {
            "151673": {
                "platform": "Visium",
                "tissue": "DLPFC",
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
                "label_key": "domain_truth",
                "dataset_visibility": "public",
                "n_domains": 7,
            }
        },
        "environment": {"os": "linux", "python": "3.11"},
    }
    sidecar_path = tmp_path / f"{node_id}.sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return csv_path, sidecar_path


def test_fed_consensus_check_landscape_and_text_output(tmp_path: Path, capsys) -> None:
    # Full path to a consensus with --check-landscape, using the non-JSON output.
    fed = tmp_path / "federation"
    node_aris = {"lab-alpha": (0.42, 0.43), "lab-beta": (0.44, 0.45)}
    for node_id, aris in node_aris.items():
        csv_path, sidecar_path = _write_node_inputs(tmp_path, node_id, aris)
        assert (
            main(
                [
                    "fed", "init-node",
                    "--node-id", node_id,
                    "--display-name", node_id,
                    "--evidence-feed", str(tmp_path / f"{node_id}.json"),
                    "--out-dir", str(fed),
                ]
            )
            == 0
        )
        capsys.readouterr()
        assert (
            main(
                [
                    "fed", "sign",
                    "--in", str(csv_path),
                    "--sidecar", str(sidecar_path),
                    "--key", str(fed / f"{node_id}.key.json"),
                    "--out", str(tmp_path / f"{node_id}.json"),
                ]
            )
            == 0
        )
        capsys.readouterr()

    store = fed / "store.jsonl"
    assert main(["fed", "pull", "--nodes-dir", str(fed / "nodes"), "--store", str(store)]) == 0
    capsys.readouterr()

    rc = main(
        [
            "fed", "consensus",
            "--store", str(store),
            "--out", str(fed / "consensus.json"),
            "--tolerance", "0.05",
            "--check-landscape",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "consensus written" in out
    assert "landscape OK" in out


def test_fed_status_text_output_empty_store(tmp_path: Path, capsys) -> None:
    rc = main(
        ["fed", "status", "--store", str(tmp_path / "nope.jsonl"), "--nodes-dir", str(tmp_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "status" in out
    assert "store empty" in out
