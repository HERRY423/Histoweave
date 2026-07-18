"""Tests for method failure fingerprint atlas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from histoweave.benchmark import (
    FINGERPRINT_ORDER,
    classify_domain_failure,
    run_failure_fingerprint_probe,
    structural_severity,
    write_fingerprint_atlas,
)
from histoweave.cli import main


def test_classify_fragmentation_failure():
    # One true domain split into 6 predicted fragments.
    truth = np.array([0] * 60 + [1] * 40)
    pred = np.array([0, 1, 2, 3, 4, 5] * 10 + [6] * 40)
    profile = classify_domain_failure(truth, pred, frag_min_clusters=5)
    assert profile.fragmentation_flag
    assert profile.fragmentation >= 1.0
    assert profile.max_true_fragments >= 5


def test_classify_merge_failure():
    # Three true domains collapsed into one predicted cluster.
    truth = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 10)
    pred = np.array([0] * 90 + [1] * 10)
    profile = classify_domain_failure(truth, pred, merge_min_domains=3)
    assert profile.merge_flag
    assert profile.merge >= 1.0
    assert profile.max_pred_true_domains >= 3


def test_classify_noise_failure():
    # Several micro-clusters < 5% of n.
    n = 200
    truth = np.array([i % 4 for i in range(n)])
    pred = np.array([i % 4 for i in range(n)], dtype=int)
    # Sprinkle 10 singleton noise clusters.
    for i in range(10):
        pred[i] = 100 + i
    profile = classify_domain_failure(truth, pred, noise_frac=0.05)
    assert profile.noise_flag
    assert profile.n_micro_clusters >= 10
    assert profile.noise > 0.0


def test_structural_severity():
    assert structural_severity(0.95, 0.05, tau=0.7) == 1.0
    assert structural_severity(0.50, 0.05, tau=0.7) == 0.0  # never worked on easy
    assert 0.0 < structural_severity(0.90, 0.50, tau=0.7) < 1.0


def test_fingerprint_probe_and_artifacts(tmp_path: Path):
    atlas = run_failure_fingerprint_probe(
        methods=["kmeans"],
        seeds=(0, 1),
        tau=0.7,
        progress=False,
    )
    assert len(atlas.fingerprints) == 1
    fp = atlas.fingerprints[0]
    assert fp.method == "kmeans"
    assert list(fp.vector.keys()) == list(FINGERPRINT_ORDER)
    assert all(0.0 <= fp.vector[k] <= 1.0 for k in FINGERPRINT_ORDER)
    assert len(FINGERPRINT_ORDER) == 4

    paths = write_fingerprint_atlas(atlas, tmp_path)
    assert Path(paths["json"]).is_file()
    assert Path(paths["markdown"]).is_file()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["matrix"]["kmeans"]
    assert len(payload["matrix"]["kmeans"]) == 4


def test_failure_fingerprint_cli(tmp_path: Path):
    out = tmp_path / "fp"
    rc = main(
        [
            "failure-fingerprint",
            "--methods",
            "kmeans",
            "--seeds",
            "1",
            "--out-dir",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    assert (out / "failure_fingerprints.json").is_file()
