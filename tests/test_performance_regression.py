"""Performance regression gates for pure-Python hot paths.

These are *not* scientific ARI gates.  They track wall-time ceilings for kNN,
z-score, PCA, and feature extraction so accidental O(n²) regressions fail CI.

Baselines live in ``tests/perf_baselines.json``.  Slack absorbs CI hardware noise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from histoweave._math import knn_indices, pca, zscore
from histoweave.benchmark.features import extract_features
from histoweave.datasets import make_synthetic

pytestmark = pytest.mark.perf

BASELINE_PATH = Path(__file__).with_name("perf_baselines.json")


def _baselines() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _time(fn, *, repeats: int = 3) -> float:
    # Warm-up once so import/JIT noise does not dominate the first sample.
    fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _assert_under(name: str, elapsed: float) -> None:
    payload = _baselines()
    spec = payload["benchmarks"][name]
    ceiling = float(spec["max_seconds"]) * float(payload["slack_factor"])
    assert elapsed <= ceiling, (
        f"performance regression on {name}: {elapsed:.4f}s > ceiling {ceiling:.4f}s "
        f"(baseline {spec['max_seconds']}s × slack {payload['slack_factor']})"
    )


def test_knn_indices_under_ceiling() -> None:
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(2000, 2))

    def run() -> None:
        knn_indices(coords, 15)

    _assert_under("knn_indices_n2000_k15", _time(run))


def test_zscore_under_ceiling() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(4000, 2000))

    def run() -> None:
        zscore(matrix, axis=0)

    _assert_under("zscore_n4000_g2000", _time(run))


def test_pca_under_ceiling() -> None:
    rng = np.random.default_rng(2)
    # genes >> spots triggers the Gram path.
    matrix = rng.normal(size=(4000, 2000))

    def run() -> None:
        pca(matrix, n_components=20, random_state=0)

    _assert_under("pca_n4000_g2000_k20", _time(run))


def test_extract_features_under_ceiling() -> None:
    data = make_synthetic(n_cells=800, n_genes=500, n_domains=4, seed=3)

    def run() -> None:
        extract_features(data, include_domain=False)

    _assert_under("extract_features_n800_g500", _time(run))


def test_baseline_file_is_complete() -> None:
    payload = _baselines()
    assert payload["schema_version"] == 1
    assert payload["slack_factor"] >= 1.0
    required = {
        "knn_indices_n2000_k15",
        "zscore_n4000_g2000",
        "pca_n4000_g2000_k20",
        "extract_features_n800_g500",
    }
    assert required <= set(payload["benchmarks"])
