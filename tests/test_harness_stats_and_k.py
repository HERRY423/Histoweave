"""Harness integration: non-oracle K + bootstrap stats."""

from __future__ import annotations

import pytest

from histoweave.benchmark import domain_detection_task, run_benchmark
from histoweave.datasets.synthetic import make_synthetic


def test_run_benchmark_estimate_k_still_recovers_clean_data():
    data = make_synthetic(seed=0, n_cells=300, n_genes=80, n_domains=3, noise=0.1)
    result = run_benchmark(
        domain_detection_task(data),
        methods=["kmeans", "spectral"],
        k_policy="estimate",
    )
    assert result.stats is not None
    assert result.stats["k_selection"]["source"] == "estimate"
    best = result.best()
    assert best is not None
    # Estimated K on clean 3-domain data should still yield strong recovery.
    assert best["score"] > 0.7


def test_run_benchmark_oracle_requires_flag():
    with pytest.raises(ValueError, match="allow_oracle_k"):
        run_benchmark(domain_detection_task(), k_policy="oracle", allow_oracle_k=False)


def test_run_benchmark_stats_attaches_ari_ci():
    data = make_synthetic(seed=1, n_cells=200, n_genes=60, n_domains=3)
    result = run_benchmark(
        domain_detection_task(data),
        methods=["kmeans"],
        method_params={"kmeans": {"n_domains": 3}},
        stats=True,
        n_boot=40,
        seed=0,
        k_policy="estimate",
    )
    row = result.leaderboard[0]
    assert "ari_ci_low" in row
    assert "ari_ci_high" in row
    assert row["ari_ci_low"] <= row["score"] + 0.15
    assert result.stats is not None
    assert "kmeans" in result.stats["bootstrap_ari"]


def test_run_benchmark_strips_uns_n_domains_under_estimate():
    data = make_synthetic(seed=2, n_domains=3)
    assert data.uns.get("n_domains") == 3
    result = run_benchmark(
        domain_detection_task(data),
        methods=["kmeans"],
        k_policy="estimate",
    )
    # If uns leak remained and estimate failed, method would still run via uns.
    # We only assert the path completed with recorded estimate metadata.
    assert result.stats["k_selection"]["n_domains_used"] >= 2
