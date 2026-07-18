"""Statistical review layer: bootstrap ranks, permutation tests, FDR."""

from __future__ import annotations

import numpy as np
import pytest

from histoweave._math import adjusted_rand_index
from histoweave.benchmark.multiple_testing import fdr_adjust, pairwise_fdr_table, reject_nulls
from histoweave.benchmark.stats_review import (
    bootstrap_ari,
    bootstrap_rank_stability,
    paired_permutation_pvalue,
    ranks_from_scores,
    review_landscape,
)


def test_fdr_bh_controls_known_case():
    # Classic BH example: first two discoveries at alpha=0.05.
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.07, 0.08, 0.09, 0.10])
    q = fdr_adjust(p, method="bh")
    assert q[0] <= q[1] <= 1.0
    assert q[0] < 0.05
    # Monotone non-decreasing after sort is enforced in original order via clip.
    assert np.all(q >= p - 1e-12)
    assert reject_nulls(p, method="bh", alpha=0.05).sum() >= 1


def test_fdr_holm_stricter_than_bh():
    p = np.array([0.01, 0.02, 0.03, 0.04])
    q_bh = fdr_adjust(p, method="bh")
    q_holm = fdr_adjust(p, method="holm")
    assert np.all(q_holm + 1e-12 >= q_bh)


def test_fdr_rejects_out_of_range():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        fdr_adjust([0.1, 1.5])


def test_ranks_from_scores_higher_is_better():
    ranks = ranks_from_scores(np.array([0.9, 0.2, 0.5]), higher_is_better=True)
    assert ranks[0] == 1.0
    assert ranks[1] == 3.0
    assert ranks[2] == 2.0


def test_ranks_nan_gets_worst():
    ranks = ranks_from_scores(np.array([0.5, np.nan, 0.8]), higher_is_better=True)
    assert ranks[1] == 3.0
    assert ranks[2] == 1.0


def test_bootstrap_ari_identical_labels():
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    boot = bootstrap_ari(labels, labels, n_boot=50, seed=0)
    assert boot.point == pytest.approx(1.0)
    assert boot.ci_low > 0.9
    assert boot.ci_high <= 1.0 + 1e-9


def test_bootstrap_ari_random_lower_than_identical():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 3, size=80)
    pred = rng.integers(0, 3, size=80)
    boot = bootstrap_ari(truth, pred, n_boot=80, seed=2)
    assert boot.mean < 0.5
    assert boot.ci_low <= boot.mean <= boot.ci_high


def test_bootstrap_rank_stability_orders_methods():
    # Method A dominates every dataset.
    performance = {
        "d1": {"A": 0.9, "B": 0.4, "C": 0.3},
        "d2": {"A": 0.85, "B": 0.5, "C": 0.2},
        "d3": {"A": 0.95, "B": 0.45, "C": 0.25},
        "d4": {"A": 0.8, "B": 0.55, "C": 0.35},
    }
    summaries, posterior, samples = bootstrap_rank_stability(performance, n_boot=200, seed=0)
    assert summaries[0].method == "A"
    assert summaries[0].p_best > 0.8
    assert abs(sum(posterior["A"]) - 1.0) < 1e-9
    assert samples.shape == (200, 3)


def test_paired_permutation_detects_consistent_winner():
    # Need ≥8–10 paired datasets for two-sided sign-flip power at α=0.05.
    a = np.array([0.90, 0.85, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93, 0.86, 0.94])
    b = np.array([0.40, 0.45, 0.50, 0.42, 0.48, 0.44, 0.41, 0.47, 0.43, 0.46])
    p = paired_permutation_pvalue(a, b, n_perm=1000, seed=0)
    assert p < 0.05


def test_paired_permutation_null_not_significant():
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.05, size=12)
    noise = rng.normal(0, 0.01, size=12)
    p = paired_permutation_pvalue(base, base + noise, n_perm=400, seed=1)
    assert p > 0.05


def test_review_landscape_emits_fdr_pairs():
    rng = np.random.default_rng(0)
    performance = {}
    for i in range(12):
        performance[f"d{i}"] = {
            "A": float(0.88 + 0.02 * rng.random()),
            "B": float(0.40 + 0.05 * rng.random()),
            "C": float(0.35 + 0.05 * rng.random()),
        }
    report = review_landscape(performance, n_boot=100, n_perm=500, seed=0, alpha=0.05)
    payload = report.to_dict()
    assert payload["protocol"] == "histoweave.stats_review.v1"
    assert payload["n_datasets"] == 12
    assert payload["pairwise"]["n_tests"] == 3  # C(3,2)
    assert payload["rank_summary"][0]["method"] == "A"
    # At least one pair should survive FDR when A dominates across many datasets.
    assert payload["pairwise"]["n_significant"] >= 1


def test_pairwise_fdr_table_shape():
    pmat = np.array(
        [
            [np.nan, 0.01, 0.2],
            [0.01, np.nan, 0.4],
            [0.2, 0.4, np.nan],
        ]
    )
    table = pairwise_fdr_table(pmat, ["A", "B", "C"], method="bh", alpha=0.05)
    assert table["n_tests"] == 3
    assert any(p["significant"] for p in table["pairs"])


def test_bootstrap_ari_matches_point_metric():
    truth = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    pred = np.array([0, 0, 1, 2, 2, 2, 0, 1])
    point = adjusted_rand_index(truth, pred)
    boot = bootstrap_ari(truth, pred, n_boot=30, seed=3)
    assert boot.point == pytest.approx(point)
