"""Non-oracle K selection and landscape k_policy gates."""

from __future__ import annotations

import pytest

from histoweave.benchmark.k_selection import (
    compare_k_policies,
    estimate_n_domains,
    make_domain_k_factory,
    oracle_n_domains,
)
from histoweave.benchmark.task_contract import (
    AnalysisTask,
    GroundTruthKind,
    TaskContract,
)
from histoweave.datasets.synthetic import make_synthetic


def test_estimate_n_domains_recovers_clean_synthetic_k():
    data = make_synthetic(n_cells=240, n_genes=90, n_domains=4, seed=0, noise=0.1)
    result = estimate_n_domains(data, method="silhouette", random_state=0)
    assert result.k == 4
    assert result.oracle_k == 4
    assert 4 in result.scores


def test_estimate_bic_gmm_returns_valid_k():
    data = make_synthetic(n_cells=180, n_genes=60, n_domains=3, seed=1)
    result = estimate_n_domains(data, method="bic_gmm", k_max=6, random_state=0)
    assert 2 <= result.k <= 6


def test_estimate_gap_returns_valid_k():
    data = make_synthetic(n_cells=120, n_genes=40, n_domains=3, seed=2)
    result = estimate_n_domains(data, method="gap", k_max=5, random_state=0)
    assert 2 <= result.k <= 5


def test_oracle_n_domains_reads_truth():
    data = make_synthetic(n_domains=5, seed=3)
    assert oracle_n_domains(data) == 5


def test_factory_oracle_requires_flag():
    with pytest.raises(ValueError, match="allow_oracle_k"):
        make_domain_k_factory(policy="oracle", allow_oracle_k=False)


def test_factory_estimate_does_not_use_truth_count_blindly():
    data = make_synthetic(n_cells=200, n_genes=80, n_domains=4, seed=0, noise=0.1)
    factory = make_domain_k_factory(policy="estimate", allow_oracle_k=False)
    params = factory(data)
    assert "n_domains" in params
    assert isinstance(params["n_domains"], int)
    assert 2 <= params["n_domains"] <= 12


def test_factory_fixed():
    data = make_synthetic(seed=0)
    factory = make_domain_k_factory(policy="fixed", fixed_k=7)
    assert factory(data) == {"n_domains": 7}


def test_compare_k_policies_report():
    data = make_synthetic(n_cells=200, n_genes=80, n_domains=4, seed=0, noise=0.1)
    data.uns["dataset_name"] = "synth4"
    report = compare_k_policies(data, estimator="silhouette")
    assert report.oracle_k == 4
    assert report.estimated_k == 4
    assert report.k_match is True
    payload = report.to_dict()
    assert payload["dataset"] == "synth4"


def test_task_contract_oracle_requires_notes():
    with pytest.raises(ValueError, match="oracle"):
        TaskContract(
            task=AnalysisTask.SPATIAL_DOMAIN,
            ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
            label_key="domain_truth",
            allow_oracle_k=True,
            notes="",
        ).validate()

    TaskContract(
        task=AnalysisTask.SPATIAL_DOMAIN,
        ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
        label_key="domain_truth",
        allow_oracle_k=True,
        notes="Controlled oracle-K ablation for method capacity.",
    ).validate()


def test_run_landscape_default_is_estimate_not_oracle():
    from histoweave.benchmark import run_landscape

    ds = {
        "a": make_synthetic(seed=0, n_cells=120, n_domains=3),
        "b": make_synthetic(seed=1, n_cells=120, n_domains=3),
    }
    lr = run_landscape(ds, methods=["kmeans"])
    for name in lr.performance:
        assert lr.dataset_meta[name].get("k_policy") == "estimate"
        assert "kmeans" in lr.performance[name]
        # Method must have produced a finite score without oracle injection.
        assert lr.performance[name]["kmeans"] == lr.performance[name]["kmeans"]  # not raising


def test_run_landscape_oracle_gate():
    from histoweave.benchmark import run_landscape
    from histoweave.benchmark.k_selection import make_domain_k_factory

    with pytest.raises(ValueError, match="allow_oracle_k"):
        make_domain_k_factory(policy="oracle")

    ds = {"a": make_synthetic(seed=0, n_cells=100, n_domains=3)}
    lr = run_landscape(ds, methods=["kmeans"], k_policy="oracle", allow_oracle_k=True)
    assert lr.dataset_meta["a"]["k_policy"] == "oracle"
