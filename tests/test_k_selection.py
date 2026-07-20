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
    assert result.geometry == "expression"
    assert result.spatial_used is False


def test_estimate_bic_gmm_returns_valid_k():
    data = make_synthetic(n_cells=180, n_genes=60, n_domains=3, seed=1)
    result = estimate_n_domains(data, method="bic_gmm", k_max=6, random_state=0)
    assert 2 <= result.k <= 6


def test_estimate_gap_returns_valid_k():
    data = make_synthetic(n_cells=120, n_genes=40, n_domains=3, seed=2)
    result = estimate_n_domains(data, method="gap", k_max=5, random_state=0)
    assert 2 <= result.k <= 5


def test_estimate_calinski_and_davies_bouldin():
    data = make_synthetic(n_cells=160, n_genes=50, n_domains=3, seed=4)
    ch = estimate_n_domains(data, method="calinski_harabasz", k_max=6, random_state=0)
    db = estimate_n_domains(data, method="davies_bouldin", k_max=6, random_state=0)
    assert 2 <= ch.k <= 6
    assert 2 <= db.k <= 6


def test_spatial_silhouette_uses_coordinates():
    data = make_synthetic(n_cells=200, n_genes=70, n_domains=4, seed=0, noise=0.12)
    result = estimate_n_domains(
        data, method="spatial_silhouette", k_max=8, random_state=0
    )
    assert result.spatial_used is True
    assert result.geometry == "spatial_smooth"
    assert 2 <= result.k <= 8
    assert result.method == "spatial_silhouette"


def test_spatial_coherence_uses_coordinates():
    data = make_synthetic(n_cells=200, n_genes=70, n_domains=4, seed=0, noise=0.12)
    result = estimate_n_domains(
        data, method="spatial_coherence", k_max=8, random_state=0
    )
    assert result.spatial_used is True
    assert 2 <= result.k <= 8
    # Scores are fractions of agreeing kNN edges.
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in result.scores.values())


def test_ensemble_default_is_spatial_aware_and_votes():
    data = make_synthetic(n_cells=220, n_genes=80, n_domains=4, seed=0, noise=0.1)
    result = estimate_n_domains(data, random_state=0)  # default ensemble
    assert result.method == "ensemble"
    assert result.spatial_used is True
    assert result.geometry == "ensemble"
    assert len(result.component_votes) >= 3
    assert "spatial_silhouette" in result.component_votes
    assert "spatial_coherence" in result.component_votes
    assert 2 <= result.k <= 12
    payload = result.to_dict()
    assert "component_votes" in payload
    assert payload["spatial_used"] is True


def test_ensemble_falls_back_without_spatial_coords():
    data = make_synthetic(n_cells=160, n_genes=50, n_domains=3, seed=5)
    # Strip spatial coordinates to simulate expression-only table.
    data.obsm = {}
    if hasattr(data, "_spatial"):
        data._spatial = None
    # SpatialTable may expose .spatial from obsm — clear if possible.
    try:
        if "spatial" in getattr(data, "obsm", {}):
            del data.obsm["spatial"]
    except Exception:
        pass
    # Force no coords by monkeypatching via empty spatial if property is read-only.
    from histoweave.benchmark import k_selection as ks

    original = ks._coords_from_table
    ks._coords_from_table = lambda _d: None  # type: ignore[assignment]
    try:
        result = estimate_n_domains(data, method="ensemble", k_max=6, random_state=0)
    finally:
        ks._coords_from_table = original  # type: ignore[assignment]
    assert result.method == "ensemble"
    assert result.spatial_used is False
    assert "spatial_silhouette" not in result.component_votes
    assert any("no spatial" in f for f in result.flags)


def test_joint_geometry():
    data = make_synthetic(n_cells=150, n_genes=40, n_domains=3, seed=6)
    result = estimate_n_domains(
        data,
        method="silhouette",
        geometry="joint",
        k_max=6,
        random_state=0,
    )
    assert result.geometry == "joint"
    assert result.spatial_used is True


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
        assert lr.performance[name]["kmeans"] == lr.performance[name]["kmeans"]


def test_run_landscape_oracle_gate():
    from histoweave.benchmark import run_landscape
    from histoweave.benchmark.k_selection import make_domain_k_factory

    with pytest.raises(ValueError, match="allow_oracle_k"):
        make_domain_k_factory(policy="oracle")

    ds = {"a": make_synthetic(seed=0, n_cells=100, n_domains=3)}
    lr = run_landscape(ds, methods=["kmeans"], k_policy="oracle", allow_oracle_k=True)
    assert lr.dataset_meta["a"]["k_policy"] == "oracle"
