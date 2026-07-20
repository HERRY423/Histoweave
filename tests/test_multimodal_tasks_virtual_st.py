"""Multimodal AnalysisTask expansion, cross-modal evidence rules, virtual ST."""

from __future__ import annotations

import numpy as np
import pytest

from histoweave.benchmark import (
    AnalysisTask,
    CrossModalRelation,
    DOMAIN_PARTITION_TASKS,
    GroundTruthKind,
    MethodRecommender,
    TaskContract,
    cross_modal_relation,
    evidence_compatibility_report,
    get_task,
    ground_truth_admissible,
    normalize_task,
    tasks_admissible,
    virtual_st_task,
)
from histoweave.benchmark.harness import run_benchmark
from histoweave.benchmark.landscape import LandscapeResult
from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER, extract_features, feature_vector
from histoweave.plugins import MethodCategory, create_method, list_methods
from histoweave.plugins.builtin import register_all


@pytest.fixture(scope="module", autouse=True)
def _register_plugins() -> None:
    register_all()


# ---------------------------------------------------------------------------
# Task enum + contracts
# ---------------------------------------------------------------------------


def test_analysis_task_includes_multimodal_and_virtual_st() -> None:
    values = {task.value for task in AnalysisTask}
    assert "spatial_protein_domain" in values
    assert "spatial_chromatin_domain" in values
    assert "virtual_st" in values
    assert AnalysisTask.SPATIAL_PROTEIN_DOMAIN in DOMAIN_PARTITION_TASKS
    assert AnalysisTask.SPATIAL_CHROMATIN_DOMAIN in DOMAIN_PARTITION_TASKS
    assert AnalysisTask.VIRTUAL_ST not in DOMAIN_PARTITION_TASKS


def test_protein_and_chromatin_contracts_require_matching_gt() -> None:
    TaskContract(
        task=AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
        ground_truth_kind=GroundTruthKind.SPATIAL_PROTEIN_DOMAIN,
        label_key="protein_domain_truth",
    ).validate()
    TaskContract(
        task=AnalysisTask.SPATIAL_CHROMATIN_DOMAIN,
        ground_truth_kind=GroundTruthKind.SPATIAL_CHROMATIN_DOMAIN,
        label_key="chromatin_domain_truth",
    ).validate()
    with pytest.raises(ValueError, match="cross-modal|ground_truth_kind"):
        TaskContract(
            task=AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
            ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
            label_key="domain_truth",
        ).validate()
    with pytest.raises(ValueError, match="cluster_proxy|ground_truth_kind"):
        TaskContract(
            task=AnalysisTask.SPATIAL_CHROMATIN_DOMAIN,
            ground_truth_kind=GroundTruthKind.CLUSTER_PROXY,
            label_key="proxy",
        ).validate()


def test_virtual_st_contract_requires_measured_expression_metric() -> None:
    TaskContract(
        task=AnalysisTask.VIRTUAL_ST,
        ground_truth_kind=GroundTruthKind.MEASURED_EXPRESSION,
        label_key="X",
        metric="mean_gene_pearson",
    ).validate()
    TaskContract(
        task=AnalysisTask.VIRTUAL_ST,
        ground_truth_kind=GroundTruthKind.NONE,
        label_key="",
        metric="mean_gene_pearson",
    ).validate()
    with pytest.raises(ValueError, match="mean_gene_pearson|ARI"):
        TaskContract(
            task=AnalysisTask.VIRTUAL_ST,
            ground_truth_kind=GroundTruthKind.MEASURED_EXPRESSION,
            label_key="X",
            # default metric ARI is invalid for virtual_st
        ).validate()
    with pytest.raises(ValueError, match="ground_truth_kind"):
        TaskContract(
            task=AnalysisTask.VIRTUAL_ST,
            ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
            label_key="domain_truth",
            metric="mean_gene_pearson",
        ).validate()


# ---------------------------------------------------------------------------
# Cross-modal evidence compatibility
# ---------------------------------------------------------------------------


def test_cross_modal_relation_matrix() -> None:
    assert cross_modal_relation("spatial_domain", "spatial_domain") is CrossModalRelation.SAME
    assert (
        cross_modal_relation(
            AnalysisTask.SPATIAL_DOMAIN,
            AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
        )
        is CrossModalRelation.SAME_FAMILY
    )
    assert (
        cross_modal_relation(
            AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
            AnalysisTask.SPATIAL_CHROMATIN_DOMAIN,
        )
        is CrossModalRelation.SAME_FAMILY
    )
    assert (
        cross_modal_relation(AnalysisTask.VIRTUAL_ST, AnalysisTask.SPATIAL_DOMAIN)
        is CrossModalRelation.INCOMPATIBLE
    )
    assert (
        cross_modal_relation(AnalysisTask.CELL_TYPE, AnalysisTask.SPATIAL_DOMAIN)
        is CrossModalRelation.INCOMPATIBLE
    )


def test_tasks_admissible_only_exact_match() -> None:
    assert tasks_admissible("spatial_domain", "spatial_domain")
    assert tasks_admissible("virtual_st", "he2st")  # alias
    assert not tasks_admissible("spatial_domain", "spatial_protein_domain")
    assert not tasks_admissible("spatial_protein_domain", "spatial_chromatin_domain")
    assert not tasks_admissible("virtual_st", "spatial_domain")


def test_evidence_compatibility_report_blocks_cross_modal() -> None:
    report = evidence_compatibility_report(
        AnalysisTask.SPATIAL_DOMAIN,
        AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
        reference_ground_truth_kind=GroundTruthKind.SPATIAL_PROTEIN_DOMAIN,
    )
    assert report["admissible"] is False
    assert report["relation"] == CrossModalRelation.SAME_FAMILY.value
    assert any("not transferable" in reason for reason in report["reasons"])

    ok = evidence_compatibility_report(
        AnalysisTask.VIRTUAL_ST,
        "virtual_st",
        reference_ground_truth_kind=GroundTruthKind.MEASURED_EXPRESSION,
    )
    assert ok["admissible"] is True
    assert ground_truth_admissible(AnalysisTask.VIRTUAL_ST, GroundTruthKind.MEASURED_EXPRESSION)


def test_normalize_task_aliases() -> None:
    assert normalize_task("domain_detection") == "spatial_domain"
    assert normalize_task("protein_domain") == "spatial_protein_domain"
    assert normalize_task("he2st") == "virtual_st"


def test_recommender_hard_filters_cross_modal_neighbours() -> None:
    """Protein-domain reference must not enter RNA spatial_domain neighbourhood."""
    from histoweave.datasets import make_synthetic

    rna = make_synthetic(seed=1, n_cells=40, n_genes=20)
    protein = make_synthetic(seed=2, n_cells=40, n_genes=20)
    order = list(RECOMMENDATION_FEATURE_ORDER)
    features = {
        "rna_ref": feature_vector(extract_features(rna, include_domain=False), order=order),
        "protein_ref": feature_vector(
            extract_features(protein, include_domain=False), order=order
        ),
    }
    landscape = LandscapeResult(
        performance={
            "rna_ref": {"kmeans": 0.8, "spectral": 0.6},
            "protein_ref": {"kmeans": 0.1, "spectral": 0.95},
        },
        features=features,
        embedding={},
        best_method={"rna_ref": "kmeans", "protein_ref": "spectral"},
        niches={"kmeans": ["rna_ref"], "spectral": ["protein_ref"]},
        timings={},
        feature_order=order,
        method_count=2,
        dataset_count=2,
        task="spatial_domain",
        metric="ARI",
        dataset_meta={
            "rna_ref": {
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
                "platform": "visium",
            },
            "protein_ref": {
                "task": "spatial_protein_domain",
                "ground_truth_kind": "spatial_protein_domain",
                "platform": "codex",
            },
        },
    )
    recommendation = MethodRecommender(landscape, k_neighbours=2).recommend(
        rna,
        dataset_name="query",
        task=AnalysisTask.SPATIAL_DOMAIN,
        platform="visium",
    )
    neighbour_names = {item["name"] for item in recommendation.neighbours}
    assert "protein_ref" not in neighbour_names
    assert "rna_ref" in neighbour_names


# ---------------------------------------------------------------------------
# Virtual ST methods + harness
# ---------------------------------------------------------------------------


def test_virtual_st_methods_are_registered() -> None:
    methods = {m["name"] for m in list_methods(category=MethodCategory.VIRTUAL_ST)}
    assert "virtual_st_morphology" in methods
    assert "virtual_st_scellst" in methods
    assert "virtual_st_storm" in methods


@pytest.mark.parametrize(
    "method_name",
    ["virtual_st_morphology", "virtual_st_scellst", "virtual_st_storm"],
)
def test_virtual_st_paired_prediction_writes_layer(method_name: str) -> None:
    task = virtual_st_task()
    data = task.dataset
    assert "image" in data.images
    result = create_method(MethodCategory.VIRTUAL_ST, method_name, mode="paired", seed=0).run(data)
    assert "virtual_st" in result.layers
    pred = np.asarray(result.layers["virtual_st"])
    assert pred.shape == (data.n_obs, data.n_vars)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
    meta = result.uns["virtual_st"][method_name]
    assert meta["supervision"] == "paired_measured_expression"
    assert "mean_gene_pearson" in meta
    assert meta["mean_gene_pearson"] > 0.0  # morphology is informative on synthetic data
    assert "X_virtual_st" in result.obsm


def test_virtual_st_inference_mode_without_fitting_targets() -> None:
    task = virtual_st_task()
    data = task.dataset
    result = create_method(
        MethodCategory.VIRTUAL_ST,
        "virtual_st_morphology",
        mode="inference",
        n_genes=8,
        seed=1,
    ).run(data)
    pred = np.asarray(result.layers["virtual_st"])
    assert pred.shape[0] == data.n_obs
    assert pred.shape[1] == data.n_vars
    meta = result.uns["virtual_st"]["virtual_st_morphology"]
    assert meta["supervision"] == "morphology_only"


def test_virtual_st_benchmark_task_scores_methods() -> None:
    task = get_task("virtual_st")
    assert task.name == "virtual_st"
    assert task.category is MethodCategory.VIRTUAL_ST
    result = run_benchmark(
        task,
        methods=["virtual_st_morphology", "virtual_st_scellst", "virtual_st_storm"],
        method_params={
            "virtual_st_morphology": {"mode": "paired", "seed": 0},
            "virtual_st_scellst": {"mode": "paired", "seed": 0},
            "virtual_st_storm": {"mode": "paired", "seed": 0},
        },
    )
    assert len(result.leaderboard) == 3
    assert all("error" not in row for row in result.leaderboard)
    scores = [row["score"] for row in result.leaderboard]
    assert all(np.isfinite(s) for s in scores)
    # Synthetic data is morphology-driven — all methods should recover positive correlation.
    assert max(scores) > 0.2
