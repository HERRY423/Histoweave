from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from histoweave.benchmark.phenomenology_metrics import evaluate_method_output
from histoweave.datasets import (
    ObservationCondition,
    SpatialPhenomenon,
    default_scenario_manifest,
    make_phenomenology_scenario,
)
from histoweave.plugins import MethodCategory


def _reference(phenomenon: SpatialPhenomenon = SpatialPhenomenon.MIXTURE):
    manifest = default_scenario_manifest(
        phenomenon,
        ObservationCondition.CLEAN,
        seed=7,
        n_obs=80,
        n_genes=64,
        image_size=48,
    )
    return make_phenomenology_scenario(manifest)


def _by_name(values):
    return {value.name: value for value in values}


def test_perfect_annotation_scores_one() -> None:
    reference = _reference()
    result = reference.copy()
    result.obs["cell_type"] = reference.obs["cell_type_truth"].copy()
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.ANNOTATION,
            result,
            reference,
            SpatialPhenomenon.MIXTURE,
        )
    )
    assert metrics["macro_f1"].value == pytest.approx(1.0)
    assert metrics["balanced_accuracy"].value == pytest.approx(1.0)


def test_perfect_domain_labels_score_one() -> None:
    reference = _reference(SpatialPhenomenon.COMPARTMENT)
    result = reference.copy()
    result.obs["domain"] = reference.obs["domain_truth"].copy()
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.DOMAIN_DETECTION,
            result,
            reference,
            SpatialPhenomenon.COMPARTMENT,
        )
    )
    assert metrics["phenomenon_recovery"].value == pytest.approx(1.0)
    assert metrics["adjusted_rand_index"].value == pytest.approx(1.0)


def test_perfect_deconvolution_scores_one_and_zero_rmse() -> None:
    reference = _reference()
    result = reference.copy()
    result.obsm["proportions"] = reference.obsm["proportions_truth"].copy()
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.DECONVOLUTION,
            result,
            reference,
            SpatialPhenomenon.MIXTURE,
        )
    )
    assert metrics["proportion_rmse"].value == pytest.approx(0.0)
    assert metrics["proportion_rmse"].normalized_value == pytest.approx(1.0)
    assert metrics["jensen_shannon_similarity"].value == pytest.approx(1.0)


def test_perfect_svg_ranking_scores_one() -> None:
    reference = _reference(SpatialPhenomenon.HOTSPOT)
    result = reference.copy()
    result.var["test_score"] = reference.var["spatial_truth"].astype(float)
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.SPATIALLY_VARIABLE_GENES,
            result,
            reference,
            SpatialPhenomenon.HOTSPOT,
        )
    )
    assert metrics["gene_pr_auc"].value == pytest.approx(1.0)
    assert metrics["precision_at_k"].value == pytest.approx(1.0)
    assert metrics["null_fdr_calibration"].value == pytest.approx(1.0)


def test_svg_false_discoveries_reduce_fdr_calibration() -> None:
    reference = _reference(SpatialPhenomenon.HOTSPOT)
    result = reference.copy()
    result.var["test_score"] = (~reference.var["spatial_truth"]).astype(float)
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.SPATIALLY_VARIABLE_GENES,
            result,
            reference,
            SpatialPhenomenon.HOTSPOT,
        )
    )
    assert metrics["precision_at_k"].value == pytest.approx(0.0)
    assert metrics["null_fdr_calibration"].value == pytest.approx(0.0)


def test_perfect_neighborhood_edges_score_one() -> None:
    reference = _reference(SpatialPhenomenon.GRADIENT)
    result = reference.copy()
    result.uns["spatial_graph"] = {"edges": np.asarray(reference.uns["truth_graph_edges"]).tolist()}
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.NEIGHBORHOOD,
            result,
            reference,
            SpatialPhenomenon.GRADIENT,
        )
    )
    assert metrics["edge_f1"].value == pytest.approx(1.0)


def test_perfect_segmentation_scores_one() -> None:
    reference = _reference(SpatialPhenomenon.GRADIENT)
    result = reference.copy()
    result.images["predicted"] = reference.images["segmentation_truth"].copy()
    result.uns["segmentation"] = {"mask_key": "predicted"}
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.SEGMENTATION,
            result,
            reference,
            SpatialPhenomenon.GRADIENT,
        )
    )
    assert metrics["instance_ap50"].value == pytest.approx(1.0)
    assert metrics["mean_matched_iou"].value == pytest.approx(1.0)
    assert metrics["count_error"].value == pytest.approx(0.0)


def test_qc_removed_truth_anomalies_scores_one() -> None:
    reference = _reference(SpatialPhenomenon.GRADIENT)
    keep = ~reference.obs["qc_truth"].to_numpy(dtype=bool)
    result = reference.subset_obs(keep)
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.QC,
            result,
            reference,
            SpatialPhenomenon.GRADIENT,
        )
    )
    assert metrics["qc_auprc"].value == pytest.approx(1.0)
    assert metrics["normal_retention"].value == pytest.approx(1.0)


def test_missing_required_output_becomes_explicit_invalid_output() -> None:
    reference = _reference()
    with pytest.raises(ValueError, match="missing required result key"):
        evaluate_method_output(
            MethodCategory.DECONVOLUTION,
            reference.copy(),
            reference,
            SpatialPhenomenon.MIXTURE,
        )


def test_ccc_truth_precision_uses_real_result_columns() -> None:
    reference = _reference(SpatialPhenomenon.MIXTURE)
    truth = [row for row in reference.uns["lr_truth"] if row["spatially_active"]]
    assert truth
    result = reference.copy()
    result.uns["liana_res"] = pd.DataFrame(
        {
            "source": [row["source"] for row in truth],
            "target": [row["target"] for row in truth],
            "ligand_complex": [row["ligand"] for row in truth],
            "receptor_complex": [row["receptor"] for row in truth],
        }
    )
    result.uns["ccc"] = {"result_key": "liana_res"}
    metrics = _by_name(
        evaluate_method_output(
            MethodCategory.CELL_CELL_COMMUNICATION,
            result,
            reference,
            SpatialPhenomenon.MIXTURE,
        )
    )
    assert metrics["lr_precision_at_k"].value == pytest.approx(1.0)
    assert metrics["null_fdr_calibration"].value == pytest.approx(1.0)
