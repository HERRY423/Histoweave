"""Preregistered metrics for phenomenon-centred method evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)

from ..data import SpatialTable
from ..datasets import SpatialPhenomenon
from ..plugins import MethodCategory


@dataclass(frozen=True)
class MetricValue:
    """One normalized or raw metric emitted by an evaluation adapter."""

    name: str
    value: float
    direction: str
    primary: bool
    normalized_value: float


def evaluate_method_output(
    category: MethodCategory | str,
    result: SpatialTable,
    reference: SpatialTable,
    phenomenon: SpatialPhenomenon | str,
) -> list[MetricValue]:
    """Evaluate a method output without using truth for method configuration."""

    category = MethodCategory(category)
    phenomenon = SpatialPhenomenon(phenomenon)
    evaluators = {
        MethodCategory.QC: _evaluate_qc,
        MethodCategory.NORMALIZATION: _evaluate_normalization,
        MethodCategory.SEGMENTATION: _evaluate_segmentation,
        MethodCategory.ANNOTATION: _evaluate_annotation,
        MethodCategory.DOMAIN_DETECTION: _evaluate_domains,
        MethodCategory.DECONVOLUTION: _evaluate_deconvolution,
        MethodCategory.SPATIALLY_VARIABLE_GENES: _evaluate_svg,
        MethodCategory.NEIGHBORHOOD: _evaluate_neighborhood,
        MethodCategory.CELL_CELL_COMMUNICATION: _evaluate_ccc,
        MethodCategory.INTEGRATION: _evaluate_integration,
        MethodCategory.INGESTION: _evaluate_ingestion,
    }
    try:
        return evaluators[category](result, reference, phenomenon)
    except KeyError as exc:
        raise ValueError(f"method output is missing required result key: {exc}") from exc


def _metric(
    name: str,
    value: float,
    *,
    primary: bool = False,
    direction: str = "maximize",
    normalized: float | None = None,
) -> MetricValue:
    value = float(value)
    normalized_value = value if normalized is None else float(normalized)
    if not np.isfinite(value) or not np.isfinite(normalized_value):
        raise ValueError(f"metric {name} is not finite")
    return MetricValue(
        name=name,
        value=value,
        direction=direction,
        primary=primary,
        normalized_value=float(np.clip(normalized_value, 0.0, 1.0)),
    )


def _fdr_control_score(false_discovery_proportion: float, alpha: float = 0.05) -> float:
    """Score empirical false-discovery control at a preregistered alpha level."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    fdp = float(np.clip(false_discovery_proportion, 0.0, 1.0))
    if fdp <= alpha:
        return 1.0
    return float(1.0 - (fdp - alpha) / (1.0 - alpha))


def _evaluate_ingestion(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    expected = np.asarray(reference.X, dtype=float)
    observed = np.asarray(result.X, dtype=float)
    if expected.shape != observed.shape:
        roundtrip = 0.0
    else:
        scale = max(float(np.mean(np.abs(expected))), 1e-12)
        relative_error = float(np.mean(np.abs(expected - observed)) / scale)
        roundtrip = 1.0 - min(relative_error, 1.0)

    expected_coords = np.asarray(reference.obsm["spatial"], dtype=float)
    observed_coords = np.asarray(result.obsm["spatial"], dtype=float)
    if expected_coords.shape != observed_coords.shape or len(expected_coords) < 3:
        coordinate_fidelity = float(expected_coords.shape == observed_coords.shape)
    else:
        expected_distances = _condensed_distances(expected_coords)
        observed_distances = _condensed_distances(observed_coords)
        coordinate_fidelity = (_safe_spearman(expected_distances, observed_distances) + 1.0) / 2.0

    metadata_checks = (
        bool(result.uns.get("assay")),
        result.obs_names.is_unique,
        result.var_names.is_unique,
        "spatial" in result.obsm,
    )
    metadata_fidelity = float(np.mean(metadata_checks))
    return [
        _metric("roundtrip_fidelity", roundtrip, primary=True),
        _metric("coordinate_fidelity", coordinate_fidelity),
        _metric("metadata_fidelity", metadata_fidelity),
    ]


def _evaluate_qc(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    truth = reference.obs["qc_truth"].to_numpy(dtype=bool)
    retained = reference.obs_names.isin(result.obs_names)
    predicted = ~retained
    if not predicted.any():
        boolean_columns = [
            column
            for column in result.obs.columns
            if column not in reference.obs.columns and result.obs[column].dtype == bool
        ]
        if boolean_columns:
            aligned = result.obs[boolean_columns[0]].reindex(reference.obs_names, fill_value=False)
            predicted = aligned.to_numpy(dtype=bool)
    auprc = average_precision_score(truth, predicted.astype(float))
    normal_retention = float(np.mean(retained[~truth])) if np.any(~truth) else 1.0
    signal_retention = _signal_retention(result, reference)
    return [
        _metric("qc_auprc", auprc, primary=True),
        _metric("normal_retention", normal_retention),
        _metric("phenomenon_signal_retention", signal_retention),
    ]


def _evaluate_normalization(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    signal = _signal_retention(result, reference)
    raw_gene_means = np.asarray(reference.X, dtype=float).mean(axis=0)
    out_gene_means = np.asarray(result.X, dtype=float).mean(axis=0)
    marker_rank = _safe_spearman(raw_gene_means, out_gene_means)
    marker_rank = (marker_rank + 1.0) / 2.0
    raw_totals = np.asarray(reference.X, dtype=float).sum(axis=1)
    out_totals = np.asarray(result.X, dtype=float).sum(axis=1)
    raw_cv = _coefficient_of_variation(raw_totals)
    out_cv = _coefficient_of_variation(out_totals)
    nuisance_removal = 1.0 - min(out_cv / max(raw_cv, 1e-12), 1.0)
    return [
        _metric("phenomenon_signal_retention", signal, primary=True),
        _metric("marker_rank_preservation", marker_rank),
        _metric("library_nuisance_removal", nuisance_removal),
    ]


def _evaluate_segmentation(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    truth = np.asarray(reference.images["segmentation_truth"])
    segmentation = result.uns.get("segmentation", {})
    mask_key = segmentation.get("mask_key")
    if not mask_key or mask_key not in result.images:
        candidates = [key for key in result.images if key not in reference.images]
        if not candidates:
            raise ValueError("segmentation result has no output mask")
        mask_key = candidates[0]
    predicted = np.asarray(result.images[mask_key])
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth segmentation masks have different shapes")
    matched_ious = _matched_instance_ious(truth, predicted)
    truth_count = len(np.unique(truth)) - int(0 in truth)
    pred_count = len(np.unique(predicted)) - int(0 in predicted)
    true_positive = sum(iou >= 0.5 for iou in matched_ious)
    denominator = true_positive + (pred_count - true_positive) + (truth_count - true_positive)
    ap50 = true_positive / denominator if denominator else 1.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    count_error = abs(pred_count - truth_count) / max(truth_count, 1)
    return [
        _metric("instance_ap50", ap50, primary=True),
        _metric("mean_matched_iou", mean_iou),
        _metric(
            "count_error",
            count_error,
            direction="minimize",
            normalized=1.0 - min(count_error, 1.0),
        ),
    ]


def _evaluate_annotation(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    predicted = result.obs["cell_type"].astype(str).to_numpy()
    truth = reference.obs.loc[result.obs_names, "cell_type_truth"].astype(str).to_numpy()
    macro = f1_score(truth, predicted, average="macro", zero_division=0)
    balanced = balanced_accuracy_score(truth, predicted)
    counts = reference.obs["cell_type_truth"].value_counts()
    rare = str(counts.index[-1])
    rare_mask = truth == rare
    rare_recall = (
        float(np.mean(predicted[rare_mask] == truth[rare_mask])) if rare_mask.any() else 0.0
    )
    return [
        _metric("macro_f1", macro, primary=True),
        _metric("balanced_accuracy", balanced),
        _metric("rare_type_recall", rare_recall),
    ]


def _evaluate_domains(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    predicted = result.obs["domain"].astype(str).to_numpy()
    ref_obs = reference.obs.loc[result.obs_names]
    discrete_truth = _discrete_phenomenon_truth(ref_obs, phenomenon)
    ari = adjusted_rand_score(discrete_truth, predicted)
    nmi = normalized_mutual_info_score(discrete_truth, predicted)
    recovery = max(0.0, (ari + nmi) / 2.0)
    boundary_f1 = _boundary_f1(reference, result.obs_names, predicted, discrete_truth)
    return [
        _metric("phenomenon_recovery", recovery, primary=True),
        _metric("adjusted_rand_index", ari, normalized=(ari + 1.0) / 2.0),
        _metric("boundary_f1", boundary_f1),
    ]


def _evaluate_deconvolution(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    predicted = np.asarray(result.obsm["proportions"], dtype=float)
    truth = np.asarray(reference.obsm["proportions_truth"], dtype=float)
    truth = truth[reference.obs_names.get_indexer(result.obs_names)]
    if predicted.shape != truth.shape:
        raise ValueError(
            f"deconvolution shape mismatch: predicted={predicted.shape}, truth={truth.shape}"
        )
    rmse = float(np.sqrt(np.mean((truth - predicted) ** 2)))
    midpoint = (truth + predicted) / 2.0
    eps = 1e-12
    js = 0.5 * np.sum(truth * np.log((truth + eps) / (midpoint + eps)), axis=1)
    js += 0.5 * np.sum(predicted * np.log((predicted + eps) / (midpoint + eps)), axis=1)
    js_similarity = 1.0 - float(np.mean(js) / np.log(2.0))
    correlations = [
        _safe_spearman(truth[:, column], predicted[:, column]) for column in range(truth.shape[1])
    ]
    correlation = (float(np.mean(correlations)) + 1.0) / 2.0
    return [
        _metric(
            "proportion_rmse",
            rmse,
            primary=True,
            direction="minimize",
            normalized=1.0 - min(rmse, 1.0),
        ),
        _metric("jensen_shannon_similarity", js_similarity),
        _metric("cell_type_correlation", correlation),
    ]


def _evaluate_svg(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    truth = reference.var["spatial_truth"].to_numpy(dtype=bool)
    scores = _extract_gene_scores(result, reference.var_names)
    pr_auc = average_precision_score(truth, scores)
    k = int(truth.sum())
    top = np.argsort(scores)[::-1][:k]
    precision = float(np.mean(truth[top])) if k else 0.0
    false_discovery_proportion = 1.0 - precision
    return [
        _metric("gene_pr_auc", pr_auc, primary=True),
        _metric("precision_at_k", precision),
        _metric(
            "null_fdr_calibration",
            _fdr_control_score(false_discovery_proportion),
        ),
    ]


def _evaluate_neighborhood(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    truth = _edge_set(np.asarray(reference.uns["truth_graph_edges"]))
    predicted = _edge_set(np.asarray(result.uns["spatial_graph"]["edges"]))
    intersection = len(truth & predicted)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [
        _metric("edge_f1", f1, primary=True),
        _metric("edge_precision", precision),
        _metric("edge_recall", recall),
    ]


def _evaluate_ccc(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    del phenomenon
    ccc = result.uns.get("ccc", {})
    result_key = ccc.get("result_key")
    if not result_key or result_key not in result.uns:
        raise ValueError("CCC result table is unavailable")
    table = result.uns[result_key]
    if not hasattr(table, "columns"):
        raise ValueError("CCC result is not tabular")
    required = {"source", "target", "ligand_complex", "receptor_complex"}
    if not required <= set(table.columns):
        raise ValueError("CCC result lacks source/target/ligand/receptor columns")
    truth = {
        (row["source"], row["target"], row["ligand"], row["receptor"])
        for row in reference.uns["lr_truth"]
        if row["spatially_active"]
    }
    predicted = [
        (str(row.source), str(row.target), str(row.ligand_complex), str(row.receptor_complex))
        for row in table.itertuples()
    ]
    k = max(1, len(truth))
    precision = len(set(predicted[:k]) & truth) / k
    false_discovery_proportion = 1.0 - precision
    # LIANA tables expose multiple method-specific scores; rank aggregation is already
    # ordered, so precision@truth-size is the defensible common denominator here.
    return [
        _metric("lr_pr_auc", precision, primary=True),
        _metric("lr_precision_at_k", precision),
        _metric(
            "null_fdr_calibration",
            _fdr_control_score(false_discovery_proportion),
        ),
    ]


def _evaluate_integration(
    result: SpatialTable, reference: SpatialTable, phenomenon: SpatialPhenomenon
) -> list[MetricValue]:
    embedding = _integration_representation(result, reference)
    labels = _discrete_phenomenon_truth(reference.obs.loc[result.obs_names], phenomenon)
    batches = reference.obs.loc[result.obs_names, "batch"].astype(str).to_numpy()
    biological = _knn_label_agreement(embedding, labels, k=8)
    mixing = _knn_batch_mixing(embedding, batches, k=8)
    recoverability = normalized_mutual_info_score(
        labels,
        _knn_majority_prediction(embedding, labels, k=8),
    )
    oversmoothing = max(0.0, 1.0 - _variance_ratio(embedding))
    return [
        _metric("biological_neighborhood_conservation", biological, primary=True),
        _metric("batch_mixing", mixing),
        _metric("phenomenon_recoverability", recoverability),
        _metric(
            "oversmoothing_penalty",
            oversmoothing,
            direction="minimize",
            normalized=1.0 - oversmoothing,
        ),
    ]


def _signal_retention(result: SpatialTable, reference: SpatialTable) -> float:
    common = reference.obs_names.intersection(result.obs_names)
    if len(common) < 3:
        return 0.0
    ref_index = reference.obs_names.get_indexer(common)
    out_index = result.obs_names.get_indexer(common)
    truth = reference.var["spatial_truth"].to_numpy(dtype=bool)
    ref_scores = _gene_spatial_scores(reference, ref_index)
    out_scores = _gene_spatial_scores(result, out_index)
    if not truth.any():
        return 0.0
    ref_ap = average_precision_score(truth, ref_scores)
    out_ap = average_precision_score(truth, out_scores)
    return float(np.clip(out_ap / max(ref_ap, 1e-12), 0.0, 1.0))


def _gene_spatial_scores(table: SpatialTable, rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(table.X, dtype=float)[rows]
    coords = np.asarray(table.obsm["spatial"], dtype=float)[rows]
    tree = cKDTree(coords)
    _, neighbors = tree.query(coords, k=min(7, len(rows)))
    local = matrix[neighbors[:, 1:]].mean(axis=1)
    scores = np.array(
        [abs(_safe_spearman(matrix[:, idx], local[:, idx])) for idx in range(matrix.shape[1])]
    )
    return np.nan_to_num(scores)


def _extract_gene_scores(result: SpatialTable, expected_index: Iterable[str]) -> np.ndarray:
    expected = list(expected_index)
    numeric_candidates = [
        column
        for column in result.var.columns
        if column not in {"mito", "spatial_truth", "batch_shift_truth"}
        and is_numeric_dtype(result.var[column].dtype)
    ]
    if numeric_candidates:
        return result.var[numeric_candidates[-1]].reindex(expected).to_numpy(dtype=float)
    ranked: list[str] = []
    for value in result.uns.values():
        if isinstance(value, dict) and "top_genes" in value:
            for entry in value["top_genes"]:
                ranked.append(str(entry.get("gene")) if isinstance(entry, dict) else str(entry))
    if not ranked:
        raise ValueError("SVG result has neither per-gene scores nor a ranked gene list")
    rank = {gene: len(ranked) - idx for idx, gene in enumerate(ranked)}
    return np.array([rank.get(gene, 0.0) for gene in expected], dtype=float)


def _discrete_phenomenon_truth(obs, phenomenon: SpatialPhenomenon) -> np.ndarray:
    if phenomenon in {
        SpatialPhenomenon.COMPARTMENT,
        SpatialPhenomenon.BOUNDARY,
        SpatialPhenomenon.BRANCHING,
    }:
        return obs["domain_truth"].astype(str).to_numpy()
    if phenomenon is SpatialPhenomenon.HOTSPOT:
        return obs["hotspot_truth"].astype(str).to_numpy()
    if phenomenon is SpatialPhenomenon.MIXTURE:
        return obs["cell_type_truth"].astype(str).to_numpy()
    continuous = obs["continuous_truth"].to_numpy(dtype=float)
    quantiles = np.unique(np.quantile(continuous, [0.0, 1 / 3, 2 / 3, 1.0]))
    return np.digitize(continuous, quantiles[1:-1]).astype(str)


def _boundary_f1(
    reference: SpatialTable,
    names,
    predicted: np.ndarray,
    truth: np.ndarray,
) -> float:
    coords = np.asarray(reference.obsm["spatial"])[reference.obs_names.get_indexer(names)]
    tree = cKDTree(coords)
    _, neighbors = tree.query(coords, k=min(7, len(coords)))
    truth_boundary = np.any(truth[neighbors[:, 1:]] != truth[:, None], axis=1)
    pred_boundary = np.any(predicted[neighbors[:, 1:]] != predicted[:, None], axis=1)
    return f1_score(truth_boundary, pred_boundary, zero_division=0)


def _matched_instance_ious(truth: np.ndarray, predicted: np.ndarray) -> list[float]:
    truth_labels = np.array([value for value in np.unique(truth) if value != 0])
    pred_labels = np.array([value for value in np.unique(predicted) if value != 0])
    if not len(truth_labels) or not len(pred_labels):
        return []
    iou: np.ndarray = np.zeros((len(truth_labels), len(pred_labels)), dtype=float)
    for i, truth_label in enumerate(truth_labels):
        truth_mask = truth == truth_label
        overlapping = np.unique(predicted[truth_mask])
        for pred_label in overlapping:
            if pred_label == 0:
                continue
            j = int(np.where(pred_labels == pred_label)[0][0])
            pred_mask = predicted == pred_label
            intersection = np.count_nonzero(truth_mask & pred_mask)
            union = np.count_nonzero(truth_mask | pred_mask)
            iou[i, j] = intersection / union
    rows, columns = linear_sum_assignment(1.0 - iou)
    return [float(iou[row, column]) for row, column in zip(rows, columns, strict=True)]


def _condensed_distances(coordinates: np.ndarray) -> np.ndarray:
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    upper = np.triu_indices(len(coordinates), k=1)
    return distances[upper]


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    if edges.size == 0:
        return set()
    return {
        (min(int(source), int(target)), max(int(source), int(target))) for source, target in edges
    }


def _integration_representation(result: SpatialTable, reference: SpatialTable) -> np.ndarray:
    new_keys = [key for key in result.obsm if key not in reference.obsm]
    if new_keys:
        return np.asarray(result.obsm[new_keys[-1]], dtype=float)
    return np.asarray(result.X, dtype=float)


def _knn_indices(embedding: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(embedding)
    _, neighbors = tree.query(embedding, k=min(k + 1, len(embedding)))
    return neighbors[:, 1:]


def _knn_label_agreement(embedding: np.ndarray, labels: np.ndarray, k: int) -> float:
    neighbors = _knn_indices(embedding, k)
    return float(np.mean(labels[neighbors] == labels[:, None]))


def _knn_batch_mixing(embedding: np.ndarray, batches: np.ndarray, k: int) -> float:
    if len(np.unique(batches)) < 2:
        return 1.0
    neighbors = _knn_indices(embedding, k)
    observed = float(np.mean(batches[neighbors] != batches[:, None]))
    frequencies = np.array([np.mean(batches == batch) for batch in np.unique(batches)])
    expected = 1.0 - float(np.sum(frequencies**2))
    return min(observed / max(expected, 1e-12), 1.0)


def _knn_majority_prediction(embedding: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    neighbors = _knn_indices(embedding, k)
    predicted: list[str] = []
    for row in neighbors:
        values, counts = np.unique(labels[row], return_counts=True)
        predicted.append(str(values[np.argmax(counts)]))
    return np.asarray(predicted)


def _variance_ratio(embedding: np.ndarray) -> float:
    total = float(np.var(embedding, axis=0).sum())
    return min(total / max(embedding.shape[1], 1), 1.0)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def _coefficient_of_variation(values: np.ndarray) -> float:
    return float(np.std(values) / max(abs(float(np.mean(values))), 1e-12))
