"""Dependency-light research methods for spatial context modelling.

These methods combine established numerical building blocks in project-specific ways.
They are research-track baselines, not claims of scientific novelty.  Every method is
therefore explicitly experimental and carries an ``unvalidated`` novelty marker.

The module is imported into the built-in registry so candidates can be exercised by
the normal compiler and benchmark machinery. Its research metadata keeps it outside
the release-maturity denominator until the documented graduation gates are met.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodMaturity, MethodSpec, ParamSpec
from ..registry import register
from ._markers import ResolvedMarkers, resolve_markers

_RESEARCH_METADATA = {"track": "research", "novelty": "unvalidated"}


def _dense_finite_matrix(data: SpatialTable, method: str) -> np.ndarray:
    """Return a finite dense matrix without mutating the input table."""
    from scipy.sparse import issparse

    source: Any = data.X
    matrix = source.toarray() if issparse(source) else np.asarray(source)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape != data.shape or matrix.shape[1] == 0:
        raise ValueError(f"{method}: X must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{method}: X must contain only finite values")
    return matrix


def _spatial_coordinates(data: SpatialTable, method: str) -> np.ndarray:
    """Validate and return finite two- or three-dimensional coordinates."""
    if data.spatial is None:
        raise ValueError(f"{method}: obsm['spatial'] is required")
    coordinates = np.asarray(data.spatial, dtype=float)
    if data.n_obs < 2:
        raise ValueError(f"{method}: at least two observations are required")
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != data.n_obs
        or coordinates.shape[1] not in (2, 3)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError(
            f"{method}: spatial coordinates must be finite, align with observations, "
            "and have two or three columns"
        )
    return coordinates


def _effective_k(k: int, n_obs: int) -> int:
    return max(1, min(int(k), n_obs - 1))


def _knn_indices(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic single-threaded nearest-neighbour distances and indices."""
    effective_k = _effective_k(k, coordinates.shape[0])
    raw_distances, raw_indices = cKDTree(coordinates).query(
        coordinates,
        k=effective_k + 1,
        workers=1,
    )
    raw_distances = np.asarray(raw_distances, dtype=float)
    raw_indices = np.asarray(raw_indices, dtype=int)
    distances = np.empty((coordinates.shape[0], effective_k), dtype=float)
    indices = np.empty((coordinates.shape[0], effective_k), dtype=int)
    for row in range(coordinates.shape[0]):
        candidates = sorted(
            (
                (float(distance), int(index))
                for distance, index in zip(
                    raw_distances[row], raw_indices[row], strict=True
                )
                if int(index) != row
            ),
            key=lambda item: (item[0], item[1]),
        )
        if len(candidates) < effective_k:
            full_distances = np.linalg.norm(coordinates - coordinates[row], axis=1)
            candidates = sorted(
                (
                    (float(distance), int(index))
                    for index, distance in enumerate(full_distances)
                    if index != row
                ),
                key=lambda item: (item[0], item[1]),
            )
        selected = candidates[:effective_k]
        distances[row] = [distance for distance, _ in selected]
        indices[row] = [index for _, index in selected]
    return distances, indices


def _positive_scale(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return float(np.median(positive)) if positive.size else 1.0


def _sampled_cross_distance_scale(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_samples: int = 128,
) -> float:
    """Estimate a cross-set distance scale with bounded temporary memory."""

    left_indices = np.linspace(
        0, left.shape[0] - 1, num=min(left.shape[0], max_samples), dtype=int
    )
    right_indices = np.linspace(
        0, right.shape[0] - 1, num=min(right.shape[0], max_samples), dtype=int
    )
    return _positive_scale(cdist(left[left_indices], right[right_indices]))


def _mutual_cross_batch_anchors(
    target_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    target_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    *,
    spatial_weight: float,
    block_size: int = 256,
) -> list[tuple[int, int]]:
    """Find exact mutual nearest anchors without a full cross-batch matrix."""

    expression_scale = _sampled_cross_distance_scale(
        target_embedding, reference_embedding
    )
    spatial_scale = _sampled_cross_distance_scale(
        target_coordinates, reference_coordinates
    )
    target_to_reference = np.empty(target_embedding.shape[0], dtype=int)
    reference_to_target = np.full(reference_embedding.shape[0], -1, dtype=int)
    reference_best = np.full(reference_embedding.shape[0], np.inf, dtype=float)

    for start in range(0, target_embedding.shape[0], block_size):
        stop = min(start + block_size, target_embedding.shape[0])
        fused = cdist(target_embedding[start:stop], reference_embedding) / expression_scale
        fused += (
            spatial_weight
            * cdist(target_coordinates[start:stop], reference_coordinates)
            / spatial_scale
        )
        target_to_reference[start:stop] = np.argmin(fused, axis=1)

        local_rows = np.argmin(fused, axis=0)
        columns = np.arange(reference_embedding.shape[0])
        local_best = fused[local_rows, columns]
        global_rows = start + local_rows
        better = local_best < reference_best
        tied = (local_best == reference_best) & (
            (reference_to_target < 0) | (global_rows < reference_to_target)
        )
        update = better | tied
        reference_best[update] = local_best[update]
        reference_to_target[update] = global_rows[update]

    return [
        (target_index, int(reference_index))
        for target_index, reference_index in enumerate(target_to_reference)
        if reference_to_target[int(reference_index)] == target_index
    ]


def _edge_list(edge_weights: Mapping[tuple[int, int], float]) -> list[list[int | float]]:
    return [
        [int(left), int(right), float(edge_weights[(left, right)])]
        for left, right in sorted(edge_weights)
    ]


def _store_graph(
    result: SpatialTable,
    *,
    name: str,
    edge_weights: Mapping[tuple[int, int], float],
    metadata: Mapping[str, Any],
) -> None:
    degrees: np.ndarray = np.zeros(result.n_obs, dtype=int)
    for left, right in edge_weights:
        degrees[left] += 1
        degrees[right] += 1
    result.obs[f"{name}_degree"] = degrees
    result.uns[name] = {
        "method": name,
        "n_nodes": int(result.n_obs),
        "n_edges": int(len(edge_weights)),
        "edges": _edge_list(edge_weights),
        **dict(metadata),
    }


def _batch_groups(data: SpatialTable, batch_key: str, method: str) -> dict[str, np.ndarray]:
    if batch_key not in data.obs:
        raise ValueError(f"{method}: obs[{batch_key!r}] is required")
    values = data.obs[batch_key]
    if values.isna().any():
        raise ValueError(f"{method}: obs[{batch_key!r}] must not contain missing values")
    labels = values.astype(str).to_numpy()
    groups = {
        label: np.flatnonzero(labels == label)
        for label in sorted(set(labels.tolist()))
    }
    if len(groups) < 2:
        raise ValueError(f"{method}: at least two batches are required")
    return groups


def _marker_inputs(
    data: SpatialTable,
    configured: object,
    method: str,
) -> ResolvedMarkers:
    markers = configured or data.uns.get("marker_genes")
    if not markers:
        raise ValueError(
            f"{method}: marker_genes must be supplied or stored in uns['marker_genes']"
        )
    marker_mapping = cast(Mapping[str, Iterable[str]], markers)
    return resolve_markers(data, marker_mapping)


def _standardized_marker_scores(
    matrix: np.ndarray,
    resolved: ResolvedMarkers,
) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    standardized = np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > 0,
    )
    return np.column_stack(
        [standardized[:, indices].mean(axis=1) for indices in resolved.indices]
    )


def _project_simplex_rows(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row onto the probability simplex."""
    projected = np.empty_like(values, dtype=float)
    for row_index, row in enumerate(values):
        ordered = np.sort(row)[::-1]
        cumulative = np.cumsum(ordered) - 1.0
        positions: np.ndarray = np.arange(1, row.size + 1, dtype=float)
        valid = ordered - cumulative / positions > 0
        if not valid.any():
            projected[row_index] = 1.0 / row.size
            continue
        rho = int(np.flatnonzero(valid)[-1])
        threshold = cumulative[rho] / float(rho + 1)
        clipped = np.maximum(row - threshold, 0.0)
        projected[row_index] = clipped / clipped.sum()
    return projected


@register
class WeaveAdaptiveRadiusGraph(Method):
    """Build a graph whose radius adapts to local sampling density."""

    spec = MethodSpec(
        name="weave_adaptive_radius_graph",
        category=MethodCategory.NEIGHBORHOOD,
        version="0.1.0",
        summary="Local-k-distance radii with symmetric Gaussian-weighted edges.",
        params=(
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("radius_scale", "float", 1.25, minimum=0.05),
        ),
        assumptions=("Finite obsm['spatial'] coordinates.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        coordinates = _spatial_coordinates(data, self.spec.name)
        k = _effective_k(self.params["k"], data.n_obs)
        distances, _ = _knn_indices(coordinates, k)
        radii = np.maximum(
            distances[:, -1] * float(self.params["radius_scale"]),
            np.finfo(float).eps,
        )
        tree = cKDTree(coordinates)
        edges: dict[tuple[int, int], float] = {}
        for left, radius in enumerate(radii):
            for raw_right in tree.query_ball_point(
                coordinates[left], float(radius), return_sorted=True
            ):
                right = int(raw_right)
                if left == right:
                    continue
                edge = (left, right) if left < right else (right, left)
                distance: float = float(
                    np.linalg.norm(coordinates[left] - coordinates[right])
                )
                sigma: float = max(
                    float((radii[left] + radii[right]) / 2.0),
                    float(np.finfo(float).eps),
                )
                weight = float(np.exp(-(distance * distance) / (2.0 * sigma * sigma)))
                edges[edge] = max(edges.get(edge, 0.0), weight)

        result = data.copy()
        _store_graph(
            result,
            name=self.spec.name,
            edge_weights=edges,
            metadata={
                "k": int(k),
                "radius_scale": float(self.params["radius_scale"]),
                "local_radius": [float(value) for value in radii],
            },
        )
        return self.finalize(result, step="neighborhood")


@register
class WeaveMutualKNNGraph(Method):
    """Keep only reciprocal spatial nearest-neighbour relationships."""

    spec = MethodSpec(
        name="weave_mutual_knn_graph",
        category=MethodCategory.NEIGHBORHOOD,
        version="0.1.0",
        summary="Reciprocal spatial k-nearest-neighbour graph.",
        params=(ParamSpec("k", "int", 6, minimum=1),),
        assumptions=("Finite obsm['spatial'] coordinates.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        coordinates = _spatial_coordinates(data, self.spec.name)
        distances, indices = _knn_indices(coordinates, self.params["k"])
        neighbour_sets = [set(row.tolist()) for row in indices]
        scale = _positive_scale(distances)
        edges: dict[tuple[int, int], float] = {}
        for left, neighbours in enumerate(neighbour_sets):
            for right in neighbours:
                if left >= right or left not in neighbour_sets[right]:
                    continue
                distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
                edges[(left, right)] = float(np.exp(-distance / scale))

        result = data.copy()
        _store_graph(
            result,
            name=self.spec.name,
            edge_weights=edges,
            metadata={"k": int(indices.shape[1]), "weight_scale": float(scale)},
        )
        return self.finalize(result, step="neighborhood")


@register
class WeaveExpressionSpatialGraph(Method):
    """Fuse expression and physical distances before selecting graph edges."""

    spec = MethodSpec(
        name="weave_expression_spatial_graph",
        category=MethodCategory.NEIGHBORHOOD,
        version="0.1.0",
        summary="Spatial-expression fused k-nearest-neighbour graph.",
        params=(
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("spatial_weight", "float", 0.5, minimum=0.0, maximum=1.0),
            ParamSpec("n_components", "int", 8, minimum=1),
        ),
        assumptions=("Finite X and obsm['spatial'] coordinates.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix = _dense_finite_matrix(data, self.spec.name)
        coordinates = _spatial_coordinates(data, self.spec.name)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scale = matrix.std(axis=0, keepdims=True)
        standardized = np.divide(
            centered,
            scale,
            out=np.zeros_like(centered),
            where=scale > 0,
        )
        component_count = min(
            int(self.params["n_components"]), data.n_obs - 1, data.n_vars
        )
        left_vectors, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
        embedding = left_vectors[:, :component_count] * singular_values[:component_count]
        k = _effective_k(self.params["k"], data.n_obs)
        spatial_scale = _positive_scale(_knn_indices(coordinates, k)[0])
        expression_scale = _positive_scale(_knn_indices(embedding, k)[0])
        spatial_weight = float(self.params["spatial_weight"])
        edges: dict[tuple[int, int], float] = {}
        for left in range(data.n_obs):
            spatial_distances = np.linalg.norm(coordinates - coordinates[left], axis=1)
            expression_distances = np.linalg.norm(embedding - embedding[left], axis=1)
            fused = (
                spatial_weight * spatial_distances / spatial_scale
                + (1.0 - spatial_weight) * expression_distances / expression_scale
            )
            fused[left] = np.inf
            for raw_right in np.argsort(fused, kind="mergesort")[:k]:
                right = int(raw_right)
                edge = (left, right) if left < right else (right, left)
                edges[edge] = max(edges.get(edge, 0.0), float(np.exp(-fused[right])))

        result = data.copy()
        _store_graph(
            result,
            name=self.spec.name,
            edge_weights=edges,
            metadata={
                "k": int(k),
                "spatial_weight": spatial_weight,
                "n_components": int(component_count),
                "spatial_scale": float(spatial_scale),
                "expression_scale": float(expression_scale),
            },
        )
        return self.finalize(result, step="neighborhood")


@register
class WeaveSpatialQuantileIntegrate(Method):
    """Align batch-wise empirical quantiles and blend local spatial context."""

    spec = MethodSpec(
        name="weave_spatial_quantile_integrate",
        category=MethodCategory.INTEGRATION,
        version="0.1.0",
        summary="Batch quantile alignment with within-batch spatial blending.",
        params=(
            ParamSpec("batch_key", "str", "batch"),
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("spatial_blend", "float", 0.2, minimum=0.0, maximum=1.0),
            ParamSpec("output_key", "str", "X_weave_spatial_quantile"),
        ),
        assumptions=("Finite X and coordinates; at least two batches.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix = _dense_finite_matrix(data, self.spec.name)
        coordinates = _spatial_coordinates(data, self.spec.name)
        groups = _batch_groups(data, self.params["batch_key"], self.spec.name)
        aligned = np.empty_like(matrix)
        for gene in range(data.n_vars):
            batch_values = [matrix[indices, gene] for indices in groups.values()]
            for indices in groups.values():
                quantiles = (rankdata(matrix[indices, gene], method="average") - 0.5) / len(indices)
                targets = np.vstack(
                    [np.quantile(values, quantiles, method="linear") for values in batch_values]
                )
                aligned[indices, gene] = targets.mean(axis=0)

        blended = aligned.copy()
        spatial_blend = float(self.params["spatial_blend"])
        if spatial_blend > 0:
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                _, local_neighbours = _knn_indices(
                    coordinates[indices], min(int(self.params["k"]), len(indices) - 1)
                )
                local_mean = aligned[indices[local_neighbours]].mean(axis=1)
                blended[indices] = (
                    (1.0 - spatial_blend) * aligned[indices] + spatial_blend * local_mean
                )
        if not np.isfinite(blended).all():
            raise RuntimeError(f"{self.spec.name}: integration produced non-finite values")

        result = data.copy()
        result.obsm[self.params["output_key"]] = blended
        result.uns[self.spec.name] = {
            "batch_key": self.params["batch_key"],
            "batches": list(groups),
            "k": int(min(int(self.params["k"]), max(len(v) - 1 for v in groups.values()))),
            "spatial_blend": spatial_blend,
            "output_key": self.params["output_key"],
        }
        return self.finalize(result, step="integration")


@register
class WeaveAnchorResidualIntegrate(Method):
    """Correct batch residuals estimated from mutual spatial-expression anchors."""

    spec = MethodSpec(
        name="weave_anchor_residual_integrate",
        category=MethodCategory.INTEGRATION,
        version="0.1.0",
        summary="Mutual cross-batch anchors followed by median residual correction.",
        params=(
            ParamSpec("batch_key", "str", "batch"),
            ParamSpec("spatial_weight", "float", 1.0, minimum=0.0),
            ParamSpec("n_components", "int", 8, minimum=1),
            ParamSpec("output_key", "str", "X_weave_anchor_residual"),
        ),
        assumptions=("Registered finite coordinates and at least two batches.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix = _dense_finite_matrix(data, self.spec.name)
        coordinates = _spatial_coordinates(data, self.spec.name)
        groups = _batch_groups(data, self.params["batch_key"], self.spec.name)

        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scale = matrix.std(axis=0, keepdims=True)
        standardized = np.divide(
            centered,
            scale,
            out=np.zeros_like(centered),
            where=scale > 0,
        )
        component_count = min(
            int(self.params["n_components"]), data.n_obs - 1, data.n_vars
        )
        left_vectors, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
        embedding = left_vectors[:, :component_count] * singular_values[:component_count]

        reference = sorted(groups, key=lambda label: (-len(groups[label]), label))[0]
        reference_indices = groups[reference]
        corrected = matrix.copy()
        anchor_records: dict[str, list[list[int]]] = {}
        for label, target_indices in groups.items():
            if label == reference:
                continue
            anchors = _mutual_cross_batch_anchors(
                embedding[target_indices],
                embedding[reference_indices],
                coordinates[target_indices],
                coordinates[reference_indices],
                spatial_weight=float(self.params["spatial_weight"]),
            )
            target_rows = np.asarray([target_indices[left] for left, _ in anchors], dtype=int)
            reference_rows = np.asarray(
                [reference_indices[right] for _, right in anchors], dtype=int
            )
            residual = np.median(matrix[target_rows] - matrix[reference_rows], axis=0)
            corrected[target_indices] -= residual
            anchor_records[label] = [
                [int(target_indices[left]), int(reference_indices[right])]
                for left, right in anchors
            ]
        if not np.isfinite(corrected).all():
            raise RuntimeError(f"{self.spec.name}: integration produced non-finite values")

        result = data.copy()
        result.obsm[self.params["output_key"]] = corrected
        result.uns[self.spec.name] = {
            "batch_key": self.params["batch_key"],
            "reference_batch": reference,
            "spatial_weight": float(self.params["spatial_weight"]),
            "n_components": int(component_count),
            "anchors": anchor_records,
            "output_key": self.params["output_key"],
        }
        return self.finalize(result, step="integration")


@register
class WeaveNeighborMarkerAnnotate(Method):
    """Smooth marker evidence over spatial neighbours before annotation."""

    spec = MethodSpec(
        name="weave_neighbor_marker_annotate",
        category=MethodCategory.ANNOTATION,
        version="0.1.0",
        summary="Marker-score annotation with deterministic spatial score smoothing.",
        params=(
            ParamSpec("marker_genes", "dict|None", None),
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("neighbor_weight", "float", 0.35, minimum=0.0, maximum=1.0),
            ParamSpec("key_added", "str", "cell_type"),
            ParamSpec("score_key", "str", "X_weave_neighbor_marker_scores"),
        ),
        assumptions=("Finite X, coordinates, and resolvable marker sets.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix = _dense_finite_matrix(data, self.spec.name)
        coordinates = _spatial_coordinates(data, self.spec.name)
        resolved = _marker_inputs(data, self.params["marker_genes"], self.spec.name)
        raw_scores = _standardized_marker_scores(matrix, resolved)
        _, neighbours = _knn_indices(coordinates, self.params["k"])
        neighbour_scores = raw_scores[neighbours].mean(axis=1)
        neighbor_weight = float(self.params["neighbor_weight"])
        scores = (1.0 - neighbor_weight) * raw_scores + neighbor_weight * neighbour_scores
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{self.spec.name}: annotation produced non-finite scores")

        best = np.argmax(scores, axis=1)
        ordered = np.sort(scores, axis=1)
        margin = ordered[:, -1] - ordered[:, -2] if scores.shape[1] > 1 else abs(ordered[:, -1])
        result = data.copy()
        result.obs[self.params["key_added"]] = pd.Categorical(
            [resolved.labels[index] for index in best],
            categories=resolved.labels,
        )
        result.obs[f"{self.params['key_added']}_confidence"] = margin
        result.obsm[self.params["score_key"]] = scores
        result.uns[self.spec.name] = {
            "labels": list(resolved.labels),
            "marker_resolution": resolved.diagnostics(),
            "k": int(neighbours.shape[1]),
            "neighbor_weight": neighbor_weight,
            "score_key": self.params["score_key"],
        }
        return self.finalize(result, step="annotation")


@register
class WeaveSpatialSimplexDeconv(Method):
    """Estimate simplex proportions with graph-Laplacian spatial regularization."""

    spec = MethodSpec(
        name="weave_spatial_simplex_deconv",
        category=MethodCategory.DECONVOLUTION,
        version="0.1.0",
        summary="Marker evidence projected onto a spatially regularized simplex.",
        params=(
            ParamSpec("marker_genes", "dict|None", None),
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("spatial_strength", "float", 0.5, minimum=0.0),
            ParamSpec("n_iter", "int", 100, minimum=1),
            ParamSpec("proportion_key", "str", "proportions"),
        ),
        assumptions=("Non-negative finite X, coordinates, and resolvable marker sets.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=_RESEARCH_METADATA.copy(),
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix = _dense_finite_matrix(data, self.spec.name)
        coordinates = _spatial_coordinates(data, self.spec.name)
        if (matrix < 0).any():
            raise ValueError(f"{self.spec.name}: X must contain non-negative expression values")
        resolved = _marker_inputs(data, self.params["marker_genes"], self.spec.name)
        logged = np.log1p(matrix)
        evidence = np.column_stack(
            [logged[:, indices].mean(axis=1) for indices in resolved.indices]
        )
        totals = evidence.sum(axis=1, keepdims=True)
        target = np.divide(
            evidence,
            totals,
            out=np.full_like(evidence, 1.0 / evidence.shape[1]),
            where=totals > 0,
        )

        distances, neighbours = _knn_indices(coordinates, self.params["k"])
        distance_scale = _positive_scale(distances)
        edge_weights: dict[tuple[int, int], float] = {}
        for left in range(data.n_obs):
            for position, raw_right in enumerate(neighbours[left]):
                right = int(raw_right)
                edge = (left, right) if left < right else (right, left)
                weight = float(np.exp(-distances[left, position] / distance_scale))
                edge_weights[edge] = max(edge_weights.get(edge, 0.0), weight)

        degree: np.ndarray = np.zeros(data.n_obs, dtype=float)
        for (left, right), weight in edge_weights.items():
            degree[left] += weight
            degree[right] += weight
        strength = float(self.params["spatial_strength"])
        step_size = 1.0 / (1.0 + 2.0 * strength * max(float(degree.max()), 0.0))
        proportions = target.copy()
        for _ in range(int(self.params["n_iter"])):
            laplacian = degree[:, None] * proportions
            for (left, right), weight in edge_weights.items():
                laplacian[left] -= weight * proportions[right]
                laplacian[right] -= weight * proportions[left]
            gradient = proportions - target + strength * laplacian
            proportions = _project_simplex_rows(proportions - step_size * gradient)
        if not np.isfinite(proportions).all():
            raise RuntimeError(f"{self.spec.name}: deconvolution produced non-finite values")

        result = data.copy()
        result.obsm[self.params["proportion_key"]] = proportions
        result.uns[self.spec.name] = {
            "cell_types": list(resolved.labels),
            "marker_resolution": resolved.diagnostics(),
            "spatial_strength": strength,
            "iterations": int(self.params["n_iter"]),
            "graph": {
                "n_nodes": int(data.n_obs),
                "n_edges": int(len(edge_weights)),
                "k": int(neighbours.shape[1]),
                "edges": _edge_list(edge_weights),
            },
            "proportion_key": self.params["proportion_key"],
        }
        return self.finalize(result, step="deconvolution")


__all__ = [
    "WeaveAdaptiveRadiusGraph",
    "WeaveAnchorResidualIntegrate",
    "WeaveExpressionSpatialGraph",
    "WeaveMutualKNNGraph",
    "WeaveNeighborMarkerAnnotate",
    "WeaveSpatialQuantileIntegrate",
    "WeaveSpatialSimplexDeconv",
]
