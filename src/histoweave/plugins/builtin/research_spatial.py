"""Experimental native spatial-analysis methods.

The methods in this module are registered as experimental built-ins so the compiler
and benchmark machinery can exercise them. They are research prototypes with a
stable plugin contract, deterministic reference implementations, and explicit
``unvalidated`` metadata, but they have not passed HistoWeave's release evidence
gates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from ..._math import kmeans, pca, zscore
from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodMaturity, MethodSpec, ParamSpec
from ..registry import register
from ._validation import validate_nonnegative_matrix, validate_spatial_coordinates

_RESEARCH_METADATA = {"track": "research", "novelty": "unvalidated"}


def _matrix_and_coordinates(data: SpatialTable, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite dense expression and aligned 2-D coordinates."""

    validate_nonnegative_matrix(data.X, method=method)
    matrix = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape != data.shape:
        raise ValueError(f"{method}: X must be a two-dimensional observation-by-gene matrix")
    if data.spatial is None:
        raise ValueError(f"{method}: obsm['spatial'] is required")
    coordinates = validate_spatial_coordinates(data.spatial, method=method)[:, :2]
    if coordinates.shape[0] != data.n_obs:
        raise ValueError(f"{method}: spatial coordinates must align with observations")
    if data.n_obs < 3:
        raise ValueError(f"{method}: at least three observations are required")
    return matrix, coordinates


def _neighbors(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic nearest-neighbour indices and squared distances."""

    effective_k = min(max(1, int(k)), coordinates.shape[0] - 1)
    raw_distances, raw_indices = cKDTree(coordinates).query(
        coordinates,
        k=effective_k + 1,
        workers=1,
    )
    raw_distances = np.atleast_2d(np.asarray(raw_distances, dtype=float))
    raw_indices = np.atleast_2d(np.asarray(raw_indices, dtype=int))
    distances = np.empty((coordinates.shape[0], effective_k), dtype=float)
    indices = np.empty((coordinates.shape[0], effective_k), dtype=int)
    for row in range(coordinates.shape[0]):
        candidates = sorted(
            (
                (float(distance), int(index))
                for distance, index in zip(raw_distances[row], raw_indices[row], strict=True)
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
    return indices, np.square(distances)


def _expression_embedding(matrix: np.ndarray, n_pcs: int) -> np.ndarray:
    """Build a finite log-expression PCA embedding with constant-gene handling."""

    logged = np.log1p(matrix)
    standardized = zscore(logged)
    width = min(int(n_pcs), matrix.shape[0] - 1, matrix.shape[1])
    if width < 1:
        raise ValueError("an expression embedding requires at least one component")
    embedding = pca(standardized, width)
    return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)


def _local_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[indices].mean(axis=1)


def _categorical(labels: np.ndarray) -> pd.Categorical:
    from typing import cast

    return cast(pd.Categorical, pd.Categorical([f"domain_{int(label)}" for label in labels]))


def _domain_spec(
    name: str,
    summary: str,
    params: Sequence[ParamSpec],
) -> MethodSpec:
    return MethodSpec(
        name=name,
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary=summary,
        params=tuple(params),
        assumptions=("Finite non-negative expression and aligned 2-D spatial coordinates.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=dict(_RESEARCH_METADATA),
        modalities=("expression", "spatial"),
        model_family="machine_learning",
    )


def _svg_spec(name: str, summary: str, params: Sequence[ParamSpec]) -> MethodSpec:
    return MethodSpec(
        name=name,
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        version="0.1.0",
        summary=summary,
        params=tuple(params),
        assumptions=("Finite non-negative expression and aligned 2-D spatial coordinates.",),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=dict(_RESEARCH_METADATA),
        modalities=("expression", "spatial"),
    )


_DOMAIN_BASE_PARAMS = (
    ParamSpec("n_domains", "int", 3, "Requested number of spatial domains.", minimum=2),
    ParamSpec("n_pcs", "int", 8, "Expression embedding width.", minimum=1),
    ParamSpec("key_added", "str", "domain", "Observation column for domain labels."),
    ParamSpec("random_state", "int", 0, "Deterministic seed."),
)


@register
class WeaveMultiscaleConsensusDomains(Method):
    """Cluster a consensus embedding assembled from several spatial resolutions."""

    spec = _domain_spec(
        "weave_multiscale_consensus_domains",
        "Consensus domains from expression embeddings smoothed at three spatial scales.",
        (*_DOMAIN_BASE_PARAMS, ParamSpec("base_k", "int", 4, minimum=1)),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        embedding = _expression_embedding(matrix, self.params["n_pcs"])
        scales = sorted(
            {self.params["base_k"], self.params["base_k"] * 2, self.params["base_k"] * 4}
        )
        views = [zscore(embedding)]
        for scale in scales:
            indices, _ = _neighbors(coordinates, scale)
            views.append(zscore(_local_mean(embedding, indices)))
        consensus = np.concatenate(views, axis=1) / np.sqrt(len(views))
        labels = kmeans(
            consensus,
            self.params["n_domains"],
            n_init=6,
            random_state=self.params["random_state"],
        )
        result = data.copy()
        result.obs[self.params["key_added"]] = _categorical(labels)
        result.obsm["X_weave_multiscale_consensus"] = consensus
        result.uns[self.spec.name] = {"scales": scales}
        return self.finalize(result)


@register
class WeaveBoundaryAwareDomains(Method):
    """Reduce cross-boundary smoothing using local transcriptomic discontinuity."""

    spec = _domain_spec(
        "weave_boundary_aware_domains",
        "Domain detection with an adaptive barrier against transcriptomic boundaries.",
        (
            *_DOMAIN_BASE_PARAMS,
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("spatial_weight", "float", 0.5, minimum=0.0, maximum=1.0),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        embedding = _expression_embedding(matrix, self.params["n_pcs"])
        indices, distances = _neighbors(coordinates, self.params["k"])
        differences = embedding[:, None, :] - embedding[indices]
        discontinuity = np.sqrt(np.square(differences).sum(axis=2))
        finite_distances = np.sqrt(distances)
        spatial_scale = max(float(np.median(finite_distances)), 1e-12)
        expression_scale = max(float(np.median(discontinuity)), 1e-12)
        weights = np.exp(-finite_distances / spatial_scale - discontinuity / expression_scale)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        local = np.einsum("ij,ijk->ik", weights, embedding[indices])
        boundary = discontinuity.mean(axis=1)
        boundary /= max(float(boundary.max()), 1e-12)
        blend = self.params["spatial_weight"] * (1.0 - boundary[:, None])
        adaptive = zscore((1.0 - blend) * embedding + blend * local)
        labels = kmeans(
            adaptive, self.params["n_domains"], n_init=6, random_state=self.params["random_state"]
        )
        result = data.copy()
        result.obs[self.params["key_added"]] = _categorical(labels)
        result.obs["weave_boundary_score"] = boundary
        result.obsm["X_weave_boundary_aware"] = adaptive
        return self.finalize(result)


@register
class WeaveTopologyRegularizedDomains(Method):
    """Refine expression clusters with deterministic graph-label regularization."""

    spec = _domain_spec(
        "weave_topology_regularized_domains",
        "Expression domains refined by a spatial graph Potts objective.",
        (
            *_DOMAIN_BASE_PARAMS,
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("topology_weight", "float", 0.35, minimum=0.0),
            ParamSpec("n_refine", "int", 6, minimum=1),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        embedding = _expression_embedding(matrix, self.params["n_pcs"])
        indices, _ = _neighbors(coordinates, self.params["k"])
        n_domains = min(self.params["n_domains"], data.n_obs)
        labels = kmeans(embedding, n_domains, n_init=6, random_state=self.params["random_state"])
        for _ in range(self.params["n_refine"]):
            centers = np.vstack(
                [
                    embedding[labels == group].mean(axis=0)
                    if np.any(labels == group)
                    else embedding.mean(axis=0)
                    for group in range(n_domains)
                ]
            )
            expression_cost = np.square(embedding[:, None, :] - centers[None, :, :]).sum(axis=2)
            scale = max(float(np.median(expression_cost)), 1e-12)
            topology_cost = np.stack(
                [(labels[indices] != group).mean(axis=1) for group in range(n_domains)], axis=1
            )
            updated = np.argmin(
                expression_cost / scale + self.params["topology_weight"] * topology_cost, axis=1
            )
            if np.array_equal(updated, labels):
                break
            labels = updated
        coherence = (labels[indices] == labels[:, None]).mean(axis=1)
        result = data.copy()
        result.obs[self.params["key_added"]] = _categorical(labels)
        result.obs["weave_topology_coherence"] = coherence
        result.obsm["X_weave_topology"] = embedding
        return self.finalize(result)


@register
class WeaveUncertaintyDomains(Method):
    """Quantify domain uncertainty from aligned deterministic clustering restarts."""

    spec = _domain_spec(
        "weave_uncertainty_domains",
        "Domain labels with per-observation uncertainty from aligned consensus runs.",
        (
            *_DOMAIN_BASE_PARAMS,
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("n_ensembles", "int", 7, minimum=3),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        embedding = _expression_embedding(matrix, self.params["n_pcs"])
        indices, _ = _neighbors(coordinates, self.params["k"])
        joint = zscore(np.concatenate([embedding, _local_mean(embedding, indices)], axis=1))
        n_domains = min(self.params["n_domains"], data.n_obs)
        runs = [
            kmeans(joint, n_domains, n_init=3, random_state=self.params["random_state"] + offset)
            for offset in range(self.params["n_ensembles"])
        ]
        reference = runs[0]
        aligned = [reference]
        fallback_center = joint.mean(axis=0)
        reference_centers = np.vstack(
            [
                joint[reference == group].mean(axis=0)
                if np.any(reference == group)
                else fallback_center
                for group in range(n_domains)
            ]
        )
        for labels in runs[1:]:
            centers = np.vstack(
                [
                    joint[labels == group].mean(axis=0)
                    if np.any(labels == group)
                    else joint.mean(axis=0)
                    for group in range(n_domains)
                ]
            )
            costs = np.square(centers[:, None, :] - reference_centers[None, :, :]).sum(axis=2)
            rows, columns = linear_sum_assignment(costs)
            mapping = np.arange(n_domains)
            mapping[rows] = columns
            aligned.append(mapping[labels])
        votes = np.stack(aligned)
        probabilities = np.stack(
            [(votes == group).mean(axis=0) for group in range(n_domains)], axis=1
        )
        labels = probabilities.argmax(axis=1)
        uncertainty = 1.0 - probabilities.max(axis=1)
        result = data.copy()
        result.obs[self.params["key_added"]] = _categorical(labels)
        result.obs["weave_domain_uncertainty"] = uncertainty
        result.obsm["X_weave_uncertainty"] = joint
        result.obsm["weave_domain_probabilities"] = probabilities
        return self.finalize(result)


_SVG_BASE_PARAMS = (
    ParamSpec("k", "int", 6, "Spatial neighbours.", minimum=1),
    ParamSpec("n_top", "int", 50, "Number of ranked genes.", minimum=1),
)


def _standardized_genes(matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(zscore(np.log1p(matrix)), nan=0.0, posinf=0.0, neginf=0.0)


def _rank_svg(result: SpatialTable, method: str, scores: np.ndarray, n_top: int) -> None:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    score_key = f"{method}_score"
    result.var[score_key] = scores
    order = np.argsort(-scores, kind="mergesort")[: min(int(n_top), result.n_vars)]
    ranking = [
        {"gene": str(result.var.index[index]), "score": float(scores[index])} for index in order
    ]
    result.uns[method] = {"score_key": score_key, "top_genes": ranking}
    result.uns["svg"] = {"method": method, "score_key": score_key, "top_genes": ranking}


@register
class WeaveMultiscaleSVG(Method):
    spec = _svg_spec(
        "weave_multiscale_svg",
        "Spatial association aggregated across short, medium, and long neighbourhoods.",
        (*_SVG_BASE_PARAMS, ParamSpec("n_scales", "int", 3, minimum=2, maximum=5)),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        values = _standardized_genes(matrix)
        scale_scores = []
        for multiplier in range(1, self.params["n_scales"] + 1):
            indices, _ = _neighbors(coordinates, self.params["k"] * multiplier)
            scale_scores.append((values * _local_mean(values, indices)).mean(axis=0))
        scores = np.maximum(np.mean(scale_scores, axis=0), 0.0)
        result = data.copy()
        _rank_svg(result, self.spec.name, scores, self.params["n_top"])
        return self.finalize(result)


@register
class WeaveBoundarySVG(Method):
    spec = _svg_spec(
        "weave_boundary_svg",
        "Genes whose local expression jumps coincide with multigene spatial boundaries.",
        _SVG_BASE_PARAMS,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        values = _standardized_genes(matrix)
        indices, _ = _neighbors(coordinates, self.params["k"])
        gene_jumps = np.square(values[:, None, :] - values[indices]).mean(axis=1)
        boundary = gene_jumps.mean(axis=1)
        centered_boundary = boundary - boundary.mean()
        centered_jumps = gene_jumps - gene_jumps.mean(axis=0, keepdims=True)
        numerator = (centered_jumps * centered_boundary[:, None]).mean(axis=0)
        denominator = gene_jumps.std(axis=0) * max(float(boundary.std()), 1e-12)
        scores = np.maximum(
            np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0),
            0.0,
        )
        result = data.copy()
        _rank_svg(result, self.spec.name, scores, self.params["n_top"])
        return self.finalize(result)


@register
class WeaveHotspotSVG(Method):
    spec = _svg_spec(
        "weave_hotspot_svg",
        "Local high/low expression concentration measured without a global domain model.",
        _SVG_BASE_PARAMS,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        values = _standardized_genes(matrix)
        indices, _ = _neighbors(coordinates, self.params["k"])
        local = (values + values[indices].sum(axis=1)) / (indices.shape[1] + 1)
        scores = np.square(local).mean(axis=0)
        result = data.copy()
        _rank_svg(result, self.spec.name, scores, self.params["n_top"])
        return self.finalize(result)


@register
class WeaveAnisotropySVG(Method):
    spec = _svg_spec(
        "weave_anisotropy_svg",
        "Directional spatial association contrast across four orientation sectors.",
        _SVG_BASE_PARAMS,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        values = _standardized_genes(matrix)
        indices, _ = _neighbors(coordinates, self.params["k"])
        offsets = coordinates[indices] - coordinates[:, None, :]
        angles = np.mod(np.arctan2(offsets[..., 1], offsets[..., 0]), np.pi)
        sectors = np.floor(angles / (np.pi / 4.0)).astype(int).clip(0, 3)
        products = values[:, None, :] * values[indices]
        directional = []
        for sector in range(4):
            mask = sectors == sector
            counts = mask.sum(axis=1)
            per_cell = np.divide(
                (products * mask[..., None]).sum(axis=1),
                counts[:, None],
                out=np.zeros_like(values),
                where=counts[:, None] > 0,
            )
            directional.append(per_cell.mean(axis=0))
        directional_array = np.stack(directional)
        scores = directional_array.max(axis=0) - directional_array.min(axis=0)
        result = data.copy()
        _rank_svg(result, self.spec.name, scores, self.params["n_top"])
        return self.finalize(result)


@register
class WeaveBootstrapRobustSVG(Method):
    spec = _svg_spec(
        "weave_bootstrap_robust_svg",
        "Spatial association ranked by deterministic bootstrap stability.",
        (
            *_SVG_BASE_PARAMS,
            ParamSpec("n_bootstraps", "int", 12, minimum=3),
            ParamSpec("sample_fraction", "float", 0.75, minimum=0.5, maximum=1.0),
            ParamSpec("random_state", "int", 0),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        matrix, coordinates = _matrix_and_coordinates(data, self.spec.name)
        values = _standardized_genes(matrix)
        sample_size = max(3, int(np.ceil(data.n_obs * self.params["sample_fraction"])))
        rng = np.random.default_rng(self.params["random_state"])
        estimates = []
        for _ in range(self.params["n_bootstraps"]):
            selected = np.sort(rng.choice(data.n_obs, size=sample_size, replace=False))
            subset_values = values[selected]
            indices, _ = _neighbors(coordinates[selected], self.params["k"])
            estimates.append((subset_values * _local_mean(subset_values, indices)).mean(axis=0))
        estimates_array = np.stack(estimates)
        mean = estimates_array.mean(axis=0)
        spread = estimates_array.std(axis=0)
        stability = np.mean(estimates_array > 0.0, axis=0)
        scores = np.maximum(mean, 0.0) * stability / (1.0 + spread)
        result = data.copy()
        _rank_svg(result, self.spec.name, scores, self.params["n_top"])
        result.uns[self.spec.name]["n_bootstraps"] = self.params["n_bootstraps"]
        return self.finalize(result)


RESEARCH_METHODS: dict[str, type[Method]] = {
    cls.spec.name: cls
    for cls in (
        WeaveMultiscaleConsensusDomains,
        WeaveBoundaryAwareDomains,
        WeaveTopologyRegularizedDomains,
        WeaveUncertaintyDomains,
        WeaveMultiscaleSVG,
        WeaveBoundarySVG,
        WeaveHotspotSVG,
        WeaveAnisotropySVG,
        WeaveBootstrapRobustSVG,
    )
}

__all__ = ["RESEARCH_METHODS", *(cls.__name__ for cls in RESEARCH_METHODS.values())]
