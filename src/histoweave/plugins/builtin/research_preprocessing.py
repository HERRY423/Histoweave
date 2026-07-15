"""Dependency-light research preprocessing methods.

The methods in this module are registered as experimental built-ins so the compiler
and benchmark machinery can exercise them. They remain unvalidated research
candidates, are excluded from release-maturity ratios, copy their inputs, and use
only NumPy/SciPy operations so experiments remain easy to audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from ...data import SpatialTable
from ..interfaces import (
    Method,
    MethodCategory,
    MethodImplementation,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register

_RESEARCH_METADATA = {"track": "research", "novelty": "unvalidated"}


def _research_spec(
    *,
    name: str,
    category: MethodCategory,
    summary: str,
    params: tuple[ParamSpec, ...],
    spatial: bool = False,
) -> MethodSpec:
    """Build the common, explicitly unvalidated research-method metadata."""

    assumptions = ["X contains finite non-negative expression values."]
    modalities = ["expression"]
    if spatial:
        assumptions.append("obsm['spatial'] contains aligned finite coordinates.")
        modalities.append("spatial")
    return MethodSpec(
        name=name,
        category=category,
        version="0.1.0",
        summary=summary,
        params=params,
        assumptions=tuple(assumptions),
        maturity=MethodMaturity.EXPERIMENTAL,
        metadata=dict(_RESEARCH_METADATA),
        language="python",
        modalities=tuple(modalities),
        model_family="statistical",
        implementation=MethodImplementation.NATIVE,
    )


def _expression_matrix(data: SpatialTable, *, method: str) -> np.ndarray:
    """Return a finite, non-negative dense matrix after structural validation."""

    source: Any = data.X
    if getattr(source, "ndim", None) != 2:
        raise ValueError(f"{method}: X must be a two-dimensional matrix")
    if data.n_obs < 1 or data.n_vars < 1:
        raise ValueError(f"{method}: X must contain at least one observation and one feature")
    if source.shape != (data.n_obs, data.n_vars):
        raise ValueError(
            f"{method}: X shape {source.shape} does not match "
            f"({data.n_obs}, {data.n_vars})"
        )
    if len(data.obs) != data.n_obs or len(data.var) != data.n_vars:
        raise ValueError(f"{method}: X must align with obs and var")
    if hasattr(source, "toarray"):
        source = source.toarray()
    try:
        matrix = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{method}: X must contain numeric values") from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"{method}: X must contain finite values")
    if np.any(matrix < 0):
        raise ValueError(f"{method}: X must contain non-negative values")
    return matrix


def _spatial_neighbors(data: SpatialTable, *, k: int, method: str) -> np.ndarray:
    """Return deterministic non-self kNN indices after spatial validation."""

    if data.spatial is None:
        raise ValueError(f"{method}: obsm['spatial'] is required")
    try:
        coordinates = np.asarray(data.spatial, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{method}: spatial coordinates must be numeric") from exc
    if coordinates.ndim != 2 or coordinates.shape[0] != data.n_obs:
        raise ValueError(f"{method}: spatial coordinates must align with observations")
    if coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"{method}: spatial coordinates must be finite and at least 2D")
    if data.n_obs < 2:
        raise ValueError(f"{method}: at least two observations are required")

    effective_k = min(int(k), data.n_obs - 1)
    _, indices = cKDTree(coordinates[:, :2]).query(
        coordinates[:, :2],
        k=effective_k + 1,
    )
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        indices = indices[:, None]

    neighbors: np.ndarray = np.empty((data.n_obs, effective_k), dtype=int)
    for row_index, candidates in enumerate(indices):
        without_self = candidates[candidates != row_index]
        if without_self.size < effective_k:
            raise ValueError(f"{method}: failed to construct a non-self spatial neighborhood")
        neighbors[row_index] = without_self[:effective_k]
    return neighbors


def _row_compositions(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def _quantile_flags(
    scores: np.ndarray,
    quantile: float,
    *,
    high_is_bad: bool,
) -> tuple[np.ndarray, float]:
    threshold_quantile = 1.0 - quantile if high_is_bad else quantile
    threshold = float(np.quantile(scores, threshold_quantile))
    flags = scores > threshold if high_is_bad else scores < threshold
    return flags, threshold


def _record_research_metadata(
    result: SpatialTable,
    method: str,
    **metadata: object,
) -> None:
    result.uns.setdefault("research_preprocessing", {})[method] = {
        "track": "research",
        "novelty": "unvalidated",
        **metadata,
    }


class _ResearchNormalization(Method):
    """Shared copy, validation, raw-layer, metadata, and provenance contract."""

    def transform(self, data: SpatialTable, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = _expression_matrix(result, method=self.spec.name)
        result.layers.setdefault("counts", result.X.copy())
        transformed = np.asarray(self.transform(result, matrix), dtype=float)
        if transformed.shape != matrix.shape:
            raise RuntimeError(
                f"{self.spec.name}: transform returned {transformed.shape}, expected {matrix.shape}"
            )
        if not np.isfinite(transformed).all():
            raise RuntimeError(f"{self.spec.name}: transform produced non-finite values")
        result.X = transformed
        result.uns["normalization"] = {
            "method": self.spec.name,
            "research": True,
        }
        _record_research_metadata(result, self.spec.name)
        return self.finalize(result, step="normalization")


@register
class WeaveSpatialEntropyQC(Method):
    """Flag locally low transcript-distribution entropy."""

    spec = _research_spec(
        name="weave_spatial_entropy_qc",
        category=MethodCategory.QC,
        summary="Local expression-composition entropy for spatial QC.",
        params=(
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("flag_fraction", "float", 0.05, minimum=0.0, maximum=1.0),
            ParamSpec("score_key", "str", "weave_spatial_entropy"),
            ParamSpec("flag_key", "str", "weave_low_spatial_entropy"),
        ),
        spatial=True,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = _expression_matrix(result, method=self.spec.name)
        neighbors = _spatial_neighbors(result, k=self.params["k"], method=self.spec.name)
        local_profile = 0.5 * (matrix + matrix[neighbors].mean(axis=1))
        proportions = _row_compositions(local_profile)
        entropy = -np.sum(
            np.where(proportions > 0, proportions * np.log(proportions + 1e-15), 0.0),
            axis=1,
        )
        active = np.sum(local_profile > 0, axis=1)
        denominator = np.log(np.maximum(active, 2))
        scores = np.divide(entropy, denominator, out=np.zeros_like(entropy), where=active > 1)
        scores = np.clip(scores, 0.0, 1.0)
        flags, threshold = _quantile_flags(
            scores,
            self.params["flag_fraction"],
            high_is_bad=False,
        )
        result.obs[self.params["score_key"]] = scores
        result.obs[self.params["flag_key"]] = flags
        _record_research_metadata(result, self.spec.name, threshold=threshold, k=self.params["k"])
        return self.finalize(result, step="qc")


@register
class WeaveNeighborDiscordanceQC(Method):
    """Flag observations whose composition diverges from their neighborhood."""

    spec = _research_spec(
        name="weave_neighbor_discordance_qc",
        category=MethodCategory.QC,
        summary="Jensen-Shannon discordance between each observation and its neighbors.",
        params=(
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("flag_fraction", "float", 0.05, minimum=0.0, maximum=1.0),
            ParamSpec("score_key", "str", "weave_neighbor_discordance"),
            ParamSpec("flag_key", "str", "weave_high_neighbor_discordance"),
        ),
        spatial=True,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = _expression_matrix(result, method=self.spec.name)
        neighbors = _spatial_neighbors(result, k=self.params["k"], method=self.spec.name)
        own = _row_compositions(matrix)
        neighborhood = _row_compositions(matrix[neighbors].mean(axis=1))
        midpoint = 0.5 * (own + neighborhood)
        own_term = np.where(own > 0, own * np.log((own + 1e-15) / (midpoint + 1e-15)), 0.0)
        neighbor_term = np.where(
            neighborhood > 0,
            neighborhood * np.log((neighborhood + 1e-15) / (midpoint + 1e-15)),
            0.0,
        )
        scores = 0.5 * (own_term.sum(axis=1) + neighbor_term.sum(axis=1)) / np.log(2.0)
        scores = np.clip(scores, 0.0, 1.0)
        flags, threshold = _quantile_flags(
            scores,
            self.params["flag_fraction"],
            high_is_bad=True,
        )
        result.obs[self.params["score_key"]] = scores
        result.obs[self.params["flag_key"]] = flags
        _record_research_metadata(result, self.spec.name, threshold=threshold, k=self.params["k"])
        return self.finalize(result, step="qc")


@register
class WeaveAdaptiveSaturationQC(Method):
    """Flag low feature saturation relative to dataset-adaptive library depth."""

    spec = _research_spec(
        name="weave_adaptive_saturation_qc",
        category=MethodCategory.QC,
        summary="Detected-feature saturation relative to an adaptive depth envelope.",
        params=(
            ParamSpec("flag_fraction", "float", 0.05, minimum=0.0, maximum=1.0),
            ParamSpec("score_key", "str", "weave_adaptive_saturation"),
            ParamSpec("flag_key", "str", "weave_low_adaptive_saturation"),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = _expression_matrix(result, method=self.spec.name)
        totals = matrix.sum(axis=1)
        detected = np.sum(matrix > 0, axis=1).astype(float)
        positive_totals = totals[totals > 0]
        depth_scale = (
            float(np.median(positive_totals)) / np.log(2.0)
            if positive_totals.size
            else 1.0
        )
        expected = result.n_vars * (1.0 - np.exp(-totals / max(depth_scale, 1e-12)))
        scores = np.divide(detected, expected, out=np.zeros_like(detected), where=expected > 0)
        scores = np.clip(scores, 0.0, 2.0)
        flags, threshold = _quantile_flags(
            scores,
            self.params["flag_fraction"],
            high_is_bad=False,
        )
        result.obs[self.params["score_key"]] = scores
        result.obs[self.params["flag_key"]] = flags
        _record_research_metadata(
            result,
            self.spec.name,
            threshold=threshold,
            depth_scale=depth_scale,
        )
        return self.finalize(result, step="qc")


@register
class WeaveSpatialMedianNormalize(_ResearchNormalization):
    """Normalize against a shrinkage blend of own and local-median depth."""

    spec = _research_spec(
        name="weave_spatial_median_normalize",
        category=MethodCategory.NORMALIZATION,
        summary="Library-size scaling with spatial local-median shrinkage.",
        params=(
            ParamSpec("target_sum", "float", 10000.0, minimum=1e-12),
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("spatial_weight", "float", 0.5, minimum=0.0, maximum=1.0),
        ),
        spatial=True,
    )

    def transform(self, data: SpatialTable, matrix: np.ndarray) -> np.ndarray:
        neighbors = _spatial_neighbors(data, k=self.params["k"], method=self.spec.name)
        totals = matrix.sum(axis=1)
        local_depths = np.concatenate((totals[:, None], totals[neighbors]), axis=1)
        local_median = np.median(local_depths, axis=1)
        weight = float(self.params["spatial_weight"])
        effective_depth = (1.0 - weight) * totals + weight * local_median
        factors = np.divide(
            self.params["target_sum"],
            effective_depth,
            out=np.zeros_like(effective_depth),
            where=effective_depth > 0,
        )
        return matrix * factors[:, None]


@register
class WeaveGraphDiffusionNormalize(_ResearchNormalization):
    """Diffuse log-normalized expression over a spatial kNN graph."""

    spec = _research_spec(
        name="weave_graph_diffusion_normalize",
        category=MethodCategory.NORMALIZATION,
        summary="Bounded spatial diffusion of library-size-normalized expression.",
        params=(
            ParamSpec("target_sum", "float", 10000.0, minimum=1e-12),
            ParamSpec("k", "int", 8, minimum=1),
            ParamSpec("diffusion", "float", 0.25, minimum=0.0, maximum=1.0),
            ParamSpec("steps", "int", 2, minimum=1, maximum=50),
        ),
        spatial=True,
    )

    def transform(self, data: SpatialTable, matrix: np.ndarray) -> np.ndarray:
        neighbors = _spatial_neighbors(data, k=self.params["k"], method=self.spec.name)
        totals = matrix.sum(axis=1, keepdims=True)
        scaled = np.divide(
            matrix * self.params["target_sum"],
            totals,
            out=np.zeros_like(matrix),
            where=totals > 0,
        )
        state = np.log1p(scaled)
        diffusion = float(self.params["diffusion"])
        for _ in range(self.params["steps"]):
            state = (1.0 - diffusion) * state + diffusion * state[neighbors].mean(axis=1)
        return state


@register
class WeaveRankStabilize(_ResearchNormalization):
    """Replace positive expression magnitudes with within-observation percentile ranks."""

    spec = _research_spec(
        name="weave_rank_stabilize",
        category=MethodCategory.NORMALIZATION,
        summary="Tie-aware within-observation rank stabilization with structural zeros.",
        params=(ParamSpec("scale", "float", 1.0, minimum=1e-12),),
    )

    def transform(self, data: SpatialTable, matrix: np.ndarray) -> np.ndarray:
        del data
        transformed = np.zeros_like(matrix)
        for row_index, row in enumerate(matrix):
            positive = row > 0
            count = int(positive.sum())
            if count:
                transformed[row_index, positive] = (
                    rankdata(row[positive], method="average") / count * self.params["scale"]
                )
        return transformed


@register
class WeaveRobustPearsonResidual(_ResearchNormalization):
    """Compute clipped negative-binomial Pearson residuals with gene-frequency shrinkage."""

    spec = _research_spec(
        name="weave_robust_pearson_residual",
        category=MethodCategory.NORMALIZATION,
        summary="Prior-shrunk, clipped negative-binomial Pearson residual normalization.",
        params=(
            ParamSpec("theta", "float", 100.0, minimum=1e-12),
            ParamSpec("gene_prior", "float", 1.0, minimum=0.0),
            ParamSpec("clip", "float", 10.0, minimum=1e-12),
        ),
    )

    def transform(self, data: SpatialTable, matrix: np.ndarray) -> np.ndarray:
        del data
        row_totals = matrix.sum(axis=1, keepdims=True)
        column_totals = matrix.sum(axis=0, keepdims=True)
        grand_total = float(column_totals.sum())
        prior = float(self.params["gene_prior"])
        denominator = grand_total + prior * matrix.shape[1]
        if denominator > 0:
            gene_probability = (column_totals + prior) / denominator
        else:
            gene_probability = np.full((1, matrix.shape[1]), 1.0 / matrix.shape[1])
        expected = row_totals * gene_probability
        variance = expected + np.square(expected) / self.params["theta"]
        residual = np.divide(
            matrix - expected,
            np.sqrt(variance),
            out=np.zeros_like(matrix),
            where=variance > 0,
        )
        return np.clip(residual, -self.params["clip"], self.params["clip"])


__all__ = [
    "WeaveAdaptiveSaturationQC",
    "WeaveGraphDiffusionNormalize",
    "WeaveNeighborDiscordanceQC",
    "WeaveRankStabilize",
    "WeaveRobustPearsonResidual",
    "WeaveSpatialEntropyQC",
    "WeaveSpatialMedianNormalize",
]
