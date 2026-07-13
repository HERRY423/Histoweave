"""Additional dependency-light production methods.

The implementations are deterministic NumPy references with explicit validation and
provenance. They provide useful baselines even when optional ecosystem packages are
not installed.
"""

from __future__ import annotations

import numpy as np

from ...data import SpatialTable
from ..interfaces import (
    Method,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register


def _nonnegative_matrix(data: SpatialTable, method: str) -> np.ndarray:
    matrix = np.asarray(data.X, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{method}: X must be a finite two-dimensional matrix")
    if (matrix < 0).any():
        raise ValueError(f"{method}: X must contain non-negative values")
    return matrix


@register
class LibrarySizeQC(Method):
    spec = MethodSpec(
        name="library_size_qc",
        category=MethodCategory.QC,
        version="1.0.0",
        summary="Library-size metric and configurable low-count flag.",
        params=(ParamSpec("min_counts", "float", 500.0, minimum=0.0),),
        maturity=MethodMaturity.PRODUCTION,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        totals = _nonnegative_matrix(result, self.spec.name).sum(axis=1)
        result.obs["total_counts"] = totals
        result.obs["low_library_size"] = totals < self.params["min_counts"]
        return self.finalize(result)


@register
class GeneComplexityQC(Method):
    spec = MethodSpec(
        name="gene_complexity_qc",
        category=MethodCategory.QC,
        version="1.0.0",
        summary="Detected-feature metric and low-complexity flag.",
        params=(ParamSpec("min_genes", "int", 100, minimum=0),),
        maturity=MethodMaturity.PRODUCTION,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        detected = (_nonnegative_matrix(result, self.spec.name) > 0).sum(axis=1)
        result.obs["detected_genes"] = detected
        result.obs["low_gene_complexity"] = detected < self.params["min_genes"]
        return self.finalize(result)


@register
class MitochondrialQC(Method):
    spec = MethodSpec(
        name="mitochondrial_qc",
        category=MethodCategory.QC,
        version="1.0.0",
        summary="Mitochondrial expression fraction and high-fraction flag.",
        params=(
            ParamSpec("prefix", "str", "MT-"),
            ParamSpec("max_fraction", "float", 0.20, minimum=0.0, maximum=1.0),
        ),
        maturity=MethodMaturity.PRODUCTION,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = _nonnegative_matrix(result, self.spec.name)
        genes = result.var.index.astype(str).str.upper()
        mask = np.asarray(genes.str.startswith(self.params["prefix"].upper()))
        totals = matrix.sum(axis=1)
        mito = matrix[:, mask].sum(axis=1) if mask.any() else np.zeros(result.n_obs)
        fraction = np.divide(mito, totals, out=np.zeros_like(mito), where=totals > 0)
        result.obs["mitochondrial_fraction"] = fraction
        result.obs["high_mitochondrial_fraction"] = fraction > self.params["max_fraction"]
        return self.finalize(result)


class _Transform(Method):
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        result.X = self.transform(_nonnegative_matrix(result, self.spec.name))
        return self.finalize(result)


@register
class LibrarySizeScale(_Transform):
    spec = MethodSpec(
        name="library_size_scale",
        category=MethodCategory.NORMALIZATION,
        version="1.0.0",
        summary="Scale every observation to a fixed library size.",
        params=(ParamSpec("target_sum", "float", 10000.0, minimum=1e-12),),
        maturity=MethodMaturity.PRODUCTION,
    )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        totals = matrix.sum(axis=1, keepdims=True)
        return np.divide(
            matrix * self.params["target_sum"],
            totals,
            out=np.zeros_like(matrix),
            where=totals > 0,
        )


@register
class CLRPerCell(_Transform):
    spec = MethodSpec(
        name="clr_per_cell",
        category=MethodCategory.NORMALIZATION,
        version="1.0.0",
        summary="Centered log-ratio transform with a stable pseudocount.",
        params=(ParamSpec("pseudocount", "float", 1.0, minimum=1e-12),),
        maturity=MethodMaturity.PRODUCTION,
    )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        logged = np.log(matrix + self.params["pseudocount"])
        return logged - logged.mean(axis=1, keepdims=True)


@register
class SquareRootTransform(_Transform):
    spec = MethodSpec(
        name="sqrt_transform",
        category=MethodCategory.NORMALIZATION,
        version="1.0.0",
        summary="Variance-stabilizing square-root count transform.",
        maturity=MethodMaturity.PRODUCTION,
    )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.sqrt(matrix)


@register
class ArcsinhTransform(_Transform):
    spec = MethodSpec(
        name="arcsinh_transform",
        category=MethodCategory.NORMALIZATION,
        version="1.0.0",
        summary="Cofactor-scaled inverse hyperbolic sine transform.",
        params=(ParamSpec("cofactor", "float", 5.0, minimum=1e-12),),
        maturity=MethodMaturity.PRODUCTION,
    )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.arcsinh(matrix / self.params["cofactor"])


@register
class TFIDFNormalization(_Transform):
    spec = MethodSpec(
        name="tfidf_l2",
        category=MethodCategory.NORMALIZATION,
        version="1.0.0",
        summary="TF-IDF weighting followed by per-observation L2 normalization.",
        maturity=MethodMaturity.PRODUCTION,
    )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        row_sum = matrix.sum(axis=1, keepdims=True)
        tf = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
        document_frequency = (matrix > 0).sum(axis=0)
        idf = np.log1p(matrix.shape[0] / (1.0 + document_frequency))
        weighted = tf * idf
        norm = np.linalg.norm(weighted, axis=1, keepdims=True)
        return np.divide(weighted, norm, out=np.zeros_like(weighted), where=norm > 0)


def _spatial_neighbors(data: SpatialTable, k: int) -> np.ndarray:
    coords = data.spatial
    if coords is None:
        raise ValueError("obsm['spatial'] is required")
    coords = np.asarray(coords, dtype=float)
    if data.n_obs < 2 or coords.shape[0] != data.n_obs or not np.isfinite(coords).all():
        raise ValueError("spatial coordinates must be finite and align with observations")
    k = min(k, data.n_obs - 1)
    squared = ((coords[:, None, :2] - coords[None, :, :2]) ** 2).sum(axis=2)
    np.fill_diagonal(squared, np.inf)
    return np.argpartition(squared, kth=k - 1, axis=1)[:, :k]


def _rank_svg(result: SpatialTable, key: str, scores: np.ndarray, n_top: int) -> None:
    order = np.argsort(scores)[::-1][: min(n_top, result.n_vars)]
    result.var[key] = scores
    result.uns[key] = {
        "top_genes": [
            {"gene": str(result.var.index[index]), key: float(scores[index])} for index in order
        ]
    }


@register
class GearysCSVG(Method):
    spec = MethodSpec(
        name="gearys_c",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        version="1.0.0",
        summary="Per-gene Geary C converted to a positive spatial-association score.",
        params=(
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("n_top", "int", 50, minimum=1),
        ),
        maturity=MethodMaturity.PRODUCTION,
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = np.asarray(result.X, dtype=float)
        neighbors = _spatial_neighbors(result, self.params["k"])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        denominator = np.square(centered).sum(axis=0)
        differences = matrix[:, None, :] - matrix[neighbors, :]
        numerator = np.square(differences).sum(axis=(0, 1))
        weight_sum = neighbors.size
        geary = np.divide(
            (result.n_obs - 1) * numerator,
            2.0 * weight_sum * denominator,
            out=np.ones(result.n_vars, dtype=float),
            where=denominator > 0,
        )
        score = 1.0 - geary
        _rank_svg(result, "gearys_c_score", score, self.params["n_top"])
        return self.finalize(result)


@register
class SpatialVarianceRatio(Method):
    spec = MethodSpec(
        name="spatial_variance_ratio",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        version="1.0.0",
        summary="Variance retained after local spatial averaging.",
        params=(
            ParamSpec("k", "int", 6, minimum=1),
            ParamSpec("n_top", "int", 50, minimum=1),
        ),
        maturity=MethodMaturity.PRODUCTION,
        modalities=("expression", "spatial"),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        result = data.copy()
        matrix = np.asarray(result.X, dtype=float)
        neighbors = _spatial_neighbors(result, self.params["k"])
        local_mean = matrix[neighbors].mean(axis=1)
        total_variance = matrix.var(axis=0)
        local_variance = local_mean.var(axis=0)
        score = np.divide(
            local_variance,
            total_variance,
            out=np.zeros(result.n_vars, dtype=float),
            where=total_variance > 0,
        )
        _rank_svg(result, "spatial_variance_ratio", score, self.params["n_top"])
        return self.finalize(result)
