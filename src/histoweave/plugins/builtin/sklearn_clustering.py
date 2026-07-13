"""scikit-learn-powered spatial domain detection methods.

These are real, production-quality clustering methods that wrap scikit-learn
behind HistoWeave's plugin interface.  Each method is now a thin declarative
specialisation of :class:`SklearnClusterMethod` — adding a new sklearn
clusterer costs ~12 lines instead of the previous ~50.

Methods registered
------------------
* **kmeans** — K-Means on spatial-PCA embedding (implemented in ``domains.py``
  but listed here for completeness; it shares the same embedding helpers).
* **dbscan** — Density-based spatial clustering (DBSCAN).
* **agglomerative** — Hierarchical agglomerative clustering (Ward linkage).
* **spectral** — Spectral clustering on a spatial k-NN graph.
* **gaussian_mixture** — Gaussian Mixture Model (soft assignments).
* **mean_shift** — Mean Shift (no n_domains required).
* **optics** — OPTICS density-based (handles varying density).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...data import SpatialTable
from ..interfaces import ParamSpec
from ._sklearn_base import (
    SklearnClusterMethod,
    register_sklearn_clusterer,
)


# ---------------------------------------------------------------------------
# DBSCAN — density-based, noise-aware
# ---------------------------------------------------------------------------
@register_sklearn_clusterer(
    name="dbscan",
    clusterer_cls="sklearn.cluster.DBSCAN",
    method_params=(
        ParamSpec("eps", "float", 0.5, "DBSCAN neighbourhood radius."),
        ParamSpec("min_samples", "int", 5, "Min points to form a core point."),
    ),
    param_mapping={"eps": "eps", "min_samples": "min_samples"},
    summary="DBSCAN clustering on PCA + spatial-neighbourhood embedding.",
    wraps="sklearn.cluster.DBSCAN",
)
class DBSCANDomains(SklearnClusterMethod):
    """DBSCAN: naturally handles arbitrarily-shaped domains and labels noise as -1."""


# ---------------------------------------------------------------------------
# Agglomerative — hierarchical, tree-cutting
# ---------------------------------------------------------------------------
@register_sklearn_clusterer(
    name="agglomerative",
    clusterer_cls="sklearn.cluster.AgglomerativeClustering",
    method_params=(
        ParamSpec("n_domains", "int", 3, "Number of clusters to cut the tree at."),
    ),
    param_mapping={"n_domains": "n_clusters"},
    summary="Ward hierarchical clustering on spatial-PCA embedding.",
    wraps="sklearn.cluster.AgglomerativeClustering",
)
class AgglomerativeDomains(SklearnClusterMethod):
    """Agglomerative (Ward): hierarchical tissue organisation via dendrogram cut."""


# ---------------------------------------------------------------------------
# Spectral — graph Laplacian, manifold-respecting
# ---------------------------------------------------------------------------
def _adapt_spectral_kwargs(
    self: SklearnClusterMethod,
    kwargs: dict,
    data: SpatialTable,
    embedding: np.ndarray,
) -> dict:
    """Clamp ``n_neighbors`` and forward ``random_state`` for determinism."""
    n_neighbors = self.params.get("n_neighbors", 12)
    kwargs["n_neighbors"] = min(n_neighbors, data.n_obs - 1)
    kwargs["random_state"] = self.params.get("random_state", 0)
    return kwargs


@register_sklearn_clusterer(
    name="spectral",
    clusterer_cls="sklearn.cluster.SpectralClustering",
    method_params=(
        ParamSpec("n_domains", "int", 3, "Number of clusters."),
    ),
    param_mapping={"n_domains": "n_clusters"},
    static_kwargs={"affinity": "nearest_neighbors", "assign_labels": "kmeans"},
    adapt_kwargs="_adapt_spectral_kwargs",
    summary="Spectral clustering on spatial k-NN graph.",
    wraps="sklearn.cluster.SpectralClustering",
)
class SpectralDomains(SklearnClusterMethod):
    """Spectral: graph-Laplacian respects non-convex tissue manifold geometry."""

    _adapt_spectral_kwargs = _adapt_spectral_kwargs


# ---------------------------------------------------------------------------
# Gaussian Mixture Model — soft assignments
# ---------------------------------------------------------------------------
def _gmm_post_fit(
    data: SpatialTable,
    embedding: np.ndarray,
    clusterer: Any,  # type: ignore[explicit-any]  # sklearn GaussianMixture
) -> None:
    """Store soft (posterior) assignments for downstream uncertainty analysis."""
    data.obsm["domain_posterior"] = clusterer.predict_proba(embedding)


@register_sklearn_clusterer(
    name="gaussian_mixture",
    clusterer_cls="sklearn.mixture.GaussianMixture",
    method_params=(
        ParamSpec("n_domains", "int", 3, "Number of mixture components."),
    ),
    param_mapping={"n_domains": "n_components", "random_state": "random_state"},
    static_kwargs={"n_init": 5},
    post_fit_hook="_gmm_post_fit",
    summary="GMM soft clustering on spatial-PCA embedding.",
    wraps="sklearn.mixture.GaussianMixture",
)
class GaussianMixtureDomains(SklearnClusterMethod):
    """GMM: probabilistic domain memberships, natural for gradual boundaries."""

    _gmm_post_fit = staticmethod(_gmm_post_fit)


# ---------------------------------------------------------------------------
# Mean Shift — automatic cluster count discovery
# ---------------------------------------------------------------------------
@register_sklearn_clusterer(
    name="mean_shift",
    clusterer_cls="sklearn.cluster.MeanShift",
    method_params=(
        ParamSpec(
            "bandwidth", "float|None", None,
            "Kernel bandwidth; None lets sklearn estimate it.",
        ),
    ),
    param_mapping={"bandwidth": "bandwidth"},
    summary="Mean Shift density-based clustering (no n_domains required).",
    wraps="sklearn.cluster.MeanShift",
)
class MeanShiftDomains(SklearnClusterMethod):
    """Mean Shift: discovers domain count automatically from density modes."""


# ---------------------------------------------------------------------------
# OPTICS — varying-density DBSCAN
# ---------------------------------------------------------------------------
@register_sklearn_clusterer(
    name="optics",
    clusterer_cls="sklearn.cluster.OPTICS",
    method_params=(
        ParamSpec("min_samples", "int", 5, "Min points to form a core point."),
    ),
    param_mapping={"min_samples": "min_samples"},
    summary="OPTICS density-based clustering (handles varying density).",
    wraps="sklearn.cluster.OPTICS",
)
class OPTICSDomains(SklearnClusterMethod):
    """OPTICS: reachability-plot clustering for tissues with varying cell density."""


# ---------------------------------------------------------------------------
# Additional scalable / hierarchical baselines for the Figure 3 benchmark
# ---------------------------------------------------------------------------
@register_sklearn_clusterer(
    name="birch",
    clusterer_cls="sklearn.cluster.Birch",
    method_params=(
        ParamSpec("n_domains", "int", 3, "Number of final clusters."),
        ParamSpec("threshold", "float", 0.5, "Radius of each clustering feature."),
        ParamSpec("branching_factor", "int", 50, "Maximum CF-tree branching factor."),
    ),
    param_mapping={
        "n_domains": "n_clusters",
        "threshold": "threshold",
        "branching_factor": "branching_factor",
    },
    summary="BIRCH CF-tree clustering on spatial-PCA embedding.",
    wraps="sklearn.cluster.Birch",
)
class BirchDomains(SklearnClusterMethod):
    """BIRCH: memory-efficient hierarchical clustering for large tissues."""


@register_sklearn_clusterer(
    name="minibatch_kmeans",
    clusterer_cls="sklearn.cluster.MiniBatchKMeans",
    method_params=(
        ParamSpec("n_domains", "int", 3, "Number of clusters."),
        ParamSpec("batch_size", "int", 256, "Mini-batch size."),
    ),
    param_mapping={
        "n_domains": "n_clusters",
        "batch_size": "batch_size",
        "random_state": "random_state",
    },
    static_kwargs={"n_init": 10},
    summary="Mini-batch k-means on spatial-PCA embedding.",
    wraps="sklearn.cluster.MiniBatchKMeans",
)
class MiniBatchKMeansDomains(SklearnClusterMethod):
    """MiniBatchKMeans: scalable centroid baseline for large spatial assays."""


@register_sklearn_clusterer(
    name="bisecting_kmeans",
    clusterer_cls="sklearn.cluster.BisectingKMeans",
    method_params=(
        ParamSpec("n_domains", "int", 3, "Number of clusters."),
    ),
    param_mapping={
        "n_domains": "n_clusters",
        "random_state": "random_state",
    },
    static_kwargs={"n_init": 10},
    summary="Bisecting k-means on spatial-PCA embedding.",
    wraps="sklearn.cluster.BisectingKMeans",
)
class BisectingKMeansDomains(SklearnClusterMethod):
    """BisectingKMeans: divisive hierarchy with deterministic k-means splits."""


# ---------------------------------------------------------------------------
# Deep-learning method stubs (document the wrapping pattern for Phase-2)
# ---------------------------------------------------------------------------
class _DeepLearningStub(SklearnClusterMethod):
    """Abstract base documenting the wrapping pattern for deep-learning methods.

    Real implementations live in separate plugin packages (e.g.
    ``histoweave-banksy``, ``histoweave-stagate``) and register via the
    ``histoweave.plugins`` entry-point group.  The stub pattern below is a
    template contributors follow; the concrete classes are NOT registered
    by default to avoid unmet dependency errors.

    NOTE: this stub cannot be instantiated — it is documentation-only.
    The ``spec`` type annotation is inherited from :class:`Method`.
    """

    # spec is inherited from Method (type annotation only — concrete
    # subclasses must assign a MethodSpec instance).

    def run(self, data):  # type: ignore[override]
        raise NotImplementedError(
            "This is a documentation stub — install the corresponding plugin package"
        )


# Example pattern — copy this into a real plugin package:
#
# @register
# class BANKSYDomains(Method):
#     spec = MethodSpec(
#         name="banksy",
#         category=MethodCategory.DOMAIN_DETECTION,
#         version="0.1.0",
#         summary="BANKSY: Building Agglomerates of Neighbourhoods for Spatial analYsis.",
#         params=(
#             ParamSpec("n_domains", "int", 5, "Number of spatial domains."),
#             ParamSpec("lambda_param", "float", 0.3, "Spatial weight (BANKSY λ)."),
#             ParamSpec("resolution", "float", 0.8, "Leiden clustering resolution."),
#         ),
#         assays=("visium", "xenium", "cosmx"),
#         wraps="BANKSY (Singhal et al., 2024)",
#         language="python",
#     )
#     def run(self, data: SpatialTable) -> SpatialTable:
#         import banksy  # deferred import
#         ...
