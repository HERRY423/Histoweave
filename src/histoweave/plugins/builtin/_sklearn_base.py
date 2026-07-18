"""Parameterised sklearn-clustering base that compresses 7 near-identical domain-detection
methods into a single template.

Before this module every sklearn clustering method duplicated ~45 lines of embedding,
fitting, labeling, and provenance boilerplate.  The :class:`SklearnClusterMethod` base
collapses those steps into a generic ``run``; subclasses only supply a clusterer class,
static kwargs, and a parameter-name mapping.  The :func:`register_sklearn_clusterer`
decorator further compresses a new method to ~12 lines by auto-generating both the spec
and the class registration.

Maturity
--------
This is an **internal implementation detail** of the builtin plugin package — it is
NOT part of HistoWeave's public plugin API.  Third-party plugins should subclass
:class:`~histoweave.plugins.interfaces.Method` directly.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from ..._math import zscore
from ...data import SpatialTable
from ..interfaces import (
    BackendRequirement,
    Method,
    MethodCategory,
    MethodImplementation,
    MethodSpec,
    ParamSpec,
)
from ..registry import register

# ---------------------------------------------------------------------------
# Shared embedding helpers (same as before, kept here so the base is self-contained)
# ---------------------------------------------------------------------------


def _spatial_embedding(
    data: SpatialTable,
    n_pcs: int = 15,
    n_neighbors: int = 12,
    spatial_weight: float = 0.3,
    random_state: int = 0,
) -> np.ndarray:
    """PCA + optional spatial neighbourhood smoothing — identical to the old helpers."""
    from ..._math import neighborhood_mean, pca

    feats = zscore(data.X)
    scores = pca(feats, n_pcs, random_state)
    w = float(spatial_weight)
    coords = data.spatial
    if coords is not None and w > 0:
        nbr = neighborhood_mean(scores, coords, n_neighbors)
        return (1 - w) * zscore(scores) + w * zscore(nbr)
    if coords is None and w > 0:
        warnings.warn(
            f"spatial_weight={w} > 0 but obsm['spatial'] is missing; "
            "falling back to expression-only clustering",
            stacklevel=2,
        )
    return zscore(scores)


def _categorical_labels(labels: np.ndarray) -> pd.Categorical:
    """Convert integer cluster labels to categorical strings (noise = -1 → domain_-1)."""
    return cast(pd.Categorical, pd.Categorical([f"domain_{int(lab)}" for lab in labels]))


# ---------------------------------------------------------------------------
# Common parameter specs — every sklearn clusterer shares these
# ---------------------------------------------------------------------------

_COMMON_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("n_pcs", "int", 15, "PCs for the embedding."),
    ParamSpec("n_neighbors", "int", 12, "Spatial neighbours for smoothing."),
    ParamSpec("spatial_weight", "float", 0.3, "0=expression only, 1=space only."),
    ParamSpec("key_added", "str", "domain", "obs column for cluster labels."),
    ParamSpec("random_state", "int", 0, "Seed for PCA reproducibility."),
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SklearnClusterMethod(Method):
    """Generic domain-detection method backed by an sklearn clusterer.

    Subclasses override **class-level** attributes to declare what differs
    between clusterers; the shared ``run`` implementation handles embedding,
    fitting, labelling, and provenance in one place.

    Class Attributes (set by subclasses or the decorator)
    -----------------------------------------------------
    _clusterer_import : str
        Fully-qualified import path (``"sklearn.cluster.SpectralClustering"``).
    _static_kwargs : dict
        Kwargs passed to the clusterer constructor on every call (e.g.
        ``{"affinity": "nearest_neighbors"}``).
    _param_mapping : dict[str, str]
        Map user-facing parameter names → clusterer constructor argument names
        (e.g. ``{"n_domains": "n_clusters"}``).
    _post_fit_hook : str | None
        Name of an optional method (on the subclass) called as
        ``_post_fit(data, embedding, fitted_clusterer)`` after fitting.
    """

    # --- class-level declarations (set by subclasses or the decorator) -------
    _clusterer_import: str = ""
    _static_kwargs: dict[str, Any] = {}
    _param_mapping: dict[str, str] = {}
    _post_fit_hook: str | None = None

    def run(self, data: SpatialTable) -> SpatialTable:
        """Fit the sklearn clusterer on a spatial-PCA embedding and label cells."""

        clusterer_cls = _resolve_import(self._clusterer_import)
        data = data.copy()

        # 1. Build the joint expression + spatial embedding -----------------
        embedding = _spatial_embedding(
            data,
            n_pcs=self.params.get("n_pcs", 15),
            n_neighbors=self.params.get("n_neighbors", 12),
            spatial_weight=self.params.get("spatial_weight", 0.3),
            random_state=self.params.get("random_state", 0),
        )

        # 2. Assemble clusterer constructor kwargs --------------------------
        kwargs = dict(self._static_kwargs)
        for user_key, sklearn_key in self._param_mapping.items():
            kwargs[sklearn_key] = self.params[user_key]
        kwargs = self._adapt_clusterer_kwargs(kwargs, data, embedding)

        # 3. Fit + predict ------------------------------------------------
        clusterer = clusterer_cls(**kwargs)
        labels = clusterer.fit_predict(embedding)

        # 4. Store results -------------------------------------------------
        data.obs[self.params.get("key_added", "domain")] = _categorical_labels(labels)
        data.obsm["X_pca"] = embedding

        # 5. Optional post-fit hook (e.g. GMM soft assignments) -------------
        if self._post_fit_hook is not None:
            hook = getattr(self, self._post_fit_hook)
            hook(data, embedding, clusterer)

        return self.finalize(data, step="domain_detection")

    # -- hooks that individual specialisations can override ------------------

    def _adapt_clusterer_kwargs(
        self,
        kwargs: dict[str, Any],
        data: SpatialTable,
        embedding: np.ndarray,
    ) -> dict[str, Any]:
        """Override point: adjust kwargs before clusterer construction.

        The default is a no-op.  Override in a subclass (or set via the
        decorator's ``adapt_kwargs`` parameter) to handle special cases like
        SpectralClustering's ``n_neighbors`` clamping.
        """
        return kwargs


# ---------------------------------------------------------------------------
# Decorator that auto-generates a registered sklearn-clustering method
# ---------------------------------------------------------------------------


def register_sklearn_clusterer(
    *,
    name: str,
    clusterer_cls: str,
    method_params: tuple[ParamSpec, ...] = (),
    param_mapping: dict[str, str] | None = None,
    static_kwargs: dict[str, Any] | None = None,
    summary: str = "",
    wraps: str | None = None,
    assumptions: tuple[str, ...] = (),
    adapt_kwargs: str | None = None,
    post_fit_hook: str | None = None,
) -> Callable[[type], type]:
    """Class decorator that registers an sklearn-clustering domain-detection method.

    The decorated class body can be empty (``pass``); all logic lives in the
    :class:`SklearnClusterMethod` base.  This reduces a new sklearn method from
    ~50 lines down to ~12:

    .. code-block:: python

        @register_sklearn_clusterer(
            name="spectral",
            clusterer_cls="sklearn.cluster.SpectralClustering",
            static_kwargs={"affinity": "nearest_neighbors", "assign_labels": "kmeans"},
            param_mapping={"n_domains": "n_clusters"},
            summary="Spectral clustering on spatial k-NN graph.",
            wraps="sklearn.cluster.SpectralClustering",
        )
        class SpectralDomains(SklearnClusterMethod):
            pass

    Parameters
    ----------
    name : str
        Registered method name (the ``spec.name`` and registry key).
    clusterer_cls : str
        Fully-qualified import path for the sklearn clusterer class.
    method_params : tuple of ParamSpec
        Extra user-facing parameters, on top of ``_COMMON_PARAMS``.
    param_mapping : dict
        Map user-facing param names → clusterer constructor arg names.
    static_kwargs : dict
        Kwargs always passed to the clusterer constructor.
    summary : str
        One-line description for ``spec.summary``.
    wraps : str or None
        What the method wraps (``spec.wraps``).
    assumptions : tuple of str
        Stated assumptions for the method.
    adapt_kwargs : str or None
        Name of a method on the class that adjusts kwargs before construction
        (signature: ``(kwargs, data, embedding) -> kwargs``).
    post_fit_hook : str or None
        Name of a method on the class called after fitting
        (signature: ``(data, embedding, fitted_clusterer) -> None``).
    """

    static_kwargs = dict(static_kwargs or {})
    param_mapping = dict(param_mapping or {})

    def _decorate(cls: type[Any]) -> type[Any]:
        # Build the full parameter list: method-specific params + common params.
        # Reverse order so method-specific params appear first in documentation.
        specific_names = {p.name for p in method_params}
        all_params = tuple(method_params) + tuple(
            p for p in _COMMON_PARAMS if p.name not in specific_names
        )

        spec = MethodSpec(
            name=name,
            category=MethodCategory.DOMAIN_DETECTION,
            version="0.1.0",
            summary=summary or f"{name} clustering on spatial-PCA embedding.",
            params=all_params,
            assumptions=assumptions or ("scikit-learn installed.",),
            wraps=wraps or clusterer_cls,
            implementation=MethodImplementation.EXTERNAL,
            backends=(BackendRequirement("scikit-learn", ">=1.3", "scanpy"),),
        )

        # Attach configuration to the class (or its parent via direct assignment).
        cls.spec = spec
        cls._clusterer_import = clusterer_cls
        cls._static_kwargs = static_kwargs
        cls._param_mapping = param_mapping
        if adapt_kwargs is not None:
            cls._adapt_clusterer_kwargs = getattr(cls, adapt_kwargs)
        if post_fit_hook is not None:
            cls._post_fit_hook = post_fit_hook

        return register(cls)

    return _decorate


def _resolve_import(dotted_path: str) -> type:
    """Import and return the class at *dotted_path* (``pkg.mod.Cls``)."""
    module_path, _, cls_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
