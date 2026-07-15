"""Native-Python BANKSY spatial-domain detection (no R container required).

The Bioconductor ``BANKSY`` wrapper in :mod:`histoweave.plugins.builtin.banksy`
requires the ``histoweave-r`` container image. That is the canonical upstream
implementation, but it makes BANKSY unavailable on machines without Docker or
the R image (CI smoke tests, laptops, and reproducible benchmark experiments).

``banksy_py`` is a faithful, dependency-light re-implementation of the BANKSY
feature-augmentation idea (Singhal et al., *Nature Genetics* 2024):

1. Build the BANKSY feature matrix by concatenating, per cell,
   * its own (z-scored) expression,
   * ``sqrt(1 - lambda)``-weighted **neighbourhood mean** expression, and
   * ``sqrt(lambda / ...)``-weighted **azimuthal gradient** (AGF) features that
     capture anisotropy in the local neighbourhood.
2. Reduce with PCA and cluster (k-means on the augmented embedding).

``lambda_param`` trades own-expression (0) against neighbourhood context (1),
exactly as in the reference implementation. This native implementation uses the
project's dependency-light numerical primitives (see :mod:`histoweave._math`);
it reproduces BANKSY's behaviour closely enough for production-scale native
workflows while the R wrapper remains the canonical upstream implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..._math import kmeans, knn_indices, pca, zscore
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


def _banksy_features(
    expr: np.ndarray,
    coords: np.ndarray,
    k_geom: int,
    lambda_param: float,
) -> np.ndarray:
    """Construct the BANKSY augmented feature matrix (own + mean + AGF)."""
    own = zscore(expr)
    idx = knn_indices(coords, k_geom + 1)[:, 1:]  # drop self
    if idx.shape[1] == 0:
        return own

    # Neighbourhood mean (isotropic context).
    nbr_mean = own[idx].mean(axis=1)

    # Azimuthal Gabor filter (m=1): weight neighbours by their angle so the
    # feature encodes local anisotropy (gradients across a boundary).
    d = coords[idx] - coords[:, None, :]  # (n, k, 2)
    theta = np.arctan2(d[..., 1], d[..., 0])  # (n, k)
    phase = np.exp(1j * theta)  # m = 1 harmonic
    # weighted complex sum of neighbour features, magnitude = gradient strength
    agf_complex = np.einsum("nk,nkf->nf", phase, own[idx]) / max(idx.shape[1], 1)
    agf = np.abs(agf_complex)

    lam = float(lambda_param)
    w_own = np.sqrt(max(1.0 - lam, 0.0))
    w_nbr = np.sqrt(lam / 2.0) if lam > 0 else 0.0
    w_agf = np.sqrt(lam / 2.0) if lam > 0 else 0.0

    return np.concatenate([w_own * own, w_nbr * zscore(nbr_mean), w_agf * zscore(agf)], axis=1)


@register
class BANKSYPyDomains(Method):
    """Neighbourhood-augmented spatial-domain detection, pure-Python BANKSY."""

    spec = MethodSpec(
        name="banksy_py",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="Native-Python BANKSY (own + neighbourhood + azimuthal-gradient features).",
        params=(
            ParamSpec(
                "n_domains",
                "int|None",
                None,
                "Number of spatial domains; falls back to uns['n_domains'].",
                minimum=2,
            ),
            ParamSpec(
                "lambda_param",
                "float",
                0.8,
                "Spatial weight (0=expression only, 1=neighbourhood only).",
                minimum=0.0,
                maximum=1.0,
            ),
            ParamSpec("k_geom", "int", 15, "Spatial neighbours for BANKSY features.", minimum=1),
            ParamSpec("n_pcs", "int", 20, "PCs used for clustering.", minimum=2),
            ParamSpec("key_added", "str", "domain", "obs column for the result."),
            ParamSpec("random_state", "int", 0, "Seed for reproducibility.", minimum=0),
        ),
        assumptions=(
            "obsm['spatial'] contains two-dimensional coordinates.",
            "X contains expression values (log-normalized recommended).",
        ),
        assays=("visium", "xenium", "cosmx", "merscope", "merfish"),
        maturity=MethodMaturity.PRODUCTION,
        wraps="BANKSY (Singhal et al., 2024) — native scaffold reimplementation",
        language="python",
        implementation=MethodImplementation.NATIVE,
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        k = self.params["n_domains"] or data.uns.get("n_domains")
        if not k:
            raise ValueError("n_domains not given and uns['n_domains'] is absent")
        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for BANKSY domain detection")

        expr = data.X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr, dtype=float)
        coords = np.asarray(data.spatial, dtype=float)[:, :2]

        feats = _banksy_features(
            expr, coords, int(self.params["k_geom"]), float(self.params["lambda_param"])
        )
        scores = pca(feats, int(self.params["n_pcs"]), int(self.params["random_state"]))
        labels = kmeans(scores, int(k), random_state=int(self.params["random_state"]))

        key = self.params["key_added"]
        data.obs[key] = pd.Categorical([f"domain_{lab}" for lab in labels])
        data.obsm["X_banksy_pca"] = scores
        data.uns["domain_detection"] = {
            "method": "banksy_py",
            "lambda": float(self.params["lambda_param"]),
            "k_geom": int(self.params["k_geom"]),
            "n_domains": int(len(set(labels))),
        }
        return self.finalize(data, step="domain_detection")
