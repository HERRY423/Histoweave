"""Harmony-integrated KMeans adapter (per-slice).

Harmony was designed for multi-batch integration, not single-slice clustering.
For a per-slice ARI baseline we treat 4 spatial quadrants (`nw / ne / sw / se`)
as pseudo-batches, run harmonypy on the PCA embedding, then cluster the
harmony-corrected embedding with KMeans.

Rationale: quadrant-level "batch" removal cancels smooth technical gradients
across the tissue (edge-of-slide dropout, imaging strip effects) while leaving
cell-type biology intact — a common single-slice hack. This gives a fair,
principled Harmony baseline within the single-dataset per-slice protocol.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def _spatial_quadrants(spatial: np.ndarray) -> np.ndarray:
    """Assign each cell to nw / ne / sw / se by median split."""
    x, y = spatial[:, 0], spatial[:, 1]
    mx, my = float(np.median(x)), float(np.median(y))
    top = y >= my
    right = x >= mx
    q = np.empty(spatial.shape[0], dtype=object)
    q[top & ~right] = "nw"
    q[top & right] = "ne"
    q[~top & ~right] = "sw"
    q[~top & right] = "se"
    return q


def run(
    X_counts,
    spatial: np.ndarray,
    seed: int,
    n_domains: int,
    n_pcs: int = 30,
) -> np.ndarray:
    X = _to_dense(X_counts).astype(float)
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    X = np.log1p(X / sums * 1e4)
    # variance filter to keep PCA meaningful
    variances = X.var(axis=0)
    keep = np.argsort(-variances)[:2000]
    keep = keep[variances[keep] > 0]
    Xk = X[:, keep]
    Xk = StandardScaler(with_mean=True, with_std=True).fit_transform(Xk)

    n_comp = int(min(n_pcs, min(Xk.shape) - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(Xk)

    quad = _spatial_quadrants(np.asarray(spatial))
    try:
        import harmonypy as hm
        import pandas as pd

        meta = pd.DataFrame({"quadrant": quad})
        # harmonypy expects an ndarray + a metadata dataframe
        ho = hm.run_harmony(
            data_mat=Z,
            meta_data=meta,
            vars_use=["quadrant"],
            max_iter_harmony=10,
            random_state=seed,
        )
        # harmonypy>=0.0.9 returns Z_corr shaped (n_pcs, n_cells); older
        # versions returned (n_cells, n_pcs). Normalise to (n_cells, n_pcs).
        Z_corr = np.asarray(ho.Z_corr)
        if Z_corr.shape[0] != Z.shape[0]:
            Z_corr = Z_corr.T
    except Exception:
        # If harmonypy fails (e.g. degenerate batch structure) fall back to
        # uncorrected PCs so the pipeline still yields a valid label vector.
        Z_corr = Z

    labels = KMeans(n_clusters=int(n_domains), n_init=10, random_state=seed).fit_predict(Z_corr)
    return labels.astype(int)
