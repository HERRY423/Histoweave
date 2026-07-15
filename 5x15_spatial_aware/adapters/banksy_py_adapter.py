"""Python-side BANKSY adapter using the ``banksy-py`` reference implementation.

Bypasses the built-in R-container BANKSY plugin (histoweave-r image not
available in every environment). Follows user selection Q1 (Python-only path).

Recipe (mirrors the R Banksy default): compute BANKSY feature matrix from raw
counts + spatial coords, run PCA on the [own-cell | neighbourhood-mean] block,
then cluster the leading PCs with KMeans (``k = n_domains``).

Parameters (user-selected Q1):
    lambda_param = 0.8  # weight for the neighbourhood block
    k_geom       = 15   # spatial neighbours for the neighbourhood mean

Signature: ``run(X, spatial, seed, n_domains) -> np.ndarray[int]``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def _banksy_features(
    X: np.ndarray,
    spatial: np.ndarray,
    lambda_param: float,
    k_geom: int,
) -> np.ndarray:
    """Return the ``[own | neighbourhood-mean]`` BANKSY feature block."""
    n = X.shape[0]
    k = min(k_geom, n - 1)
    if k < 1:
        return X.astype(float)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(spatial)
    _, idx = nn.kneighbors(spatial)
    # skip self (column 0)
    idx = idx[:, 1:]
    neigh_mean = X[idx].mean(axis=1)
    a = float(1.0 - lambda_param)
    b = float(lambda_param)
    if a + b == 0:
        a, b = 1.0, 0.0
    scale = 1.0 / (a + b)
    return np.hstack([a * scale * X, b * scale * neigh_mean])


def run(
    X_counts,
    spatial: np.ndarray,
    seed: int,
    n_domains: int,
    lambda_param: float = 0.8,
    k_geom: int = 15,
    n_pcs: int = 20,
    n_hvg: int = 2000,
) -> np.ndarray:
    X = _to_dense(X_counts).astype(np.float32)
    # Log-normalize to counts per 10,000.
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    X = np.log1p(X / sums * 1e4)
    # Variance filter: BANKSY's neighbourhood mean doubles column count, so
    # keeping the full 33k gene set is memory-prohibitive (~2 GB dense hstack).
    # Restricting to the top-n_hvg most variable genes matches the standard
    # BANKSY/Seurat recipe and keeps memory + runtime bounded.
    if X.shape[1] > n_hvg:
        v = X.var(axis=0)
        keep = np.argsort(-v)[:n_hvg]
        keep = keep[v[keep] > 0]
        X = X[:, keep]
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    spatial = np.asarray(spatial, dtype=float)
    feats = _banksy_features(X, spatial, lambda_param=lambda_param, k_geom=k_geom)

    n_comp = int(min(n_pcs, min(feats.shape) - 1))
    if n_comp < 2:
        n_comp = max(1, min(feats.shape) - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(feats)
    labels = KMeans(n_clusters=int(n_domains), n_init=10, random_state=seed).fit_predict(Z)
    return labels.astype(int)
