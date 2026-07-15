"""Moran's-I-filtered spectral clustering adapter.

A cheap-but-competitive spatial-aware baseline that has no external heavy
dependencies (no R, no Gaussian-process inference). Steps:

1. Compute Moran's I per gene against a k-NN spatial graph.
2. Keep the top-N genes by Moran's I.
3. log1p(cp10k) + z-score + PCA to 20 components.
4. Spectral clustering with ``n_domains`` clusters on the k-NN affinity graph.

The k-NN graph used by Moran's I is reused as the affinity for spectral
clustering, so spatial coordinates directly shape the clustering — hence
"spatial-aware".
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def _knn_weights(spatial: np.ndarray, k: int) -> sparse.csr_matrix:
    """Row-stochastic k-NN weight matrix (self excluded)."""
    n = spatial.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(spatial)
    _, idx = nn.kneighbors(spatial)
    idx = idx[:, 1:]  # drop self
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    data = np.ones_like(rows, dtype=float) / k
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return W


def _morans_i(X: np.ndarray, W: sparse.csr_matrix) -> np.ndarray:
    """Vectorised Moran's I per gene."""
    n = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean  # (n, g)
    denom = (Xc**2).sum(axis=0)  # (g,)
    # numerator: sum_ij w_ij * xc_i * xc_j = xc^T W xc  (per gene, column-wise)
    WXc = W @ Xc  # (n, g)
    numer = (Xc * WXc).sum(axis=0)
    W_sum = W.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (n / W_sum) * (numer / denom)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def run(
    X_counts,
    spatial: np.ndarray,
    seed: int,
    n_domains: int,
    k_geom: int = 15,
    n_top: int = 500,
    n_pcs: int = 20,
    n_prefilter: int = 3000,
) -> np.ndarray:
    X = _to_dense(X_counts).astype(np.float32)
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    Xn = np.log1p(X / sums * 1e4)
    # Dispersion prefilter: Moran's I via vectorised (Xc^T W Xc) is fast but
    # allocates (n_cells, n_genes). Restricting to the most variable genes
    # keeps Moran's I ~O(n * n_prefilter) which is comfortable at 3611 x 3000.
    if Xn.shape[1] > n_prefilter:
        v = Xn.var(axis=0)
        keep = np.argsort(-v)[:n_prefilter]
        keep = keep[v[keep] > 0]
        Xn = Xn[:, keep]
    spatial = np.asarray(spatial, dtype=float)

    W = _knn_weights(spatial, k=k_geom)
    scores = _morans_i(Xn, W)
    keep = np.argsort(-scores)[: max(2, n_top)]
    keep = keep[scores[keep] > 0]  # drop non-spatially-varying genes
    if keep.size < 2:
        # nothing spatial — degrade to random spectral on a dispersion filter
        v = Xn.var(axis=0)
        keep = np.argsort(-v)[: max(2, n_top)]

    Xk = StandardScaler(with_mean=True, with_std=True).fit_transform(Xn[:, keep])
    n_comp = int(min(n_pcs, min(Xk.shape) - 1))
    Z = PCA(n_components=n_comp, random_state=seed).fit_transform(Xk)

    # Symmetrise the k-NN graph for spectral (mutual + weighted)
    W_sym = 0.5 * (W + W.T)

    try:
        labels = SpectralClustering(
            n_clusters=int(n_domains),
            affinity="precomputed",
            random_state=seed,
            assign_labels="kmeans",
            n_init=10,
        ).fit_predict(W_sym)
    except Exception:
        # Fall back to KMeans on PCA (spectral can fail on degenerate graphs)
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=int(n_domains), n_init=10, random_state=seed).fit_predict(Z)
    return labels.astype(int)
