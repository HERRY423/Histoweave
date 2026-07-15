"""Shared helper: turn an SVG ranking into KMeans domain labels.

Used by SpatialDE, nnSVG and Moran's-I pipelines. The recipe is deliberately
identical across the three so that observed ARI differences reflect the
*ranking* quality, not the downstream clustering.

Recipe (matches user selection Q5, top-N SVG → PCA → KMeans):
    1. Take the top ``n_top`` genes from the caller-supplied ranking.
    2. log1p + per-cell sum-to-1e4 normalise, then z-score per gene.
    3. PCA to ``n_pcs`` components (or ``min(n_top-1, n_pcs)``).
    4. KMeans with ``n_domains`` clusters and the supplied ``seed``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def cluster_from_svg_ranking(
    X_counts,
    ranked_genes: Sequence[str],
    all_genes: Sequence[str],
    n_domains: int,
    seed: int = 42,
    n_top: int = 500,
    n_pcs: int = 20,
) -> np.ndarray:
    """Cluster cells using the top-N genes from an SVG ranking.

    Parameters
    ----------
    X_counts : (n_cells, n_genes) counts matrix (sparse or dense).
    ranked_genes : gene identifiers ordered best-first by the SVG method.
    all_genes : column ordering of ``X_counts``.
    n_domains : number of clusters (typically the true layer count).
    seed : RNG seed used by both PCA and KMeans.
    n_top : how many top-ranked genes to keep.
    n_pcs : PCA components to feed KMeans.

    Returns
    -------
    labels : (n_cells,) int array in ``[0, n_domains)``.
    """
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    # keep the ranking order, drop missing/duplicate names
    top_idx: list[int] = []
    seen: set[int] = set()
    for g in ranked_genes:
        j = gene_to_idx.get(g)
        if j is None or j in seen:
            continue
        seen.add(j)
        top_idx.append(j)
        if len(top_idx) >= n_top:
            break
    if len(top_idx) < 2:
        # SVG ranking gave nothing usable; fall back to random projection
        rng = np.random.default_rng(seed)
        top_idx = list(rng.choice(len(all_genes), size=min(50, len(all_genes)), replace=False))

    X = _to_dense(X_counts)[:, top_idx].astype(float)
    # log1p(cp10k) — matches the rest of the harness
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    X = np.log1p(X / sums * 1e4)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    n_comp = int(min(n_pcs, min(X.shape) - 1))
    if n_comp < 2:
        n_comp = 2 if min(X.shape) >= 2 else 1
    if n_comp < 1:
        return np.zeros(X.shape[0], dtype=int)
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(X)
    labels = KMeans(
        n_clusters=int(n_domains),
        n_init=10,
        random_state=seed,
    ).fit_predict(Z)
    return labels.astype(int)
