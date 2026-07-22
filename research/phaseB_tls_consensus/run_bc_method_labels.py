"""Phase B: produce spatial-domain labels for the breast-cancer Visium slice
from one method, using the SAME adapters as the DLPFC benchmark.

Runs ONE method in whatever interpreter invokes it (so heavy methods run in
their isolated env python). Writes labels_<method>.npy + meta json.

Usage:
    <env_python> run_bc_method_labels.py <method> <bc_h5ad> <out_dir> <n_domains> [seed]

method in: stagate, graphst, kmeans, gmm, spectral, agglomerative
(sklearn methods are spatial-weighted: features = z(HVG PCA) concat sw*z(xy))
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
ADAPTERS = REPO / "5x15_spatial_aware" / "adapters"
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(REPO / "src"))


def _load(bc_h5ad: str):
    import anndata as ad

    a = ad.read_h5ad(bc_h5ad)
    X = a.X
    # counts expected in X (raw integers per Phase B build); fall back to layer
    if "counts" in a.layers:
        X = a.layers["counts"]
    spatial = np.asarray(a.obsm["spatial"], dtype=float)
    genes = a.var_names.tolist()
    return a, X, spatial, genes


def _spatial_weighted_features(X, spatial, seed, sw=0.3, n_pcs=30):
    import anndata as ad
    import scanpy as sc
    from sklearn.preprocessing import StandardScaler

    ann = ad.AnnData(X=X.copy())
    ann.var_names = [f"g{i}" for i in range(ann.n_vars)]
    sc.pp.normalize_total(ann, target_sum=1e4)
    sc.pp.log1p(ann)
    sc.pp.highly_variable_genes(ann, n_top_genes=3000, flavor="seurat")
    ann = ann[:, ann.var.highly_variable].copy()
    sc.pp.scale(ann, max_value=10)
    sc.tl.pca(ann, n_comps=n_pcs, random_state=seed)
    expr = StandardScaler().fit_transform(ann.obsm["X_pca"])
    xy = StandardScaler().fit_transform(spatial)
    return np.hstack([expr, sw * xy])


def main() -> None:
    method = sys.argv[1]
    bc_h5ad = sys.argv[2]
    out_dir = Path(sys.argv[3])
    n_domains = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out_dir.mkdir(parents=True, exist_ok=True)

    a, X, spatial, genes = _load(bc_h5ad)
    n = X.shape[0]

    if method == "stagate":
        import stagate_adapter

        labels = stagate_adapter.run(X, spatial, genes, seed=seed, n_domains=n_domains)
    elif method == "graphst":
        import graphst_adapter

        labels = graphst_adapter.run(X, spatial, genes, seed=seed, n_domains=n_domains)
    elif method in {"kmeans", "gmm", "spectral", "agglomerative"}:
        feats = _spatial_weighted_features(X, spatial, seed, sw=0.3)
        if method == "kmeans":
            from sklearn.cluster import KMeans

            labels = KMeans(n_clusters=n_domains, random_state=seed, n_init=10).fit_predict(feats)
        elif method == "gmm":
            from sklearn.mixture import GaussianMixture

            labels = GaussianMixture(
                n_components=n_domains, random_state=seed, covariance_type="full", max_iter=200
            ).fit_predict(feats)
        elif method == "spectral":
            from sklearn.cluster import SpectralClustering

            labels = SpectralClustering(
                n_clusters=n_domains,
                random_state=seed,
                affinity="nearest_neighbors",
                n_neighbors=15,
                assign_labels="kmeans",
            ).fit_predict(feats)
        else:
            from sklearn.cluster import AgglomerativeClustering

            labels = AgglomerativeClustering(n_clusters=n_domains).fit_predict(feats)
    else:
        raise SystemExit(f"unknown method {method!r}")

    labels = np.asarray(labels).astype(int)
    if labels.shape[0] != n:
        raise SystemExit(f"label length {labels.shape[0]} != n_spots {n}")
    np.save(out_dir / f"labels_{method}.npy", labels)
    (out_dir / f"labels_{method}.json").write_text(
        json.dumps(
            {
                "method": method,
                "n_domains": int(n_domains),
                "seed": seed,
                "n_spots": int(n),
                "n_labels": int(len(set(labels.tolist()))),
            }
        )
    )
    _log(f"[{method}] wrote labels: {len(set(labels.tolist()))} domains over {n} spots")


if __name__ == "__main__":
    main()
