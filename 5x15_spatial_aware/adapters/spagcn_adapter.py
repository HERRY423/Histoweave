"""Adapter for the official :mod:`SpaGCN` spatial-domain implementation."""

from __future__ import annotations

import importlib
import random

import numpy as np

from ._sota_common import make_adata


def run(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    seed: int,
    n_domains: int,
    array_coords: np.ndarray | None = None,
    p: float = 0.5,
    max_epochs: int = 200,
) -> np.ndarray:
    """Run the official SpaGCN recipe without histology-image features."""
    try:
        spg = importlib.import_module("SpaGCN")
        torch = importlib.import_module("torch")
        sc = importlib.import_module("scanpy")
    except ImportError as exc:
        raise ImportError(
            "SpaGCN backend is missing; install SpaGCN==1.2.7 in its isolated "
            "benchmark environment"
        ) from exc

    adata = make_adata(X_counts, spatial, gene_names, n_genes=3000)
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    coords = np.asarray(spatial, dtype=float)
    adj = spg.calculate_adj_matrix(x=coords[:, 0], y=coords[:, 1], histology=False)
    length_scale = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    if length_scale is None:
        raise RuntimeError("SpaGCN could not find a graph length scale")
    resolution = spg.search_res(
        adata,
        adj,
        length_scale,
        int(n_domains),
        start=0.7,
        step=0.1,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=int(seed),
        t_seed=int(seed),
        n_seed=int(seed),
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = spg.SpaGCN()
    model.set_l(length_scale)
    model.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=resolution,
        tol=5e-3,
        lr=0.05,
        max_epochs=int(max_epochs),
    )
    labels, _ = model.predict()

    if array_coords is not None:
        grid = np.asarray(array_coords, dtype=float)
        grid_adj = spg.calculate_adj_matrix(x=grid[:, 0], y=grid[:, 1], histology=False)
        labels = spg.refine(
            sample_id=adata.obs_names.tolist(),
            pred=np.asarray(labels).tolist(),
            dis=grid_adj,
            shape="hexagon",
        )
    return np.asarray(labels, dtype=int)
