"""Adapter for the official PyTorch-Geometric STAGATE implementation."""

from __future__ import annotations

import importlib

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ._sota_common import cluster_embedding, make_adata, torch_device


def _adaptive_radius(spatial: np.ndarray, k: int = 6) -> float:
    coords = np.asarray(spatial, dtype=float)
    n_neighbors = min(k + 1, coords.shape[0])
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(coords)
    radius = float(np.median(distances[:, -1]) * 1.05)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("could not derive a positive STAGATE spatial radius")
    return radius


def run(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    seed: int,
    n_domains: int,
    n_epochs: int = 1000,
) -> np.ndarray:
    """Learn the STAGATE embedding and apply fixed-q Gaussian-mixture clustering."""
    try:
        stagate = importlib.import_module("STAGATE_pyG")
        torch = importlib.import_module("torch")
        sc = importlib.import_module("scanpy")
    except ImportError as exc:
        raise ImportError(
            "STAGATE backend is missing; install the official QIFEIDKN/STAGATE_pyG "
            "package in its isolated benchmark environment"
        ) from exc

    adata = make_adata(X_counts, spatial, gene_names, n_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, adata.n_vars))
    stagate.Cal_Spatial_Net(adata, rad_cutoff=_adaptive_radius(spatial))
    result = stagate.train_STAGATE(
        adata,
        n_epochs=int(n_epochs),
        random_seed=int(seed),
        device=torch_device(torch),
        verbose=False,
    )
    if "STAGATE" not in result.obsm:
        raise RuntimeError("STAGATE completed without adata.obsm['STAGATE']")
    return cluster_embedding(result.obsm["STAGATE"], n_domains=n_domains, seed=seed)
