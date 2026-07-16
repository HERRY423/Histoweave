"""Shared helpers for external state-of-the-art domain methods."""

from __future__ import annotations

import os
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.mixture import GaussianMixture


def _to_csr(counts) -> sparse.csr_matrix:
    if sparse.issparse(counts):
        return counts.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(counts, dtype=np.float32))


def _top_variable_indices(counts: sparse.csr_matrix, n_genes: int) -> np.ndarray:
    """Select a bounded matrix before an external backend starts."""
    mean = np.asarray(counts.mean(axis=0)).ravel()
    mean_sq = np.asarray(counts.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    n_keep = min(int(n_genes), counts.shape[1])
    order = np.argsort(-variance, kind="stable")[:n_keep]
    return order[variance[order] > 0]


def make_adata(
    counts,
    spatial: np.ndarray,
    gene_names: Sequence[str],
    *,
    n_genes: int = 3000,
    array_coords: np.ndarray | None = None,
) -> ad.AnnData:
    """Build the minimal AnnData contract used by the official packages."""
    matrix = _to_csr(counts)
    keep = _top_variable_indices(matrix, n_genes=n_genes)
    if keep.size < 2:
        raise ValueError("SOTA adapter requires at least two non-constant genes")
    obs = pd.DataFrame(index=[f"spot_{i}" for i in range(matrix.shape[0])])
    if array_coords is not None:
        coords = np.asarray(array_coords)
        if coords.shape != (matrix.shape[0], 2):
            raise ValueError("array_coords must have shape (n_spots, 2)")
        obs["array_row"] = coords[:, 0]
        obs["array_col"] = coords[:, 1]
    var_names = [str(gene_names[i]) for i in keep]
    adata = ad.AnnData(
        X=matrix[:, keep],
        obs=obs,
        var=pd.DataFrame(index=var_names),
        obsm={"spatial": np.asarray(spatial, dtype=np.float32)},
    )
    adata.layers["counts"] = adata.X.copy()
    adata.var_names_make_unique()
    return adata


def torch_device(torch_module):
    """Use CPU by default; opt into CUDA with HISTOWEAVE_SOTA_DEVICE=cuda."""
    requested = os.environ.get("HISTOWEAVE_SOTA_DEVICE", "cpu").lower()
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("HISTOWEAVE_SOTA_DEVICE=cuda but CUDA is unavailable")
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def cluster_embedding(embedding, *, n_domains: int, seed: int) -> np.ndarray:
    """Fixed-q mclust-style downstream clustering for learned embeddings."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < int(n_domains):
        raise ValueError("embedding is incompatible with the requested domain count")
    if not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    labels = GaussianMixture(
        n_components=int(n_domains),
        covariance_type="full",
        n_init=3,
        random_state=int(seed),
        reg_covar=1e-6,
    ).fit_predict(values)
    return labels.astype(int)
