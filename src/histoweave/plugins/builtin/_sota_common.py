"""Shared helpers for first-class SOTA spatial-domain plugins.

These utilities keep optional heavy backends out of the import path and avoid
duplicating AnnData construction across SpaGCN / GraphST / STAGATE adapters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse


def to_csr(counts: Any) -> sparse.csr_matrix:
    if sparse.issparse(counts):
        return counts.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(counts, dtype=np.float32))


def top_variable_indices(counts: sparse.csr_matrix, n_genes: int) -> np.ndarray:
    mean = np.asarray(counts.mean(axis=0)).ravel()
    mean_sq = np.asarray(counts.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    n_keep = min(int(n_genes), counts.shape[1])
    order = np.argsort(-variance, kind="stable")[:n_keep]
    return order[variance[order] > 0]


def make_adata(
    counts: Any,
    spatial: np.ndarray,
    gene_names: Sequence[str],
    obs_names: Sequence[str] | None = None,
    *,
    n_genes: int = 3000,
    array_coords: np.ndarray | None = None,
):
    """Build the minimal AnnData contract used by official SOTA packages."""
    from anndata import AnnData

    matrix = to_csr(counts)
    keep = top_variable_indices(matrix, n_genes=n_genes)
    if keep.size < 2:
        raise ValueError("SOTA domain methods require at least two non-constant genes")
    if obs_names is None:
        index = pd.Index([f"spot_{i}" for i in range(matrix.shape[0])], dtype=object)
    else:
        index = pd.Index([str(name) for name in obs_names], dtype=object)
    obs = pd.DataFrame(index=index)
    if array_coords is not None:
        coords = np.asarray(array_coords)
        if coords.shape != (matrix.shape[0], 2):
            raise ValueError("array_coords must have shape (n_spots, 2)")
        obs["array_row"] = coords[:, 0]
        obs["array_col"] = coords[:, 1]
    var_names = [str(gene_names[i]) for i in keep]
    spatial_arr = np.asarray(spatial, dtype=np.float32)
    selected = matrix[:, keep]
    adata = AnnData(
        X=selected,
        obs=obs,
        var=pd.DataFrame(index=var_names),
        obsm=cast(Any, {"spatial": spatial_arr}),
    )
    # Densify or copy sparse counts explicitly for SOTA backends that still
    # assume a concrete array/sparse matrix with .copy().
    if hasattr(selected, "copy"):
        adata.layers["counts"] = selected.copy()
    else:  # pragma: no cover - defensive
        adata.layers["counts"] = np.asarray(selected).copy()
    adata.var_names_make_unique()
    return adata


def cluster_embedding(embedding: Any, *, n_domains: int, seed: int) -> np.ndarray:
    """Fixed-q Gaussian-mixture clustering for learned embeddings."""
    from sklearn.mixture import GaussianMixture

    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < int(n_domains):
        raise ValueError("embedding is incompatible with the requested domain count")
    if not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    labels = GaussianMixture(
        n_components=int(n_domains),
        covariance_type="diag",
        random_state=int(seed),
        n_init=1,
        max_iter=200,
    ).fit_predict(values)
    return np.asarray(labels, dtype=int)


def resolve_n_domains(data, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if data.uns.get("n_domains"):
        return int(data.uns["n_domains"])
    raise ValueError("n_domains not given and uns['n_domains'] is absent")


def torch_device(torch_module):
    import os

    requested = os.environ.get("HISTOWEAVE_SOTA_DEVICE", "cpu").lower()
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("HISTOWEAVE_SOTA_DEVICE=cuda but CUDA is unavailable")
        return torch_module.device("cuda")
    return torch_module.device("cpu")
