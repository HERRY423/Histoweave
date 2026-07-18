"""Unit tests for SOTA shared helpers."""

import numpy as np
from scipy import sparse

from histoweave.plugins.builtin._sota_common import (
    cluster_embedding,
    make_adata,
    top_variable_indices,
)


def test_top_variable_indices_prefers_high_variance() -> None:
    matrix = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 10.0], [0.0, 100.0]]))
    keep = top_variable_indices(matrix, n_genes=1)
    assert keep.tolist() == [1]


def test_make_adata_selects_variable_genes() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(30, 50)).astype(float)
    counts[:, 0] = 0.0
    adata = make_adata(
        counts,
        np.column_stack([np.arange(30), np.zeros(30)]),
        [f"g{i}" for i in range(50)],
        n_genes=10,
    )
    assert adata.n_obs == 30
    assert adata.n_vars <= 10
    assert "spatial" in adata.obsm


def test_cluster_embedding_returns_k_labels() -> None:
    rng = np.random.default_rng(1)
    emb = np.vstack([rng.normal(0, 0.1, size=(20, 4)), rng.normal(3, 0.1, size=(20, 4))])
    labels = cluster_embedding(emb, n_domains=2, seed=0)
    assert labels.shape == (40,)
    assert len(set(labels.tolist())) == 2
