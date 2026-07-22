"""Adapter for the official GraphST graph-contrastive representation model."""

from __future__ import annotations

import importlib

import numpy as np

from ._sota_common import cluster_embedding, make_adata, torch_device


def _model_class(package):
    candidate = getattr(package, "GraphST", None)
    if callable(candidate):
        return candidate
    if candidate is not None and callable(getattr(candidate, "GraphST", None)):
        return candidate.GraphST
    # Some GraphST releases do not re-export the model at the package top level;
    # the ``GraphST.GraphST`` submodule is not auto-imported as an attribute.
    # Import it explicitly and resolve the ``GraphST`` class from there.
    submodule = importlib.import_module("GraphST.GraphST")
    model_cls = getattr(submodule, "GraphST", None)
    if callable(model_cls):
        return model_cls
    raise ImportError("GraphST package does not expose the GraphST model class")


def run(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    seed: int,
    n_domains: int,
    epochs: int = 600,
) -> np.ndarray:
    """Learn a GraphST embedding and cluster it with a fixed-q Gaussian mixture."""
    try:
        package = importlib.import_module("GraphST")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "GraphST backend is missing; install the official "
            "JinmiaoChenLab/GraphST package in its isolated benchmark environment"
        ) from exc

    adata = make_adata(X_counts, spatial, gene_names, n_genes=3000)
    model = _model_class(package)(
        adata,
        device=torch_device(torch),
        epochs=int(epochs),
        random_seed=int(seed),
        datatype="10X",
    )
    result = model.train()
    if "emb" not in result.obsm:
        raise RuntimeError("GraphST completed without adata.obsm['emb']")
    return cluster_embedding(result.obsm["emb"], n_domains=n_domains, seed=seed)
