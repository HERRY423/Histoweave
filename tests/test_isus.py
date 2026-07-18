from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from histoweave.benchmark.isus import (
    compute_isus,
    compute_isus_from_table,
    isus_band,
    mi_discrete_continuous,
)


def test_isus_distinguishes_expression_only_and_spatial_signal():
    rng = np.random.default_rng(0)
    n_obs, n_domains = 600, 3
    labels = rng.integers(0, n_domains, size=n_obs)
    centers = rng.normal(size=(n_domains, 8)) * 3.0
    expression = centers[labels] + rng.normal(size=(n_obs, 8))
    expression_only = compute_isus(
        expression,
        rng.uniform(0, 100, size=(n_obs, 2)),
        labels,
        n_pcs=6,
    )

    coordinates = rng.uniform(0, 100, size=(n_obs, 2))
    spatial_labels = np.clip((coordinates[:, 0] // 34).astype(int), 0, 2)
    weak_centers = rng.normal(size=(n_domains, 8)) * 0.35
    weak_expression = weak_centers[spatial_labels] + rng.normal(size=(n_obs, 8))
    spatial = compute_isus(
        weak_expression,
        coordinates,
        spatial_labels,
        n_pcs=6,
    )

    assert expression_only.isus is not None and expression_only.isus < 0.1
    assert expression_only.band == "expression-sufficient"
    assert spatial.isus is not None and spatial.isus > 0.3
    assert spatial.band == "spatial-critical"
    json.dumps(spatial.to_dict(), allow_nan=False)


def test_public_mi_estimator_and_bands():
    labels = np.repeat([0, 1], 20)
    separated = np.r_[np.zeros((20, 1)), np.ones((20, 1)) * 10]
    assert mi_discrete_continuous(separated, labels) > 0
    assert isus_band(None) == "undetermined"
    assert isus_band(0.05) == "expression-sufficient"
    assert isus_band(0.2) == "modest-spatial-signal"
    assert isus_band(0.4) == "spatial-critical"


def test_compute_from_anndata_like_sparse_table():
    rng = np.random.default_rng(4)
    n_obs = 90
    coordinates = rng.uniform(size=(n_obs, 2))
    labels = (coordinates[:, 0] > 0.5).astype(int)
    table = SimpleNamespace(
        X=csr_matrix(rng.poisson(2, size=(n_obs, 12))),
        obs=pd.DataFrame({"domain_truth": labels}),
        obsm={"spatial": coordinates},
        uns={"dataset_name": "sparse-toy"},
    )
    result = compute_isus_from_table(table, n_pcs=5)
    assert result.dataset == "sparse-toy"
    assert result.n_obs == n_obs
    assert result.n_pcs == 5

