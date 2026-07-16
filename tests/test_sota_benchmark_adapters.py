"""Contracts for the optional SOTA DLPFC benchmark adapters."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "5x15_spatial_aware"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

common = importlib.import_module("adapters._sota_common")


def _counts() -> np.ndarray:
    return np.array(
        [
            [8, 0, 1, 0],
            [7, 0, 2, 0],
            [6, 1, 2, 0],
            [0, 7, 0, 0],
            [0, 8, 0, 0],
            [1, 6, 0, 0],
        ],
        dtype=float,
    )


def test_sota_common_builds_sparse_adata_and_preserves_array_coordinates():
    spatial = np.arange(12, dtype=float).reshape(6, 2)
    array_coords = spatial + 100
    adata = common.make_adata(
        _counts(), spatial, ["g0", "g1", "g2", "constant"], array_coords=array_coords
    )

    assert adata.shape == (6, 3)
    assert adata.obsm["spatial"].shape == (6, 2)
    assert adata.obs[["array_row", "array_col"]].to_numpy().tolist() == array_coords.tolist()
    assert "counts" in adata.layers


def test_fixed_q_embedding_clustering_is_seed_deterministic():
    embedding = np.array(
        [[-2, -2], [-2, -1.8], [-1.8, -2], [2, 2], [2, 1.8], [1.8, 2]], dtype=float
    )
    first = common.cluster_embedding(embedding, n_domains=2, seed=11)
    second = common.cluster_embedding(embedding, n_domains=2, seed=11)

    assert np.array_equal(first, second)
    assert np.unique(first).size == 2


def test_sota_methods_are_wired_to_official_backend_contracts():
    experiment = (BENCHMARK / "experiment_5x15_methods.py").read_text(encoding="utf-8")
    r_driver = (BENCHMARK / "run_bayesspace.R").read_text(encoding="utf-8")
    adapters = BENCHMARK / "adapters"

    for name in ("spagcn", "graphst", "bayesspace", "stagate"):
        assert f'"{name}"' in experiment
        assert (adapters / f"{name}_adapter.py").is_file()
    assert 'import_module("SpaGCN")' in (adapters / "spagcn_adapter.py").read_text()
    assert 'import_module("GraphST")' in (adapters / "graphst_adapter.py").read_text()
    assert 'import_module("STAGATE_pyG")' in (adapters / "stagate_adapter.py").read_text()
    assert "spatialPreprocess" in r_driver
    assert "spatialCluster" in r_driver
    assert "nrep = nrep" in r_driver


def test_benchmark_records_backend_failures_in_long_form_output():
    experiment = (BENCHMARK / "experiment_5x15_methods.py").read_text(encoding="utf-8")
    runner = (BENCHMARK / "sota_runner.py").read_text(encoding="utf-8")
    assert '"status": status' in experiment
    assert '"error": error' in experiment
    assert "HISTOWEAVE_SOTA_TIMEOUT" in runner
    assert "HISTOWEAVE_{method.upper()}_PYTHON" in runner
