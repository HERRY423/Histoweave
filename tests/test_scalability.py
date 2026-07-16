import json

import numpy as np
from scipy import sparse

from histoweave.benchmark import ScalingConfig, fit_complexity, run_scaling, write_scaling_artifacts
from histoweave.datasets import make_scalable_synthetic


def test_scalable_synthetic_is_sparse_and_annotated():
    data = make_scalable_synthetic(240, 120, n_domains=4, density=0.1, seed=7, chunk_size=50)
    assert sparse.isspmatrix_csr(data.X)
    assert data.X.dtype == np.float32
    assert data.X.shape == (240, 120)
    assert 0.07 < data.X.nnz / (data.X.shape[0] * data.X.shape[1]) <= 0.1
    assert {"domain_truth", "batch"} <= set(data.obs)
    assert data.obsm["spatial"].shape == (240, 2)
    assert len(data.uns["marker_genes"]) == 4


def test_complexity_fit_and_small_isolated_sweep(tmp_path):
    fit = fit_complexity((100, 1000, 10000), (1.0, 100.0, 10000.0))
    assert fit.status == "ok"
    assert abs((fit.exponent or 0.0) - 2.0) < 0.01
    result = run_scaling(ScalingConfig(
        scales=(40, 80), n_genes=40, density=0.1,
        methods=(("qc", "basic_qc"),), per_method_timeout_s=60.0, per_method_mem_cap_gb=4.0,
    ))
    assert len(result.records) == 2
    assert {record.status for record in result.records} <= {"ok", "timeout", "oom", "error"}
    artifacts = write_scaling_artifacts(result, tmp_path)
    assert all(path.exists() for path in artifacts.values())
    summary = json.loads(artifacts["summary_json"].read_text(encoding="utf-8"))
    assert summary["totals"]["cells_measured"] == 2
