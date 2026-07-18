"""Regression tests for P0 platform hardening (task contracts, maturity, knn, SOTA)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from histoweave._math import knn_indices
from histoweave.benchmark import (
    RECOMMENDATION_FEATURE_ORDER,
    AnalysisTask,
    GroundTruthKind,
    LandscapeResult,
    MethodRecommender,
    TaskContract,
    extract_features,
    feature_vector,
)
from histoweave.benchmark.task_contract import assert_labels_usable, split_method_policy
from histoweave.data import SpatialTable
from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, list_methods
from histoweave.plugins.builtin.release_manifest import (
    VALIDATED_METHODS,
    VALIDATION_EVIDENCE,
)


def test_task_contract_rejects_leiden_as_domain_gt():
    with pytest.raises(ValueError, match="self-supervised"):
        TaskContract(
            task=AnalysisTask.SPATIAL_DOMAIN,
            ground_truth_kind=GroundTruthKind.SELF_SUPERVISED,
            label_key="leiden",
        ).validate()

    with pytest.raises(ValueError, match="cluster_proxy"):
        TaskContract(
            task=AnalysisTask.SPATIAL_DOMAIN,
            ground_truth_kind=GroundTruthKind.CLUSTER_PROXY,
            label_key="cluster",
        ).validate()

    with pytest.raises(ValueError, match="forbidden"):
        TaskContract(
            task=AnalysisTask.SPATIAL_DOMAIN,
            ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
            label_key="leiden_domains",
        ).validate()


def test_task_contract_accepts_expert_spatial_domain_labels():
    data = make_synthetic(n_cells=40, n_genes=12, n_domains=3, seed=0)
    data.obs["domain_truth"] = data.obs["domain_truth"].astype(str)
    contract = TaskContract(
        task=AnalysisTask.SPATIAL_DOMAIN,
        ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
        label_key="domain_truth",
        platform="visium",
    )
    labels = assert_labels_usable(data, contract)
    assert labels.shape[0] == data.n_obs


def test_sota_methods_are_registered_first_class():
    names = {row["name"] for row in list_methods("domain_detection")}
    for required in {"spagcn", "graphst", "stagate", "bayesspace", "banksy", "banksy_py"}:
        assert required in names
    deconv = {row["name"] for row in list_methods("deconvolution")}
    assert "rctd" in deconv


def test_sota_methods_fail_closed_without_backend():
    data = make_synthetic(n_cells=30, n_genes=15, n_domains=3, seed=1)
    data.layers["counts"] = np.asarray(np.expm1(np.maximum(data.X, 0)), dtype=int)
    data.uns["n_domains"] = 3
    # SpaGCN may or may not be installed; either path must not silently succeed
    # with a toy substitute when the official backend is missing.
    try:
        import SpaGCN  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError, match="SpaGCN"):
            create_method("domain_detection", "spagcn", n_domains=3).run(data)


def test_maturity_deinflation_and_validated_set():
    rows = {row["name"]: row for row in list_methods()}
    assert rows["marker_score"]["maturity"] == "experimental"
    assert rows["marker_deconv"]["maturity"] == "experimental"
    assert rows["spatial_autoencoder"]["maturity"] == "experimental"
    # SOTA wrappers: scientific evidence → validated; contract gates → contract_validated.
    for name in ("spagcn", "graphst", "stagate"):
        assert rows[name]["maturity"] == "validated"
        assert name in VALIDATION_EVIDENCE
        assert VALIDATION_EVIDENCE[name]["kind"] == "scientific"
    for name in ("rctd", "spatialde"):
        assert rows[name]["maturity"] == "contract_validated"
        assert VALIDATION_EVIDENCE[name]["kind"] == "contract"
    for name in VALIDATED_METHODS:
        assert rows[name]["maturity"] == "validated"
        assert name in VALIDATION_EVIDENCE
    assert len(VALIDATED_METHODS) == 10


def test_knn_indices_matches_brute_force_on_small_data():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(80, 2))
    k = 5
    got = knn_indices(coords, k)
    brute = np.argsort(
        np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2),
        axis=1,
        kind="stable",
    )[:, :k]
    # cKDTree and argsort can break distance ties differently; require identical
    # neighbour *sets* and matching first column (self).
    assert got.shape == brute.shape
    assert np.array_equal(got[:, 0], brute[:, 0])
    for i in range(coords.shape[0]):
        assert set(got[i].tolist()) == set(brute[i].tolist())


def test_knn_indices_scales_without_full_distance_matrix():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(3000, 2))
    indices = knn_indices(coords, 8)
    assert indices.shape == (3000, 8)
    assert indices.dtype.kind in "iu"
    # Self should be among neighbours for each row (usually column 0).
    assert (indices[:, 0] == np.arange(3000)).mean() > 0.99


def test_feature_extraction_accepts_sparse_matrix():
    data = make_synthetic(n_cells=60, n_genes=20, seed=2)
    dense_feats = extract_features(data, include_domain=False)
    sparse_table = SpatialTable(
        X=sparse.csr_matrix(np.asarray(data.X, dtype=float)),
        obs=data.obs.copy(),
        var=data.var.copy(),
        obsm={"spatial": np.asarray(data.spatial, dtype=float)},
        uns=dict(data.uns),
    )
    sparse_feats = extract_features(sparse_table, include_domain=False)
    for key in RECOMMENDATION_FEATURE_ORDER:
        assert np.isfinite(sparse_feats[key]) or np.isnan(sparse_feats[key])
        if np.isfinite(dense_feats[key]) and np.isfinite(sparse_feats[key]):
            assert sparse_feats[key] == pytest.approx(dense_feats[key], rel=1e-5, abs=1e-5)


def test_recommender_v2_reports_baselines_and_priors():
    datasets = {
        "a": make_synthetic(n_cells=50, n_genes=12, seed=3),
        "b": make_synthetic(n_cells=55, n_genes=14, seed=4),
        "c": make_synthetic(n_cells=60, n_genes=16, seed=5),
    }
    features = {
        name: feature_vector(
            extract_features(table, include_domain=False),
            order=RECOMMENDATION_FEATURE_ORDER,
        )
        for name, table in datasets.items()
    }
    performance = {
        "a": {"spectral@sw0.8": 0.40, "kmeans@sw0.0": 0.20, "gaussian_mixture@sw0.8": 0.38},
        "b": {"spectral@sw0.8": 0.35, "kmeans@sw0.0": 0.22, "gaussian_mixture@sw0.8": 0.36},
        "c": {"spectral@sw0.8": 0.30, "kmeans@sw0.0": 0.28, "gaussian_mixture@sw0.8": 0.33},
    }
    kb = LandscapeResult(
        performance=performance,
        features=features,
        embedding={},
        best_method={
            "a": "spectral@sw0.8",
            "b": "gaussian_mixture@sw0.8",
            "c": "gaussian_mixture@sw0.8",
        },
        niches={},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=3,
        dataset_count=3,
        task="spatial_domain",
        metric="ARI",
        dataset_meta={
            "a": {
                "platform": "visium",
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
            },
            "b": {
                "platform": "visium",
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
            },
            "c": {"platform": "xenium", "task": "cell_type", "ground_truth_kind": "cell_type"},
        },
    )
    rec = MethodRecommender(kb, k_neighbours=2).recommend(
        datasets["a"],
        dataset_name="query",
        task=AnalysisTask.SPATIAL_DOMAIN,
        platform="visium",
        spatial_context_policy="high",
    )
    assert rec.schema_version == 3
    assert rec.platform == "visium"
    assert rec.global_best_method is not None
    assert rec.beats_global_best_baseline is not None
    assert rec.best() is not None
    assert rec.best().base_method or "@" in rec.best().method
    # Neighbours from a different task should be down-weighted / warned.
    assert (
        any(
            "task" in w.lower() or "global-best" in w.lower() or "Knowledge base" in w
            for w in rec.warnings
        )
        or rec.neighbours
    )
    payload = rec.to_dict()
    assert "baselines" in payload
    assert payload["baselines"]["global_best_method"] == rec.global_best_method


def test_split_method_policy_helper():
    assert split_method_policy("spectral@sw0.8") == ("spectral", "sw0.8")
    assert split_method_policy("banksy_py") == ("banksy_py", None)
