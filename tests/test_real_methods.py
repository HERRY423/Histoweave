"""Tests for scikit-learn/scipy/networkx-backed real method wrappers.

These tests verify that the new Phase-1 plugin methods (DBSCAN, spectral clustering,
Moran's I SVG detection, spatial graph analysis, ComBat integration) produce valid,
deterministic outputs — exercising the real libraries that replace the Phase-0 toy
implementations.
"""

import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, list_methods


def _normalize(data):
    return create_method("normalization", "log1p_cp10k").run(data)


# ---------------------------------------------------------------------------
# Domain detection — sklearn clustering
# ---------------------------------------------------------------------------
class TestSklearnClustering:
    """Each clustering method: runs, writes 'domain', is deterministic."""

    @pytest.mark.parametrize("method_name", ["dbscan", "agglomerative", "spectral"])
    def test_runs_and_writes_domain(self, method_name):
        data = _normalize(make_synthetic(n_cells=200, n_genes=30, n_domains=3, seed=0))
        result = create_method("domain_detection", method_name).run(data)
        assert "domain" in result.obs
        assert result.obs["domain"].notna().all()
        assert result.n_obs == data.n_obs

    @pytest.mark.parametrize("method_name", ["dbscan", "agglomerative", "spectral"])
    def test_is_deterministic(self, method_name):
        data = _normalize(make_synthetic(n_cells=150, n_genes=25, n_domains=3, seed=1))
        a = create_method("domain_detection", method_name, random_state=0).run(data)
        b = create_method("domain_detection", method_name, random_state=0).run(data)
        assert list(a.obs["domain"]) == list(b.obs["domain"])

    def test_dbscan_may_produce_noise(self):
        """DBSCAN labels noise as -1, which should appear as a distinct category."""
        data = _normalize(make_synthetic(n_cells=200, n_genes=30, n_domains=3, seed=2))
        result = create_method("domain_detection", "dbscan", eps=0.3, min_samples=10).run(data)
        labels = set(result.obs["domain"])
        # DBSCAN can produce noise (domain_-1); all should be categorical.
        assert all(str(label).startswith("domain_") for label in labels)

    def test_agglomerative_respects_n_domains(self):
        data = _normalize(make_synthetic(n_cells=200, n_genes=30, n_domains=3, seed=3))
        result = create_method(
            "domain_detection", "agglomerative", n_domains=5, spatial_weight=0.0
        ).run(data)
        # Ward with n_clusters=5 should produce ~5 clusters (allow ±1 for bad splits).
        n = result.obs["domain"].nunique()
        assert 3 <= n <= 5


# ---------------------------------------------------------------------------
# SVG detection — Moran's I
# ---------------------------------------------------------------------------
class TestMoransISVG:
    def test_runs_and_writes_var_column(self):
        data = _normalize(make_synthetic(n_cells=200, n_genes=30, seed=4))
        result = create_method("svg", "morans_i").run(data)
        assert "morans_i" in result.var
        scores = result.var["morans_i"].to_numpy()
        # Moran's I is typically in [-1, 1] for normalised data.
        assert -1.1 <= scores.min() <= scores.max() <= 1.1
        assert "svg" in result.uns
        assert len(result.uns["svg"]["top_genes"]) > 0

    def test_top_genes_are_ranked(self):
        data = _normalize(make_synthetic(n_cells=300, n_genes=50, n_domains=4, seed=5))
        result = create_method("svg", "morans_i", n_top=10).run(data)
        scores = [g["morans_i"] for g in result.uns["svg"]["top_genes"]]
        assert scores == sorted(scores, reverse=True)
        assert len(scores) == 10

    def test_marker_genes_score_high(self):
        """Genes that define spatial domains should have high Moran's I."""
        data = _normalize(make_synthetic(
            n_cells=300, n_genes=40, n_domains=3,
            marker_genes_per_domain=5, noise=0.1, seed=6,
        ))
        result = create_method("svg", "morans_i", n_top=15).run(data)
        top_genes = {g["gene"] for g in result.uns["svg"]["top_genes"]}
        # At least some of the marker genes from uns['marker_genes'] should be top.
        all_markers = set()
        for genes in data.uns["marker_genes"].values():
            all_markers.update(genes)
        assert len(top_genes & all_markers) >= 2


# ---------------------------------------------------------------------------
# Spatial graph — networkx neighborhood analysis
# ---------------------------------------------------------------------------
class TestSpatialGraph:
    def test_constructs_graph_and_writes_metrics(self):
        data = _normalize(make_synthetic(n_cells=200, n_genes=20, seed=7))
        result = create_method("neighborhood", "spatial_graph", k=6).run(data)
        for col in ("spatial_degree", "spatial_clustering", "spatial_centrality"):
            assert col in result.obs
            assert result.obs[col].notna().all()
        g = result.uns["spatial_graph"]
        assert g["n_nodes"] == 200
        assert g["n_edges"] > 0

    def test_radius_mode(self):
        data = _normalize(make_synthetic(n_cells=100, n_genes=15, seed=8))
        result = create_method(
            "neighborhood", "spatial_graph", mode="radius", radius=50.0
        ).run(data)
        assert result.uns["spatial_graph"]["mode"] == "radius"
        assert result.obs["spatial_degree"].max() > 0

    def test_graph_is_symmetric_undirected(self):
        data = _normalize(make_synthetic(n_cells=80, n_genes=15, seed=9))
        result = create_method("neighborhood", "spatial_graph", k=5).run(data)
        edges = result.uns["spatial_graph"]["edges"]
        # Every edge should have u < v (canonical form for undirected).
        for u, v in edges:
            assert u < v


# ---------------------------------------------------------------------------
# Integration — ComBat
# ---------------------------------------------------------------------------
class TestComBat:
    def test_single_batch_is_noop(self):
        data = _normalize(make_synthetic(n_cells=100, n_genes=20, seed=10))
        # No batch column: add a dummy one with all same.
        data.obs["batch"] = "A"
        before = data.X.copy()
        result = create_method("integration", "combat", batch_key="batch").run(data)
        # Single batch — should be a no-op.
        assert np.allclose(result.X, before)

    def test_multi_batch_reduces_batch_effect(self):
        """With two artificial 'batches' (shifted means), ComBat should reduce the gap."""
        data = _normalize(make_synthetic(n_cells=200, n_genes=30, seed=11))
        # Split into two artificial batches by adding a shift to the second half.
        data.obs["batch"] = ["A"] * 100 + ["B"] * 100
        X = np.asarray(data.X, dtype=float)
        X[100:] += 1.5  # batch B is systematically higher
        data.X = X

        result = create_method("integration", "combat", batch_key="batch").run(data)

        # After correction, the per-batch means should be closer together.
        mean_a = result.X[:100].mean()
        mean_b = result.X[100:].mean()
        gap_after = abs(mean_a - mean_b)
        gap_before = abs(X[:100].mean() - X[100:].mean())
        assert gap_after < gap_before * 0.5  # at least 50% reduction

    def test_leaves_pre_combat_layer(self):
        data = _normalize(make_synthetic(n_cells=100, n_genes=20, seed=12))
        data.obs["batch"] = ["A"] * 50 + ["B"] * 50
        X = np.asarray(data.X, dtype=float)
        X[50:] *= 2.0
        data.X = X

        result = create_method("integration", "combat", batch_key="batch").run(data)
        assert "pre_combat" in result.layers
        assert result.layers["pre_combat"].shape == data.shape

    def test_nonparametric_mode_uses_batch_labels_and_reduces_shift(self):
        data = _normalize(make_synthetic(n_cells=200, n_genes=20, seed=13))
        data.obs["batch"] = ["A"] * 100 + ["B"] * 100
        shifted = np.asarray(data.X, dtype=float)
        shifted[100:] += 2.0
        data.X = shifted

        result = create_method(
            "integration", "combat", batch_key="batch", parametric=False
        ).run(data)
        gap_before = abs(shifted[:100].mean() - shifted[100:].mean())
        gap_after = abs(result.X[:100].mean() - result.X[100:].mean())

        assert gap_after < gap_before * 0.25


# ---------------------------------------------------------------------------
# Integration: new methods appear in list_methods
# ---------------------------------------------------------------------------
def test_new_methods_are_listed():
    methods = {m["name"] for m in list_methods()}
    expected = {
        "dbscan", "agglomerative", "spectral",
        "morans_i", "spatial_graph", "combat",
    }
    assert expected.issubset(methods)
