"""Vitessce view config generation and report integration tests."""

import json

import numpy as np
import pandas as pd
import pytest

from histoweave.data import SpatialTable
from histoweave.datasets import make_synthetic
from histoweave.report.vitessce_data import build_vitessce_view_config, vitessce_data_json


def _make_processed_table():
    data = make_synthetic(n_cells=100, n_genes=30, n_domains=3, seed=0)
    gene_names = [f"gene_{i}" for i in range(data.n_vars)]
    data.var.index = pd.Index(gene_names, name="feature_id")
    data.uns["svg"] = {
        "top_genes": [
            {"gene": gene_names[0], "fsv": 0.9},
            {"gene": gene_names[1], "fsv": 0.8},
            {"gene": gene_names[2], "fsv": 0.7},
            {"gene": gene_names[3], "fsv": 0.6},
            {"gene": gene_names[4], "fsv": 0.5},
        ]
    }
    return data


class TestVitessceViewConfig:
    """Vitessce view config generation contract."""

    def test_returns_config_and_data_keys(self):
        data = _make_processed_table()
        vc = build_vitessce_view_config(data, top_genes=10)
        assert "config" in vc
        assert "data" in vc
        assert "gene_names" in vc

    def test_config_has_required_fields(self):
        data = _make_processed_table()
        vc = build_vitessce_view_config(data, top_genes=5)
        config = vc["config"]
        assert config["version"] == "1.0.16"
        assert len(config["layout"]) >= 2
        assert len(config["datasets"]) >= 2
        components = [c["component"] for c in config["layout"]]
        assert "scatterplot" in components
        assert "heatmap" in components

    def test_data_contains_cells_json(self):
        data = _make_processed_table()
        vc = build_vitessce_view_config(data, top_genes=5)
        cells = json.loads(vc["data"]["cells.json"])
        assert isinstance(cells, list)
        assert len(cells) <= 100  # max spots
        assert "x" in cells[0]
        assert "y" in cells[0]
        assert "mappings" in cells[0]

    def test_cells_have_domain_mapping(self):
        data = _make_processed_table()
        data.obs["domain"] = pd.Categorical(["domain_0"] * data.n_obs)
        vc = build_vitessce_view_config(data, top_genes=5)
        cells = json.loads(vc["data"]["cells.json"])
        assert "domain" in cells[0]["mappings"]

    def test_subsamples_when_too_many_spots(self):
        data = make_synthetic(n_cells=300, n_genes=10, n_domains=3, seed=0)
        vc = build_vitessce_view_config(data, top_genes=5, max_spots=50)
        cells = json.loads(vc["data"]["cells.json"])
        assert len(cells) <= 50

    def test_no_spatial_raises(self):
        data = _make_processed_table()
        del data.obsm["spatial"]
        with pytest.raises(ValueError, match="spatial"):
            build_vitessce_view_config(data)

    def test_gene_names_present(self):
        data = _make_processed_table()
        vc = build_vitessce_view_config(data, top_genes=3)
        assert len(vc["gene_names"]) > 0

    def test_vitessce_data_json_roundtrips(self):
        data = _make_processed_table()
        json_str = vitessce_data_json(data, top_genes=5)
        parsed = json.loads(json_str)
        assert "config" in parsed
        assert "data" in parsed

    def test_empty_report_falls_back_gracefully(self):
        """Even with minimal data, config generation should not crash."""
        X = np.ones((5, 3))
        obs = pd.DataFrame(index=pd.Index(["c0","c1","c2","c3","c4"], name="barcode"))
        var = pd.DataFrame(index=pd.Index(["a","b","c"], name="feature_id"))
        st = SpatialTable(
            X=X, obs=obs, var=var,
            obsm={"spatial": np.arange(10).reshape(5, 2).astype(float)},
            uns={"assay": "test"},
        )
        vc = build_vitessce_view_config(st, top_genes=3)
        assert vc["config"]["version"] == "1.0.16"


class TestBuildReportIntegration:
    """build_report includes Vitessce data without crashing."""

    def test_report_includes_vitessce_in_context(self, tmp_path):
        from histoweave import build_report, run_pipeline
        from histoweave.datasets import make_synthetic

        data = make_synthetic(seed=0)
        result = run_pipeline(data)
        out = build_report(result, tmp_path / "report.html")
        assert out.exists()
        text = out.read_text(encoding="utf-8")

        # Vitessce container
        assert "vitessce-container" in text
        # Vitessce CDN
        assert "unpkg.com/vitessce" in text
        # Fallback SVG still present
        assert "<svg" in text
        # Static report still renders
        assert "HistoWeave Analysis Report" in text
