"""Vitessce v3 config generation and report integration tests."""

import json
from io import StringIO

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
    data.obs["cell_type"] = pd.Categorical(["neuron"] * 50 + ["glia"] * 50)
    data.obs["domain"] = data.obs["domain_truth"].astype("category")
    data.uns["svg"] = {
        "top_genes": [
            {"gene": gene_names[index], "fsv": 0.9 - index / 10}
            for index in range(5)
        ]
    }
    return data


class TestVitessceViewConfig:
    def test_config_uses_v3_native_csv_file_types(self):
        vc = build_vitessce_view_config(_make_processed_table(), top_genes=5)
        config = vc["config"]
        assert config["version"] == "1.0.16"
        assert len(config["datasets"]) == 1
        file_types = {item["fileType"] for item in config["datasets"][0]["files"]}
        assert file_types == {
            "obsEmbedding.csv",
            "obsSpots.csv",
            "obsFeatureMatrix.csv",
            "obsSets.csv",
        }
        matrix_file = next(
            item for item in config["datasets"][0]["files"]
            if item["fileType"] == "obsFeatureMatrix.csv"
        )
        assert matrix_file["options"] is None

    def test_coordination_space_and_views_are_schema_shaped(self):
        config = build_vitessce_view_config(_make_processed_table())["config"]
        assert set(config["coordinationSpace"]) >= {
            "dataset",
            "embeddingType",
            "obsType",
            "featureType",
            "featureValueType",
            "obsColorEncoding",
        }
        assert {view["component"] for view in config["layout"]} == {
            "scatterplot",
            "obsSets",
            "heatmap",
        }
        for view in config["layout"]:
            assert view["coordinationScopes"]["dataset"] == "A"
            assert "props" not in view or view["component"] == "heatmap"

    def test_inline_csv_payloads_parse_and_align(self):
        vc = build_vitessce_view_config(_make_processed_table(), top_genes=5)
        frames = {
            key: pd.read_csv(StringIO(value))
            for key, value in vc["data"].items()
        }
        assert set(frames) == {
            "obs_spots.csv",
            "obs_embedding.csv",
            "obs_sets.csv",
            "obs_matrix.csv",
        }
        assert all(len(frame) == 100 for frame in frames.values())
        assert frames["obs_spots.csv"].columns.tolist() == ["obs_id", "x", "y"]
        assert frames["obs_embedding.csv"].columns.tolist() == ["obs_id", "e0", "e1"]
        assert frames["obs_matrix.csv"].shape == (100, 6)
        expected_ids = frames["obs_spots.csv"]["obs_id"].tolist()
        assert all(frame["obs_id"].tolist() == expected_ids for frame in frames.values())

    def test_obs_sets_include_domain_and_cell_type(self):
        vc = build_vitessce_view_config(_make_processed_table())
        sets = pd.read_csv(StringIO(vc["data"]["obs_sets.csv"]))
        assert {"domain", "cell_type"}.issubset(sets.columns)
        sets_file = next(
            item for item in vc["config"]["datasets"][0]["files"]
            if item["fileType"] == "obsSets.csv"
        )
        assert [item["name"] for item in sets_file["options"]["obsSets"]][:2] == [
            "Domain",
            "Cell Type",
        ]

    def test_subsampling_is_deterministic_and_applies_to_every_file(self):
        data = make_synthetic(n_cells=300, n_genes=10, n_domains=3, seed=0)
        first = build_vitessce_view_config(data, top_genes=5, max_spots=50)
        second = build_vitessce_view_config(data, top_genes=5, max_spots=50)
        assert first["data"] == second["data"]
        for payload in first["data"].values():
            assert len(pd.read_csv(StringIO(payload))) == 50

    def test_no_spatial_raises(self):
        data = _make_processed_table()
        del data.obsm["spatial"]
        with pytest.raises(ValueError, match="spatial"):
            build_vitessce_view_config(data)

    def test_json_round_trip(self):
        parsed = json.loads(vitessce_data_json(_make_processed_table(), top_genes=5))
        assert parsed["config"]["datasets"][0]["files"][2]["options"] is None

    def test_minimal_table_without_sets_uses_description_view(self):
        st = SpatialTable(
            X=np.ones((5, 3)),
            obs=pd.DataFrame(index=pd.Index([f"c{i}" for i in range(5)])),
            var=pd.DataFrame(index=pd.Index(["a", "b", "c"])),
            obsm={"spatial": np.arange(10).reshape(5, 2).astype(float)},
            uns={"assay": "test"},
        )
        config = build_vitessce_view_config(st)["config"]
        assert "description" in {view["component"] for view in config["layout"]}
        assert "obsSets.csv" not in {
            item["fileType"] for item in config["datasets"][0]["files"]
        }


class TestBidirectionalLinking:
    def test_shared_interaction_scopes_present(self):
        config = build_vitessce_view_config(_make_processed_table())["config"]
        cs = config["coordinationSpace"]
        # shared brushing/linking scopes for scatterplot <-> obsSets <-> heatmap
        for key in ("obsSetSelection", "obsHighlight", "featureSelection"):
            assert key in cs
        # every view must reference the shared selection + feature scopes
        for view in config["layout"]:
            scopes = view["coordinationScopes"]
            assert scopes.get("obsSetSelection") == "A"
            assert scopes.get("featureSelection") == "A"

    def test_cluster_top_markers_payload(self):
        vc = build_vitessce_view_config(_make_processed_table(), top_genes=10)
        markers = vc["cluster_top_markers"]
        assert vc["primary_labelPlaceholder" if False else "primary_label"] is not None
        # keyed by the display name of each label column
        assert markers, "expected a non-empty per-cluster marker map"
        # each cluster maps to a list of gene names drawn from the selected genes
        gene_pool = set(vc["gene_names"])
        for _hierarchy, by_cluster in markers.items():
            assert by_cluster
            for _cluster, genes in by_cluster.items():
                assert isinstance(genes, list) and genes
                assert set(genes) <= gene_pool

    def test_report_embeds_linking_js(self, tmp_path):
        from histoweave import build_report, run_pipeline

        result = run_pipeline(make_synthetic(seed=0))
        text = build_report(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "cluster_top_markers" in text
        assert "onConfigChange" in text
        assert "obsSetSelection" in text


class TestBuildReportIntegration:
    def test_report_uses_official_esm_bundle_and_keeps_svg_fallback(self, tmp_path):
        from histoweave import build_report, run_pipeline

        result = run_pipeline(make_synthetic(seed=0))
        out = build_report(result, tmp_path / "report.html")
        text = out.read_text(encoding="utf-8")
        assert 'id="vitessce-payload"' in text
        assert "cdn.jsdelivr.net/npm/vitessce@3.5.7/dist/index.min.js" in text
        assert "unpkg.com/vitessce" not in text
        assert "dist/index.min.css" not in text
        assert "import('vitessce')" in text
        assert "createRoot(container).render" in text
        assert "type: 'text/csv'" in text
        assert "<svg" in text
