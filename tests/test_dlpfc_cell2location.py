"""Cell2location contract validation with real DLPFC brain cell-type reference.

Builds a biologically plausible sc reference from known marker genes in the
DLPFC gene space, then validates the cell2location method contract:
reference shape, shared genes, abundance proportions, and metadata.
"""

import os
import sys
import tempfile
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix

from histoweave.data import SpatialTable
from histoweave.plugins import create_method, list_methods

_MARKER_DEFS: dict[str, list[str]] = {
    "Neurons": ["SNAP25", "SYT1", "GRIN1", "GAD1", "GAD2",
                "SLC17A7", "RGS4", "NEFL", "SST", "PVALB"],
    "Astrocytes": ["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "GLUL", "GJA1"],
    "Oligodendrocytes": ["MBP", "MOBP", "PLP1", "MOG", "MAG", "SOX10", "OLIG1", "CNP"],
    "Microglia": ["C1QA", "C1QB", "C1QC", "TREM2", "CX3CR1",
                  "ITGAM", "P2RY12", "TMEM119"],
    "Endothelial": ["CLDN5", "PECAM1", "CDH5", "VWF", "FLT1", "ENG"],
    "OPC": ["PDGFRA", "CSPG4", "VCAN", "OLIG2", "SOX6", "NKX2-2"],
}


def _dlpfc_cache_path():
    return os.path.join(tempfile.gettempdir(), "histoweave_dlpfc_cache",
                        "151507_filtered_feature_bc_matrix.h5")


def _cached_or_skip():
    path = _dlpfc_cache_path()
    if not os.path.exists(path):
        pytest.skip("DLPFC data not cached — run scripts/dlpfc_cell2location_reference.py first")
    return path


def _load_dlpfc_matrix():
    import h5py
    with h5py.File(_cached_or_skip(), 'r') as f:
        barcodes = [b.decode('utf-8') for b in f['matrix/barcodes'][:]]
        features = [feat.decode('utf-8') for feat in f['matrix/features/name'][:]]
        data_arr = np.array(f['matrix/data'][:])
        indices = np.array(f['matrix/indices'][:])
        indptr = np.array(f['matrix/indptr'][:])
        shape = tuple(f['matrix/shape'][:])
    X = csc_matrix((data_arr, indices, indptr), shape=shape).tocsr().T
    return X, barcodes, features


def _build_reference(features):
    """Build brain cell-type reference from marker genes in the DLPFC feature space."""
    feature_set = set(features)
    found = {ct: [g for g in markers if g in feature_set] for ct, markers in _MARKER_DEFS.items()}
    ref_genes = sorted(set().union(*found.values()))
    ref_df = pd.DataFrame(1.0, index=ref_genes, columns=sorted(found.keys()))
    return ref_df, found


def _install_mock_cell2location(monkeypatch):
    """Install a mock cell2location module that records calls and returns test data."""
    calls = {}

    class FakeCell2location:
        @staticmethod
        def setup_anndata(**kwargs):
            calls["setup"] = kwargs

        def __init__(self, adata, **kwargs):
            calls["init"] = kwargs
            self.adata = adata

        def train(self, **kwargs):
            calls["train"] = kwargs

        def export_posterior(self, adata, **kwargs):
            calls["posterior"] = kwargs
            n_cell_types = len(kwargs.get("sample_kwargs", {}))
            n_cell_types = calls["init"].get("cell_state_df", pd.DataFrame()).shape[1]
            adata.obsm["q05_cell_abundance_w_sf"] = np.ones((adata.n_obs, max(n_cell_types, 1)))
            return adata

    fake_c2l = ModuleType("cell2location")
    fake_models = ModuleType("cell2location.models")
    fake_models.Cell2location = FakeCell2location
    fake_c2l.models = fake_models
    monkeypatch.setitem(sys.modules, "cell2location", fake_c2l)
    monkeypatch.setitem(sys.modules, "cell2location.models", fake_models)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCell2locationReference:
    """Reference matrix construction and validation."""

    def test_marker_genes_found_in_dlpfc(self):
        _, _, features = _load_dlpfc_matrix()
        _, found = _build_reference(features)
        for ct, genes in found.items():
            assert len(genes) >= 5, f"{ct} has only {len(genes)} marker genes in DLPFC"

    def test_reference_matrix_shape(self):
        _, _, features = _load_dlpfc_matrix()
        ref_df, _ = _build_reference(features)
        assert ref_df.shape[0] >= 40  # at least 40 shared marker genes
        assert ref_df.shape[1] == 6   # 6 cell types
        assert list(ref_df.columns) == ["Astrocytes", "Endothelial", "Microglia",
                                        "Neurons", "OPC", "Oligodendrocytes"]

    def test_reference_values_are_positive(self):
        _, _, features = _load_dlpfc_matrix()
        ref_df, _ = _build_reference(features)
        assert (ref_df.to_numpy() > 0).all()


class TestCell2locationDLPFCContract:
    """Validate the cell2location method contract on real DLPFC data."""

    def test_cell2location_registered(self):
        methods = {m["name"] for m in list_methods(category="deconvolution")}
        assert "cell2location" in methods

    def test_cell2location_on_dlpfc_with_reference(self, monkeypatch):
        X, barcodes, features = _load_dlpfc_matrix()
        ref_df, _ = _build_reference(features)

        # Subset to 500 genes overlapping reference + highly expressed
        overlap_genes = [g for g in features if g in ref_df.index]
        if len(overlap_genes) < 10:
            pytest.skip("Too few reference genes overlap with top expressed genes")

        N_GENES = min(500, len(features))
        gene_means = np.array(X.mean(axis=0)).flatten()
        top_idx = np.argsort(gene_means)[-N_GENES:]
        X_sub = X[:, top_idx].toarray()
        sub_features = [features[i] for i in top_idx]
        X_norm = np.log1p(X_sub / (X_sub.sum(axis=1, keepdims=True) + 1) * 10000)

        spatial = np.column_stack([
            np.arange(len(barcodes)) % 65 * 100,
            np.arange(len(barcodes)) // 65 * 100,
        ]).astype(float)

        obs = pd.DataFrame(index=pd.Index(barcodes, name='barcode'))
        var = pd.DataFrame(
            {'feature_name': sub_features},
            index=pd.Index(sub_features, name='feature_id')
        )

        st = SpatialTable(
            X=X_norm, obs=obs, var=var,
            obsm={'spatial': spatial},
            layers={'counts': X_sub.astype(np.float64)},
            uns={'assay': 'visium', 'n_domains': 7, 'cell2location_reference': ref_df},
        )

        calls = _install_mock_cell2location(monkeypatch)

        result = create_method(
            "deconvolution", "cell2location",
            max_epochs=5, n_cells_per_location=5.0, use_gpu=False,
            reference_key="cell2location_reference",
        ).run(st.copy())

        # Check outputs
        assert "cell_abundance" in result.obsm
        assert "proportions" in result.obsm
        assert result.obsm["cell_abundance"].shape[1] == 6  # 6 cell types
        assert np.allclose(result.obsm["proportions"].sum(axis=1), 1.0)
        assert "deconvolution" in result.uns
        assert result.uns["deconvolution"]["method"] == "cell2location"

        # Check mock calls exercised the real code paths
        assert "setup" in calls
        assert calls["setup"]["layer"] == "counts"
        assert calls["train"]["max_epochs"] == 5
        assert calls["train"]["accelerator"] == "cpu"
        assert "use_gpu" not in calls["train"]
        assert calls["posterior"]["sample_kwargs"]["use_gpu"] is False


class TestCell2locationValidation:
    """Input validation for cell2location."""

    def test_missing_reference_rejected(self, monkeypatch):
        data = _make_minimal_table()
        _install_mock_cell2location(monkeypatch)
        with pytest.raises(KeyError, match="cell2location_reference"):
            create_method("deconvolution", "cell2location").run(data.copy())

    def test_empty_reference_rejected(self, monkeypatch):
        data = _make_minimal_table()
        data.uns["cell2location_reference"] = pd.DataFrame()
        _install_mock_cell2location(monkeypatch)
        with pytest.raises(ValueError, match="empty"):
            create_method("deconvolution", "cell2location").run(data.copy())


def _make_minimal_table(n=10):
    X = np.ones((n, 5))
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n)], name='barcode'))
    var = pd.DataFrame(index=pd.Index([f"g{i}" for i in range(5)], name='feature_id'))
    return SpatialTable(
        X=X, obs=obs, var=var,
        obsm={'spatial': np.column_stack([np.arange(n), np.arange(n)]).astype(float)},
        layers={'counts': X.astype(np.float64)},
        uns={'assay': 'test'},
    )
