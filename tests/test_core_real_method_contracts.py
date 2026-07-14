"""Hard failure-mode contracts for the six core external method adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method
from histoweave.plugins.builtin._validation import validate_count_matrix


def _module(monkeypatch, name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _adata(n_obs: int = 6) -> AnnData:
    counts = np.arange(n_obs * 3, dtype=float).reshape(n_obs, 3)
    adata = AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    adata.layers["counts"] = counts.copy()
    adata.obsm["spatial"] = np.arange(n_obs * 2, dtype=float).reshape(n_obs, 2)
    return adata


def test_count_validation_handles_sparse_and_rejects_normalized_values():
    validate_count_matrix(csr_matrix([[0.0, 1.0], [2.0, 0.0]]), method="test")
    with pytest.raises(ValueError, match="integer-like raw counts"):
        validate_count_matrix(csr_matrix([[0.0, 1.25], [2.0, 0.0]]), method="test")


def test_cell2location_aligns_posterior_rows_and_cell_type_columns(monkeypatch):
    class FakeCell2location:
        @staticmethod
        def setup_anndata(**kwargs):
            return None

        def __init__(self, adata, **kwargs):
            self.adata = adata

        def train(self, **kwargs):
            return None

        def export_posterior(self, adata, **kwargs):
            adata.obsm["q05_cell_abundance_w_sf"] = pd.DataFrame(
                {"B": np.arange(1, adata.n_obs + 1), "T": np.arange(11, 11 + adata.n_obs)},
                index=adata.obs_names,
            )
            return adata

    package = _module(monkeypatch, "cell2location")
    package.models = _module(monkeypatch, "cell2location.models", Cell2location=FakeCell2location)
    adata = _adata()
    adata.uns["cell2location_reference"] = pd.DataFrame(
        {"T": [1.0, 0.2, 0.1], "B": [0.1, 0.5, 1.0]}, index=adata.var_names
    )

    result = create_method("deconvolution", "cell2location", max_epochs=1).run_on_anndata(adata)
    # Posterior columns must follow the reference order T, B.
    assert result.obsm["cell_abundance"][0].tolist() == [11.0, 1.0]
    assert result.uns["deconvolution"]["cell_types"] == ["T", "B"]


def test_banksy_and_sctransform_reject_fractional_counts_before_bridge(monkeypatch):
    data = make_synthetic(n_cells=20, n_genes=10, seed=11)
    data.X = np.asarray(data.X, dtype=float)
    data.X[0, 0] += 0.25
    for method_name, category in (("banksy", "domain_detection"), ("sctransform", "normalization")):
        method = create_method(category, method_name)
        monkeypatch.setattr(
            method, "_find_r_script", lambda: pytest.fail("bridge started before validation")
        )
        with pytest.raises(ValueError, match="integer-like raw counts"):
            method.run(data)


def test_cellpose_accepts_auto_detected_rgb_axis_and_counts_sparse_labels(monkeypatch):
    calls = {}

    class FakeCellposeModel:
        def __init__(self, **kwargs):
            pass

        def eval(self, image, **kwargs):
            calls.update(kwargs)
            masks = np.zeros(image.shape[:2], dtype=np.int32)
            masks[0, 0] = 1
            masks[-1, -1] = 4
            return masks, {}, np.zeros(1)

    _module(monkeypatch, "cellpose", models=SimpleNamespace(CellposeModel=FakeCellposeModel))
    data = make_synthetic(n_cells=10, n_genes=5, seed=12)
    data.images["image"] = np.ones((8, 9, 3), dtype=np.float32)
    result = create_method("segmentation", "cellpose2").run(data)

    assert result.images["cellpose_masks"].shape == (8, 9)
    assert result.uns["segmentation"]["n_instances"] == 2
    assert result.uns["segmentation"]["max_label"] == 4
    assert calls["batch_size"] == 8


def test_scanvi_rejects_single_labelled_class_before_training(monkeypatch):
    scvi = _module(monkeypatch, "scvi")
    scvi.settings = SimpleNamespace(seed=None)
    scvi.model = SimpleNamespace()
    adata = _adata()
    adata.obs["cell_type_seed"] = pd.Categorical(["T", "Unknown"] * 3)

    with pytest.raises(ValueError, match="at least two labelled"):
        create_method("annotation", "scanvi").run_on_anndata(adata)


def test_spatialde_filters_constant_genes_and_maps_only_tested_results(monkeypatch):
    adata = AnnData(
        X=np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 2, 3, 1],
                [0, 4, 1, 1],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["constant_zero", "g1", "g2", "constant_one"]),
    )
    adata.obsm["spatial"] = np.arange(8, dtype=float).reshape(4, 2)

    _module(
        monkeypatch,
        "NaiveDE",
        stabilize=lambda values: values,
        regress_out=lambda sample_info, values, formula: values,
    )

    def run_spatialde(coordinates, values):
        assert values.columns.tolist() == ["g1", "g2"]
        return pd.DataFrame(
            {
                "g": ["g2", "g1"],
                "FSV": [0.8, 0.5],
                "pval": [0.001, 0.02],
                "qval": [0.002, 0.03],
                "l": [2.0, 1.0],
            }
        )

    _module(monkeypatch, "SpatialDE", run=run_spatialde)
    result = create_method("svg", "spatialde", min_cells=2).run_on_anndata(adata)

    assert np.isnan(result.var.loc["constant_zero", "spatialde_qval"])
    assert result.uns["svg"]["n_tested"] == 2
    assert [row["gene"] for row in result.uns["svg"]["top_genes"]] == ["g2", "g1"]


def test_r_scripts_call_real_backends_with_required_contracts():
    root = Path(__file__).parents[1]
    banksy = (root / "workflows/containers/histoweave-r/histoweave-banksy.R").read_text()
    sct = (root / "workflows/containers/histoweave-r/histoweave-sctransform.R").read_text()
    dockerfile = (root / "workflows/containers/histoweave-r/Dockerfile").read_text()

    assert "Banksy::computeBanksy" in banksy
    assert "Banksy::clusterBanksy" in banksy
    assert "k_neighbors = effective_k_neighbors" in banksy
    assert "do.call(sctransform::vst, vst_args)" in sct
    assert "return_only_var_genes" not in sct
    assert 'adata$layers[["counts"]] <- source' in sct
    assert "glmGamPoi" in dockerfile


def test_count_validation_rejects_zero_library_observations():
    with pytest.raises(ValueError, match="positive library size"):
        validate_count_matrix(np.array([[0.0, 0.0], [1.0, 2.0]]), method="test")


def test_release_images_include_real_backends_and_r_bridge_cli():
    root = Path(__file__).parents[1]
    python_image = (root / "workflows/containers/histoweave-python/Dockerfile").read_text()
    r_image = (root / "workflows/containers/histoweave-r/Dockerfile").read_text()
    workflow = (root / ".github/workflows/containers.yml").read_text()
    nextflow = (root / "workflows/nextflow/main.nf").read_text(encoding="utf-8")

    for extra in ("spatialde", "cell2location", "scanvi", "cellpose2"):
        assert extra in python_image
    assert "COPY pyproject.toml README.md LICENSE ./" in r_image
    assert 'pip install ".[${HISTOWEAVE_EXTRAS}]"' in r_image
    assert "RETICULATE_PYTHON=/opt/histoweave-venv/bin/python" in r_image
    assert "histoweave list >/dev/null" in r_image
    assert "- name: histoweave-r\n            context: ." in workflow
    assert "method in ['sctransform', 'r_lognorm'] ? params.r_image" in nextflow
    assert "method == 'banksy' ? params.r_image" in nextflow
