import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import (
    METHOD_MATURITY_POLICIES,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
    create_method,
    list_methods,
)
from histoweave.plugins.builtin._r_base import RContainerMethod


class FakeAnnData:
    def __init__(self, n_obs=6, genes=("g0", "g1", "g2")):
        self.obs_names = pd.Index([f"c{i}" for i in range(n_obs)])
        self.var_names = pd.Index(genes)
        self.X = np.arange(n_obs * len(genes), dtype=float).reshape(n_obs, len(genes))
        self.obs = pd.DataFrame(index=self.obs_names)
        self.var = pd.DataFrame(index=self.var_names)
        self.obsm = {"spatial": np.arange(n_obs * 2, dtype=float).reshape(n_obs, 2)}
        self.layers = {"counts": self.X.copy()}
        self.uns = {}

    @property
    def n_obs(self):
        return len(self.obs_names)

    def copy(self):
        clone = FakeAnnData(len(self.obs_names), tuple(self.var_names))
        clone.X = self.X.copy()
        clone.obs = self.obs.copy()
        clone.var = self.var.copy()
        clone.obsm = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self.obsm.items()
        }
        clone.layers = {key: value.copy() for key, value in self.layers.items()}
        clone.uns = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self.uns.items()
        }
        return clone

    def __getitem__(self, index):
        obs_index, var_index = index
        assert obs_index == slice(None)
        selected = pd.Index(var_index)
        positions = self.var_names.get_indexer(selected)
        clone = FakeAnnData(self.n_obs, tuple(selected))
        clone.X = self.X[:, positions].copy()
        clone.obs = self.obs.copy()
        clone.layers = {
            key: value[:, positions].copy() for key, value in self.layers.items()
        }
        clone.uns = dict(self.uns)
        clone.obsm = {key: value.copy() for key, value in self.obsm.items()}
        return clone


def _install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_maturity_policy_is_ordered_queryable_and_string_compatible():
    spec = MethodSpec(
        name="quality_contract",
        category=MethodCategory.QC,
        version="1",
        maturity="production",
    )
    assert spec.maturity is MethodMaturity.PRODUCTION
    ranks = [
        METHOD_MATURITY_POLICIES[level].rank
        for level in (
            MethodMaturity.EXPERIMENTAL,
            MethodMaturity.BETA,
            MethodMaturity.PRODUCTION,
            MethodMaturity.VALIDATED,
        )
    ]
    assert ranks == sorted(ranks)
    beta_or_higher = list_methods(minimum_maturity="beta")
    assert beta_or_higher
    assert all(item["maturity_rank"] >= 20 for item in beta_or_higher)
    assert all(item["quality_requirements"] for item in beta_or_higher)


def test_cell2location_calls_real_model_contract(monkeypatch):
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
            adata.obsm["q05_cell_abundance_w_sf"] = np.tile([2.0, 1.0], (adata.n_obs, 1))
            return adata

    package = _install_module(monkeypatch, "cell2location")
    models = _install_module(monkeypatch, "cell2location.models", Cell2location=FakeCell2location)
    package.models = models
    adata = FakeAnnData()
    adata.uns["cell2location_reference"] = pd.DataFrame(
        {"T": [1.0, 0.2, 0.1], "B": [0.1, 0.5, 1.0]}, index=adata.var_names
    )

    result = create_method("deconvolution", "cell2location", max_epochs=2).run_on_anndata(adata)
    assert calls["setup"]["layer"] == "counts"
    assert calls["train"]["max_epochs"] == 2
    assert calls["posterior"]["use_quantiles"] is True
    assert result.obsm["cell_abundance"].shape == (adata.n_obs, 2)
    assert np.allclose(result.obsm["proportions"].sum(axis=1), 1.0)
    assert result.uns["deconvolution"]["cell_types"] == ["T", "B"]


def test_liana_plus_calls_rank_aggregate_and_validates_result(monkeypatch):
    calls = {}

    def rank_aggregate(adata, **kwargs):
        calls.update(kwargs)
        adata.uns[kwargs["key_added"]] = pd.DataFrame(
            {
                "source": ["T"],
                "target": ["B"],
                "ligand_complex": ["L"],
                "receptor_complex": ["R"],
                "magnitude_rank": [0.1],
            }
        )

    _install_module(
        monkeypatch,
        "liana",
        mt=SimpleNamespace(rank_aggregate=rank_aggregate),
    )
    adata = FakeAnnData()
    adata.obs["cell_type"] = pd.Categorical(["T", "T", "T", "B", "B", "B"])
    result = create_method("ccc", "liana_plus", n_perms=None).run_on_anndata(adata)
    assert calls["spatial_key"] == "spatial"
    assert calls["inplace"] is True
    assert result.uns["ccc"]["n_interactions"] == 1


def test_scanvi_runs_scvi_pretraining_and_soft_prediction(monkeypatch):
    calls = {"train": []}

    class FakeSCVI:
        @staticmethod
        def setup_anndata(adata, **kwargs):
            calls["setup"] = kwargs

        def __init__(self, adata, **kwargs):
            self.adata = adata
            calls["scvi_init"] = kwargs

        def train(self, **kwargs):
            calls["train"].append(("scvi", kwargs))

    class FakeSCANVI:
        def __init__(self, adata):
            self.adata = adata

        @classmethod
        def from_scvi_model(cls, model, **kwargs):
            calls["from_scvi"] = kwargs
            return cls(model.adata)

        def train(self, **kwargs):
            calls["train"].append(("scanvi", kwargs))

        def predict(self, adata, soft=False):
            if soft:
                return pd.DataFrame(
                    np.tile([0.8, 0.2], (adata.n_obs, 1)),
                    index=adata.obs_names,
                    columns=["T", "B"],
                )
            return pd.Series(["T"] * adata.n_obs, index=adata.obs_names)

        def get_latent_representation(self, adata):
            return np.ones((adata.n_obs, 3))

    scvi = _install_module(
        monkeypatch,
        "scvi",
        settings=SimpleNamespace(seed=None),
        model=SimpleNamespace(SCVI=FakeSCVI, SCANVI=FakeSCANVI),
    )
    assert scvi is sys.modules["scvi"]
    adata = FakeAnnData()
    adata.obs["cell_type_seed"] = pd.Categorical(["T", "B", "Unknown"] * 2)
    result = create_method(
        "annotation", "scanvi", scvi_epochs=2, scanvi_epochs=3
    ).run_on_anndata(adata)
    assert [entry[1]["max_epochs"] for entry in calls["train"]] == [2, 3]
    assert result.obs["cell_type"].astype(str).tolist() == ["T"] * adata.n_obs
    assert result.obsm["scanvi_probabilities"].shape == (adata.n_obs, 2)
    assert result.obsm["X_scanvi"].shape == (adata.n_obs, 3)


def test_celltypist_maps_labels_confidence_and_probability(monkeypatch):
    def annotate(adata, **kwargs):
        labels = pd.DataFrame(
            {
                "predicted_labels": ["T"] * adata.n_obs,
                "majority_voting": ["B"] * adata.n_obs,
                "conf_score": [0.9] * adata.n_obs,
            },
            index=adata.obs_names,
        )
        probability = pd.DataFrame(
            np.tile([0.1, 0.9], (adata.n_obs, 1)),
            index=adata.obs_names,
            columns=["T", "B"],
        )
        return SimpleNamespace(predicted_labels=labels, probability_matrix=probability)

    _install_module(monkeypatch, "celltypist", annotate=annotate)
    adata = FakeAnnData()
    result = create_method("annotation", "celltypist").run_on_anndata(adata)
    assert result.obs["cell_type"].astype(str).tolist() == ["B"] * adata.n_obs
    assert np.allclose(result.obs["celltypist_confidence"], 0.9)
    assert result.uns["annotation"]["classes"] == ["T", "B"]


def test_cellpose2_calls_model_and_stores_label_image(monkeypatch):
    calls = {}

    class FakeCellposeModel:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def eval(self, image, **kwargs):
            calls["eval"] = kwargs
            masks = np.zeros(image.shape, dtype=np.int32)
            masks[1:3, 1:3] = 1
            return masks, {}, np.zeros(1)

    _install_module(
        monkeypatch,
        "cellpose",
        models=SimpleNamespace(CellposeModel=FakeCellposeModel),
    )
    data = make_synthetic(n_cells=10, n_genes=5, seed=4)
    data.images["image"] = np.arange(16, dtype=float).reshape(4, 4)
    result = create_method("segmentation", "cellpose2").run(data)
    assert calls["init"]["model_type"] == "cyto2"
    assert calls["eval"]["do_3D"] is False
    assert result.images["cellpose_masks"].shape == (4, 4)
    assert result.uns["segmentation"]["n_instances"] == 1


def test_sctransform_is_real_r_container_wrapper_and_validates_before_io(monkeypatch):
    method = create_method("normalization", "sctransform", vst_flavor="v2")
    assert isinstance(method, RContainerMethod)
    data = make_synthetic(n_cells=10, n_genes=5, seed=5)
    args = method._build_r_args(data)
    assert "vst_flavor=v2" in args
    data.X[0, 0] = -1
    monkeypatch.setattr(
        method, "_find_r_script", lambda: pytest.fail("R bridge started before validation")
    )
    with pytest.raises(ValueError, match="non-negative"):
        method.run(data)


def test_new_wrappers_are_beta_not_self_declared_production():
    names = {"cell2location", "liana_plus", "scanvi", "celltypist", "cellpose2", "sctransform"}
    specs = {item["name"]: item for item in list_methods() if item["name"] in names}
    assert set(specs) == names
    assert all(item["maturity"] == "beta" for item in specs.values())
