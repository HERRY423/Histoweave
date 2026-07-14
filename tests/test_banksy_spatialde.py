import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, list_methods
from histoweave.plugins.builtin._r_base import RContainerMethod


def test_banksy_is_an_r_container_method_and_builds_arguments():
    data = make_synthetic(n_cells=20, n_genes=10, seed=1)
    method = create_method(
        "domain_detection", "banksy", lambda_param=0.6, algorithm="kmeans", n_domains=4
    )
    assert isinstance(method, RContainerMethod)
    args = method._build_r_args(data)
    assert "lambda=0.6" in args
    assert "algorithm=kmeans" in args
    assert "n_domains=4" in args


def test_banksy_validates_2d_coordinates_before_bridge_io(monkeypatch):
    data = make_synthetic(n_cells=20, n_genes=10, seed=2)
    data.obsm["spatial"] = np.c_[data.spatial, np.zeros(data.n_obs)]
    method = create_method("domain_detection", "banksy")
    monkeypatch.setattr(
        method, "_find_r_script", lambda: pytest.fail("R bridge started before validation")
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        method.run(data)


def test_spatialde_delegates_to_anndata_bridge(monkeypatch):
    data = make_synthetic(n_cells=10, n_genes=5, seed=3)
    method = create_method("svg", "spatialde")
    monkeypatch.setattr(method, "_run_via_anndata", lambda value: ("bridged", value))
    assert method.run(data) == ("bridged", data)


def test_spatialde_maps_results_back_to_var(monkeypatch):
    genes = pd.Index(["g0", "g1", "g2"])

    class FakeAnnData:
        def __init__(self):
            self.X = np.arange(12, dtype=float).reshape(4, 3)
            self.obs_names = pd.Index(["c0", "c1", "c2", "c3"])
            self.var_names = genes
            self.obs = pd.DataFrame(index=self.obs_names)
            self.var = pd.DataFrame(index=self.var_names)
            self.obsm = {"spatial": np.arange(8).reshape(4, 2)}
            self.layers = {}
            self.uns = {}

        def copy(self):
            clone = FakeAnnData()
            clone.var = self.var.copy()
            clone.uns = dict(self.uns)
            return clone

    monkeypatch.setitem(
        sys.modules,
        "NaiveDE",
        SimpleNamespace(
            stabilize=lambda values: values,
            regress_out=lambda sample_info, values, formula: values,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "SpatialDE",
        SimpleNamespace(
            run=lambda coordinates, values: pd.DataFrame(
                {
                    "g": ["g1", "g0", "g2"],
                    "FSV": [0.9, 0.4, 0.2],
                    "pval": [0.001, 0.02, 0.5],
                    "qval": [0.003, 0.03, 0.7],
                    "l": [2.0, 1.0, 3.0],
                }
            )
        ),
    )

    result = create_method("svg", "spatialde", n_top=2).run_on_anndata(FakeAnnData())
    assert result.var.loc["g1", "spatialde_fsv"] == pytest.approx(0.9)
    assert result.var["spatialde_significant"].tolist() == [True, True, False]
    assert [item["gene"] for item in result.uns["svg"]["top_genes"]] == ["g1", "g0"]


def test_banksy_and_spatialde_are_registered():
    methods = {(item["category"], item["name"]): item for item in list_methods()}
    assert methods[("domain_detection", "banksy")]["language"] == "container"
    assert methods[("svg", "spatialde")]["language"] == "python"


def test_banksy_r_script_normalizes_before_spatial_clustering():
    script_path = (
        Path(__file__).parents[1]
        / "workflows"
        / "containers"
        / "histoweave-r"
        / "histoweave-banksy.R"
    )
    script = script_path.read_text(encoding="utf-8")
    library_factors = script.index("scuttle::computeLibraryFactors")
    log_normalization = script.index("scuttle::logNormCounts")
    compute_banksy = script.index("Banksy::computeBanksy")
    assert library_factors < log_normalization < compute_banksy
    assert script.count('assay_name = "logcounts"') == 3
