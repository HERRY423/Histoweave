from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from histoweave.data import SpatialTable
from histoweave.datasets import make_synthetic
from histoweave.plugins import MethodMaturity
from histoweave.plugins.builtin.research_spatial import RESEARCH_METHODS

DOMAIN_NAMES = (
    "weave_multiscale_consensus_domains",
    "weave_boundary_aware_domains",
    "weave_topology_regularized_domains",
    "weave_uncertainty_domains",
)
SVG_NAMES = (
    "weave_multiscale_svg",
    "weave_boundary_svg",
    "weave_hotspot_svg",
    "weave_anisotropy_svg",
    "weave_bootstrap_robust_svg",
)


@pytest.fixture(scope="module")
def spatial_data() -> SpatialTable:
    return make_synthetic(n_cells=48, n_genes=18, n_domains=3, seed=321)


def test_research_catalog_has_nine_explicitly_unvalidated_native_methods():
    assert set(RESEARCH_METHODS) == set(DOMAIN_NAMES) | set(SVG_NAMES)
    for name, cls in RESEARCH_METHODS.items():
        assert cls.spec.name == name
        assert cls.spec.version == "0.1.0"
        assert cls.spec.maturity is MethodMaturity.EXPERIMENTAL
        assert cls.spec.implementation.value == "native"
        assert cls.spec.metadata == {"track": "research", "novelty": "unvalidated"}
        assert cls.__module__ == "histoweave.plugins.builtin.research_spatial"


@pytest.mark.parametrize("name", DOMAIN_NAMES)
def test_domain_methods_are_deterministic_finite_and_preserve_input(name, spatial_data):
    cls = RESEARCH_METHODS[name]
    before_x = np.asarray(spatial_data.X).copy()
    before_obs = spatial_data.obs.copy(deep=True)
    first = cls(n_domains=3).run(spatial_data)
    second = cls(n_domains=3).run(spatial_data)

    assert first is not spatial_data
    assert np.array_equal(np.asarray(spatial_data.X), before_x)
    pd.testing.assert_frame_equal(spatial_data.obs, before_obs)
    assert first.shape == spatial_data.shape
    assert first.obs["domain"].dtype.name == "category"
    assert first.obs["domain"].notna().all()
    assert first.obs["domain"].tolist() == second.obs["domain"].tolist()
    embeddings = [key for key in first.obsm if key.startswith("X_weave_")]
    assert embeddings
    for key in embeddings:
        assert first.obsm[key].shape[0] == spatial_data.n_obs
        assert np.isfinite(first.obsm[key]).all()
        assert np.allclose(first.obsm[key], second.obsm[key])
    assert first.provenance[-1]["method"] == name
    assert len(first.provenance) == len(spatial_data.provenance) + 1


def test_uncertainty_domains_publish_valid_probabilities(spatial_data):
    result = RESEARCH_METHODS["weave_uncertainty_domains"](n_domains=3).run(spatial_data)
    probabilities = result.obsm["weave_domain_probabilities"]
    uncertainty = result.obs["weave_domain_uncertainty"].to_numpy()
    assert probabilities.shape == (spatial_data.n_obs, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((uncertainty >= 0.0) & (uncertainty <= 1.0))


def test_uncertainty_domains_handle_constant_expression_and_empty_clusters():
    data = make_synthetic(n_cells=12, n_genes=5, seed=91)
    data.X = np.zeros((data.n_obs, data.n_vars), dtype=float)

    result = RESEARCH_METHODS["weave_uncertainty_domains"](
        n_domains=4,
        n_ensembles=5,
    ).run(data)

    probabilities = result.obsm["weave_domain_probabilities"]
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.obs["domain"].notna().all()


def test_kdtree_neighbors_handle_duplicate_coordinates_deterministically():
    data = make_synthetic(n_cells=12, n_genes=6, seed=22)
    data.obsm["spatial"] = np.zeros((data.n_obs, 2), dtype=float)

    domain = RESEARCH_METHODS["weave_multiscale_consensus_domains"](
        n_domains=3
    ).run(data)
    svg = RESEARCH_METHODS["weave_hotspot_svg"](n_top=4).run(data)

    assert domain.obs["domain"].notna().all()
    assert np.isfinite(domain.obsm["X_weave_multiscale_consensus"]).all()
    assert np.isfinite(svg.var["weave_hotspot_svg_score"]).all()


@pytest.mark.parametrize("name", SVG_NAMES)
def test_svg_methods_are_deterministic_finite_and_rank_genes(name, spatial_data):
    cls = RESEARCH_METHODS[name]
    before_x = np.asarray(spatial_data.X).copy()
    first = cls(n_top=7).run(spatial_data)
    second = cls(n_top=7).run(spatial_data)
    score_key = f"{name}_score"

    assert first is not spatial_data
    assert np.array_equal(np.asarray(spatial_data.X), before_x)
    assert first.shape == spatial_data.shape
    assert score_key in first.var
    assert first.var[score_key].shape == (spatial_data.n_vars,)
    assert np.isfinite(first.var[score_key]).all()
    assert np.allclose(first.var[score_key], second.var[score_key])
    ranking = first.uns[name]["top_genes"]
    assert len(ranking) == 7
    assert [row["score"] for row in ranking] == sorted(
        (row["score"] for row in ranking), reverse=True
    )
    assert {row["gene"] for row in ranking}.issubset(set(spatial_data.var_names.astype(str)))
    assert first.uns["svg"]["method"] == name
    assert first.provenance[-1]["method"] == name
    assert len(first.provenance) == len(spatial_data.provenance) + 1


@pytest.mark.parametrize("name", (*DOMAIN_NAMES, *SVG_NAMES))
def test_research_methods_require_finite_spatial_coordinates(name):
    data = make_synthetic(n_cells=12, n_genes=6, seed=11)
    coordinates = np.asarray(data.spatial).copy()
    coordinates[0, 0] = np.nan
    data.obsm["spatial"] = coordinates
    with pytest.raises(ValueError, match="spatial coordinates must be finite"):
        RESEARCH_METHODS[name]().run(data)
