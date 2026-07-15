from __future__ import annotations

import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method, get_method
from histoweave.plugins.interfaces import MethodImplementation, MethodMaturity

_METHODS = (
    ("qc", "weave_spatial_entropy_qc"),
    ("qc", "weave_neighbor_discordance_qc"),
    ("qc", "weave_adaptive_saturation_qc"),
    ("normalization", "weave_spatial_median_normalize"),
    ("normalization", "weave_graph_diffusion_normalize"),
    ("normalization", "weave_rank_stabilize"),
    ("normalization", "weave_robust_pearson_residual"),
)
_QC_OUTPUTS = (
    (
        "weave_spatial_entropy_qc",
        "weave_spatial_entropy",
        "weave_low_spatial_entropy",
    ),
    (
        "weave_neighbor_discordance_qc",
        "weave_neighbor_discordance",
        "weave_high_neighbor_discordance",
    ),
    (
        "weave_adaptive_saturation_qc",
        "weave_adaptive_saturation",
        "weave_low_adaptive_saturation",
    ),
)


def _data():
    data = make_synthetic(n_cells=30, n_genes=16, seed=17)
    data.X = np.asarray(data.X, dtype=float)
    return data


@pytest.mark.parametrize(("category", "name"), _METHODS)
def test_research_method_specs_are_explicitly_unvalidated_native(category, name):
    spec = get_method(category, name).spec
    assert spec.version == "0.1.0"
    assert spec.maturity is MethodMaturity.EXPERIMENTAL
    assert spec.implementation is MethodImplementation.NATIVE
    assert spec.metadata == {"track": "research", "novelty": "unvalidated"}


@pytest.mark.parametrize(("name", "score_key", "flag_key"), _QC_OUTPUTS)
def test_research_qc_methods_are_finite_deterministic_and_non_mutating(
    name,
    score_key,
    flag_key,
):
    data = _data()
    original = data.X.copy()

    first = create_method("qc", name).run(data)
    second = create_method("qc", name).run(data)

    np.testing.assert_array_equal(data.X, original)
    np.testing.assert_allclose(first.obs[score_key], second.obs[score_key])
    np.testing.assert_array_equal(first.obs[flag_key], second.obs[flag_key])
    assert np.isfinite(first.obs[score_key].to_numpy()).all()
    assert first.obs[flag_key].dtype == bool
    assert int(first.obs[flag_key].sum()) <= int(np.ceil(0.05 * data.n_obs))
    assert first.n_obs == data.n_obs
    assert first.n_vars == data.n_vars
    assert first.obs.index.equals(data.obs.index)
    assert first.var.equals(data.var)
    np.testing.assert_allclose(first.obsm["spatial"], data.obsm["spatial"])
    assert first.uns["assay"] == data.uns["assay"]
    assert first.provenance[-1]["method"] == name
    assert first.uns["research_preprocessing"][name]["novelty"] == "unvalidated"


@pytest.mark.parametrize(
    "name",
    [name for category, name in _METHODS if category == "normalization"],
)
def test_research_normalizers_are_finite_deterministic_and_preserve_counts(name):
    data = _data()
    original = data.X.copy()

    first = create_method("normalization", name).run(data)
    second = create_method("normalization", name).run(data)

    np.testing.assert_array_equal(data.X, original)
    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_allclose(first.layers["counts"], original)
    assert first.X.shape == original.shape
    assert np.isfinite(first.X).all()
    assert first.obs.equals(data.obs)
    assert first.var.equals(data.var)
    np.testing.assert_allclose(first.obsm["spatial"], data.obsm["spatial"])
    assert first.uns["assay"] == data.uns["assay"]
    assert first.provenance[-1]["method"] == name
    assert first.uns["normalization"] == {"method": name, "research": True}


@pytest.mark.parametrize(
    ("category", "name"),
    [
        ("qc", "weave_spatial_entropy_qc"),
        ("qc", "weave_neighbor_discordance_qc"),
        ("normalization", "weave_spatial_median_normalize"),
        ("normalization", "weave_graph_diffusion_normalize"),
    ],
)
def test_spatial_research_methods_fail_closed_without_coordinates(category, name):
    data = _data()
    data.obsm.pop("spatial")
    with pytest.raises(ValueError, match="spatial.*required"):
        create_method(category, name).run(data)


@pytest.mark.parametrize(("category", "name"), _METHODS)
@pytest.mark.parametrize("invalid", [np.nan, -1.0])
def test_research_methods_reject_nonfinite_and_negative_expression(category, name, invalid):
    data = _data()
    data.X[0, 0] = invalid
    with pytest.raises(ValueError, match="finite|non-negative"):
        create_method(category, name).run(data)


def test_rank_stabilization_preserves_zeros_and_ties():
    data = _data()
    data.X[:] = 0.0
    data.X[0, :4] = [1.0, 4.0, 4.0, 9.0]

    result = create_method("normalization", "weave_rank_stabilize").run(data)

    assert np.count_nonzero(result.X[0]) == 4
    assert result.X[0, 1] == pytest.approx(result.X[0, 2])
    assert np.count_nonzero(result.X[1:]) == 0


def test_robust_pearson_residual_is_bounded_by_clip():
    result = create_method(
        "normalization",
        "weave_robust_pearson_residual",
        clip=1.5,
    ).run(_data())
    assert np.max(np.abs(result.X)) <= 1.5


@pytest.mark.parametrize(
    ("category", "name"),
    [
        ("qc", "weave_adaptive_saturation_qc"),
        ("normalization", "weave_rank_stabilize"),
        ("normalization", "weave_robust_pearson_residual"),
    ],
)
def test_nonspatial_research_methods_support_one_observation(category, name):
    data = _data().subset_obs(np.arange(30) == 0)
    result = create_method(category, name).run(data)
    assert result.X.shape == data.X.shape
    assert np.isfinite(result.X).all()


@pytest.mark.parametrize(
    ("category", "name"),
    [
        ("qc", "weave_spatial_entropy_qc"),
        ("qc", "weave_neighbor_discordance_qc"),
        ("normalization", "weave_spatial_median_normalize"),
        ("normalization", "weave_graph_diffusion_normalize"),
    ],
)
def test_spatial_research_methods_reject_one_observation(category, name):
    data = _data().subset_obs(np.arange(30) == 0)
    with pytest.raises(ValueError, match="at least two observations"):
        create_method(category, name).run(data)
