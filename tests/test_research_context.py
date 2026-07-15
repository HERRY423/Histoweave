"""Contracts for the experimental built-in research-context methods."""

from __future__ import annotations

import json
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from histoweave.benchmark import deconvolution_task, run_benchmark
from histoweave.data import SpatialTable
from histoweave.report import build_report

MARKERS = {"type_a": ["g0", "g1"], "type_b": ["g2", "g3"]}

METHODS = {
    "WeaveAdaptiveRadiusGraph": ("weave_adaptive_radius_graph", "neighborhood"),
    "WeaveMutualKNNGraph": ("weave_mutual_knn_graph", "neighborhood"),
    "WeaveExpressionSpatialGraph": ("weave_expression_spatial_graph", "neighborhood"),
    "WeaveSpatialQuantileIntegrate": ("weave_spatial_quantile_integrate", "integration"),
    "WeaveAnchorResidualIntegrate": ("weave_anchor_residual_integrate", "integration"),
    "WeaveNeighborMarkerAnnotate": ("weave_neighbor_marker_annotate", "annotation"),
    "WeaveSpatialSimplexDeconv": ("weave_spatial_simplex_deconv", "deconvolution"),
}


@pytest.fixture(scope="module")
def research_module() -> ModuleType:
    """Return the built-in module without mutating global registry state."""
    from histoweave.plugins.builtin import research_context

    return research_context


def _table(*, aligned_batches: bool = False, spatial: bool = True) -> SpatialTable:
    profiles = np.array(
        [
            [9.0, 8.0, 1.0, 1.0, 2.0],
            [8.0, 9.0, 1.0, 1.0, 2.5],
            [1.0, 1.0, 9.0, 8.0, 2.0],
            [1.0, 1.0, 8.0, 9.0, 2.5],
        ]
    )
    if aligned_batches:
        matrix = np.vstack([profiles, profiles + 3.0])
        coordinates = np.vstack(
            [
                np.array([[0.0, 0.0], [1.1, 0.0], [0.0, 1.2], [1.1, 1.2]]),
                np.array([[0.0, 0.0], [1.1, 0.0], [0.0, 1.2], [1.1, 1.2]]),
            ]
        )
    else:
        matrix = np.vstack(
            [profiles[:2], profiles[:2] - 0.25, profiles[2:], profiles[2:] - 0.25]
        )
        coordinates = np.array(
            [
                [0.0, 0.0],
                [0.9, 0.1],
                [0.2, 1.1],
                [1.2, 1.0],
                [8.0, 0.0],
                [9.1, 0.2],
                [8.2, 1.3],
                [9.3, 1.1],
            ]
        )
    obs = pd.DataFrame(
        {"batch": ["batch_a"] * 4 + ["batch_b"] * 4},
        index=[f"spot_{index}" for index in range(8)],
    )
    var = pd.DataFrame(index=["g0", "g1", "g2", "g3", "g4"])
    obsm = {"spatial": coordinates} if spatial else {}
    return SpatialTable(matrix, obs=obs, var=var, obsm=obsm)


def test_research_specs_are_explicitly_unvalidated_and_native(research_module: ModuleType):
    for class_name, (method_name, category) in METHODS.items():
        method_class = getattr(research_module, class_name)
        assert method_class.spec.name == method_name
        assert method_class.spec.category.value == category
        assert method_class.spec.version == "0.1.0"
        assert method_class.spec.maturity.value == "experimental"
        assert method_class.spec.metadata == {
            "track": "research",
            "novelty": "unvalidated",
        }
        assert method_class.spec.implementation.value == "native"
        assert method_class.spec.wraps is None
        assert "spatial" in method_class.spec.modalities


@pytest.mark.parametrize(
    ("class_name", "method_name", "extra_params"),
    [
        ("WeaveAdaptiveRadiusGraph", "weave_adaptive_radius_graph", {}),
        ("WeaveMutualKNNGraph", "weave_mutual_knn_graph", {}),
        (
            "WeaveExpressionSpatialGraph",
            "weave_expression_spatial_graph",
            {"n_components": 2},
        ),
    ],
)
def test_graph_methods_are_deterministic_and_emit_serializable_edges(
    research_module: ModuleType,
    class_name: str,
    method_name: str,
    extra_params: dict[str, object],
):
    data = _table(aligned_batches=True)
    method = getattr(research_module, class_name)(k=2, **extra_params)
    first = method.run(data)
    second = method.run(data)

    first_payload = first.uns[method_name]
    second_payload = second.uns[method_name]
    assert json.dumps(first_payload, sort_keys=True) == json.dumps(
        second_payload, sort_keys=True
    )
    edges = first_payload["edges"]
    assert first_payload["n_nodes"] == data.n_obs
    assert first_payload["n_edges"] == len(edges)
    assert edges == sorted(edges, key=lambda edge: (edge[0], edge[1]))
    assert all(left < right and np.isfinite(weight) for left, right, weight in edges)

    degree_key = f"{method_name}_degree"
    degrees = first.obs[degree_key].to_numpy()
    assert np.isfinite(degrees).all()
    assert int(degrees.sum()) == 2 * len(edges)
    assert first.provenance[-1]["method"] == method_name
    assert degree_key not in data.obs
    np.testing.assert_array_equal(first.X, data.X)


def test_spatial_quantile_integration_aligns_batch_distributions(
    research_module: ModuleType,
):
    data = _table(aligned_batches=True)
    method = research_module.WeaveSpatialQuantileIntegrate(k=2, spatial_blend=0.0)
    first = method.run(data)
    second = method.run(data)
    output_key = "X_weave_spatial_quantile"
    integrated = np.asarray(first.obsm[output_key])

    assert integrated.shape == data.shape
    assert np.isfinite(integrated).all()
    np.testing.assert_allclose(integrated, second.obsm[output_key])
    for gene in range(data.n_vars):
        np.testing.assert_allclose(
            np.sort(integrated[:4, gene]),
            np.sort(integrated[4:, gene]),
        )
    assert first.provenance[-1]["method"] == "weave_spatial_quantile_integrate"
    assert json.loads(json.dumps(first.uns["weave_spatial_quantile_integrate"]))[
        "batches"
    ] == ["batch_a", "batch_b"]
    assert output_key not in data.obsm


def test_anchor_residual_integration_removes_a_registered_additive_shift(
    research_module: ModuleType,
):
    data = _table(aligned_batches=True)
    method = research_module.WeaveAnchorResidualIntegrate(
        spatial_weight=10.0,
        n_components=3,
    )
    first = method.run(data)
    second = method.run(data)
    output_key = "X_weave_anchor_residual"
    integrated = np.asarray(first.obsm[output_key])

    assert np.isfinite(integrated).all()
    np.testing.assert_allclose(integrated, second.obsm[output_key])
    np.testing.assert_allclose(integrated[:4], integrated[4:])
    np.testing.assert_allclose(integrated[:4], np.asarray(data.X)[:4])
    metadata = first.uns["weave_anchor_residual_integrate"]
    assert metadata["reference_batch"] == "batch_a"
    assert len(metadata["anchors"]["batch_b"]) == 4
    json.dumps(metadata)
    assert first.provenance[-1]["method"] == "weave_anchor_residual_integrate"


def test_neighbor_marker_annotation_is_deterministic_and_fails_closed(
    research_module: ModuleType,
):
    data = _table()
    method = research_module.WeaveNeighborMarkerAnnotate(
        marker_genes=MARKERS,
        k=2,
        neighbor_weight=0.4,
    )
    first = method.run(data)
    second = method.run(data)

    assert first.obs["cell_type"].astype(str).tolist() == [
        "type_a",
        "type_a",
        "type_a",
        "type_a",
        "type_b",
        "type_b",
        "type_b",
        "type_b",
    ]
    assert first.obs["cell_type"].astype(str).tolist() == second.obs[
        "cell_type"
    ].astype(str).tolist()
    scores = np.asarray(first.obsm["X_weave_neighbor_marker_scores"])
    assert scores.shape == (data.n_obs, 2)
    assert np.isfinite(scores).all()
    assert (first.obs["cell_type_confidence"].to_numpy() >= 0).all()
    assert first.provenance[-1]["method"] == "weave_neighbor_marker_annotate"
    assert "cell_type" not in data.obs

    with pytest.raises(ValueError, match="marker_genes must be supplied"):
        research_module.WeaveNeighborMarkerAnnotate(marker_genes=None).run(data)


def test_spatial_simplex_deconvolution_has_valid_deterministic_proportions(
    research_module: ModuleType,
):
    data = _table()
    method = research_module.WeaveSpatialSimplexDeconv(
        marker_genes=MARKERS,
        k=2,
        spatial_strength=0.25,
        n_iter=60,
    )
    first = method.run(data)
    second = method.run(data)
    proportions = np.asarray(first.obsm["proportions"])

    assert proportions.shape == (data.n_obs, 2)
    assert np.isfinite(proportions).all()
    assert (proportions >= 0).all()
    np.testing.assert_allclose(proportions.sum(axis=1), 1.0)
    np.testing.assert_allclose(proportions, second.obsm["proportions"])
    assert (proportions[:4, 0] > proportions[:4, 1]).all()
    assert (proportions[4:, 1] > proportions[4:, 0]).all()
    metadata = first.uns["weave_spatial_simplex_deconv"]
    assert metadata["cell_types"] == ["type_a", "type_b"]
    assert metadata["graph"]["n_edges"] == len(metadata["graph"]["edges"])
    json.dumps(metadata)
    assert first.provenance[-1]["method"] == "weave_spatial_simplex_deconv"
    assert "proportions" not in data.obsm

    with pytest.raises(ValueError, match="marker_genes must be supplied"):
        research_module.WeaveSpatialSimplexDeconv(marker_genes=None).run(data)


def test_standard_output_keys_feed_report_and_benchmark(research_module, tmp_path):
    data = _table()
    annotated = research_module.WeaveNeighborMarkerAnnotate(
        marker_genes=MARKERS,
        k=2,
    ).run(data)
    report_path = build_report(annotated, tmp_path / "research-report.html")
    assert "type_a" in report_path.read_text(encoding="utf-8")

    benchmark_data = _table()
    benchmark_data.uns["marker_genes"] = MARKERS
    benchmark_data.obsm["proportions_truth"] = np.vstack(
        [np.tile([1.0, 0.0], (4, 1)), np.tile([0.0, 1.0], (4, 1))]
    )
    benchmark = run_benchmark(
        deconvolution_task(dataset=benchmark_data),
        methods=["weave_spatial_simplex_deconv"],
        method_params={
            "weave_spatial_simplex_deconv": {"k": 2, "n_iter": 30}
        },
    )
    assert len(benchmark.leaderboard) == 1
    assert np.isfinite(benchmark.leaderboard[0]["score"])


@pytest.mark.parametrize(
    ("class_name", "params"),
    [
        ("WeaveAdaptiveRadiusGraph", {}),
        ("WeaveMutualKNNGraph", {}),
        ("WeaveExpressionSpatialGraph", {}),
        ("WeaveSpatialQuantileIntegrate", {}),
        ("WeaveAnchorResidualIntegrate", {}),
        ("WeaveNeighborMarkerAnnotate", {"marker_genes": MARKERS}),
        ("WeaveSpatialSimplexDeconv", {"marker_genes": MARKERS}),
    ],
)
def test_every_research_method_fails_closed_without_spatial_coordinates(
    research_module: ModuleType,
    class_name: str,
    params: dict[str, object],
):
    method = getattr(research_module, class_name)(**params)
    with pytest.raises(ValueError, match="obsm\\['spatial'\\] is required"):
        method.run(_table(spatial=False))
