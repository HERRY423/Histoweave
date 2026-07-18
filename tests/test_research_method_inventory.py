from __future__ import annotations

from histoweave.plugins import list_methods, method_coverage_report

EXPECTED_RESEARCH_METHODS = {
    "weave_adaptive_radius_graph",
    "weave_adaptive_saturation_qc",
    "weave_anchor_residual_integrate",
    "weave_anisotropy_svg",
    "weave_bootstrap_robust_svg",
    "weave_boundary_aware_domains",
    "weave_boundary_svg",
    "weave_expression_spatial_graph",
    "weave_graph_diffusion_normalize",
    "weave_hotspot_svg",
    "weave_multiscale_consensus_domains",
    "weave_multiscale_svg",
    "weave_mutual_knn_graph",
    "weave_neighbor_discordance_qc",
    "weave_neighbor_marker_annotate",
    "weave_rank_stabilize",
    "weave_robust_pearson_residual",
    "weave_spatial_entropy_qc",
    "weave_spatial_median_normalize",
    "weave_spatial_quantile_integrate",
    "weave_spatial_simplex_deconv",
    "weave_topology_regularized_domains",
    "weave_uncertainty_domains",
}


def test_research_method_inventory_is_explicit_and_unvalidated() -> None:
    rows = {row["name"]: row for row in list_methods()}
    assert EXPECTED_RESEARCH_METHODS <= set(rows)
    for name in EXPECTED_RESEARCH_METHODS:
        row = rows[name]
        assert row["maturity"] == "experimental"
        assert row["implementation"] == "native"
        assert row["metadata"]["track"] == "research"
        assert row["metadata"]["novelty"] == "unvalidated"


def test_release_coverage_separates_research_candidates() -> None:
    report = method_coverage_report()
    assert report["counts"]["research_candidates"] >= 23
    assert report["counts"]["release_methods"] >= 40
    assert report["ratios"]["beta_plus"] == 1.0
    assert report["counts"]["validated"] >= 3
    assert report["research_targets"]["candidates_at_least_20"] is True
    assert report["passes_all_targets"] is True
