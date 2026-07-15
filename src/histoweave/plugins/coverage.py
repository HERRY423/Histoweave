"""Release-gate metrics for method coverage, maturity, and modalities."""

from __future__ import annotations

from .builtin.release_manifest import BETA_METHODS, PRODUCTION_METHODS, RESEARCH_METHODS
from .interfaces import METHOD_MATURITY_POLICIES, MethodMaturity
from .registry import list_methods


def method_coverage_report() -> dict[str, object]:
    """Return auditable coverage metrics and the requested release gates."""

    methods = list_methods()
    total = len(methods)
    release_names = PRODUCTION_METHODS | BETA_METHODS
    research = [
        method
        for method in methods
        if method["name"] in RESEARCH_METHODS
        and method.get("metadata", {}).get("track") == "research"
    ]
    release_methods = [
        method for method in methods if method["name"] in release_names
    ]
    release_total = len(release_methods)
    classified_names = {method["name"] for method in (*release_methods, *research)}
    unclassified = sum(1 for method in methods if method["name"] not in classified_names)
    beta_rank = METHOD_MATURITY_POLICIES[MethodMaturity.BETA].rank
    production_rank = METHOD_MATURITY_POLICIES[MethodMaturity.PRODUCTION].rank

    beta_plus = sum(
        1 for method in release_methods if int(method["maturity_rank"]) >= beta_rank
    )
    production_plus = sum(
        1 for method in release_methods if int(method["maturity_rank"]) >= production_rank
    )
    experimental = sum(
        1
        for method in release_methods
        if method["maturity"] == MethodMaturity.EXPERIMENTAL.value
    )
    deep_learning = sum(
        1 for method in release_methods if method["model_family"] == "deep_learning"
    )
    image_expression = sum(
        1
        for method in release_methods
        if {"image", "expression"}.issubset(method["modalities"])
    )
    external = sum(
        1 for method in release_methods if method["implementation"] == "external"
    )
    critical_names = {"cell2location", "banksy", "spatialde", "cellpose2", "scanvi"}
    critical = {method["name"]: method for method in methods if method["name"] in critical_names}
    critical_external = (
        set(critical) == critical_names
        and all(method["implementation"] == "external" for method in critical.values())
        and all(method["backends"] for method in critical.values())
    )

    def ratio(value: int) -> float:
        return value / release_total if release_total else 0.0

    ratios = {
        "beta_plus": ratio(beta_plus),
        "production_plus": ratio(production_plus),
        "experimental": ratio(experimental),
    }
    targets = {
        "total_methods_at_least_50": release_total >= 50,
        "beta_plus_is_100_percent": ratios["beta_plus"] == 1.0,
        "production_plus_above_80_percent": ratios["production_plus"] > 0.80,
        "experimental_below_5_percent": ratios["experimental"] < 0.05,
        "deep_learning_at_least_10": deep_learning >= 10,
        "image_expression_multimodal": image_expression > 0,
        "external_wrappers_at_least_15": external >= 15,
        "critical_methods_use_real_backends": critical_external,
    }
    research_targets = {"candidates_at_least_20": len(research) >= 20}
    return {
        "total_methods": total,
        "counts": {
            "beta_plus": beta_plus,
            "production_plus": production_plus,
            "experimental": experimental,
            "deep_learning": deep_learning,
            "image_expression": image_expression,
            "external_wrappers": external,
            "release_methods": release_total,
            "research_candidates": len(research),
            "unclassified_methods": unclassified,
        },
        "ratios": ratios,
        "targets": targets,
        "research_targets": research_targets,
        "passes_all_targets": all(targets.values()),
    }
