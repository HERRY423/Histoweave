"""Release-gate metrics for method coverage, maturity, and modalities."""

from __future__ import annotations

from .builtin.release_manifest import (
    BETA_METHODS,
    EXPERIMENTAL_BASELINES,
    PRODUCTION_METHODS,
    RESEARCH_METHODS,
    VALIDATED_METHODS,
)
from .interfaces import METHOD_MATURITY_POLICIES, MethodMaturity
from .registry import list_methods


def _track_for(method: dict) -> str:
    """Return the release-manifest track for one method row."""
    name = method["name"]
    meta = method.get("metadata") or {}
    if name in VALIDATED_METHODS:
        return "validated"
    if name in PRODUCTION_METHODS:
        return "production"
    if name in BETA_METHODS:
        return "beta"
    if name in RESEARCH_METHODS or meta.get("track") == "research":
        return "research"
    if name in EXPERIMENTAL_BASELINES or meta.get("track") == "baseline":
        return "baseline"
    if meta.get("track") == "sota":
        # SOTA plugins are beta in the maturity manifest.
        return "beta"
    return "unclassified"


def method_coverage_report() -> dict[str, object]:
    """Return auditable coverage metrics and the requested release gates.

    After maturity de-inflation, the **release denominator** is
    production ∪ beta ∪ validated (not experimental baselines or research
    incubator methods).  Gates are calibrated to that honest inventory.

    Every registered method must fall into exactly one track
    (``production`` / ``beta`` / ``validated`` / ``research`` / ``baseline``).
    Unclassified methods are listed by name and fail the release gate.
    """

    methods = list_methods()
    total = len(methods)

    by_track: dict[str, list[str]] = {
        "validated": [],
        "production": [],
        "beta": [],
        "research": [],
        "baseline": [],
        "unclassified": [],
    }
    for method in methods:
        track = _track_for(method)
        by_track.setdefault(track, []).append(method["name"])

    for names in by_track.values():
        names.sort()

    unclassified_names = list(by_track["unclassified"])
    unclassified = len(unclassified_names)

    release_names = PRODUCTION_METHODS | BETA_METHODS | VALIDATED_METHODS
    release_methods = [method for method in methods if method["name"] in release_names]
    release_total = len(release_methods)
    research = [method for method in methods if method["name"] in by_track["research"]]

    beta_rank = METHOD_MATURITY_POLICIES[MethodMaturity.BETA].rank
    production_rank = METHOD_MATURITY_POLICIES[MethodMaturity.PRODUCTION].rank
    validated_rank = METHOD_MATURITY_POLICIES[MethodMaturity.VALIDATED].rank

    beta_plus = sum(1 for method in release_methods if int(method["maturity_rank"]) >= beta_rank)
    production_plus = sum(
        1 for method in release_methods if int(method["maturity_rank"]) >= production_rank
    )
    validated = sum(
        1 for method in release_methods if int(method["maturity_rank"]) >= validated_rank
    )
    experimental = sum(
        1 for method in release_methods if method["maturity"] == MethodMaturity.EXPERIMENTAL.value
    )
    deep_learning = sum(
        1 for method in release_methods if method["model_family"] == "deep_learning"
    )
    image_expression = sum(
        1 for method in release_methods if {"image", "expression"}.issubset(method["modalities"])
    )
    external = sum(1 for method in release_methods if method["implementation"] == "external")
    sota = sum(1 for method in methods if method.get("metadata", {}).get("track") == "sota")
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
        "validated_plus": ratio(validated),
        "experimental": ratio(experimental),
    }
    # Honest post-deinflation gates: release track is mature; experimental mass
    # lives in the research incubator / baseline track outside this denominator.
    targets = {
        "total_methods_at_least_50": total >= 50,
        "release_methods_at_least_40": release_total >= 40,
        "beta_plus_is_100_percent": ratios["beta_plus"] == 1.0,
        "production_plus_above_60_percent": ratios["production_plus"] > 0.60,
        "validated_at_least_3": validated >= 3,
        "experimental_below_5_percent": ratios["experimental"] < 0.05,
        "deep_learning_at_least_8": deep_learning >= 8,
        "image_expression_multimodal": image_expression > 0,
        "external_wrappers_at_least_15": external >= 15,
        "sota_plugins_at_least_5": sota >= 5,
        "critical_methods_use_real_backends": critical_external,
        "unclassified_is_zero": unclassified == 0,
        "all_methods_tracked": total
        == sum(len(v) for k, v in by_track.items() if k != "unclassified") + unclassified
        and unclassified == 0,
    }
    research_targets = {"candidates_at_least_20": len(research) >= 20}
    return {
        "total_methods": total,
        "counts": {
            "beta_plus": beta_plus,
            "production_plus": production_plus,
            "validated": validated,
            "experimental": experimental,
            "deep_learning": deep_learning,
            "image_expression": image_expression,
            "external_wrappers": external,
            "release_methods": release_total,
            "research_candidates": len(research),
            "baseline_methods": len(by_track["baseline"]),
            "sota_plugins": sota,
            "unclassified_methods": unclassified,
        },
        "tracks": {key: list(names) for key, names in by_track.items()},
        "unclassified_names": unclassified_names,
        "ratios": ratios,
        "targets": targets,
        "research_targets": research_targets,
        "passes_all_targets": all(targets.values()),
    }
