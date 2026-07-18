"""Unit tests for benchmark harness default candidate filtering."""

from histoweave.benchmark.harness import _default_benchmark_candidates
from histoweave.plugins import MethodCategory, list_methods


def test_default_candidates_exclude_sota_and_research() -> None:
    names = set(_default_benchmark_candidates(MethodCategory.DOMAIN_DETECTION))
    assert names
    rows = {row["name"]: row for row in list_methods("domain_detection")}
    for name in names:
        meta = rows[name].get("metadata") or {}
        assert meta.get("track") not in {"sota", "research"}
    # First-class SOTA plugins must be opt-in.
    assert "spagcn" not in names
    assert "graphst" not in names
