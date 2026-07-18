"""Tests for the causal performance landscape (interventional do-analysis)."""

import json

import numpy as np
import pytest

from histoweave.benchmark import (
    CausalLandscapeResult,
    causal_graph_svg,
    run_causal_landscape,
)

# A small, fast configuration: strong-signal methods that run without R containers.
_FAST_METHODS = ["kmeans", "spectral", "gaussian_mixture"]
_FAST_KW = dict(grid=(2.0, 7.0, 12.0), n_seeds=3, methods=_FAST_METHODS)


@pytest.fixture(scope="module")
def result() -> CausalLandscapeResult:
    return run_causal_landscape(**_FAST_KW)


def test_returns_result_with_expected_shape(result: CausalLandscapeResult):
    assert result.knob == "marker_gene_lift"
    assert result.lo == 2.0 and result.hi == 12.0
    assert set(result.effects) == set(_FAST_METHODS)
    # Full dose-response grid retained for every method.
    for m in _FAST_METHODS:
        assert set(result.grid_means[m]) == set(result.grid)
        for lv in result.grid:
            assert len(result.seed_replicates[m][lv]) == result.n_seeds


def test_determinism():
    """Identical seeds -> identical ACE (the interventional loop is deterministic)."""
    r1 = run_causal_landscape(**_FAST_KW)
    r2 = run_causal_landscape(**_FAST_KW)
    for m in _FAST_METHODS:
        assert r1.effects[m].ace == pytest.approx(r2.effects[m].ace, abs=1e-12)
        assert r1.effects[m].ci_low == pytest.approx(r2.effects[m].ci_low, abs=1e-12)
        assert r1.effects[m].ci_high == pytest.approx(r2.effects[m].ci_high, abs=1e-12)


def test_json_round_trip_is_strict(result: CausalLandscapeResult):
    """to_dict() must serialize under allow_nan=False (all NaN -> null)."""
    payload = json.dumps(result.to_dict(), allow_nan=False)
    reloaded = json.loads(payload)
    assert reloaded["schema_version"] == 1
    assert reloaded["knob"] == "marker_gene_lift"
    assert set(reloaded["effects"]) == set(_FAST_METHODS)


def test_ci_validity_and_significance_flag(result: CausalLandscapeResult):
    for e in result.effects.values():
        if not np.isfinite(e.ace):
            continue
        assert e.ci_low <= e.ace <= e.ci_high
        # significance is exactly "CI excludes zero"
        assert e.significant == bool(e.ci_low > 0.0 or e.ci_high < 0.0)


def test_scientific_direction_stronger_signal_helps(result: CausalLandscapeResult):
    """Acceptance check: raising marker_gene_lift must make domains easier.

    The strong clustering methods must show a positive, significant causal effect,
    and their dose-response must be broadly increasing.
    """
    for m in ["kmeans", "spectral"]:
        e = result.effects[m]
        assert e.ace > 0, f"{m} ACE should be positive (stronger signal -> higher ARI)"
        assert e.significant, f"{m} ACE should be significant"
    # Monotone-ish grid for the top method: hi anchor beats lo anchor.
    top = result.ranked_effects()[0]
    gm = result.grid_means[top.method]
    assert gm[result.hi] >= gm[result.lo]


def test_feature_displacement_records_confounding(result: CausalLandscapeResult):
    """The knob must move spatial_autocorrelation, and the co-moved features must be visible."""
    sa_lo, sa_hi = result.feature_shift("spatial_autocorrelation")
    assert np.isfinite(sa_lo) and np.isfinite(sa_hi)
    assert sa_hi > sa_lo, "higher lift should raise spatial autocorrelation"
    # effective_rank co-moves (the confounding we explicitly disclose).
    er_lo, er_hi = result.feature_shift("effective_rank_90")
    assert np.isfinite(er_lo) and np.isfinite(er_hi)
    # All 16 target-free features present at every level.
    for lv in result.grid:
        assert set(result.feature_displacement[lv]) == set(result.feature_order)


def test_causal_graph_svg_renders(result: CausalLandscapeResult):
    svg = causal_graph_svg(result)
    assert svg.strip().startswith("<svg")
    assert "do(marker_gene_lift)" in svg
    # editable text, not outlined paths
    assert "<text" in svg


def test_layout_knob_rejected():
    with pytest.raises(ValueError):
        run_causal_landscape(knob="layout", grid=(0, 1), n_seeds=2, methods=["kmeans"])
