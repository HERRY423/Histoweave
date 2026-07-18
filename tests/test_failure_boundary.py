"""Tests for adversarial failure-boundary mapping.

These exercise the engine (axes, corrected SVG scorer, boundary detection) and
the ``histoweave benchmark-boundary`` CLI on a small, fast slice so the whole
path is guarded without running the full multi-minute study.
"""

from __future__ import annotations

import json

from histoweave.benchmark.failure_boundary import (
    DEFAULT_TAU,
    SweepPoint,
    build_axes,
    detect_boundary,
    make_svg_task_fixed,
    run_boundary_study,
    write_study_outputs,
)
from histoweave.benchmark.harness import run_benchmark
from histoweave.cli import main as cli_main
from histoweave.datasets import make_synthetic


def test_build_axes_cover_all_tasks():
    axes = build_axes()
    keys = {(a.task, a.param) for a in axes}
    # one knob per (task, axis); the eight documented sweeps must all be present
    assert ("domain_detection", "marker_gene_lift") in keys
    assert ("domain_detection", "noise") in keys
    assert ("domain_detection", "n_domains") in keys
    assert ("domain_detection", "n_cells") in keys
    assert ("deconvolution", "noise") in keys
    assert ("deconvolution", "n_cell_types") in keys
    assert ("svg", "marker_gene_lift") in keys
    assert ("svg", "noise") in keys
    # every axis is direction-tagged and has a usable grid
    for a in axes:
        assert a.degrade_direction in {"increasing", "decreasing"}
        assert len(a.values) >= 3


def test_deconvolution_n_cell_types_capped_at_generator_capacity():
    # DC_BASE has n_genes=60 and 5 markers/type -> at most 12 marker-distinct
    # programs. The axis must not exceed that or the ground-truth outgrows the
    # prediction (shape mismatch) and we'd measure a generator limit, not a method.
    axis = next(a for a in build_axes() if a.task == "deconvolution" and a.param == "n_cell_types")
    assert max(axis.values) <= 12


def test_corrected_svg_scorer_is_a_true_precision_at_k():
    # The corrected scorer must be bounded in [0, 1] (the shipped harness scorer
    # could exceed 1.0 because it did not truncate to top-k).
    ds = make_synthetic(n_cells=400, n_domains=4, marker_gene_lift=6.0, noise=0.2, seed=0)
    task = make_svg_task_fixed(ds, k=10)
    result = run_benchmark(task, methods=["morans_i"])
    score = result.leaderboard[0]["score"]
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_detect_boundary_finds_monotone_crossing():
    # Synthesize a clean easy->hard decay that crosses tau=0.7 between value 3 and 4
    # on a "decreasing means harder" axis (like marker_gene_lift).
    axis = next(
        a for a in build_axes() if a.task == "domain_detection" and a.param == "marker_gene_lift"
    )
    # means: high at large lift, low at small lift
    fake = {5.0: 0.95, 4.0: 0.90, 3.0: 0.80, 2.0: 0.60, 1.0: 0.30}
    points = []
    for v, m in fake.items():
        for seed in range(3):
            points.append(SweepPoint(param_value=v, seed=seed, method="fake", score=m, seconds=0.0))
    b = detect_boundary(points, axis, "fake", tau=0.7)
    assert b.verdict == "boundary_found"
    # crossing is between lift 3.0 (0.80) and 2.0 (0.60): interpolated ~2.5
    assert 2.0 < b.x_star < 3.0
    assert b.safe_low == 3.0  # smallest lift still >= tau


def test_detect_boundary_flags_never_and_always():
    axis = next(a for a in build_axes() if a.task == "domain_detection" and a.param == "noise")
    # all below tau -> never_acceptable
    lo = [
        SweepPoint(param_value=float(v), seed=0, method="m", score=0.2, seconds=0.0)
        for v in axis.values
    ]
    assert detect_boundary(lo, axis, "m", tau=0.7).verdict == "never_acceptable"
    # all above tau -> always_acceptable
    hi = [
        SweepPoint(param_value=float(v), seed=0, method="m", score=0.95, seconds=0.0)
        for v in axis.values
    ]
    assert detect_boundary(hi, axis, "m", tau=0.7).verdict == "always_acceptable"


def test_failed_runs_count_as_unacceptable():
    # nan scores (method raised) must be treated as below tau, not ignored.
    axis = next(a for a in build_axes() if a.task == "domain_detection" and a.param == "noise")
    pts = []
    for i, v in enumerate(axis.values):
        score = 0.95 if i < 2 else float("nan")  # runs on the two easiest points only
        pts.append(SweepPoint(param_value=float(v), seed=0, method="m", score=score, seconds=None))
    b = detect_boundary(pts, axis, "m", tau=0.7)
    assert b.verdict == "boundary_found"


def test_run_boundary_study_small_slice(tmp_path):
    # End-to-end on a tiny slice: one axis, two methods, two seeds.
    result = run_boundary_study(
        tasks=["domain_detection"],
        params=["marker_gene_lift"],
        methods=["kmeans", "banksy_py"],
        tau=DEFAULT_TAU,
        n_seeds=2,
        progress=False,
        include_fingerprints=False,
    )
    cards = result.cards_dataframe()
    assert len(cards) == 2
    assert set(cards["method"]) == {"kmeans", "banksy_py"}
    # kmeans has a high ceiling on clean data
    kmeans_best = float(cards.set_index("method").loc["kmeans", "best_score"])
    assert kmeans_best > 0.9
    # writing produces all four artifacts and valid JSON
    write_study_outputs(result, tmp_path)
    for fname in (
        "boundary_long.csv",
        "safe_operating_cards.csv",
        "safe_operating_cards.json",
        "safe_operating_cards.md",
    ):
        assert (tmp_path / fname).exists()
    bundle = json.loads((tmp_path / "safe_operating_cards.json").read_text())
    assert bundle["tau"] == DEFAULT_TAU
    assert len(bundle["cards"]) == 2


def test_cli_benchmark_boundary_writes_cards(tmp_path):
    rc = cli_main(
        [
            "benchmark-boundary",
            "--task",
            "domain_detection",
            "--axis",
            "marker_gene_lift",
            "--methods",
            "kmeans",
            "--tau",
            "0.7",
            "--seeds",
            "2",
            "--out",
            str(tmp_path),
            "--json",
            "--no-fingerprints",
        ]
    )
    assert rc == 0
    bundle = json.loads((tmp_path / "safe_operating_cards.json").read_text())
    assert len(bundle["cards"]) == 1
    assert bundle["cards"][0]["method"] == "kmeans"
    # long table has 1 method x 10 grid values x 2 seeds = 20 rows
    import pandas as pd

    long_df = pd.read_csv(tmp_path / "boundary_long.csv")
    assert len(long_df) == 10 * 2
