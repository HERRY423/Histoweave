from __future__ import annotations

import json

from histoweave.benchmark.pareto import (
    OBJECTIVE_DIRECTIONS,
    ObjectiveTable,
    analyze_dataset,
    build_report,
    knee_point,
    nondominated_sort,
    pareto_frontier,
    pareto_svg,
)
from histoweave.benchmark.pareto_io import (
    load_memory_gb,
    objective_tables_from_long_csv,
)


def _known_points():
    return {
        "accuracy": {"accuracy": 0.90, "speed": 100.0, "memory": 8.0},
        "fast": {"accuracy": 0.70, "speed": 2.0, "memory": 1.0},
        "balanced": {"accuracy": 0.85, "speed": 10.0, "memory": 2.0},
        "dominated": {"accuracy": 0.60, "speed": 50.0, "memory": 6.0},
        "balanced_tie": {"accuracy": 0.85, "speed": 10.0, "memory": 2.0},
    }


def test_known_frontier_sort_and_knee():
    points = _known_points()
    directions = {name: OBJECTIVE_DIRECTIONS[name] for name in ("accuracy", "speed", "memory")}
    frontier = pareto_frontier(points, directions)
    assert "dominated" not in frontier
    assert {"accuracy", "fast", "balanced", "balanced_tie"}.issubset(frontier)
    ranks = nondominated_sort(points, directions)
    assert ranks["dominated"] >= 1
    assert knee_point(points, directions) in points


def test_report_is_json_safe_and_svg_is_valid():
    table = ObjectiveTable("toy", _known_points())
    result = analyze_dataset(table)
    report = build_report([table]).to_dict()
    json.dumps(report, allow_nan=False)
    svg = pareto_svg(result)
    assert svg.startswith("<svg")
    assert "dataset toy" in svg
    assert result.knee in svg


def test_long_csv_and_scaling_adapters(tmp_path):
    benchmark = tmp_path / "benchmark_long.csv"
    benchmark.write_text(
        "dataset,config,method,seed,ari,seconds\n"
        "d1,a@sw0.0,a,1,0.5,2.0\n"
        "d1,a@sw0.0,a,2,0.7,4.0\n"
        "d1,b@sw0.0,b,1,0.8,6.0\n"
        "d1,b@sw0.0,b,2,0.8,6.0\n",
        encoding="utf-8",
    )
    scaling = tmp_path / "scaling_metrics.csv"
    scaling.write_text(
        "category,method,n_cells,peak_rss_mb,status\n"
        "domain_detection,a,1000,1024,ok\n"
        "domain_detection,a,10000,2048,ok\n"
        "domain_detection,a,100000,60000,oom\n"
        "domain_detection,b,10000,3072,ok\n",
        encoding="utf-8",
    )
    assert load_memory_gb(tmp_path) == {"a": 2.0, "b": 3.0}
    tables = objective_tables_from_long_csv(benchmark, scaling_dir=tmp_path, n_boot=30)
    assert len(tables) == 1
    assert tables[0].table["a@sw0.0"]["accuracy"] == 0.6
    assert tables[0].table["a@sw0.0"]["speed"] == 3.0
    assert tables[0].table["a@sw0.0"]["memory"] == 2.0
    assert "robustness" in tables[0].table["a@sw0.0"]

