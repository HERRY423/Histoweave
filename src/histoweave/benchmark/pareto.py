"""Multi-objective Pareto analysis for benchmark configurations.

The ordinary recommender intentionally returns a ranked list.  This module is
the complementary interface for decisions that should not hide trade-offs:
accuracy is maximised while runtime, memory, and uncertainty are minimised.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from typing import Any, Literal

import numpy as np

Direction = Literal["max", "min"]
ObjectiveRow = dict[str, float | None]
ObjectivePoints = dict[str, ObjectiveRow]

OBJECTIVE_DIRECTIONS: dict[str, Direction] = {
    "accuracy": "max",
    "speed": "min",
    "memory": "min",
    "robustness": "min",
}

OBJECTIVE_LABELS = {
    "accuracy": "accuracy (ARI)",
    "speed": "speed (s)",
    "memory": "memory (GB)",
    "robustness": "robustness (ARI CI width)",
}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_table(points: ObjectivePoints) -> ObjectivePoints:
    return {
        str(config): {str(name): _finite(value) for name, value in row.items()}
        for config, row in points.items()
    }


@dataclass
class ObjectiveTable:
    """Objective values for all configurations on one dataset."""

    dataset: str
    table: ObjectivePoints
    directions: dict[str, Direction] = field(default_factory=lambda: dict(OBJECTIVE_DIRECTIONS))
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dataset = str(self.dataset)
        self.table = _clean_table(self.table)
        invalid = {
            name: value
            for name, value in self.directions.items()
            if value not in {"max", "min"}
        }
        if invalid:
            raise ValueError(f"objective directions must be 'max' or 'min', got {invalid}")
        self.directions = dict(self.directions)
        self.notes = list(self.notes)

    @property
    def points(self) -> ObjectivePoints:
        """Alias retained for callers that refer to the values as points."""
        return self.table

    @property
    def objectives(self) -> list[str]:
        return [
            name
            for name in self.directions
            if any(_finite(row.get(name)) is not None for row in self.table.values())
        ]


def _utilities(
    points: ObjectivePoints,
    directions: dict[str, Direction],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    objectives = list(directions)
    utilities: dict[str, dict[str, float]] = {}
    for config, row in points.items():
        converted: dict[str, float] = {}
        for objective in objectives:
            value = _finite(row.get(objective))
            if value is None:
                converted[objective] = float("nan")
            else:
                converted[objective] = value if directions[objective] == "max" else -value
        utilities[str(config)] = converted
    return utilities, objectives


def _epsilon_by_objective(
    utilities: dict[str, dict[str, float]],
    objectives: list[str],
    eps_rel: float,
) -> dict[str, float]:
    if eps_rel < 0:
        raise ValueError("eps_rel must be non-negative")
    eps: dict[str, float] = {}
    for objective in objectives:
        values = np.asarray(
            [row[objective] for row in utilities.values() if np.isfinite(row[objective])],
            dtype=float,
        )
        span = float(np.ptp(values)) if values.size else 0.0
        eps[objective] = eps_rel * span if span > 0 else 0.0
    return eps


def _dominates(
    left: dict[str, float],
    right: dict[str, float],
    objectives: list[str],
    eps_abs: dict[str, float],
) -> bool:
    """Return whether *left* dominates *right* on their shared finite axes."""
    shared = [
        objective
        for objective in objectives
        if np.isfinite(left[objective]) and np.isfinite(right[objective])
    ]
    if not shared:
        return False
    strictly_better = False
    for objective in shared:
        tolerance = eps_abs[objective]
        if left[objective] < right[objective] - tolerance:
            return False
        if left[objective] > right[objective] + tolerance:
            strictly_better = True
    return strictly_better


def pareto_frontier(
    points: ObjectivePoints,
    directions: dict[str, Direction] | None = None,
    eps_rel: float = 1e-3,
) -> list[str]:
    """Return the sorted set of non-dominated configurations.

    Missing objective values are compared pairwise on the axes shared by both
    configurations.  A pair with no shared finite axis cannot dominate.
    """
    resolved = dict(directions or OBJECTIVE_DIRECTIONS)
    utilities, objectives = _utilities(points, resolved)
    eps_abs = _epsilon_by_objective(utilities, objectives, eps_rel)
    configs = list(utilities)
    return sorted(
        config
        for config in configs
        if not any(
            _dominates(utilities[other], utilities[config], objectives, eps_abs)
            for other in configs
            if other != config
        )
    )


def nondominated_sort(
    points: ObjectivePoints,
    directions: dict[str, Direction] | None = None,
    eps_rel: float = 1e-3,
) -> dict[str, int]:
    """Assign Pareto layers (rank 0 is the frontier)."""
    resolved = dict(directions or OBJECTIVE_DIRECTIONS)
    utilities, objectives = _utilities(points, resolved)
    eps_abs = _epsilon_by_objective(utilities, objectives, eps_rel)
    remaining = set(utilities)
    ranks: dict[str, int] = {}
    layer = 0
    while remaining:
        current = sorted(
            config
            for config in remaining
            if not any(
                _dominates(utilities[other], utilities[config], objectives, eps_abs)
                for other in remaining
                if other != config
            )
        )
        if not current:
            # Pairwise comparison with missing axes can theoretically create a
            # dominance cycle.  Keep output total and deterministic if it does.
            for config in sorted(remaining):
                ranks[config] = layer
            break
        for config in current:
            ranks[config] = layer
        remaining.difference_update(current)
        layer += 1
    return ranks


def knee_point(
    points: ObjectivePoints,
    directions: dict[str, Direction] | None = None,
) -> str | None:
    """Pick the point closest to the ideal corner after per-axis min-max scaling."""
    if not points:
        return None
    resolved = dict(directions or OBJECTIVE_DIRECTIONS)
    utilities, objectives = _utilities(points, resolved)
    scaled: dict[str, dict[str, float]] = {config: {} for config in utilities}
    for objective in objectives:
        values = np.asarray(
            [row[objective] for row in utilities.values() if np.isfinite(row[objective])],
            dtype=float,
        )
        if not values.size:
            for config in utilities:
                scaled[config][objective] = float("nan")
            continue
        low, high = float(values.min()), float(values.max())
        for config, row in utilities.items():
            value = row[objective]
            if not np.isfinite(value):
                scaled[config][objective] = float("nan")
            elif high == low:
                scaled[config][objective] = 0.5
            else:
                scaled[config][objective] = (value - low) / (high - low)

    best: str | None = None
    best_distance = float("inf")
    for config in sorted(utilities):
        squared = [
            (1.0 - scaled[config][objective]) ** 2
            for objective in objectives
            if np.isfinite(scaled[config][objective])
        ]
        if not squared:
            continue
        distance = float(np.sqrt(np.mean(squared)))
        if distance < best_distance:
            best, best_distance = config, distance
    return best


@dataclass
class ParetoDatasetResult:
    dataset: str
    objectives: list[str]
    frontier: list[str]
    ranks: dict[str, int]
    knee: str | None
    table: ObjectivePoints
    notes: list[str] = field(default_factory=list)

    @property
    def n_configs(self) -> int:
        return len(self.table)

    @property
    def n_frontier(self) -> int:
        return len(self.frontier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "objectives": list(self.objectives),
            "frontier": list(self.frontier),
            "ranks": dict(self.ranks),
            "knee": self.knee,
            "table": _clean_table(self.table),
            "n_configs": self.n_configs,
            "n_frontier": self.n_frontier,
            "notes": list(self.notes),
        }


def analyze_dataset(table: ObjectiveTable, eps_rel: float = 1e-3) -> ParetoDatasetResult:
    """Analyze one objective table."""
    objectives = table.objectives
    directions = {name: table.directions[name] for name in objectives}
    cleaned = _clean_table(table.table)
    return ParetoDatasetResult(
        dataset=table.dataset,
        objectives=objectives,
        frontier=pareto_frontier(cleaned, directions, eps_rel=eps_rel),
        ranks=nondominated_sort(cleaned, directions, eps_rel=eps_rel),
        knee=knee_point(cleaned, directions),
        table=cleaned,
        notes=list(table.notes),
    )


@dataclass
class ParetoReport:
    objectives: list[str]
    directions: dict[str, Direction]
    eps_rel: float
    datasets: dict[str, dict[str, Any]]
    frontier_frequency: dict[str, int]
    notes: list[str] = field(default_factory=list)
    schema_version: int = 1

    @property
    def n_datasets(self) -> int:
        return len(self.datasets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objectives": list(self.objectives),
            "directions": dict(self.directions),
            "eps_rel": float(self.eps_rel),
            "datasets": self.datasets,
            "frontier_frequency": dict(self.frontier_frequency),
            "n_datasets": self.n_datasets,
            "notes": list(self.notes),
        }


def build_report(tables: list[ObjectiveTable], eps_rel: float = 1e-3) -> ParetoReport:
    """Build a JSON-safe multi-dataset Pareto report."""
    if eps_rel < 0:
        raise ValueError("eps_rel must be non-negative")
    datasets: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    objective_set: set[str] = set()
    report_directions: dict[str, Direction] = {}
    for table in tables:
        if table.dataset in datasets:
            raise ValueError(f"duplicate dataset in Pareto tables: {table.dataset!r}")
        result = analyze_dataset(table, eps_rel=eps_rel)
        datasets[table.dataset] = result.to_dict()
        counts.update(result.frontier)
        for objective in result.objectives:
            objective_set.add(objective)
            report_directions.setdefault(objective, table.directions[objective])

    ordered_objectives = [name for name in OBJECTIVE_DIRECTIONS if name in objective_set]
    ordered_objectives.extend(sorted(objective_set.difference(ordered_objectives)))
    ordered_directions = {name: report_directions[name] for name in ordered_objectives}
    frequency = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return ParetoReport(
        objectives=ordered_objectives,
        directions=ordered_directions,
        eps_rel=eps_rel,
        datasets=datasets,
        frontier_frequency=frequency,
        notes=[
            "Pareto frontier = non-dominated configurations; no single method is "
            "asserted as universally best."
        ],
    )


def _short_label(config: str) -> str:
    method, _, suffix = config.partition("@sw")
    aliases = {
        "agglomerative": "agg",
        "gaussian_mixture": "gmm",
        "kmeans": "km",
        "spectral": "spec",
        "birch": "birch",
    }
    label = aliases.get(method, method[:8])
    return f"{label}@{suffix}" if suffix else label


def pareto_svg(
    result: ParetoDatasetResult,
    directions: dict[str, Direction] | None = None,
) -> str:
    """Render a dependency-free SVG of accuracy against available cost axes."""
    resolved = dict(directions or OBJECTIVE_DIRECTIONS)
    x_axes = [name for name in ("speed", "memory", "robustness") if name in result.objectives]
    if "accuracy" not in result.objectives:
        x_axes = result.objectives[1:4]
        y_axis = result.objectives[0] if result.objectives else None
    else:
        y_axis = "accuracy"
    if y_axis is None or not x_axes:
        raise ValueError("Pareto SVG requires at least two finite objectives")

    panel = 230
    gap = 55
    left = 92
    top = 50
    bottom = 74
    width = left + len(x_axes) * panel + (len(x_axes) - 1) * gap + 28
    height = top + panel + bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Liberation Sans, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        f'<text x="{width / 2:.1f}" y="24" font-size="14" text-anchor="middle" '
        f'font-weight="bold">Pareto frontier — dataset {escape(result.dataset)} '
        f'({result.n_frontier}/{result.n_configs} non-dominated)</text>',
    ]
    frontier = set(result.frontier)

    for panel_index, x_axis in enumerate(x_axes):
        x0 = left + panel_index * (panel + gap)
        finite_rows: list[tuple[str, float, float]] = []
        for config, row in result.table.items():
            x_value = _finite(row.get(x_axis))
            y_value = _finite(row.get(y_axis))
            if x_value is not None and y_value is not None:
                finite_rows.append((config, x_value, y_value))
        if not finite_rows:
            continue
        xs = np.asarray([row[1] for row in finite_rows], dtype=float)
        ys = np.asarray([row[2] for row in finite_rows], dtype=float)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())

        def sx(
            value: float,
            x_origin: float = x0,
            x_minimum: float = xmin,
            x_maximum: float = xmax,
        ) -> float:
            if x_maximum == x_minimum:
                return x_origin + panel / 2
            return x_origin + panel * (value - x_minimum) / (x_maximum - x_minimum)

        def sy(
            value: float,
            y_minimum: float = ymin,
            y_maximum: float = ymax,
        ) -> float:
            if y_maximum == y_minimum:
                return top + panel / 2
            return top + panel * (1 - (value - y_minimum) / (y_maximum - y_minimum))

        lines.extend(
            [
                f'<rect x="{x0:.1f}" y="{top:.1f}" width="{panel:.1f}" height="{panel:.1f}" '
                'fill="#FAF9F3" stroke="#000000" stroke-width="1"/>',
                f'<text x="{x0 + panel / 2:.1f}" y="{top + panel + 40:.1f}" font-size="11" '
                f'text-anchor="middle">{escape(OBJECTIVE_LABELS.get(x_axis, x_axis))} '
                f'({resolved.get(x_axis, "min")})</text>',
            ]
        )
        if panel_index == 0:
            lines.append(
                f'<text x="32" y="{top + panel / 2:.1f}" font-size="11" text-anchor="middle" '
                f'transform="rotate(-90 32 {top + panel / 2:.1f})">'
                f'{escape(OBJECTIVE_LABELS.get(y_axis, y_axis))} '
                f'({resolved.get(y_axis, "max")})</text>'
            )
        frontier_rows = sorted(
            [row for row in finite_rows if row[0] in frontier],
            key=lambda row: row[1],
        )
        if len(frontier_rows) > 1:
            coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for _, x, y in frontier_rows)
            lines.append(
                f'<polyline points="{coords}" fill="none" stroke="#FF9400" '
                'stroke-width="2" stroke-dasharray="4 3"/>'
            )
        for config, x_value, y_value in finite_rows:
            is_frontier = config in frontier
            is_knee = config == result.knee
            radius = 6 if is_knee else (5 if is_frontier else 3)
            fill = "#0279EE" if is_knee else ("#FF9400" if is_frontier else "#ECE9E2")
            title = (
                f"{escape(config)} ({escape(x_axis)}={x_value:.3g}, "
                f"{escape(y_axis)}={y_value:.3g})"
            )
            lines.append(
                f'<circle cx="{sx(x_value):.1f}" cy="{sy(y_value):.1f}" '
                f'r="{radius}" fill="{fill}" stroke="#000000" stroke-width="0.7">'
                f'<title>{title}</title></circle>'
            )
            if is_frontier:
                lines.append(
                    f'<text x="{sx(x_value) + 7:.1f}" y="{sy(y_value) - 7:.1f}" '
                    f'font-size="8" fill="#111111">{escape(_short_label(config))}</text>'
                )
        for fraction in (0.0, 0.5, 1.0):
            x_value = xmin + fraction * (xmax - xmin)
            y_value = ymin + fraction * (ymax - ymin)
            lines.append(
                f'<text x="{sx(x_value):.1f}" y="{top + panel + 17:.1f}" font-size="8.5" '
                f'text-anchor="middle" fill="#333333">{x_value:.3g}</text>'
            )
            if panel_index == 0:
                lines.append(
                    f'<text x="{x0 - 7:.1f}" y="{sy(y_value) + 3:.1f}" font-size="8.5" '
                    f'text-anchor="end" fill="#333333">{y_value:.3g}</text>'
                )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"

