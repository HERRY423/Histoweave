"""Input adapters for multi-objective Pareto analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .pareto import OBJECTIVE_DIRECTIONS, Direction, ObjectiveTable


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bootstrap_ci_width(
    values: Iterable[float],
    *,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 0,
) -> float:
    """Return the percentile-bootstrap confidence-interval width of the mean."""
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    if array.size == 1:
        return 0.0
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(array, size=array.size, replace=True).mean() for _ in range(n_boot)],
        dtype=float,
    )
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(high - low)


def load_memory_gb(scaling_dir: str | Path) -> dict[str, float]:
    """Load peak memory per method from a scalability study.

    The highest successful ``n_cells`` row is used for each method.  OOM rows
    describe a ceiling rather than a completed method run and are therefore not
    treated as comparable peak-memory measurements.
    """
    path = Path(scaling_dir)
    if path.is_dir():
        path = path / "scaling_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    candidates: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"method", "peak_rss_mb"}
        if not required.issubset(fields):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            category = str(row.get("category", "") or "").strip().lower()
            if category and category not in {"domain", "domain_detection", "spatial_domain"}:
                continue
            status = str(row.get("status", "") or "").strip().lower()
            if status and status not in {"ok", "success", "completed"}:
                continue
            method = str(row.get("method", "") or "").strip()
            memory_mb = _number(row.get("peak_rss_mb"))
            n_cells_value = _number(row.get("n_cells"))
            if not method or memory_mb is None:
                continue
            n_cells = int(n_cells_value) if n_cells_value is not None else 0
            candidates[method].append((n_cells, memory_mb / 1024.0))
    return {
        method: max(rows, key=lambda item: (item[0], item[1]))[1]
        for method, rows in candidates.items()
    }


def _config_base(config: str, row: dict[str, str] | None = None) -> str:
    if row:
        method = str(row.get("method", "") or "").strip()
        if method:
            return method
    return config.split("@", 1)[0]


def objective_tables_from_long_csv(
    path: str | Path,
    *,
    scaling_dir: str | Path | None = None,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 0,
) -> list[ObjectiveTable]:
    """Aggregate per-seed benchmark rows into per-dataset objective tables."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"accuracy": [], "speed": [], "base": None})
    )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "dataset" not in fields or "ari" not in fields:
            raise ValueError(f"{path} must contain 'dataset' and 'ari' columns")
        config_field = (
            "config" if "config" in fields else ("method" if "method" in fields else None)
        )
        if config_field is None:
            raise ValueError(f"{path} must contain a 'config' or 'method' column")
        has_speed = "seconds" in fields
        has_replicates = "seed" in fields

        for row in reader:
            status = str(row.get("status", "") or "").strip().lower()
            if status in {"failed", "error", "timeout", "oom", "skipped"}:
                continue
            dataset = str(row.get("dataset", "") or "").strip()
            config = str(row.get(config_field, "") or "").strip()
            accuracy = _number(row.get("ari"))
            if not dataset or not config or accuracy is None:
                continue
            cell = grouped[dataset][config]
            cell["accuracy"].append(accuracy)
            speed = _number(row.get("seconds")) if has_speed else None
            if speed is not None:
                cell["speed"].append(speed)
            cell["base"] = _config_base(config, row)

    if not grouped:
        raise ValueError(f"{path} produced no finite benchmark rows")
    memory = load_memory_gb(scaling_dir) if scaling_dir is not None else {}
    tables: list[ObjectiveTable] = []
    for dataset in sorted(grouped):
        points: dict[str, dict[str, float | None]] = {}
        for config, values in grouped[dataset].items():
            accuracy_values = values["accuracy"]
            objective_row: dict[str, float | None] = {
                "accuracy": float(np.mean(accuracy_values)),
            }
            if values["speed"]:
                objective_row["speed"] = float(np.mean(values["speed"]))
            if memory:
                objective_row["memory"] = memory.get(str(values["base"]))
            if has_replicates:
                objective_row["robustness"] = bootstrap_ci_width(
                    accuracy_values,
                    n_boot=n_boot,
                    ci=ci,
                    seed=seed,
                )
            points[config] = objective_row

        active: dict[str, Direction] = {}
        for objective, direction in OBJECTIVE_DIRECTIONS.items():
            if any(_number(row.get(objective)) is not None for row in points.values()):
                active[objective] = direction
        notes: list[str] = []
        if scaling_dir is not None and "memory" not in active:
            notes.append("No matching successful domain-detection memory rows were found.")
        tables.append(
            ObjectiveTable(
                dataset=dataset,
                table=points,
                directions=active,
                notes=notes,
            )
        )
    return tables


def objective_tables_from_landscape(
    path: str | Path,
    *,
    scaling_dir: str | Path | None = None,
) -> list[ObjectiveTable]:
    """Build accuracy/runtime objective tables from a landscape JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    performance = payload.get("performance")
    if not isinstance(performance, dict):
        raise ValueError(f"{path} has no object-valued 'performance' field")
    timings = payload.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    memory = load_memory_gb(scaling_dir) if scaling_dir is not None else {}
    accuracy_direction: Direction = "max" if payload.get("higher_is_better", True) else "min"

    tables: list[ObjectiveTable] = []
    for dataset in sorted(str(name) for name in performance):
        performance_row = performance.get(dataset)
        if not isinstance(performance_row, dict):
            continue
        timing_row = timings.get(dataset, {})
        if not isinstance(timing_row, dict):
            timing_row = {}
        points: dict[str, dict[str, float | None]] = {}
        for config, raw_accuracy in performance_row.items():
            accuracy = _number(raw_accuracy)
            if accuracy is None:
                continue
            config_name = str(config)
            objective_row: dict[str, float | None] = {"accuracy": accuracy}
            speed = _number(timing_row.get(config_name))
            if speed is not None:
                objective_row["speed"] = speed
            if memory:
                objective_row["memory"] = memory.get(_config_base(config_name))
            points[config_name] = objective_row
        if not points:
            continue
        directions: dict[str, Direction] = {"accuracy": accuracy_direction}
        if any("speed" in row for row in points.values()):
            directions["speed"] = "min"
        if any(_number(row.get("memory")) is not None for row in points.values()):
            directions["memory"] = "min"
        tables.append(ObjectiveTable(dataset=dataset, table=points, directions=directions))
    if not tables:
        raise ValueError(f"{path} produced no finite objective tables")
    return tables

