"""Isolated, resumable computational scaling benchmark harness."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, cast

import numpy as np

from .complexity import fit_complexity

DEFAULT_COMPUTE_METHODS: tuple[tuple[str, str], ...] = (
    ("qc", "basic_qc"),
    ("qc", "gene_complexity_qc"),
    ("qc", "library_size_qc"),
    ("qc", "mitochondrial_qc"),
    ("normalization", "arcsinh_transform"),
    ("normalization", "clr_per_cell"),
    ("normalization", "library_size_scale"),
    ("normalization", "log1p_cp10k"),
    ("normalization", "sqrt_transform"),
    ("normalization", "tfidf_l2"),
    ("domain_detection", "agglomerative"),
    ("domain_detection", "birch"),
    ("domain_detection", "bisecting_kmeans"),
    ("domain_detection", "dbscan"),
    ("domain_detection", "gaussian_mixture"),
    ("domain_detection", "kmeans"),
    ("domain_detection", "mean_shift"),
    ("domain_detection", "minibatch_kmeans"),
    ("domain_detection", "optics"),
    ("domain_detection", "spectral"),
    ("svg", "gearys_c"),
    ("svg", "morans_i"),
    ("svg", "spatial_variance_ratio"),
    ("neighborhood", "spatial_graph"),
    ("integration", "combat"),
    ("integration", "denoising_spatial_autoencoder"),
    ("integration", "graph_expression_autoencoder"),
    ("integration", "harmony"),
    ("integration", "spatial_autoencoder"),
    ("integration", "variational_spatial_autoencoder"),
)


@dataclass(frozen=True)
class ScalingConfig:
    scales: tuple[int, ...] = (1_000, 10_000, 100_000, 500_000, 1_000_000)
    n_genes: int = 2_000
    density: float = 0.05
    methods: tuple[tuple[str, str], ...] = DEFAULT_COMPUTE_METHODS
    per_method_timeout_s: float = 1_800.0
    per_method_mem_cap_gb: float = 58.0
    seed: int = 42
    prep: tuple[tuple[str, str], ...] = (("normalization", "log1p_cp10k"),)

    def __post_init__(self) -> None:
        if not self.scales or any(item < 1 for item in self.scales):
            raise ValueError("scales must contain positive cell counts")
        if tuple(sorted(set(self.scales))) != self.scales:
            raise ValueError("scales must be unique and sorted")
        if self.n_genes < 1 or not 0.0 < self.density <= 1.0:
            raise ValueError("n_genes and density are invalid")
        if self.per_method_timeout_s <= 0.0 or self.per_method_mem_cap_gb <= 0.0:
            raise ValueError("timeout and memory cap must be positive")


@dataclass
class ScalingRecord:
    category: str
    method: str
    n_cells: int
    n_genes: int
    density: float
    wall_seconds: float | None
    peak_rss_mb: float | None
    throughput_cells_per_s: float | None
    status: str
    error: str = ""
    version: str = ""


@dataclass
class ScalingResult:
    config: ScalingConfig
    records: list[ScalingRecord] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        totals = {
            name: sum(record.status == name for record in self.records)
            for name in ("ok", "timeout", "oom", "error", "skipped_after_limit")
        }
        reached = sorted(
            {
                record.method
                for record in self.records
                if record.n_cells == max(self.config.scales) and record.status == "ok"
            }
        )
        return {
            "schema_version": 1,
            "config": {
                **asdict(self.config),
                "methods": None,
                "n_methods": len(self.config.methods),
            },
            "totals": {"cells_measured": len(self.records), **totals},
            "max_scale": max(self.config.scales),
            "methods_reaching_max_scale": reached,
            "n_methods_reaching_max_scale": len(reached),
        }


def _worker(category: str, method: str, config: ScalingConfig, n_cells: int, queue: Any) -> None:
    try:
        from scipy import sparse

        from histoweave.datasets import make_scalable_synthetic
        from histoweave.plugins import create_method

        data = make_scalable_synthetic(
            n_cells, config.n_genes, density=config.density, seed=config.seed
        )
        # Dataset generation is outside the timed analysis region documented in
        # the report: prep -> densify -> method.run.
        started = time.perf_counter()
        for prep_category, prep_method in config.prep:
            data = create_method(prep_category, prep_method).run(data)
        if sparse.issparse(data.X):
            sparse_matrix = cast(sparse.spmatrix, data.X)
            data.X = sparse_matrix.toarray().astype(np.float32, copy=False)
        wrapped = create_method(category, method)
        wrapped.run(data)
        queue.put(
            {
                "status": "ok",
                "wall_seconds": time.perf_counter() - started,
                "version": wrapped.spec.version,
            }
        )
    except MemoryError:
        queue.put({"status": "oom", "error": "MemoryError"})
    except BaseException as exc:
        queue.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _peak_rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        import psutil

        process = psutil.Process(pid)
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue
        return rss / 1024**2
    except Exception:
        return None


def _run_cell(category: str, method: str, config: ScalingConfig, n_cells: int) -> ScalingRecord:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    process = context.Process(target=_worker, args=(category, method, config, n_cells, queue))
    process.start()
    peak = 0.0
    deadline = time.monotonic() + config.per_method_timeout_s
    status = "error"
    payload: dict[str, Any] = {}
    while process.is_alive() and time.monotonic() < deadline:
        peak = max(peak, _peak_rss_mb(process.pid) or 0.0)
        if peak > config.per_method_mem_cap_gb * 1024:
            status, payload = (
                "oom",
                {"error": f"peak RSS exceeded {config.per_method_mem_cap_gb} GB cap"},
            )
            process.terminate()
            break
        time.sleep(0.05)
    if process.is_alive():
        if status != "oom":
            status, payload = (
                "timeout",
                {"error": f"exceeded {config.per_method_timeout_s} s timeout"},
            )
        process.terminate()
    process.join(timeout=5.0)
    peak = max(peak, _peak_rss_mb(process.pid) or 0.0)
    if not payload:
        try:
            # Allow the multiprocessing queue's feeder thread a brief flush so
            # a completed worker is not misclassified as an empty result.
            payload = queue.get(timeout=0.5)
            status = str(payload.get("status", "error"))
        except Empty:
            status = "error"
            payload = {
                "error": (
                    f"worker exited with code {process.exitcode} without result"
                    if process.exitcode
                    else "worker produced no result"
                )
            }
    seconds = float(payload["wall_seconds"]) if status == "ok" else None
    return ScalingRecord(
        category,
        method,
        n_cells,
        config.n_genes,
        config.density,
        seconds,
        peak or None,
        n_cells / seconds if seconds else None,
        status,
        str(payload.get("error", "")),
        str(payload.get("version", "")),
    )


def run_scaling(config: ScalingConfig) -> ScalingResult:
    """Run every method/scale in an isolated process with tiered failure ceilings."""
    result = ScalingResult(config)
    for category, method in config.methods:
        limited = False
        for n_cells in config.scales:
            if limited:
                result.records.append(
                    ScalingRecord(
                        category,
                        method,
                        n_cells,
                        config.n_genes,
                        config.density,
                        None,
                        None,
                        None,
                        "skipped_after_limit",
                        "prior ceiling",
                    )
                )
                continue
            record = _run_cell(category, method, config, n_cells)
            result.records.append(record)
            limited = record.status in {"oom", "timeout"}
    return result


def write_scaling_artifacts(result: ScalingResult, out_dir: str | Path) -> dict[str, Path]:
    """Write CSV and JSON artifacts without requiring pandas."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "scaling_metrics.csv"
    fields = list(ScalingRecord.__dataclass_fields__)
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in result.records)
    fits: list[dict[str, Any]] = []
    for category, method in result.config.methods:
        ok = [
            item
            for item in result.records
            if item.category == category and item.method == method and item.status == "ok"
        ]
        time_fit = fit_complexity(
            [item.n_cells for item in ok], [item.wall_seconds or 0.0 for item in ok]
        )
        mem_fit = fit_complexity(
            [item.n_cells for item in ok], [item.peak_rss_mb or 0.0 for item in ok]
        )
        fits.append(
            {
                "category": category,
                "method": method,
                "n_success": len(ok),
                "time_exponent_k": time_fit.exponent,
                "time_r2": time_fit.r_squared,
                "mem_exponent_k": mem_fit.exponent,
                "mem_r2": mem_fit.r_squared,
                "max_scale_reached": max((item.n_cells for item in ok), default=0),
                "status": time_fit.status,
            }
        )
    complexity = output / "complexity_fits.csv"
    with complexity.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fits[0]))
        writer.writeheader()
        writer.writerows(fits)
    summary = output / "scaling_summary.json"
    summary.write_text(json.dumps(result.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metrics_csv": metrics, "complexity_fits_csv": complexity, "summary_json": summary}
