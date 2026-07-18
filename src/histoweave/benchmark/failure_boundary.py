"""Adversarial failure-boundary mapping for HistoWeave methods.

Instead of asking *"which method scores best?"*, this module asks
*"where does each method break?"* — it sweeps a **single** synthetic-data
parameter across its axis, evaluates a method with the existing benchmark
harness at every point (averaged over replicate seeds), and locates the
**critical parameter value** at which the method crosses an acceptability
threshold ``tau`` (default 0.7).

The output is a set of **Safe Operating Cards**: for every (method, axis)
pair, the parameter interval where the method can be trusted, and the exact
point where it fails.

Design principles
-----------------
* **Non-invasive.** Built entirely on top of the public
  :func:`histoweave.benchmark.harness.run_benchmark` / :class:`Task` API and
  the synthetic generators in :mod:`histoweave.datasets`. It does not modify
  any registered method.
* **Metric-correct SVG scoring.** The SVG task shipped in the harness has a
  mis-specified ``precision@k`` (it does not truncate a method's output to
  top-*k*, and only reads ``uns['svg']`` while some methods write to a
  method-named key). :func:`make_svg_task_fixed` provides a corrected task so
  SVG boundaries reflect method robustness rather than plumbing bugs.
* **Direction-aware, honest boundaries.** The detector knows whether a knob
  is expected to *degrade* the score as it increases (e.g. ``noise``) or as it
  decreases (e.g. ``marker_gene_lift`` = signal strength). It reports
  monotonicity diagnostics and explicitly flags "never fails" / "always fails"
  within the tested range rather than inventing a crossing point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from histoweave.benchmark.harness import (
    Task,
    deconvolution_task,
    domain_detection_task,
    run_benchmark,
)
from histoweave.data import SpatialTable
from histoweave.datasets import (
    make_mixture_synthetic,
    make_synthetic,
)
from histoweave.logging import get_logger
from histoweave.plugins import MethodCategory
from histoweave.workflow import PipelineStep

_LOGGER = get_logger("histoweave.benchmark.failure_boundary")

DEFAULT_TAU = 0.7


# ======================================================================
# Corrected SVG task (precision@k that truncates and reads the right key)
# ======================================================================
def _collect_true_markers(dataset: SpatialTable) -> set[str]:
    markers: set[str] = set()
    for genes in dataset.uns.get("marker_genes", {}).values():
        markers.update(str(g) for g in genes)
    return markers


def _extract_top_genes(result: SpatialTable, k: int) -> list[str]:
    """Pull a method's ranked SVG list, tolerant of where it stored it.

    Preference order:
      1. ``uns['svg']['top_genes']`` (morans_i, weave_* natives)
      2. the first ``uns[<key>]`` dict that carries a ``top_genes`` list
         (gearys_c -> 'gearys_c_score', spatial_variance_ratio -> its name)
    The list is truncated to the first *k* entries so the score is a true
    precision@k in [0, 1].
    """
    candidates: list[dict] = []
    svg_block = result.uns.get("svg")
    if isinstance(svg_block, dict) and svg_block.get("top_genes"):
        candidates.append(svg_block)
    if not candidates:
        for value in result.uns.values():
            if isinstance(value, dict) and value.get("top_genes"):
                candidates.append(value)
                break
    if not candidates:
        return []
    top_raw = candidates[0]["top_genes"][:k]
    names: list[str] = []
    for entry in top_raw:
        if isinstance(entry, dict):
            names.append(str(entry.get("gene", "")))
        else:
            names.append(str(entry))
    return [n for n in names if n]


def make_svg_task_fixed(dataset: SpatialTable, k: int = 10) -> Task:
    """SVG detection task with a corrected precision@k scorer.

    ``score = |top_k(pred) ∩ true_markers| / k`` — bounded in [0, 1], and
    robust to which ``uns`` key a method uses. ``k`` is capped at the number
    of true marker genes so a perfect method can reach 1.0.
    """
    true_markers = _collect_true_markers(dataset)
    k_eff = min(k, len(true_markers)) or k

    def precision_at_k(result: SpatialTable, ref: SpatialTable) -> float:
        pred = set(_extract_top_genes(result, k_eff))
        if not pred:
            return 0.0
        hits = len(pred & true_markers)
        return hits / float(k_eff)

    return Task(
        name="svg",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        dataset=dataset,
        score=precision_at_k,
        output_key="svg",
        higher_is_better=True,
        prep=[PipelineStep(MethodCategory.NORMALIZATION, "log1p_cp10k")],
    )


# ======================================================================
# Axis definitions — one tunable knob per (task, axis)
# ======================================================================
@dataclass(frozen=True)
class SweepAxis:
    """A single synthetic-data knob to sweep for a given task.

    Parameters
    ----------
    task
        ``"domain_detection"`` | ``"deconvolution"`` | ``"svg"``.
    param
        Name of the generator argument being swept (the free variable).
    values
        Ordered parameter grid (ascending).
    degrade_direction
        ``"increasing"`` if larger values make the problem *harder* (score
        expected to fall as the value rises, e.g. ``noise``);
        ``"decreasing"`` if smaller values are harder (e.g.
        ``marker_gene_lift`` = signal strength, ``n_cells``).
    make_dataset
        ``(value, seed) -> SpatialTable`` builder; all non-swept knobs are
        pinned to the documented baseline inside the closure.
    make_task
        ``(dataset) -> Task`` builder.
    label
        Human-readable axis label for cards/figures.
    unit
        Optional unit string for display.
    """

    task: str
    param: str
    values: tuple[float, ...]
    degrade_direction: str  # "increasing" | "decreasing"
    make_dataset: Callable[[float, int], SpatialTable]
    make_task: Callable[[SpatialTable], Task]
    label: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.degrade_direction not in {"increasing", "decreasing"}:
            raise ValueError("degrade_direction must be 'increasing' or 'decreasing'")
        if len(self.values) < 3:
            raise ValueError("a sweep needs at least 3 grid points")


# --- Baselines (documented, held constant while one knob varies) --------
# domain_detection baseline: moderate difficulty blob tissue.
DD_BASE: dict[str, Any] = dict(
    n_cells=600, n_genes=50, n_domains=4, marker_gene_lift=6.0, noise=0.25, layout="blob"
)
# deconvolution baseline: mixture spots.
DC_BASE: dict[str, Any] = dict(n_spots=400, n_genes=60, n_cell_types=4, noise=0.15)
# svg baseline: same family as domain detection (marker recovery).
SVG_BASE: dict[str, Any] = dict(
    n_cells=600, n_genes=50, n_domains=4, marker_gene_lift=6.0, noise=0.25, layout="blob"
)


def _dd_dataset(
    overrides: Callable[[float], dict[str, Any]],
) -> Callable[[float, int], SpatialTable]:
    def build(value: float, seed: int) -> SpatialTable:
        cfg = dict(DD_BASE)
        cfg.update(overrides(value))
        return make_synthetic(seed=seed, **cfg)

    return build


def build_axes() -> list[SweepAxis]:
    """Construct the default set of sweep axes for all three tasks.

    Ranges are deliberately pushed wide so each sweep *crosses* tau=0.7 for
    at least some methods; the driver auto-flags axes that do not.
    """
    axes: list[SweepAxis] = []

    # ---- domain_detection (make_synthetic, ARI) -----------------------
    def dd_build(param: str):
        def build(value: float, seed: int) -> SpatialTable:
            cfg = dict(DD_BASE)
            if param == "n_domains" or param == "n_cells":
                cfg[param] = int(value)
            else:
                cfg[param] = float(value)
            return make_synthetic(seed=seed, **cfg)

        return build

    axes.append(
        SweepAxis(
            task="domain_detection",
            param="marker_gene_lift",
            values=tuple(np.round(np.arange(5.0, 0.49, -0.5), 3).tolist()),  # 5.0 -> 0.5
            degrade_direction="decreasing",
            make_dataset=dd_build("marker_gene_lift"),
            make_task=lambda ds: domain_detection_task(dataset=ds),
            label="Marker gene signal strength (lift)",
            unit="Poisson lift",
        )
    )
    axes.append(
        SweepAxis(
            task="domain_detection",
            param="noise",
            values=tuple(np.round(np.arange(0.1, 2.01, 0.15), 3).tolist()),  # 0.1 -> 2.0
            degrade_direction="increasing",
            make_dataset=dd_build("noise"),
            make_task=lambda ds: domain_detection_task(dataset=ds),
            label="Multiplicative expression noise (sigma)",
            unit="lognormal sigma",
        )
    )
    axes.append(
        SweepAxis(
            task="domain_detection",
            param="n_domains",
            values=tuple(float(x) for x in range(2, 21, 2)),  # 2 -> 20
            degrade_direction="increasing",
            make_dataset=dd_build("n_domains"),
            make_task=lambda ds: domain_detection_task(dataset=ds),
            label="Number of spatial domains",
            unit="domains",
        )
    )
    axes.append(
        SweepAxis(
            task="domain_detection",
            param="n_cells",
            values=tuple(float(x) for x in (60, 100, 150, 200, 300, 450, 600, 900)),
            degrade_direction="decreasing",
            make_dataset=dd_build("n_cells"),
            make_task=lambda ds: domain_detection_task(dataset=ds),
            label="Sample size (number of cells)",
            unit="cells",
        )
    )

    # ---- deconvolution (make_mixture_synthetic, 1 - RMSD) -------------
    def dc_build(param: str):
        def build(value: float, seed: int) -> SpatialTable:
            cfg = dict(DC_BASE)
            if param == "n_cell_types":
                cfg[param] = int(value)
            else:
                cfg[param] = float(value)
            return make_mixture_synthetic(seed=seed, **cfg)

        return build

    axes.append(
        SweepAxis(
            task="deconvolution",
            param="noise",
            values=tuple(np.round(np.arange(0.05, 1.21, 0.1), 3).tolist()),  # 0.05 -> 1.15
            degrade_direction="increasing",
            make_dataset=dc_build("noise"),
            make_task=lambda ds: deconvolution_task(dataset=ds),
            label="Mixture expression noise (sigma)",
            unit="lognormal sigma",
        )
    )
    axes.append(
        SweepAxis(
            task="deconvolution",
            param="n_cell_types",
            # Capped at 12: with DC_BASE n_genes=60 and 5 markers/type the mixture
            # generator can only synthesise 12 marker-distinct programs. Beyond that
            # the extra cell types have no unique signature and the ground-truth
            # proportion matrix outgrows any marker-based prediction (shape mismatch),
            # which would reflect a generator capacity limit, not method robustness.
            values=tuple(float(x) for x in range(2, 13, 1)),  # 2 -> 12
            degrade_direction="increasing",
            make_dataset=dc_build("n_cell_types"),
            make_task=lambda ds: deconvolution_task(dataset=ds),
            label="Number of cell types in mixture",
            unit="cell types",
        )
    )

    # ---- svg (make_synthetic, corrected precision@k) -----------------
    def svg_build(param: str):
        def build(value: float, seed: int) -> SpatialTable:
            cfg = dict(SVG_BASE)
            cfg[param] = float(value)
            return make_synthetic(seed=seed, **cfg)

        return build

    axes.append(
        SweepAxis(
            task="svg",
            param="marker_gene_lift",
            values=tuple(np.round(np.arange(5.0, 0.49, -0.5), 3).tolist()),
            degrade_direction="decreasing",
            make_dataset=svg_build("marker_gene_lift"),
            make_task=lambda ds: make_svg_task_fixed(ds, k=10),
            label="Marker gene signal strength (lift)",
            unit="Poisson lift",
        )
    )
    axes.append(
        SweepAxis(
            task="svg",
            param="noise",
            values=tuple(np.round(np.arange(0.1, 2.01, 0.15), 3).tolist()),
            degrade_direction="increasing",
            make_dataset=svg_build("noise"),
            make_task=lambda ds: make_svg_task_fixed(ds, k=10),
            label="Multiplicative expression noise (sigma)",
            unit="lognormal sigma",
        )
    )

    return axes


# ======================================================================
# Sweep execution
# ======================================================================
@dataclass
class SweepPoint:
    param_value: float
    seed: int
    method: str
    score: float
    seconds: float | None
    error: str | None = None


def run_sweep(
    axis: SweepAxis,
    methods: Sequence[str] | None = None,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    progress: bool = True,
) -> list[SweepPoint]:
    """Evaluate methods across an axis at every grid point x every seed.

    Returns a flat list of :class:`SweepPoint`. A method that raises at a
    point is recorded with ``score=nan`` and the error string, so a method
    that fails to *run* is distinguishable from one that runs but scores low.
    """
    points: list[SweepPoint] = []
    for value in axis.values:
        for seed in seeds:
            dataset = axis.make_dataset(value, seed)
            task = axis.make_task(dataset)
            result = run_benchmark(task, methods=list(methods) if methods else None)
            for row in result.leaderboard:
                raw = row.get("score")
                score = float(raw) if raw is not None and np.isfinite(raw) else float("nan")
                points.append(
                    SweepPoint(
                        param_value=float(value),
                        seed=int(seed),
                        method=str(row["method"]),
                        score=score,
                        seconds=row.get("seconds"),
                        error=row.get("error"),
                    )
                )
        if progress:
            _LOGGER.info("sweep %s/%s value=%s done", axis.task, axis.param, value)
    return points


# ======================================================================
# Boundary detection
# ======================================================================
@dataclass
class Boundary:
    """The failure boundary of one method along one axis."""

    task: str
    param: str
    method: str
    tau: float
    degrade_direction: str
    x_star: float | None  # interpolated crossing (acceptable -> unacceptable)
    safe_low: float | None  # inclusive lower end of the trusted interval
    safe_high: float | None  # inclusive upper end of the trusted interval
    verdict: str  # 'boundary_found' | 'never_acceptable' | 'always_acceptable' | 'no_runs'
    best_score: float
    worst_score: float
    monotonic: bool
    spearman: float | None
    n_points: int
    curve_values: list[float] = field(default_factory=list)
    curve_mean: list[float] = field(default_factory=list)
    curve_std: list[float] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        # keep the heavy curves out of the flat summary table
        for key in ("curve_values", "curve_mean", "curve_std"):
            d.pop(key, None)
        return d


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.all(np.isnan(y)):
        return None
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return None
    xr = np.argsort(np.argsort(x[mask]))
    yr = np.argsort(np.argsort(y[mask]))
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _interp_crossing(x0: float, s0: float, x1: float, s1: float, tau: float) -> float:
    """Linear interpolation of the parameter value where score == tau."""
    if s1 == s0:
        return x1
    frac = (tau - s0) / (s1 - s0)
    return float(x0 + frac * (x1 - x0))


def detect_boundary(
    points: list[SweepPoint],
    axis: SweepAxis,
    method: str,
    tau: float = DEFAULT_TAU,
) -> Boundary:
    """Locate the acceptable->unacceptable crossing for one method on one axis.

    The grid is first ordered by *problem difficulty* (easy -> hard) using
    ``degrade_direction`` so the crossing is always interpreted as
    "score falls through tau as the problem gets harder". The reported
    ``x_star``/``safe_*`` are converted back to the native parameter scale.
    """
    mine = [p for p in points if p.method == method]
    # aggregate mean score per parameter value
    values = sorted({p.param_value for p in mine})
    mean_by_val: dict[float, float] = {}
    std_by_val: dict[float, float] = {}
    for v in values:
        scores = np.array([p.score for p in mine if p.param_value == v], dtype=float)
        with np.errstate(invalid="ignore"):
            mean_by_val[v] = float(np.nanmean(scores)) if scores.size else float("nan")
            std_by_val[v] = float(np.nanstd(scores)) if scores.size else float("nan")

    native_vals = np.array(values, dtype=float)
    means = np.array([mean_by_val[v] for v in values], dtype=float)
    stds = np.array([std_by_val[v] for v in values], dtype=float)

    if np.all(np.isnan(means)):
        return Boundary(
            task=axis.task,
            param=axis.param,
            method=method,
            tau=tau,
            degrade_direction=axis.degrade_direction,
            x_star=None,
            safe_low=None,
            safe_high=None,
            verdict="no_runs",
            best_score=float("nan"),
            worst_score=float("nan"),
            monotonic=False,
            spearman=None,
            n_points=len(values),
            curve_values=values,
            curve_mean=means.tolist(),
            curve_std=stds.tolist(),
        )

    # Order easy -> hard.
    if axis.degrade_direction == "increasing":
        order = np.argsort(native_vals)  # small value = easy
    else:
        order = np.argsort(-native_vals)  # large value = easy
    dv = native_vals[order]
    dm = means[order]

    # Fill nan (failed runs) as worst-possible so they count as unacceptable.
    dm_filled = np.where(np.isnan(dm), 0.0, dm)

    best_score = float(np.nanmax(means))
    worst_score = float(np.nanmin(means))
    # monotonic (non-increasing along easy->hard) within tolerance
    diffs = np.diff(dm_filled)
    monotonic = bool(np.all(diffs <= 1e-9))
    sp = _spearman(native_vals, means)

    acceptable = dm_filled >= tau

    # Native safe range = all native values whose mean >= tau (contiguous or not).
    safe_native = native_vals[np.where(means >= tau)[0]] if np.any(means >= tau) else np.array([])
    safe_low = float(np.min(safe_native)) if safe_native.size else None
    safe_high = float(np.max(safe_native)) if safe_native.size else None

    # Verdicts
    if np.all(acceptable):
        verdict = "always_acceptable"
        x_star = None
    elif not np.any(acceptable):
        verdict = "never_acceptable"
        x_star = None
    else:
        verdict = "boundary_found"
        # first easy->hard index where it drops from acceptable to unacceptable
        x_star = None
        for i in range(len(acceptable) - 1):
            if acceptable[i] and not acceptable[i + 1]:
                x_star = _interp_crossing(dv[i], dm_filled[i], dv[i + 1], dm_filled[i + 1], tau)
                break
        if x_star is None:
            # acceptable region not contiguous from the easy end; take the last
            # crossing into the unacceptable regime.
            for i in range(len(acceptable) - 1):
                if acceptable[i] and not acceptable[i + 1]:
                    x_star = _interp_crossing(dv[i], dm_filled[i], dv[i + 1], dm_filled[i + 1], tau)
        # If still None (only becomes acceptable at the hard end), mark non-monotone.
        if x_star is None:
            verdict = "non_monotone"

    return Boundary(
        task=axis.task,
        param=axis.param,
        method=method,
        tau=tau,
        degrade_direction=axis.degrade_direction,
        x_star=x_star,
        safe_low=safe_low,
        safe_high=safe_high,
        verdict=verdict,
        best_score=best_score,
        worst_score=worst_score,
        monotonic=monotonic,
        spearman=sp,
        n_points=len(values),
        curve_values=values,
        curve_mean=means.tolist(),
        curve_std=stds.tolist(),
    )


# ======================================================================
# Study driver (shared by the CLI subcommand and the standalone scripts)
# ======================================================================
_TASK_CATEGORY = {
    "domain_detection": "domain_detection",
    "deconvolution": "deconvolution",
    "svg": "svg",
}

VERDICT_HUMAN = {
    "boundary_found": "Failure boundary located",
    "always_acceptable": "Robust across the entire tested range",
    "never_acceptable": "Never reaches the acceptable bar in the tested range",
    "non_monotone": "Non-monotone response (no single clean boundary)",
    "no_runs": "Did not run",
}


def probe_runnable(task_name: str) -> tuple[list[str], dict[str, str]]:
    """Return ``(runnable_methods, {excluded_method: reason})`` for a task.

    Every registered method in the task's category is run once on the baseline
    dataset. A method that raises, or returns a non-finite score, is excluded
    with the reason recorded — so backend/data-gated methods (``banksy``,
    ``cell2location``, ``nnsvg``, ``spatialde``) are reported explicitly rather
    than silently dropped.
    """
    from histoweave.plugins import list_methods

    all_methods = [m["name"] for m in list_methods(_TASK_CATEGORY[task_name])]
    if task_name == "deconvolution":
        ds = make_mixture_synthetic(**DC_BASE, seed=0)
        task = deconvolution_task(dataset=ds)
    elif task_name == "svg":
        ds = make_synthetic(**SVG_BASE, seed=0)
        task = make_svg_task_fixed(ds, k=10)
    else:
        ds = make_synthetic(**DD_BASE, seed=0)
        task = domain_detection_task(dataset=ds)

    res = run_benchmark(task, methods=all_methods)
    runnable: list[str] = []
    excluded: dict[str, str] = {}
    for row in res.leaderboard:
        score = row.get("score")
        if row.get("error"):
            excluded[row["method"]] = row["error"]
        elif score is not None and np.isfinite(score):
            runnable.append(row["method"])
        else:
            excluded[row["method"]] = "non-finite score on baseline"
    return runnable, excluded


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3g}"


@dataclass
class BoundaryStudyResult:
    """Bundle of everything a boundary study produces."""

    tau: float
    seeds: tuple[int, ...]
    boundaries: list[Boundary]
    long_rows: list[dict]
    curves: dict[str, dict]
    runnable_by_task: dict[str, list[str]]
    excluded_by_task: dict[str, dict[str, str]]
    axes_meta: list[dict]

    def cards_dataframe(self):
        import pandas as pd

        df = pd.DataFrame([b.to_row() for b in self.boundaries])
        for col in ("x_star", "safe_low", "safe_high", "best_score", "worst_score", "spearman"):
            if col in df:
                df[col] = df[col].astype(float).round(4)
        if not df.empty:
            df = df.sort_values(["task", "param", "method"]).reset_index(drop=True)
        return df

    def to_bundle(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "seeds": list(self.seeds),
            "runnable_by_task": self.runnable_by_task,
            "excluded_by_task": self.excluded_by_task,
            "axes": self.axes_meta,
            "cards": [b.to_row() for b in self.boundaries],
            "curves": self.curves,
        }


def run_boundary_study(
    *,
    axes: Sequence[SweepAxis] | None = None,
    tasks: Sequence[str] | None = None,
    params: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    tau: float = DEFAULT_TAU,
    n_seeds: int = 5,
    progress: bool = True,
) -> BoundaryStudyResult:
    """Run the adversarial failure-boundary study and return a result bundle.

    Parameters
    ----------
    axes
        Sweep axes to run. Defaults to :func:`build_axes`.
    tasks, params
        Optional filters (by task name and/or parameter name) applied to
        ``axes`` — used by the CLI to run a single ``--task``/``--axis``.
    methods
        Restrict evaluation to these methods (must be runnable). ``None`` runs
        every method that passes :func:`probe_runnable` for each axis's task.
    tau
        Absolute acceptability threshold.
    n_seeds
        Number of replicate seeds (``range(n_seeds)``).
    """
    import histoweave.plugins.builtin  # noqa: F401  (registers builtin methods)

    all_axes = list(axes) if axes is not None else build_axes()
    if tasks:
        want_tasks = set(tasks)
        all_axes = [a for a in all_axes if a.task in want_tasks]
    if params:
        want_params = set(params)
        all_axes = [a for a in all_axes if a.param in want_params]
    if not all_axes:
        raise ValueError("no sweep axes selected (check --task / --axis filters)")

    seeds = tuple(range(n_seeds))

    # Runnable methods per task (probed once per task present in the selection).
    runnable_by_task: dict[str, list[str]] = {}
    excluded_by_task: dict[str, dict[str, str]] = {}
    for task_name in {a.task for a in all_axes}:
        runnable, excluded = probe_runnable(task_name)
        if methods is not None:
            want = set(methods)
            skipped = want - set(runnable)
            for m in skipped:
                excluded.setdefault(m, "excluded by --methods filter or not runnable")
            runnable = [m for m in runnable if m in want]
        runnable_by_task[task_name] = runnable
        excluded_by_task[task_name] = excluded
        if progress:
            _LOGGER.info(
                "probe %s: %d runnable, %d excluded", task_name, len(runnable), len(excluded)
            )

    long_rows: list[dict] = []
    boundaries: list[Boundary] = []
    curves: dict[str, dict] = {}

    for axis in all_axes:
        task_methods = runnable_by_task[axis.task]
        if not task_methods:
            continue
        if progress:
            _LOGGER.info(
                "sweep %s/%s: %d methods, %d values, %d seeds",
                axis.task,
                axis.param,
                len(task_methods),
                len(axis.values),
                len(seeds),
            )
        points = run_sweep(axis, methods=task_methods, seeds=seeds, progress=progress)
        for p in points:
            long_rows.append(
                {
                    "task": axis.task,
                    "param": axis.param,
                    "param_value": p.param_value,
                    "seed": p.seed,
                    "method": p.method,
                    "score": p.score,
                    "seconds": p.seconds,
                    "error": p.error,
                    "degrade_direction": axis.degrade_direction,
                    "axis_label": axis.label,
                    "unit": axis.unit,
                }
            )
        for method in task_methods:
            b = detect_boundary(points, axis, method, tau=tau)
            boundaries.append(b)
            curves[f"{axis.task}|{axis.param}|{method}"] = {
                "task": axis.task,
                "param": axis.param,
                "method": method,
                "label": axis.label,
                "unit": axis.unit,
                "degrade_direction": axis.degrade_direction,
                "tau": tau,
                "values": b.curve_values,
                "mean": b.curve_mean,
                "std": b.curve_std,
                "x_star": b.x_star,
                "safe_low": b.safe_low,
                "safe_high": b.safe_high,
                "verdict": b.verdict,
            }

    axes_meta = [
        {
            "task": a.task,
            "param": a.param,
            "degrade_direction": a.degrade_direction,
            "values": list(a.values),
            "label": a.label,
            "unit": a.unit,
        }
        for a in all_axes
    ]

    return BoundaryStudyResult(
        tau=tau,
        seeds=seeds,
        boundaries=boundaries,
        long_rows=long_rows,
        curves=curves,
        runnable_by_task=runnable_by_task,
        excluded_by_task=excluded_by_task,
        axes_meta=axes_meta,
    )


def write_cards_md(
    path,
    boundaries: list[Boundary],
    tau: float,
    excluded_by_task: dict[str, dict[str, str]],
    seeds: tuple[int, ...],
) -> None:
    """Write the human-readable Safe Operating Cards markdown file."""
    from collections import defaultdict
    from pathlib import Path

    by_method: dict[tuple[str, str], list[Boundary]] = defaultdict(list)
    for b in boundaries:
        by_method[(b.task, b.method)].append(b)

    lines: list[str] = []
    lines.append("# Safe Operating Cards — HistoWeave methods\n")
    lines.append(
        f"Adversarial failure-boundary mapping. Acceptability threshold "
        f"**tau = {tau}** (absolute; score = ARI for domain detection, "
        f"1-RMSD for deconvolution, precision@k for SVG). Each boundary is the "
        f"mean over {len(seeds)} replicate seeds. **x\\*** is the interpolated "
        f"parameter value where a method crosses from acceptable to unacceptable "
        f"as the problem gets harder; **safe range** is the interval where mean "
        f"score >= tau.\n"
    )
    lines.append(
        "> Reading direction: for `noise`, `n_domains`, `n_cell_types` larger = "
        "harder; for `marker_gene_lift` (signal) and `n_cells` (sample size) "
        "smaller = harder.\n"
    )
    for task, method in sorted(by_method):
        cards = sorted(by_method[(task, method)], key=lambda c: c.param)
        lines.append(f"\n## `{method}`  ·  task: {task}\n")
        lines.append(
            "| Axis (knob) | Verdict | Safe range | x\\* (failure) | Best | Worst | Monotone |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for c in cards:
            safe = (
                f"{_fmt(c.safe_low)} – {_fmt(c.safe_high)}"
                if (c.safe_low is not None or c.safe_high is not None)
                else "—"
            )
            lines.append(
                f"| {c.param} | {VERDICT_HUMAN.get(c.verdict, c.verdict)} | {safe} "
                f"| {_fmt(c.x_star)} | {_fmt(c.best_score)} | {_fmt(c.worst_score)} "
                f"| {'yes' if c.monotonic else 'no'} |"
            )
    lines.append("\n## Not evaluated (backend/data unavailable in this environment)\n")
    for task, excl in excluded_by_task.items():
        if not excl:
            continue
        lines.append(f"\n**{task}**\n")
        for m, reason in excl.items():
            short = reason.split(":")[0]
            lines.append(f"- `{m}` — {short}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_study_outputs(result: BoundaryStudyResult, out_dir) -> dict[str, str]:
    """Persist the long table, cards (CSV/JSON/MD) for a study result.

    Returns a dict of ``{artifact_name: path}``.
    """
    import json
    from pathlib import Path

    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    long_path = out / "boundary_long.csv"
    pd.DataFrame(result.long_rows).to_csv(long_path, index=False)

    cards_csv = out / "safe_operating_cards.csv"
    result.cards_dataframe().to_csv(cards_csv, index=False)

    cards_json = out / "safe_operating_cards.json"
    with open(cards_json, "w") as fh:
        json.dump(result.to_bundle(), fh, indent=2, default=float)

    cards_md = out / "safe_operating_cards.md"
    write_cards_md(cards_md, result.boundaries, result.tau, result.excluded_by_task, result.seeds)

    return {
        "long_csv": str(long_path),
        "cards_csv": str(cards_csv),
        "cards_json": str(cards_json),
        "cards_md": str(cards_md),
    }
