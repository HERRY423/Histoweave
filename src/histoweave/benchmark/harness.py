"""Benchmarking harness — HistoWeave's core differentiator.

Following the Open Problems model, each analysis *task* pairs reference data with a
ground-truth proxy and quantitative metrics. Every registered method for the task's
category is evaluated, producing a leaderboard that powers in-workflow recommendations
("for data like this, methods X/Y lead on accuracy and scalability").

This scaffold implements one fully-working task — spatial domain detection scored by
Adjusted Rand Index against known domains — plus the generic machinery to add more.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .._math import adjusted_rand_index
from ..data import SpatialTable
from ..plugins import MethodCategory, create_method, get_method, list_methods

if TYPE_CHECKING:  # avoid a runtime import cycle (workflow imports from plugins too)
    from ..workflow import PipelineStep


@dataclass
class Task:
    """A benchmark task: a category of methods, a dataset, and a scoring function.

    Parameters
    ----------
    name
        Human-readable task id, e.g. ``"domain_detection"``.
    category
        The method category whose registered methods are evaluated.
    dataset
        A :class:`SpatialTable` carrying whatever ground truth ``score`` needs.
    score
        ``(result, dataset) -> float``; higher is better.
    output_key
        The ``obs`` column each method is expected to populate.
    higher_is_better
        Ranking direction for the leaderboard.
    """

    name: str
    category: MethodCategory
    dataset: SpatialTable
    score: Callable[[SpatialTable, SpatialTable], float]
    output_key: str
    higher_is_better: bool = True
    prep: list[PipelineStep] = field(default_factory=list)  # optional pre-steps (e.g. normalize)


@dataclass
class BenchmarkResult:
    task: str
    metric: str
    leaderboard: list[dict]

    def best(self) -> dict | None:
        return self.leaderboard[0] if self.leaderboard else None


def run_benchmark(
    task: Task,
    methods: list[str] | None = None,
    *,
    method_params: dict[str, dict] | None = None,
) -> BenchmarkResult:
    """Evaluate every registered method for ``task.category`` and rank them.

    Returns a :class:`BenchmarkResult` whose leaderboard is sorted best-first. In a
    real deployment the scores are written back into each method's ``MethodSpec.benchmark``
    so the registry can surface recommendations.
    """
    method_params = method_params or {}
    candidates = methods or [
        item["name"]
        for item in list_methods(task.category)
        if item["language"] != "container"
    ]

    # Apply any shared preparation (e.g. normalization) once.
    prepared = task.dataset.copy()
    for step in task.prep:
        prepared = create_method(step.category, step.method, **step.params).run(prepared)

    rows = []
    for name in candidates:
        params = method_params.get(name, {})
        entry: dict = {"method": name}
        try:
            method = create_method(task.category, name, **params)
            t0 = time.perf_counter()
            result = method.run(prepared.copy())
            entry["seconds"] = round(time.perf_counter() - t0, 4)
            entry["score"] = round(float(task.score(result, task.dataset)), 4)
            entry["version"] = method.spec.version
        except Exception as exc:  # a failing method scores worst, doesn't crash the run
            entry["seconds"] = None
            entry["score"] = float("-inf") if task.higher_is_better else float("inf")
            entry["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)

    rows.sort(key=lambda r: r["score"], reverse=task.higher_is_better)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    # Write scores back into each method's MethodSpec.benchmark so the
    # registry surfaces them (``histoweave list-methods``, recommendations).
    for row in rows:
        try:
            cls = get_method(task.category, row["method"])
            bench = dict(cls.spec.benchmark)
            bench[task.name] = {
                "score": row.get("score"),
                "rank": row["rank"],
                "seconds": row.get("seconds"),
            }
            cls.spec = replace(cls.spec, benchmark=bench)
        except Exception:
            pass  # method was unregistered mid-run or spec is pinned; skip

    return BenchmarkResult(task=task.name, metric="score", leaderboard=rows)


# ---------------------------------------------------------------------------
# A concrete, ready-to-run task.
# ---------------------------------------------------------------------------
def domain_detection_task(
    dataset: SpatialTable | None = None,
    truth_key: str = "domain_truth",
) -> Task:
    """Spatial domain detection scored by ARI against ``obs[truth_key]``."""
    from ..datasets import make_synthetic
    from ..workflow import PipelineStep

    dataset = dataset if dataset is not None else make_synthetic(seed=0)

    def ari(result: SpatialTable, ref: SpatialTable) -> float:
        pred = result.obs["domain"].to_numpy()
        # Align to the (possibly QC-filtered) observations that survived.
        truth = ref.obs.loc[result.obs_names, truth_key].to_numpy()
        return adjusted_rand_index(truth, pred)

    return Task(
        name="domain_detection",
        category=MethodCategory.DOMAIN_DETECTION,
        dataset=dataset,
        score=ari,
        output_key="domain",
        higher_is_better=True,
        prep=[PipelineStep(MethodCategory.NORMALIZATION, "log1p_cp10k")],
    )


def deconvolution_task(
    dataset: SpatialTable | None = None,
    truth_key: str = "proportions_truth",
) -> Task:
    """Cell-type deconvolution scored by 1 − RMSD against ground-truth proportions.

    Returns a Task where ``higher_is_better=True`` and score = 1 − proportions_rmsd
    (so 1.0 = perfect recovery, lower = worse).
    """
    from .._math import proportions_rmsd
    from ..datasets import make_mixture_synthetic
    from ..workflow import PipelineStep

    dataset = dataset if dataset is not None else make_mixture_synthetic(seed=0)

    def one_minus_rmsd(result: SpatialTable, ref: SpatialTable) -> float:
        pred = result.obsm["proportions"]
        truth = ref.obsm[truth_key][: len(pred)]
        return float(1.0 - proportions_rmsd(truth, pred))

    return Task(
        name="deconvolution",
        category=MethodCategory.DECONVOLUTION,
        dataset=dataset,
        score=one_minus_rmsd,
        output_key="proportions",
        higher_is_better=True,
        prep=[PipelineStep(MethodCategory.NORMALIZATION, "log1p_cp10k")],
    )


# ---------------------------------------------------------------------------
# SVG detection task
# ---------------------------------------------------------------------------
def svg_task(
    dataset: SpatialTable | None = None,
    k: int = 10,
) -> Task:
    """Spatially variable gene detection scored by precision@k.

    The ground truth is the set of marker genes defined in ``uns['marker_genes']``
    (or ``var`` columns flagged as markers).  Methods are scored by what fraction
    of their top-*k* detected SVGs are true marker genes.

    This task validates that SVG methods recover the *biological signal* that
    generated the spatial pattern, not just genes with high variance.
    """
    from ..datasets import make_synthetic
    from ..workflow import PipelineStep

    dataset = dataset if dataset is not None else make_synthetic(seed=0)

    # Collect all marker gene names from the dataset.
    marker_dict = dataset.uns.get("marker_genes", {})
    true_markers: set[str] = set()
    for genes in marker_dict.values():
        true_markers.update(str(g) for g in genes)

    def precision_at_k(result: SpatialTable, ref: SpatialTable) -> float:
        # result.uns['svg']['top_genes'] is a list of {gene, morans_i} dicts
        svg_data = result.uns.get("svg", {})
        top_genes_raw = svg_data.get("top_genes", [])
        top_gene_names: set[str] = set()
        for entry in top_genes_raw:
            if isinstance(entry, dict):
                top_gene_names.add(str(entry.get("gene", "")))
            else:
                top_gene_names.add(str(entry))
        if not top_gene_names:
            return 0.0
        hits = len(top_gene_names & true_markers)
        return hits / min(k, len(top_gene_names) or 1) if hits > 0 else 0.0

    return Task(
        name="svg",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        dataset=dataset,
        score=precision_at_k,
        output_key="svg",
        higher_is_better=True,
        prep=[PipelineStep(MethodCategory.NORMALIZATION, "log1p_cp10k")],
    )


# ---------------------------------------------------------------------------
# Task registry (maps task name → factory)
# ---------------------------------------------------------------------------
def get_task(name: str, **kwargs) -> Task:
    """Look up a benchmark task factory by name.

    ``name`` is one of ``"domain_detection"``, ``"deconvolution"``, ``"svg"``.
    """
    factories: dict[str, Any] = {
        "domain_detection": domain_detection_task,
        "deconvolution": deconvolution_task,
        "svg": svg_task,
    }
    if name not in factories:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(factories)}")
    return factories[name](**kwargs)
