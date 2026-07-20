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

import numpy as np

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
    # Optional statistical review (cell-bootstrap CIs, FDR notes). Single-dataset
    # leaderboards cannot support multi-dataset rank permutation; only ARI CIs.
    stats: dict | None = None

    def best(self) -> dict | None:
        return self.leaderboard[0] if self.leaderboard else None


def run_benchmark(
    task: Task,
    methods: list[str] | None = None,
    *,
    method_params: dict[str, dict] | None = None,
    stats: bool = False,
    n_boot: int = 200,
    seed: int = 0,
    k_policy: str = "estimate",
    allow_oracle_k: bool = False,
) -> BenchmarkResult:
    """Evaluate every registered method for ``task.category`` and rank them.

    Returns a :class:`BenchmarkResult` whose leaderboard is sorted best-first. In a
    real deployment the scores are written back into each method's ``MethodSpec.benchmark``
    so the registry can surface recommendations.

    When ``stats=True`` and the task is domain detection, each successful row
    also receives a cell-bootstrap ARI CI (refit-free).

    For domain detection, ``k_policy`` controls how ``n_domains`` is supplied
    (default ``estimate`` — no ground-truth leak).  Explicit per-method
    ``method_params[name]['n_domains']`` always wins.
    """
    method_params = {k: dict(v) for k, v in (method_params or {}).items()}
    candidates = methods or _default_benchmark_candidates(task.category)

    # Apply any shared preparation (e.g. normalization) once.
    prepared = task.dataset.copy()
    for step in task.prep:
        prepared = create_method(step.category, step.method, **step.params).run(prepared)

    # Non-oracle K for domain methods that declare n_domains and lack an override.
    estimated_k: int | None = None
    k_meta: dict[str, Any] = {"k_policy": k_policy}
    if task.name == "domain_detection":
        from .k_selection import estimate_n_domains, oracle_n_domains

        if k_policy == "oracle":
            if not allow_oracle_k:
                raise ValueError(
                    "k_policy='oracle' requires allow_oracle_k=True (oracle-K leak is opt-in only)"
                )
            estimated_k = oracle_n_domains(task.dataset)
            k_meta["n_domains_used"] = estimated_k
            k_meta["source"] = "oracle"
        elif k_policy == "estimate":
            selection = estimate_n_domains(
                task.dataset,
                method="ensemble",
                random_state=seed,
            )
            estimated_k = selection.k
            k_meta.update(
                {
                    "n_domains_used": estimated_k,
                    "source": "estimate",
                    "estimator": selection.method,
                    "geometry": selection.geometry,
                    "spatial_used": selection.spatial_used,
                    "component_votes": dict(selection.component_votes),
                    "oracle_k": selection.oracle_k,
                    "flags": list(selection.flags),
                }
            )
        # Strip generative K from uns so methods cannot silently read it.
        if "n_domains" in prepared.uns and k_policy != "oracle":
            prepared.uns = dict(prepared.uns)
            prepared.uns.pop("n_domains", None)

    rows = []
    # Keep predictions for optional bootstrap.
    predictions: dict[str, Any] = {}
    for name in candidates:
        params = dict(method_params.get(name, {}))
        if (
            estimated_k is not None
            and "n_domains" not in params
            and task.name == "domain_detection"
        ):
            # Only inject when the method declares the param.
            try:
                probe = create_method(task.category, name)
                if "n_domains" in probe.params:
                    params["n_domains"] = estimated_k
            except Exception:
                pass
        entry: dict = {"method": name}
        try:
            method = create_method(task.category, name, **params)
            t0 = time.perf_counter()
            result = method.run(prepared.copy())
            entry["seconds"] = round(time.perf_counter() - t0, 4)
            entry["score"] = round(float(task.score(result, task.dataset)), 4)
            entry["version"] = method.spec.version
            if stats and task.name == "domain_detection" and "domain" in result.obs.columns:
                predictions[name] = result
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
            method_cls = get_method(task.category, row["method"])
            bench = dict(method_cls.spec.benchmark)
            bench[task.name] = {
                "score": row.get("score"),
                "rank": row["rank"],
                "seconds": row.get("seconds"),
            }
            method_cls.spec = replace(method_cls.spec, benchmark=bench)
        except Exception:
            pass  # method was unregistered mid-run or spec is pinned; skip

    stats_payload: dict | None = None
    if stats and predictions:
        from .stats_review import bootstrap_ari

        truth_key = "domain_truth"
        if truth_key not in task.dataset.obs.columns:
            truth_key = next(
                (c for c in task.dataset.obs.columns if "truth" in c.lower()),
                "domain_truth",
            )
        cis: dict[str, dict] = {}
        for name, result in predictions.items():
            try:
                pred = result.obs["domain"].to_numpy()
                truth = task.dataset.obs.loc[result.obs_names, truth_key].to_numpy()
                boot = bootstrap_ari(truth, pred, n_boot=n_boot, seed=seed)
                cis[name] = boot.to_dict()
                for row in rows:
                    if row["method"] == name:
                        row["ari_ci_low"] = round(boot.ci_low, 4)
                        row["ari_ci_high"] = round(boot.ci_high, 4)
            except Exception:
                continue
        stats_payload = {
            "protocol": "histoweave.benchmark_stats.v1",
            "scope": "single_dataset_cell_bootstrap",
            "n_boot": n_boot,
            "bootstrap_ari": cis,
            "k_selection": k_meta if task.name == "domain_detection" else None,
            "notes": [
                "Single-dataset leaderboard: rank permutation / FDR require "
                "a multi-dataset landscape (see review_landscape)."
            ],
        }
    elif task.name == "domain_detection":
        stats_payload = {
            "protocol": "histoweave.benchmark_stats.v1",
            "scope": "k_selection_only",
            "k_selection": k_meta,
        }

    return BenchmarkResult(task=task.name, metric="score", leaderboard=rows, stats=stats_payload)


def _default_benchmark_candidates(category: MethodCategory | str) -> list[str]:
    """Methods safe for the default synthetic smoke harness.

    Optional SOTA backends (SpaGCN/GraphST/…) and research-incubator candidates
    are opt-in via ``methods=[...]`` so a missing optional dependency does not
    poison the leaderboard with ``-inf`` scores under ``--fail-on-error``.
    """
    names: list[str] = []
    for item in list_methods(category):
        if item["language"] == "container":
            continue
        meta = item.get("metadata") or {}
        if meta.get("track") in {"sota", "research"}:
            continue
        names.append(item["name"])
    return names


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
def virtual_st_task(
    dataset: SpatialTable | None = None,
    *,
    image_key: str = "image",
    measured_layer: str | None = None,
) -> Task:
    """H&E → spatial transcriptomics prediction scored by mean gene Pearson.

    Methods under :class:`~histoweave.plugins.interfaces.MethodCategory.VIRTUAL_ST`
    write predicted expression to ``layers['virtual_st']``.  The score is the
    mean per-gene Pearson correlation against the measured matrix (``X`` or
    ``measured_layer``), matching the primary virtual-ST literature metric.
    """
    from ..plugins.builtin.virtual_st import mean_gene_pearson

    if dataset is None:
        dataset = _make_virtual_st_synthetic(seed=0, image_key=image_key)

    def pearson_score(result: SpatialTable, ref: SpatialTable) -> float:
        if "virtual_st" not in result.layers:
            raise KeyError("virtual_st methods must write layers['virtual_st']")
        predicted = np.asarray(result.layers["virtual_st"], dtype=float)
        if measured_layer is None or measured_layer in {"X", "x", "expression"}:
            measured = ref.X
        else:
            measured = ref.layers[measured_layer]
        measured_arr = np.asarray(
            measured.todense() if hasattr(measured, "todense") else measured, dtype=float
        )
        # Align to observations that survived any prep.
        if predicted.shape[0] != measured_arr.shape[0]:
            measured_arr = measured_arr[: predicted.shape[0]]
        n_genes = min(predicted.shape[1], measured_arr.shape[1])
        return mean_gene_pearson(predicted[:, :n_genes], measured_arr[:, :n_genes])

    return Task(
        name="virtual_st",
        category=MethodCategory.VIRTUAL_ST,
        dataset=dataset,
        score=pearson_score,
        output_key="virtual_st",
        higher_is_better=True,
        prep=[],
    )


def _make_virtual_st_synthetic(
    *,
    seed: int = 0,
    n_obs: int = 48,
    n_genes: int = 16,
    image_key: str = "image",
) -> SpatialTable:
    """Paired H&E + expression toy dataset for virtual_st CI."""
    from ..data import SpatialTable

    rng = np.random.default_rng(seed)
    # Grid coordinates.
    side = int(np.ceil(np.sqrt(n_obs)))
    ys, xs = np.divmod(np.arange(n_obs), side)
    coords = np.column_stack((xs.astype(float), ys.astype(float)))
    # Synthetic H&E: smooth spatial colour fields + noise.
    yy, xx = np.mgrid[0:64, 0:64]
    image = np.stack(
        (
            0.4 + 0.4 * np.sin(xx / 8.0) + 0.1 * rng.random((64, 64)),
            0.4 + 0.4 * np.cos(yy / 7.0) + 0.1 * rng.random((64, 64)),
            0.3 + 0.2 * rng.random((64, 64)),
        ),
        axis=-1,
    )
    # Expression partially driven by local morphology colour.
    unit = coords / np.maximum(coords.max(axis=0), 1.0)
    morph = np.column_stack(
        (
            np.sin(unit[:, 0] * np.pi),
            np.cos(unit[:, 1] * np.pi),
            unit[:, 0] * unit[:, 1],
        )
    )
    loadings = rng.normal(0.0, 1.0, size=(morph.shape[1], n_genes))
    expression = np.clip(np.exp(morph @ loadings + rng.normal(0.0, 0.15, size=(n_obs, n_genes))), 0.0, None)
    obs = __import__("pandas").DataFrame(index=[f"spot_{i}" for i in range(n_obs)])
    var = __import__("pandas").DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return SpatialTable(
        X=expression,
        obs=obs,
        var=var,
        obsm={"spatial": coords},
        images={image_key: image},
        uns={"platform": "histology", "analysis_task": "virtual_st"},
    )


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

    ``name`` is one of ``"domain_detection"``, ``"deconvolution"``, ``"svg"``,
    ``"virtual_st"``.
    """
    factories: dict[str, Any] = {
        "domain_detection": domain_detection_task,
        "deconvolution": deconvolution_task,
        "svg": svg_task,
        "virtual_st": virtual_st_task,
    }
    if name not in factories:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(factories)}")
    return factories[name](**kwargs)
