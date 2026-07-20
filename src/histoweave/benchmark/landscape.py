"""Performance-landscape analysis for spatial analysis tasks.

HistoWeave uses landscapes to *quantify method × spatial-context selection
uncertainty*, not to claim a single universal best method.  Each landscape is
bound to a :class:`~histoweave.benchmark.task_contract.TaskContract` so domain
recovery and cell-type recovery are never silently mixed.

Outputs
-------
* **Performance matrix** — datasets × methods (or method@policy), entry = metric.
* **Feature embedding** — datasets in 2‑D (PCA on target-free feature vectors).
* **Method niches** — regions of feature space where each configuration wins.
* **Dataset metadata** — platform / task / ground-truth kind for recommendation priors.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._math import adjusted_rand_index
from ..data import SpatialTable
from ..plugins import MethodCategory, create_method, list_methods
from .features import (
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)


@dataclass
class LandscapeResult:
    """Complete output of a landscape analysis run."""

    # Mapping: dataset name → method name → ARI score (NaN = method failed).
    performance: dict[str, dict[str, float]]

    # Dataset name → feature vector.
    features: dict[str, np.ndarray]

    # 2‑D embedding coordinates (dataset name → (x, y)).
    embedding: dict[str, tuple[float, float]]

    # Dataset name → best method.
    best_method: dict[str, str]

    # Method name → list of dataset names where it was best (or top-tier).
    niches: dict[str, list[str]]

    # Raw timings: dataset → method → seconds.
    timings: dict[str, dict[str, float | None]]

    # Metadata
    feature_order: list[str] = field(default_factory=lambda: list(RECOMMENDATION_FEATURE_ORDER))
    method_count: int = 0
    dataset_count: int = 0
    task: str = "spatial_domain"
    metric: str = "score"
    higher_is_better: bool = True
    # Per-dataset priors used by MethodRecommender v2 (platform, task, gt kind).
    dataset_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    def performance_matrix(self) -> np.ndarray:
        """Return (n_datasets × n_methods) array of ARI scores."""
        ds_names = sorted(self.performance)
        method_names = sorted(next(iter(self.performance.values())).keys()) if ds_names else []
        mat = np.full((len(ds_names), len(method_names)), np.nan)
        for i, ds in enumerate(ds_names):
            for j, m in enumerate(method_names):
                mat[i, j] = self.performance[ds].get(m, np.nan)
        return mat

    def method_order(self) -> list[str]:
        return sorted({method for row in self.performance.values() for method in row})

    def dataset_order(self) -> list[str]:
        return sorted(self.performance)

    def summary(self) -> str:
        """One-paragraph human-readable summary of the landscape."""
        lines = []
        lines.append(f"Landscape: {self.dataset_count} datasets × {self.method_count} methods")
        lines.append(f"Feature dimensions: {len(self.feature_order)}")

        # Which method wins most often?
        wins: dict[str, int] = {}
        for _ds, best in self.best_method.items():
            wins[best] = wins.get(best, 0) + 1
        ranked = sorted(wins.items(), key=lambda kv: -kv[1])
        lines.append("Method wins: " + ", ".join(f"{m}={c}" for m, c in ranked))

        # Niche size (how many datasets each method dominates)
        niche_sizes = {m: len(v) for m, v in self.niches.items()}
        lines.append(
            "Niche sizes: "
            + ", ".join(f"{m}={n}" for m, n in sorted(niche_sizes.items(), key=lambda kv: -kv[1]))
        )

        # Mean ARI per method
        means = {}
        for m in self.method_order():
            scores = [
                self.performance[ds][m]
                for ds in self.dataset_order()
                if not np.isnan(self.performance[ds].get(m, np.nan))
            ]
            if scores:
                means[m] = float(np.mean(scores))
        lines.append(
            "Mean ARI: "
            + ", ".join(f"{m}={means[m]:.3f}" for m in sorted(means, key=lambda k: -means[k]))
        )

        return "\n".join(lines)


def run_landscape(
    datasets: dict[str, SpatialTable],
    *,
    methods: list[str] | None = None,
    n_domains_override: int | None = None,
    k_policy: str = "estimate",
    allow_oracle_k: bool = False,
    k_estimator: str = "ensemble",
    random_state: int = 0,
) -> LandscapeResult:
    """Run every domain-detection method on every dataset and build the landscape.

    This is the primary entry point for the **domain detection** task.  For other
    tasks use :func:`run_task_landscape`.  For all tasks at once, use
    :func:`run_multi_landscape`.

    K (``n_domains``) policy
    ------------------------
    Historically this injected the true domain count (oracle K).  That is
    unrealistic and is no longer the default.  Pass ``k_policy='oracle'`` with
    ``allow_oracle_k=True`` only for controlled ablations, or
    ``n_domains_override`` for a fixed K.
    """
    from .k_selection import make_domain_k_factory

    if n_domains_override is not None:
        policy = "fixed"
        fixed_k = int(n_domains_override)
        oracle_ok = True
    else:
        policy = k_policy
        fixed_k = None
        oracle_ok = allow_oracle_k

    factory = make_domain_k_factory(
        policy=policy,  # type: ignore[arg-type]
        fixed_k=fixed_k,
        estimator=k_estimator,  # type: ignore[arg-type]
        allow_oracle_k=oracle_ok,
        random_state=random_state,
    )
    result = run_task_landscape(
        datasets,
        category=MethodCategory.DOMAIN_DETECTION,
        methods=methods,
        extra_params_factory=factory,
    )
    # Record the K policy on every dataset meta row for auditability.
    for name in result.performance:
        meta = dict(result.dataset_meta.get(name, {}))
        meta["k_policy"] = policy if n_domains_override is None else "fixed"
        meta["k_estimator"] = k_estimator if policy in {"estimate", "dual"} else None
        if n_domains_override is not None:
            meta["n_domains_used"] = int(n_domains_override)
        result.dataset_meta[name] = meta
    return result


def run_task_landscape(
    datasets: dict[str, SpatialTable],
    *,
    category: MethodCategory | str,
    methods: list[str] | None = None,
    extra_params_factory: Callable[[SpatialTable], dict[str, Any]] | None = None,
    dataset_meta: dict[str, dict[str, Any]] | None = None,
    task: str | None = None,
) -> LandscapeResult:
    """Run every method registered for *category* on every dataset.

    Parameters
    ----------
    datasets
        Mapping of dataset name → :class:`SpatialTable`.
    category
        The method category (e.g. ``MethodCategory.DOMAIN_DETECTION``,
        ``MethodCategory.SPATIALLY_VARIABLE_GENES``).
    methods
        Method names to evaluate.  When *None*, all registered methods for
        *category* are used.
    extra_params_factory
        Optional callable ``(data) -> dict[str, Any]`` that returns extra
        keyword arguments to pass to ``create_method()`` (e.g. ``n_domains``
        for domain-detection methods).
    dataset_meta
        Optional per-dataset priors (platform / task / ground_truth_kind).
        When omitted, HistoWeave tries the real-data registry via
        :func:`histoweave.benchmark.landscape_io.meta_from_registry`.
    task
        Analysis task name stored on the landscape (defaults from *category*).
    """
    if methods is None:
        methods = [m["name"] for m in list_methods(category)]

    performance: dict[str, dict[str, float]] = {}
    features: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float | None]] = {}

    # Sniff which params each method accepts.
    method_param_names: dict[str, set[str]] = {}
    for method_name in methods:
        try:
            cls = create_method(category, method_name)
            method_param_names[method_name] = set(cls.params.keys())
        except Exception:
            method_param_names[method_name] = set()

    for ds_name, data in datasets.items():
        feats = extract_features(data, include_domain=False)
        features[ds_name] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)

        # Pre-normalize once (log1p_cp10k) so every method gets the same input.
        normalized = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(data.copy())

        extra = extra_params_factory(data) if extra_params_factory else {}

        perf_row: dict[str, float] = {}
        time_row: dict[str, float | None] = {}
        for method_name in methods:
            try:
                t0 = time.perf_counter()
                params: dict[str, Any] = {}
                # Only pass extra params the method declares.
                for key, val in extra.items():
                    if key in method_param_names.get(method_name, set()):
                        params[key] = val
                result = create_method(
                    category,
                    method_name,
                    **params,
                ).run(normalized.copy())
                elapsed = time.perf_counter() - t0
                perf_row[method_name] = float(_score_result(result, data, category))
                time_row[method_name] = round(elapsed, 4)
            except Exception:
                perf_row[method_name] = float("nan")
                time_row[method_name] = None

        performance[ds_name] = perf_row
        timings[ds_name] = time_row

    # 2‑D embedding via PCA on the feature vectors.
    embedding = _embed_datasets(features)

    # Determine best method per dataset and method niches.
    best_method, niches = _compute_niches(performance)

    cat_value = str(getattr(category, "value", category))
    task_value = task or ("spatial_domain" if cat_value == "domain_detection" else cat_value)
    if dataset_meta is None:
        try:
            from .landscape_io import meta_from_registry

            dataset_meta = meta_from_registry(performance.keys())
            for row in dataset_meta.values():
                row.setdefault("task", task_value)
        except Exception:
            dataset_meta = {}

    return LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings,
        method_count=len(methods),
        dataset_count=len(datasets),
        task=task_value,
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        dataset_meta=dict(dataset_meta),
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _embed_datasets(
    features: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Embed datasets into 2‑D via PCA on standardised feature vectors."""
    ds_names = sorted(features)
    if not ds_names:
        return {}
    X = np.array([features[n] for n in ds_names], dtype=float)

    # Impute NaN columns with column median (0 when a column is entirely missing).
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            finite = col[~mask]
            fill = float(np.median(finite)) if finite.size else 0.0
            col[mask] = fill
            X[:, j] = col

    # All-missing / constant features → zero embedding (CSV-only landscapes).
    if not np.isfinite(X).all() or np.allclose(X, X[0]):
        return {name: (0.0, 0.0) for name in ds_names}

    # Standardise.
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    Xs = (X - mean) / std

    # PCA to 2‑D.
    if Xs.shape[0] >= 2:
        try:
            U, S, _Vt = np.linalg.svd(Xs, full_matrices=False)
            n_comp = min(2, U.shape[1], S.shape[0])
            coords_2d = np.zeros((len(ds_names), 2))
            coords_2d[:, :n_comp] = U[:, :n_comp] * S[:n_comp]
        except np.linalg.LinAlgError:
            coords_2d = np.zeros((len(ds_names), 2))
    else:
        coords_2d = np.zeros((len(ds_names), 2))

    return {
        name: (float(coords_2d[i, 0]), float(coords_2d[i, 1])) for i, name in enumerate(ds_names)
    }


# ---------------------------------------------------------------------------
# Scoring dispatcher (per task category)
# ---------------------------------------------------------------------------
def _score_result(
    result: SpatialTable,
    truth_data: SpatialTable,
    category: MethodCategory | str,
) -> float:
    """Compute a 0‑1 score for a method result against ground truth.

    The scoring function depends on the task category; this dispatcher keeps
    ``run_task_landscape`` category-agnostic.
    """
    cat = category.value if isinstance(category, MethodCategory) else str(category)

    if cat == MethodCategory.DOMAIN_DETECTION.value:
        pred = result.obs["domain"].to_numpy()
        truth = truth_data.obs.loc[result.obs_names, "domain_truth"].to_numpy()
        return float(adjusted_rand_index(truth, pred))

    if cat == MethodCategory.SPATIALLY_VARIABLE_GENES.value:
        # precision@k against known marker genes
        marker_dict = truth_data.uns.get("marker_genes", {})
        true_markers: set[str] = set()
        for genes in marker_dict.values():
            true_markers.update(str(g) for g in genes)
        svg_data = result.uns.get("svg", {})
        top_genes = svg_data.get("top_genes", [])
        top_names = {
            str(entry.get("gene", "")) if isinstance(entry, dict) else str(entry)
            for entry in top_genes
        }
        if not top_names or not true_markers:
            return 0.0
        hits = len(top_names & true_markers)
        return hits / min(len(top_names), len(true_markers))

    if cat == MethodCategory.DECONVOLUTION.value:
        from .._math import proportions_rmsd

        pred = result.obsm.get("proportions")
        truth_key = "proportions_truth"
        if pred is None or truth_key not in truth_data.obsm:
            return float("nan")
        truth = truth_data.obsm[truth_key][: len(pred)]
        return float(1.0 - proportions_rmsd(truth, pred))

    # Generic fallback: if the result carries a "score" in uns, use it.
    score = result.uns.get("score")
    if isinstance(score, int | float):
        return float(score)
    return float("nan")


# ---------------------------------------------------------------------------
# Multi-task landscape
# ---------------------------------------------------------------------------
@dataclass
class MultiLandscapeResult:
    """A collection of :class:`LandscapeResult` objects, one per task."""

    tasks: dict[str, LandscapeResult] = field(default_factory=dict)

    def task_names(self) -> list[str]:
        return sorted(self.tasks)

    def __getitem__(self, task: str) -> LandscapeResult:
        return self.tasks[task]

    def summary(self) -> str:
        lines = ["Multi-Task Landscape Summary", "=" * 30]
        for name in self.task_names():
            r = self.tasks[name]
            lines.append(f"\n--- {name} ---")
            lines.append(r.summary())
        return "\n".join(lines)


def run_multi_landscape(
    datasets: dict[str, SpatialTable],
    *,
    categories: list[MethodCategory | str] | None = None,
    methods_per_task: dict[str, list[str]] | None = None,
) -> MultiLandscapeResult:
    """Run :func:`run_task_landscape` for every relevant task category.

    This is the entry point for a comprehensive benchmark run that feeds the
    multi-task recommendation engine.
    """
    if categories is None:
        categories = [
            MethodCategory.DOMAIN_DETECTION,
            MethodCategory.SPATIALLY_VARIABLE_GENES,
            MethodCategory.DECONVOLUTION,
        ]

    tasks: dict[str, LandscapeResult] = {}
    for cat in categories:
        cat_val = cat.value if isinstance(cat, MethodCategory) else str(cat)
        methods = (methods_per_task or {}).get(cat_val)
        try:
            tasks[cat_val] = run_task_landscape(
                datasets,
                category=cat,
                methods=methods,
            )
        except Exception:
            # A missing category shouldn't block the others.
            pass

    return MultiLandscapeResult(tasks=tasks)


# ---------------------------------------------------------------------------
# Niches
# ---------------------------------------------------------------------------
def _compute_niches(
    performance: dict[str, dict[str, float]],
    threshold: float = 0.95,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Determine the best method per dataset and each method's niche.

    A dataset belongs to a method's *niche* when that method's ARI is within
    *threshold* of the best score for that dataset.  This captures a fuzzy
    "winner-takes-most" region rather than a hard winner-takes-all boundary.
    """
    best_method: dict[str, str] = {}
    niches: dict[str, list[str]] = {}

    for ds_name, scores in performance.items():
        valid = {m: s for m, s in scores.items() if not np.isnan(s)}
        if not valid:
            continue
        best = max(valid, key=lambda k: valid[k])
        best_method[ds_name] = best
        cutoff = valid[best] * threshold
        for method, score in valid.items():
            if score >= cutoff:
                niches.setdefault(method, []).append(ds_name)

    return best_method, niches


# ---------------------------------------------------------------------------
# Landscape SVG visualization
# ---------------------------------------------------------------------------
_METHOD_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def landscape_svg(result: LandscapeResult, width: int = 620, height: int = 460) -> str:
    """Render the 2‑D feature embedding as an SVG scatter plot.

    Each point is a **dataset**, coloured by the method that scored highest on
    it.  Point size encodes the winner's ARI (bigger = stronger win).  The plot
    is a static fallback; production deployment should use interactive rendering
    (e.g. Plotly or Vitessce).
    """
    import html

    ds_names = result.dataset_order()
    if not ds_names:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
            '<text x="10" y="25" font-size="12">(no data)</text></svg>'
        )

    # Build colour map for methods
    unique_methods = sorted(set(result.best_method.get(ds, "unknown") for ds in ds_names))
    color_map: dict[str, str] = {}
    for i, m in enumerate(unique_methods):
        color_map[m] = _METHOD_COLORS[i % len(_METHOD_COLORS)]

    # Collect coordinates and scores
    xs: list[float] = []
    ys: list[float] = []
    sizes: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    for ds in ds_names:
        emb = result.embedding.get(ds, (0.0, 0.0))
        best = result.best_method.get(ds, "?")
        score = result.performance[ds].get(best, 0.0)
        xs.append(emb[0])
        ys.append(emb[1])
        sizes.append(4.0 + 14.0 * max(0.0, min(1.0, score)))  # radius 4–18
        colors.append(color_map.get(best, "#999999"))
        labels.append(ds)

    pad = 50
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def sx(x: float) -> float:
        return pad + (x - xmin) / xr * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - ymin) / yr * (height - 2 * pad)

    points = ""
    for i in range(len(ds_names)):
        points += (
            f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(ys[i]):.1f}" '
            f'r="{sizes[i]:.1f}" fill="{colors[i]}" fill-opacity="0.80" '
            f'stroke="#333" stroke-width="0.5"/>'
        )
        points += (
            f'<text x="{sx(xs[i]):.1f}" y="{sy(ys[i]) - sizes[i] - 3:.1f}" '
            f'text-anchor="middle" font-size="9" fill="currentColor">'
            f"{html.escape(labels[i].replace('_', ' '))}</text>"
        )

    # Legend
    legend_items = []
    for i, (method, col) in enumerate(color_map.items()):
        ly = pad + i * 20
        legend_items.append(
            f'<rect x="{width - 130}" y="{ly}" width="12" height="12" '
            f'fill="{col}" fill-opacity="0.80" stroke="#333" stroke-width="0.5"/>'
            f'<text x="{width - 113}" y="{ly + 10}" font-size="11" '
            f'fill="currentColor">{html.escape(method)}</text>'
        )

    title = "Performance Landscape: Method Niches in Dataset Feature Space"
    legend_markup = "".join(legend_items)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width + 20} {height}" '
        f'width="{width + 20}" height="{height}" '
        f'font-family="system-ui, sans-serif">'
        f'<text x="{pad}" y="22" font-size="13" font-weight="600" fill="currentColor">'
        f"{html.escape(title)}</text>"
        f'<text x="{pad}" y="38" font-size="10" fill="#666">'
        f"Point size ∝ winner ARI.  Colour = best method.  "
        f"Distance = dataset similarity.</text>"
        f"{points}{legend_markup}</svg>"
    )
