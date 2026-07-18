"""Import, merge, and export performance landscapes with task contracts.

P1 goal: let offline SOTA adapter runs (SpaGCN / GraphST / …) and the in-process
sklearn / spatial-weight sweep share one :class:`LandscapeResult` that already
carries ``dataset_meta`` so :class:`MethodRecommender` v2 can apply task and
platform priors without re-deriving semantics from filenames.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .features import RECOMMENDATION_FEATURE_ORDER
from .landscape import LandscapeResult, _compute_niches, _embed_datasets
from .task_contract import (
    AnalysisTask,
    GroundTruthKind,
    TaskContract,
    classify_platform,
)


def attach_dataset_meta(
    landscape: LandscapeResult,
    meta: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> LandscapeResult:
    """Return a copy of *landscape* with ``dataset_meta`` filled or extended."""
    merged = dict(landscape.dataset_meta)
    for name, row in meta.items():
        if name not in landscape.performance:
            continue
        if name in merged and not overwrite:
            current = dict(merged[name])
            current.update(
                {k: v for k, v in row.items() if k not in current or current[k] in (None, "")}
            )
            merged[name] = current
        else:
            merged[name] = dict(row)
    landscape.dataset_meta = merged
    return landscape


def meta_from_registry(
    dataset_names: Iterable[str],
    *,
    name_map: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build ``dataset_meta`` rows from :mod:`histoweave.datasets.real` entries.

    Parameters
    ----------
    dataset_names
        Keys used in a landscape performance matrix (e.g. ``151673`` or
        ``dlpfc_151673``).
    name_map
        Optional alias map landscape-key → registry name.
    """
    from ..datasets.real import get_dataset, list_datasets

    available = {row["name"] for row in list_datasets()}
    name_map = dict(name_map or {})
    out: dict[str, dict[str, Any]] = {}
    for key in dataset_names:
        registry_name = name_map.get(key, key)
        if registry_name not in available:
            # Common DLPFC shorthand: bare slice id → dlpfc_<id>
            candidate = f"dlpfc_{key}" if not str(key).startswith("dlpfc_") else key
            if candidate in available:
                registry_name = candidate
            else:
                out[str(key)] = {
                    "platform": None,
                    "task": AnalysisTask.SPATIAL_DOMAIN.value,
                    "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
                    "registry_name": None,
                }
                continue
        entry = get_dataset(registry_name)
        out[str(key)] = entry.to_dataset_meta()
        out[str(key)]["registry_name"] = entry.name
    return out


def landscape_from_long_csv(
    path: str | Path,
    *,
    task: str | AnalysisTask = AnalysisTask.SPATIAL_DOMAIN,
    metric: str = "ARI",
    dataset_col: str = "dataset",
    method_col: str | None = None,
    score_col: str = "ari",
    seed_col: str = "seed",
    seconds_col: str = "seconds",
    config_col: str | None = "config",
    prefer_config_as_method: bool = True,
    dataset_meta: dict[str, dict[str, Any]] | None = None,
    features: dict[str, np.ndarray] | None = None,
    higher_is_better: bool = True,
) -> LandscapeResult:
    """Aggregate a ``benchmark_long.csv`` into a :class:`LandscapeResult`.

    When both ``config`` (e.g. ``kmeans@sw0.8``) and ``method`` columns exist,
    ``prefer_config_as_method=True`` uses the configuration key so recommender
    v2 can rank method×spatial-context policies.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        fields = set(reader.fieldnames)
        if dataset_col not in fields or score_col not in fields:
            raise ValueError(f"{path} must contain columns {dataset_col!r} and {score_col!r}")
        use_config = prefer_config_as_method and config_col is not None and config_col in fields
        resolved_method_col = method_col or ("method" if "method" in fields else None)
        if not use_config and resolved_method_col is None:
            raise ValueError(f"{path} needs a method or config column")

        for row in reader:
            dataset = str(row[dataset_col]).strip()
            if use_config and row.get(config_col):
                method = str(row[config_col]).strip()
            else:
                method = str(row[resolved_method_col]).strip()
            if not dataset or not method:
                continue
            raw = row.get(score_col, "")
            if raw in ("", None):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            # Optional status filter (SOTA adapters write status=failed).
            status = str(row.get("status", "") or "").lower()
            if status in {"failed", "error", "timeout", "oom"}:
                continue
            scores[dataset][method].append(value)
            if seconds_col in fields and row.get(seconds_col) not in ("", None):
                try:
                    times[dataset][method].append(float(row[seconds_col]))
                except (TypeError, ValueError):
                    pass

    if len(scores) < 1:
        raise ValueError(f"{path} produced no finite performance rows")

    performance = {
        ds: {method: float(np.mean(vals)) for method, vals in methods.items()}
        for ds, methods in scores.items()
    }
    timings = {
        ds: {
            method: (float(np.mean(vals)) if vals else None)
            for method, vals in times.get(ds, {}).items()
        }
        for ds in performance
    }

    if features is None:
        # Placeholder feature vectors (NaN) so the landscape object is valid;
        # callers should attach real features via attach_features_from_tables.
        order = list(RECOMMENDATION_FEATURE_ORDER)
        features = {ds: np.full(len(order), np.nan, dtype=float) for ds in performance}
        feature_order = order
    else:
        feature_order = list(RECOMMENDATION_FEATURE_ORDER)
        for ds, vector in features.items():
            if len(vector) != len(feature_order):
                raise ValueError(
                    f"feature vector for {ds!r} has length {len(vector)}; "
                    f"expected {len(feature_order)}"
                )

    embedding = _embed_datasets(features)
    best_method, niches = _compute_niches(performance)
    task_value = task.value if isinstance(task, AnalysisTask) else str(task)
    if task_value in {"domain_detection", "domain"}:
        task_value = AnalysisTask.SPATIAL_DOMAIN.value

    result = LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings,
        feature_order=feature_order,
        method_count=len({m for row in performance.values() for m in row}),
        dataset_count=len(performance),
        task=task_value,
        metric=metric,
        higher_is_better=higher_is_better,
        dataset_meta={},
    )
    if dataset_meta is None:
        dataset_meta = meta_from_registry(performance.keys())
        # Force task consistency with the landscape task.
        for row in dataset_meta.values():
            row.setdefault("task", task_value)
    attach_dataset_meta(result, dataset_meta, overwrite=True)
    return result


def merge_landscapes(
    *landscapes: LandscapeResult,
    task: str | None = None,
    prefer_later: bool = True,
) -> LandscapeResult:
    """Merge multiple landscapes that share the same analysis task.

    Performance cells from later landscapes overwrite earlier ones when
    ``prefer_later=True`` (default), so a SOTA matrix can overlay a baseline
    sklearn landscape without losing datasets that only appear in one source.
    """
    if not landscapes:
        raise ValueError("merge_landscapes requires at least one landscape")
    base_task = task or landscapes[0].task
    performance: dict[str, dict[str, float]] = {}
    features: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float | None]] = {}
    dataset_meta: dict[str, dict[str, Any]] = {}
    feature_order = list(landscapes[0].feature_order or RECOMMENDATION_FEATURE_ORDER)
    metric = landscapes[0].metric
    higher = landscapes[0].higher_is_better

    ordered = landscapes if prefer_later else reversed(landscapes)
    for land in ordered:
        if land.task not in {base_task, "domain_detection"} and base_task not in {
            land.task,
            "domain_detection",
            AnalysisTask.SPATIAL_DOMAIN.value,
        }:
            # Soft check: allow legacy domain_detection alias only.
            if {land.task, base_task} != {"domain_detection", AnalysisTask.SPATIAL_DOMAIN.value}:
                raise ValueError(
                    f"cannot merge landscapes with tasks {land.task!r} and {base_task!r}"
                )
        for ds, row in land.performance.items():
            performance.setdefault(ds, {}).update(row)
        for ds, vec in land.features.items():
            features[ds] = np.asarray(vec, dtype=float)
        for ds, timing_row in land.timings.items():
            if ds not in timings:
                timings[ds] = {}
            timings[ds].update(timing_row)
        for ds, meta in land.dataset_meta.items():
            dataset_meta.setdefault(ds, {}).update(meta)

    # Ensure every performance dataset has a feature vector.
    for ds in performance:
        if ds not in features:
            features[ds] = np.full(len(feature_order), np.nan, dtype=float)

    embedding = _embed_datasets(features)
    best_method, niches = _compute_niches(performance)
    task_value = base_task
    if task_value in {"domain_detection", "domain"}:
        task_value = AnalysisTask.SPATIAL_DOMAIN.value
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings,
        feature_order=feature_order,
        method_count=len({m for row in performance.values() for m in row}),
        dataset_count=len(performance),
        task=task_value,
        metric=metric,
        higher_is_better=higher,
        dataset_meta=dataset_meta,
    )


def attach_features_from_tables(
    landscape: LandscapeResult,
    tables: dict[str, Any],
) -> LandscapeResult:
    """Replace placeholder features with target-free vectors from SpatialTables."""
    from .features import extract_features, feature_vector

    order = list(landscape.feature_order or RECOMMENDATION_FEATURE_ORDER)
    for name, table in tables.items():
        if name not in landscape.performance:
            continue
        feats = extract_features(table, include_domain=False)
        landscape.features[name] = feature_vector(feats, order=order)
    landscape.embedding = _embed_datasets(landscape.features)
    return landscape


def write_landscape_json(landscape: LandscapeResult, path: str | Path) -> Path:
    """Persist a landscape as recommender knowledge-base schema v3."""
    from .recommend import MethodRecommender

    path = Path(path)
    MethodRecommender(landscape).save_knowledge_base(path)
    return path


def validate_landscape_contracts(landscape: LandscapeResult) -> list[str]:
    """Return human-readable contract violations (empty list = ok)."""
    problems: list[str] = []
    for ds, meta in landscape.dataset_meta.items():
        task = meta.get("task") or landscape.task
        kind = meta.get("ground_truth_kind")
        label_key = meta.get("label_key") or "domain_truth"
        if not kind:
            problems.append(f"{ds}: missing ground_truth_kind in dataset_meta")
            continue
        try:
            TaskContract(
                task=AnalysisTask(task) if not isinstance(task, AnalysisTask) else task,
                ground_truth_kind=GroundTruthKind(kind),
                label_key=str(label_key),
                platform=classify_platform(meta.get("platform")),
            ).validate()
        except ValueError as exc:
            problems.append(f"{ds}: {exc}")
        if str(task) in {AnalysisTask.SPATIAL_DOMAIN.value, "domain_detection"} and str(kind) in {
            GroundTruthKind.SELF_SUPERVISED.value,
            GroundTruthKind.CLUSTER_PROXY.value,
        }:
            problems.append(
                f"{ds}: spatial_domain landscape must not use ground_truth_kind={kind!r}"
            )
    return problems


def dlpfc_slice_name_map() -> dict[str, str]:
    """Map bare spatialLIBD slice ids to registry names."""
    return {
        "151507": "dlpfc_151507",
        "151508": "dlpfc_151508",
        "151509": "dlpfc_151509",
        "151510": "dlpfc_151510",
        "151669": "dlpfc_151669",
        "151670": "dlpfc_151670",
        "151671": "dlpfc_151671",
        "151672": "dlpfc_151672",
        "151673": "dlpfc_151673",
        "151674": "dlpfc_151674",
        "151675": "dlpfc_151675",
        "151676": "dlpfc_151676",
    }


def build_dlpfc_merged_landscape(
    *,
    baseline_csv: str | Path | None = None,
    sota_csv: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> LandscapeResult:
    """Build the P1 DLPFC landscape from on-disk CSV artefacts when present.

    * Baseline: ``5x15_spatial_aware/benchmark_long.csv`` (method@sw configs).
    * SOTA (optional): any long CSV with columns dataset/method/ari[/status]
      for SpaGCN/GraphST/BayesSpace/STAGATE/banksy_py.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    baseline_path = Path(baseline_csv or root / "5x15_spatial_aware" / "benchmark_long.csv")
    landscapes: list[LandscapeResult] = []
    if baseline_path.exists():
        base = landscape_from_long_csv(
            baseline_path,
            task=AnalysisTask.SPATIAL_DOMAIN,
            prefer_config_as_method=True,
        )
        attach_dataset_meta(
            base,
            meta_from_registry(base.performance.keys(), name_map=dlpfc_slice_name_map()),
            overwrite=True,
        )
        for meta in base.dataset_meta.values():
            meta["task"] = AnalysisTask.SPATIAL_DOMAIN.value
            meta["ground_truth_kind"] = GroundTruthKind.SPATIAL_DOMAIN.value
            meta["label_key"] = "domain_truth"
            meta["study"] = meta.get("study") or "Maynard2021_spatialLIBD"
        landscapes.append(base)

    sota_path = (
        Path(sota_csv)
        if sota_csv is not None
        else root / "5x15_spatial_aware" / "sota_benchmark_long.csv"
    )
    if sota_path.exists():
        try:
            sota = landscape_from_long_csv(
                sota_path,
                task=AnalysisTask.SPATIAL_DOMAIN,
                prefer_config_as_method=False,
                method_col="method",
            )
        except ValueError:
            # Dry-run / all-skipped SOTA grids have no finite ARI cells — skip merge.
            sota = None
        if sota is not None:
            attach_dataset_meta(
                sota,
                meta_from_registry(sota.performance.keys(), name_map=dlpfc_slice_name_map()),
                overwrite=True,
            )
            for meta in sota.dataset_meta.values():
                meta["task"] = AnalysisTask.SPATIAL_DOMAIN.value
                meta["ground_truth_kind"] = GroundTruthKind.SPATIAL_DOMAIN.value
                meta["label_key"] = "domain_truth"
                meta["study"] = meta.get("study") or "Maynard2021_spatialLIBD"
                meta["track"] = "sota"
            landscapes.append(sota)

    if not landscapes:
        raise FileNotFoundError(
            f"No DLPFC landscape CSVs found. Expected {baseline_path} and/or {sota_path}"
        )
    if len(landscapes) == 1:
        return landscapes[0]
    return merge_landscapes(*landscapes, task=AnalysisTask.SPATIAL_DOMAIN.value)
