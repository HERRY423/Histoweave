#!/usr/bin/env python
"""Build a ≥20-query multi-source landscape and run protocol endpoints.

Endpoints (docs/decision-protocol.md):
  1. Study-grouped personalisation (leave-one-study/slice-out)
  2. Selective regret–coverage
  3. Pareto membership stability (seed bootstrap)
  4. SOTA comparison under a unified resource budget
  5. Oracle-K leakage impact (non-oracle K ARI drop; dual-track long CSV)

Usage
-----
python scripts/run_protocol_endpoints.py
python scripts/run_protocol_endpoints.py --skip-dlpfc-expand   # use only cached CSVs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.features import (  # noqa: E402
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)
from histoweave.benchmark.landscape import (  # noqa: E402
    LandscapeResult,
    _compute_niches,
    _embed_datasets,
)
from histoweave.benchmark.protocol_endpoints import (  # noqa: E402
    leave_one_study_out,
    oracle_k_leakage_impact,
    pareto_stability_from_long_csv,
    selective_regret_coverage,
    sota_unified_resource_compare,
    write_protocol_bundle,
)
from histoweave.benchmark.task_contract import AnalysisTask, GroundTruthKind  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402

LOG = logging.getLogger("histoweave.run_protocol_endpoints")

# Shared method set for multi-source LOO. Density-based methods (dbscan /
# mean_shift / optics) are excluded from expansion because they dominate wall
# time on full DLPFC matrices without winning on the Maynard layer task.
COMMON_METHODS = [
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "gaussian_mixture",
    "kmeans",
    "minibatch_kmeans",
    "spectral",
]

DLPFC_ALL = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]

DLPFC_DONOR = {
    "151507": "Br5292",
    "151508": "Br5292",
    "151509": "Br5292",
    "151510": "Br5292",
    "151669": "Br5595",
    "151670": "Br5595",
    "151671": "Br5595",
    "151672": "Br5595",
    "151673": "Br8100",
    "151674": "Br8100",
    "151675": "Br8100",
    "151676": "Br8100",
}

PLATFORM_STUDIES = ("merfish", "slideseqv2", "xenium")
EXTERNAL_STUDIES = (
    "visium_hd_crc",
    "xenium_lung_cancer",
    "xenium_ovarian_cancer",
    "visium_mouse_brain",
    "allen_merfish_brain_section",
)


def _mean_perf_from_long(
    path: Path,
    *,
    methods: list[str],
    collapse_config: bool = False,
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        use_config = collapse_config and "config" in fields
        for row in reader:
            status = str(row.get("status") or "").lower()
            if status in {"failed", "error", "timeout", "oom", "skipped"}:
                continue
            dataset = str(row.get("dataset") or "").strip()
            if use_config and row.get("config"):
                raw = str(row["config"]).strip()
                method = raw.split("@", 1)[0]
            else:
                method = str(row.get("method") or "").strip()
            if method not in methods:
                continue
            try:
                ari = float(row["ari"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(ari):
                continue
            scores[dataset][method].append(ari)
    return {
        dataset: {method: float(np.mean(vals)) for method, vals in methods_map.items()}
        for dataset, methods_map in scores.items()
    }


def _mean_timings_from_long(
    path: Path,
    *,
    methods: list[str],
    collapse_config: bool = False,
) -> dict[str, dict[str, float]]:
    times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "seconds" not in fields:
            return {}
        use_config = collapse_config and "config" in fields
        for row in reader:
            dataset = str(row.get("dataset") or "").strip()
            if use_config and row.get("config"):
                method = str(row["config"]).split("@", 1)[0]
            else:
                method = str(row.get("method") or "").strip()
            if method not in methods:
                continue
            try:
                seconds = float(row["seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(seconds):
                times[dataset][method].append(seconds)
    return {
        dataset: {method: float(np.mean(vals)) for method, vals in methods_map.items()}
        for dataset, methods_map in times.items()
    }


def _features_from_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    out: dict[str, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("dataset") or "").strip()
            if not name:
                continue
            values = []
            for key in RECOMMENDATION_FEATURE_ORDER:
                raw = row.get(key, "")
                try:
                    values.append(float(raw) if raw not in ("", None) else float("nan"))
                except (TypeError, ValueError):
                    values.append(float("nan"))
            # CSV may use full DEFAULT order including domain columns — ignore extras.
            out[name] = np.asarray(values, dtype=float)
    return out


def _load_dlpfc_table(sid: str, repo: Path) -> tuple[SpatialTable, int]:
    import pandas as pd
    import scanpy as sc

    path = repo / "datasets_cache" / "dlpfc" / f"dlpfc_{sid}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    truth_col = "domain_truth" if "domain_truth" in adata.obs else "spatialLIBD_layer"
    truth = adata.obs[truth_col].astype(str)
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    tab = SpatialTable(
        X=X.astype(np.float32),
        obs=pd.DataFrame(
            {"domain_truth": pd.Categorical(truth)}, index=adata.obs_names.astype(str)
        ),
        var=pd.DataFrame(index=adata.var_names.astype(str)),
        obsm={"spatial": np.asarray(adata.obsm["spatial"], dtype=np.float32)},
        uns={
            "slice_id": sid,
            "platform": "visium",
            "assay": "visium",
            "study_group": DLPFC_DONOR.get(sid, "dlpfc"),
            "donor": DLPFC_DONOR.get(sid),
        },
        layers={"counts": X.astype(np.float32)},
    )
    return tab, int(pd.Categorical(truth).categories.size)


def _expand_missing_dlpfc(
    existing: dict[str, dict[str, float]],
    *,
    repo: Path,
    methods: list[str],
    seeds: list[int],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Run sklearn domain methods on DLPFC slices missing from *existing*."""
    from histoweave.benchmark.landscape import run_task_landscape
    from histoweave.plugins import MethodCategory

    missing = [sid for sid in DLPFC_ALL if sid not in existing]
    if not missing:
        return {}, {}, {}

    LOG.info("Expanding DLPFC landscape for missing slices: %s", missing)
    datasets: dict[str, SpatialTable] = {}
    n_domains: dict[str, int] = {}
    features: dict[str, np.ndarray] = {}
    for sid in missing:
        tab, k = _load_dlpfc_table(sid, repo)
        # Cap gene dimension for expansion speed (HVG already ~2k in bundles).
        datasets[sid] = tab
        n_domains[sid] = k
        feats = extract_features(tab, include_domain=False)
        features[sid] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
        LOG.info("  loaded %s: n_obs=%s n_domains=%s", sid, tab.n_obs, k)

    per_seed: list[dict[str, dict[str, float]]] = []
    per_seed_time: list[dict[str, dict[str, float | None]]] = []
    for seed in seeds:
        # One slice at a time keeps peak memory bounded and surfaces progress.
        seed_perf: dict[str, dict[str, float]] = {}
        seed_time: dict[str, dict[str, float | None]] = {}
        for sid, tab in datasets.items():
            LOG.info("  running methods seed=%s slice=%s", seed, sid)

            def factory(data: SpatialTable, _seed: int = seed, _sid: str = sid) -> dict[str, Any]:
                return {"n_domains": n_domains[_sid], "random_state": _seed}

            landscape = run_task_landscape(
                {sid: tab},
                category=MethodCategory.DOMAIN_DETECTION,
                methods=methods,
                extra_params_factory=factory,
            )
            seed_perf[sid] = dict(landscape.performance[sid])
            seed_time[sid] = dict(landscape.timings.get(sid, {}))
        per_seed.append(seed_perf)
        per_seed_time.append(seed_time)

    mean_perf: dict[str, dict[str, float]] = {}
    mean_time: dict[str, dict[str, float]] = {}
    for sid in missing:
        mean_perf[sid] = {}
        mean_time[sid] = {}
        for method in methods:
            vals = [
                seed_map[sid][method]
                for seed_map in per_seed
                if sid in seed_map
                and method in seed_map[sid]
                and np.isfinite(seed_map[sid][method])
            ]
            mean_perf[sid][method] = float(np.mean(vals)) if vals else float("nan")
            tvals = [
                seed_map[sid][method]
                for seed_map in per_seed_time
                if sid in seed_map
                and method in seed_map[sid]
                and seed_map[sid][method] is not None
                and np.isfinite(seed_map[sid][method])  # type: ignore[arg-type]
            ]
            if tvals:
                mean_time[sid][method] = float(np.mean(tvals))
    return mean_perf, features, mean_time


def _external_proxy_features(repo: Path) -> dict[str, np.ndarray]:
    """Build limited feature proxies for external datasets when h5ad is absent."""
    manifest_path = repo / "benchmark_external_validation" / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    platform_code = {
        "Visium HD": 0.0,
        "Xenium": 1.0,
        "Xenium Prime": 1.2,
        "Visium v2": 0.2,
        "MERFISH": 2.0,
    }
    out: dict[str, np.ndarray] = {}
    for name, meta in manifest.items():
        vec = {key: float("nan") for key in RECOMMENDATION_FEATURE_ORDER}
        vec["n_obs"] = float(meta.get("n_obs") or float("nan"))
        # Domain count is a weak size proxy only; recommendation order excludes
        # domain labels but n_obs remains admissible.
        plat = str(meta.get("platform") or "")
        # Encode platform into spatial_entropy / cluster_tendency slots as soft priors
        # when raw tables are unavailable — documented as proxy features.
        code = platform_code.get(plat, 0.5)
        vec["cluster_tendency"] = 0.4 + 0.1 * code
        vec["spatial_autocorrelation"] = 0.3 + 0.05 * code
        vec["sparsity"] = 0.85 if "Visium" in plat else 0.7
        out[name] = feature_vector(vec, order=RECOMMENDATION_FEATURE_ORDER)
    return out


def _dataset_meta_row(
    name: str,
    *,
    platform: str | None,
    study_group: str | None = None,
    donor: str | None = None,
) -> dict[str, Any]:
    return {
        "platform": platform,
        "task": AnalysisTask.SPATIAL_DOMAIN.value,
        "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
        "study_group": study_group or name,
        "donor": donor,
    }


def build_multisource_landscape(
    repo: Path,
    *,
    expand_dlpfc: bool = True,
    seeds: list[int] | None = None,
) -> LandscapeResult:
    """Assemble a 20-query multi-source landscape from cached artefacts + expansion."""
    seeds = seeds or [42]
    methods = list(COMMON_METHODS)
    performance: dict[str, dict[str, float]] = {}
    timings: dict[str, dict[str, float | None]] = {}
    features: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}

    # --- 5x10 DLPFC baseline ---
    p_5x10 = repo / "5x10_dlpfc_benchmark" / "benchmark_long.csv"
    if p_5x10.is_file():
        perf = _mean_perf_from_long(p_5x10, methods=methods)
        time_map = _mean_timings_from_long(p_5x10, methods=methods)
        feat_map = _features_from_csv(repo / "5x10_dlpfc_benchmark" / "dataset_features.csv")
        for sid, row in perf.items():
            key = str(sid)
            performance[key] = row
            timings[key] = dict(time_map.get(key, {}))
            if key in feat_map:
                features[key] = feat_map[key]
            meta[key] = _dataset_meta_row(
                key,
                platform="visium",
                study_group=f"dlpfc_{DLPFC_DONOR.get(key, 'unknown')}",
                donor=DLPFC_DONOR.get(key),
            )

    # Expand remaining DLPFC slices
    if expand_dlpfc:
        extra_perf, extra_feat, extra_time = _expand_missing_dlpfc(
            performance, repo=repo, methods=methods, seeds=seeds
        )
        for sid, row in extra_perf.items():
            performance[sid] = row
            timings[sid] = dict(extra_time.get(sid, {}))
            features[sid] = extra_feat[sid]
            meta[sid] = _dataset_meta_row(
                sid,
                platform="visium",
                study_group=f"dlpfc_{DLPFC_DONOR.get(sid, 'unknown')}",
                donor=DLPFC_DONOR.get(sid),
            )
            # Also refresh features for previously present slices if missing.
        for sid in list(performance):
            if sid in DLPFC_ALL and sid not in features:
                try:
                    tab, _ = _load_dlpfc_table(sid, repo)
                    feats = extract_features(tab, include_domain=False)
                    features[sid] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
                except Exception as exc:
                    LOG.warning("Could not extract features for %s: %s", sid, exc)

    # Ensure features for all DLPFC already in performance
    for sid in list(performance):
        if sid in DLPFC_ALL and sid not in features:
            try:
                tab, _ = _load_dlpfc_table(sid, repo)
                feats = extract_features(tab, include_domain=False)
                features[sid] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
            except Exception as exc:
                LOG.warning("feature extraction failed for %s: %s", sid, exc)

    # --- Cross-platform studies ---
    p_7x15 = repo / "7x15_cross_platform" / "benchmark_long.csv"
    if p_7x15.is_file():
        perf = _mean_perf_from_long(p_7x15, methods=methods, collapse_config=True)
        time_map = _mean_timings_from_long(p_7x15, methods=methods, collapse_config=True)
        feat_map = _features_from_csv(repo / "7x15_cross_platform" / "dataset_features.csv")
        platform_of = {
            "merfish": "merfish",
            "slideseqv2": "slideseqv2",
            "xenium": "xenium",
        }
        for name in PLATFORM_STUDIES:
            if name not in perf:
                continue
            performance[name] = perf[name]
            timings[name] = dict(time_map.get(name, {}))
            if name in feat_map:
                features[name] = feat_map[name]
            meta[name] = _dataset_meta_row(name, platform=platform_of[name], study_group=name)

    # --- External multi-study ---
    p_ext = repo / "benchmark_external_validation" / "benchmark_long.csv"
    if p_ext.is_file():
        perf = _mean_perf_from_long(p_ext, methods=methods)
        time_map = _mean_timings_from_long(p_ext, methods=methods)
        proxy_feat = _external_proxy_features(repo)
        platforms = {
            "visium_hd_crc": "visium",
            "xenium_lung_cancer": "xenium",
            "xenium_ovarian_cancer": "xenium",
            "visium_mouse_brain": "visium",
            "allen_merfish_brain_section": "merfish",
        }
        for name in EXTERNAL_STUDIES:
            if name not in perf:
                continue
            performance[name] = perf[name]
            timings[name] = dict(time_map.get(name, {}))
            if name in proxy_feat:
                features[name] = proxy_feat[name]
            meta[name] = _dataset_meta_row(name, platform=platforms.get(name), study_group=name)

    if len(performance) < 2:
        raise RuntimeError("Insufficient landscapes found to build multi-source validation")

    # Fill missing feature vectors with NaN (imputed inside recommender).
    for name in performance:
        if name not in features:
            features[name] = np.full(len(RECOMMENDATION_FEATURE_ORDER), np.nan, dtype=float)
        # Align method columns
        for method in methods:
            performance[name].setdefault(method, float("nan"))

    embedding = _embed_datasets(features)
    best_method, niches = _compute_niches(performance)
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings,
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=len(methods),
        dataset_count=len(performance),
        task=AnalysisTask.SPATIAL_DOMAIN.value,
        metric="ARI",
        higher_is_better=True,
        dataset_meta=meta,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "protocol_endpoints_results",
        help="Output directory for endpoint artefacts",
    )
    parser.add_argument(
        "--skip-dlpfc-expand",
        action="store_true",
        help="Do not run missing DLPFC slices (CSV-only mode)",
    )
    parser.add_argument("--k-neighbours", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=200, help="Pareto bootstrap draws")
    parser.add_argument(
        "--resource-seconds",
        type=float,
        default=7200.0,
        help="Shared wall-time budget for SOTA resource filter",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    LOG.info("Building multi-source landscape (expand_dlpfc=%s)", not args.skip_dlpfc_expand)
    landscape = build_multisource_landscape(
        ROOT, expand_dlpfc=not args.skip_dlpfc_expand, seeds=[42]
    )
    LOG.info(
        "Landscape ready: %d datasets × %d methods",
        landscape.dataset_count,
        landscape.method_count,
    )

    # Prefer holding out real studies first (all keys are query units).
    query_names = sorted(landscape.performance.keys())
    queries, summary = leave_one_study_out(
        landscape,
        query_names=query_names,
        methods=COMMON_METHODS,
        k_neighbours=args.k_neighbours,
    )
    LOG.info(
        "Study-grouped: n=%d top1=%.2f mean_regret=%.4f global=%.4f beats=%s",
        summary.n_queries,
        summary.top1_accuracy,
        summary.mean_selection_regret,
        summary.mean_global_best_regret,
        summary.beats_global_best,
    )

    selective = selective_regret_coverage(queries)
    LOG.info(
        "Selective: thr=%s coverage=%s hybrid_regret=%s",
        selective.get("recommended_threshold"),
        selective.get("recommended_coverage"),
        selective.get("recommended_hybrid_regret"),
    )

    # Pareto stability: prefer spatial-aware multi-seed long CSV, else 5x10.
    pareto_src = ROOT / "5x15_spatial_aware" / "benchmark_long.csv"
    if not pareto_src.is_file():
        pareto_src = ROOT / "5x10_dlpfc_benchmark" / "benchmark_long.csv"
    pareto_stability = None
    if pareto_src.is_file():
        pareto_stability = pareto_stability_from_long_csv(
            pareto_src, n_boot=args.n_boot, seed=args.seed
        )
        LOG.info(
            "Pareto stability: %d datasets, n_boot=%d",
            pareto_stability["n_datasets"],
            pareto_stability["n_boot"],
        )

    sota_csv = ROOT / "5x15_spatial_aware" / "sota_benchmark_long.csv"
    baseline_csv = ROOT / "5x10_dlpfc_benchmark" / "benchmark_long.csv"
    sota_resource = None
    if sota_csv.is_file():
        sota_resource = sota_unified_resource_compare(
            sota_csv,
            max_seconds=args.resource_seconds,
            baseline_csv=baseline_csv if baseline_csv.is_file() else None,
            resource_label=f"cpu_timeout_{int(args.resource_seconds)}s",
        )
        top = (sota_resource.get("method_ranking") or [{}])[0]
        LOG.info(
            "SOTA resource: accepted=%s top=%s ari=%s",
            sota_resource.get("n_accepted_cells"),
            top.get("method"),
            top.get("mean_ari"),
        )

    # Endpoint 5: Oracle-K leakage from dual-track non-oracle K SOTA archive.
    leak_csv = ROOT / "non_oracle_k_sota" / "benchmark_long.csv"
    oracle_k_leakage = None
    if leak_csv.is_file():
        oracle_k_leakage = oracle_k_leakage_impact(leak_csv)
        LOG.info(
            "Oracle-K leakage: mean_drop=%s methods=%s",
            oracle_k_leakage.get("mean_ari_drop_across_methods"),
            oracle_k_leakage.get("methods"),
        )
    else:
        LOG.warning(
            "Skipping oracle_k_leakage endpoint: missing %s "
            "(run non_oracle_k_sota/run_non_oracle_k_sota.py first)",
            leak_csv,
        )

    # Persist landscape snapshot
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    landscape_path = out / "multisource_landscape.json"
    landscape_payload = {
        "task": landscape.task,
        "metric": landscape.metric,
        "higher_is_better": landscape.higher_is_better,
        "feature_order": landscape.feature_order,
        "performance": landscape.performance,
        "timings": landscape.timings,
        "features": {k: v.tolist() for k, v in landscape.features.items()},
        "best_method": landscape.best_method,
        "dataset_meta": landscape.dataset_meta,
        "dataset_count": landscape.dataset_count,
        "method_count": landscape.method_count,
        "methods": COMMON_METHODS,
        "query_names": query_names,
    }
    landscape_path.write_text(json.dumps(landscape_payload, indent=2), encoding="utf-8")

    paths = write_protocol_bundle(
        out,
        study_queries=queries,
        study_summary=summary,
        selective=selective,
        pareto_stability=pareto_stability,
        sota_resource=sota_resource,
        oracle_k_leakage=oracle_k_leakage,
        landscape_meta={
            "path": str(landscape_path.name),
            "n_datasets": landscape.dataset_count,
            "datasets": query_names,
            "methods": COMMON_METHODS,
            "notes": [
                "External studies use manifest-derived proxy features when raw h5ad is absent.",
                "DLPFC slices share three donors; independence is slice-level, "
                "not fully donor-level.",
                "Cross-platform and external studies supply cross-study queries.",
                "Oracle-K leakage uses non_oracle_k_sota/benchmark_long.csv when present.",
            ],
        },
    )
    LOG.info("Wrote artefacts: %s", {k: str(v) for k, v in paths.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
