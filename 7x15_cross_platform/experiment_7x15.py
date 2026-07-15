"""HistoWeave 7-dataset x 15-config cross-platform spatial-aware domain landscape.

Extends the 5x15 spatial-aware DLPFC benchmark across sequencing platforms:

  * 7 datasets: 5 DLPFC Visium slices + MERFISH + Slide-seqV2 (+ Xenium as 8th when its
    cache is present). Cross-platform coverage: sequencing-based spot (Visium),
    imaging-based single-molecule (MERFISH/Xenium), bead-based (Slide-seqV2).
  * 15 configs: 5 core k-aware clusterers x 3 spatial weights {0.0, 0.3, 0.8}
    (same design as 5x15).
  * metric = ARI vs each dataset's domain label; per-dataset n_domains = label count.
  * >=3 seeds -> mean +/- sd; leave-one-DATASET-out recommendation validation.

Ground-truth caveat: DLPFC uses spatialLIBD manual layers (expert domains). MERFISH /
Slide-seqV2 / Xenium have no expert spatial-domain annotation, so each uses a proxy label
(published cell-type / transcriptomic cluster, or Leiden clustering for Xenium). Cross-
platform ARI therefore measures recovery of annotated cell-type structure on those, not
histological domains. This is documented in the report.

Consumes pre-built .h5ad caches: DLPFC from HISTOWEAVE_DLPFC_DATA, cross-platform from
HISTOWEAVE_XPLAT_DATA. Cross-platform datasets are subsampled to <=6,000 cells in prep so
the domain methods' brute-force spatial KNN stays within memory and matches Visium scale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from histoweave.benchmark.landscape import _compute_niches, run_task_landscape
from histoweave.benchmark.recommend import MethodRecommender
from histoweave.data import SpatialTable
from histoweave.plugins import MethodCategory

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


BASE_DIR = Path(__file__).resolve().parent
DLPFC_DIR = Path(os.environ.get("HISTOWEAVE_DLPFC_DATA", BASE_DIR / "dlpfc_cache"))
XPLAT_DIR = Path(os.environ.get("HISTOWEAVE_XPLAT_DATA", BASE_DIR / "xplat_cache"))
OUT = Path(os.environ.get("HISTOWEAVE_BENCHMARK_OUT", BASE_DIR))
OUT.mkdir(parents=True, exist_ok=True)

DLPFC_SLICES = ["151673", "151674", "151507", "151669", "151670"]
XPLAT_CANDIDATES = ["merfish", "slideseqv2", "xenium"]

CORE_METHODS = ["kmeans", "gaussian_mixture", "agglomerative", "spectral", "birch"]
SPATIAL_WEIGHTS = [0.0, 0.3, 0.8]
CONFIGS = [f"{m}@sw{w}" for w in SPATIAL_WEIGHTS for m in CORE_METHODS]
SEEDS = [42, 1, 2]
PROTOCOL = "histoweave.landscape.cross_platform_spatial_aware.v1"

PLATFORM = {
    "151673": "Visium",
    "151674": "Visium",
    "151507": "Visium",
    "151669": "Visium",
    "151670": "Visium",
    "merfish": "MERFISH",
    "slideseqv2": "Slide-seqV2",
    "xenium": "Xenium",
}
GROUND_TRUTH_KIND = {
    "Visium": "expert_manual_layer",
    "MERFISH": "proxy_cell_class",
    "Slide-seqV2": "proxy_cluster",
    "Xenium": "proxy_leiden_cluster",
}


def _cfg(method: str, w: float) -> str:
    return f"{method}@sw{w}"


def _load_h5ad(path: Path, sid: str) -> tuple[SpatialTable, int]:
    a = sc.read_h5ad(path)
    counts = a.layers["counts"]
    X = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical(a.obs["domain_truth"].astype(str).values)},
        index=a.obs_names.astype(str),
    )
    var = pd.DataFrame(index=a.var_names.astype(str))
    tab = SpatialTable(
        X=X.astype(float),
        obs=obs,
        var=var,
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=float)},
        uns={"slice_id": sid},
    )
    return tab, int(obs["domain_truth"].nunique())


def build_datasets() -> tuple[list[str], dict, dict]:
    order: list[str] = []
    datasets: dict[str, SpatialTable] = {}
    n_domains_map: dict[str, int] = {}
    for sid in DLPFC_SLICES:
        p = DLPFC_DIR / f"{sid}.h5ad"
        tab, k = _load_h5ad(p, sid)
        datasets[sid] = tab
        n_domains_map[sid] = k
        order.append(sid)
        _log(
            f"[load] {sid} ({PLATFORM[sid]}): {tab.n_obs} cells, {tab.n_vars} genes, n_domains={k}"
        )
    for sid in XPLAT_CANDIDATES:
        p = XPLAT_DIR / f"{sid}.h5ad"
        if not p.exists():
            _log(f"[skip] {sid}: cache {p} missing")
            continue
        tab, k = _load_h5ad(p, sid)
        datasets[sid] = tab
        n_domains_map[sid] = k
        order.append(sid)
        _log(
            f"[load] {sid} ({PLATFORM[sid]}): {tab.n_obs} cells, {tab.n_vars} genes, n_domains={k}"
        )
    return order, datasets, n_domains_map


def main() -> None:
    DATASETS_ORDER, datasets, n_domains_map = build_datasets()
    _log(f"\n[info] {len(DATASETS_ORDER)} datasets x {len(CONFIGS)} configs x {len(SEEDS)} seeds")

    per_seed_perf: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_time: dict[int, dict[str, dict[str, float]]] = {}
    landscape_ref = None
    for seed in SEEDS:
        _log(f"\n=== seed {seed} ===")
        merged_perf = {ds: {} for ds in DATASETS_ORDER}
        merged_time = {ds: {} for ds in DATASETS_ORDER}
        ref_for_seed = None
        for w in SPATIAL_WEIGHTS:

            def factory(data: SpatialTable, _seed=seed, _w=w) -> dict:
                return {
                    "n_domains": n_domains_map[data.uns["slice_id"]],
                    "random_state": _seed,
                    "spatial_weight": _w,
                }

            ls = run_task_landscape(
                datasets,
                category=MethodCategory.DOMAIN_DETECTION,
                methods=list(CORE_METHODS),
                extra_params_factory=factory,
            )
            if ref_for_seed is None:
                ref_for_seed = ls
            for ds in DATASETS_ORDER:
                for m in CORE_METHODS:
                    cfg = _cfg(m, w)
                    merged_perf[ds][cfg] = ls.performance[ds].get(m, float("nan"))
                    merged_time[ds][cfg] = ls.timings[ds].get(m)
            _log(
                f"  [sw={w}] "
                + " ".join(
                    f"{ds}:{ls.performance[ds].get(ls.best_method.get(ds), float('nan')):.3f}"
                    for ds in DATASETS_ORDER
                )
            )

        per_seed_perf[seed] = merged_perf
        per_seed_time[seed] = merged_time
        if seed == SEEDS[0]:
            landscape_ref = ref_for_seed
            landscape_ref.performance = merged_perf
            landscape_ref.best_method, landscape_ref.niches = _compute_niches(merged_perf)

    # --- aggregate ---
    long_rows = []
    for ds in DATASETS_ORDER:
        for cfg in CONFIGS:
            for s in SEEDS:
                long_rows.append(
                    {
                        "dataset": ds,
                        "platform": PLATFORM[ds],
                        "ground_truth": GROUND_TRUTH_KIND[PLATFORM[ds]],
                        "config": cfg,
                        "method": cfg.split("@")[0],
                        "spatial_weight": float(cfg.split("@sw")[1]),
                        "seed": s,
                        "ari": _f(per_seed_perf[s][ds].get(cfg, np.nan)),
                        "seconds": per_seed_time[s][ds].get(cfg),
                        "n_domains_truth": n_domains_map[ds],
                    }
                )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "benchmark_long.csv", index=False)

    agg = long_df.groupby(["dataset", "config"])["ari"].agg(["mean", "std", "count"]).reset_index()
    mean_mat = agg.pivot(index="dataset", columns="config", values="mean").reindex(
        index=DATASETS_ORDER, columns=CONFIGS
    )
    std_mat = agg.pivot(index="dataset", columns="config", values="std").reindex(
        index=DATASETS_ORDER, columns=CONFIGS
    )
    mean_mat.to_csv(OUT / "performance_matrix_mean.csv")
    std_mat.to_csv(OUT / "performance_matrix_std.csv")

    time_agg = (
        long_df.dropna(subset=["seconds"])
        .groupby(["dataset", "config"])["seconds"]
        .mean()
        .reset_index()
    )
    time_agg.to_csv(OUT / "timings_mean.csv", index=False)

    mean_perf = {
        ds: {
            cfg: float(mean_mat.loc[ds, cfg])
            if np.isfinite(mean_mat.loc[ds, cfg])
            else float("nan")
            for cfg in CONFIGS
        }
        for ds in DATASETS_ORDER
    }
    landscape_ref.performance = mean_perf
    landscape_ref.best_method, landscape_ref.niches = _compute_niches(mean_perf)

    feat_rows = []
    for ds in DATASETS_ORDER:
        fv = landscape_ref.features[ds]
        row = {"dataset": ds, "platform": PLATFORM[ds]}
        row.update({f: _f(fv[i]) for i, f in enumerate(landscape_ref.feature_order)})
        feat_rows.append(row)
    pd.DataFrame(feat_rows).to_csv(OUT / "dataset_features.csv", index=False)

    MethodRecommender(
        landscape_ref, k_neighbours=min(2, len(DATASETS_ORDER) - 1)
    ).save_knowledge_base(OUT / "landscape.json")

    rec_rows = leave_one_out(datasets, landscape_ref, mean_perf, DATASETS_ORDER)
    rec_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("neighbours", "feature_order")} for r in rec_rows]
    )
    for col in ("training_datasets", "oracle_methods", "recommended_methods"):
        rec_df[col] = rec_df[col].apply(lambda x: "|".join(map(str, x)))
    rec_df.to_csv(OUT / "recommendation_loocv.csv", index=False)

    summary = rec_summary(rec_rows)
    with open(OUT / "recommendation_loocv.json", "w") as f:
        json.dump({"protocol": PROTOCOL, "rows": _js(rec_rows), "summary": summary}, f, indent=2)

    master = {
        "protocol": PROTOCOL,
        "task": "cross_platform_domain_detection_spatial_aware",
        "metric": "ARI",
        "higher_is_better": True,
        "datasets": DATASETS_ORDER,
        "platforms": {ds: PLATFORM[ds] for ds in DATASETS_ORDER},
        "ground_truth_kind": {ds: GROUND_TRUTH_KIND[PLATFORM[ds]] for ds in DATASETS_ORDER},
        "configs": CONFIGS,
        "core_methods": CORE_METHODS,
        "spatial_weights": SPATIAL_WEIGHTS,
        "seeds": SEEDS,
        "n_domains_truth": n_domains_map,
        "performance_matrix_mean": _js(mean_perf),
        "best_method_seedmean": _js(landscape_ref.best_method),
        "recommendation": {"rows": _js(rec_rows), "summary": summary},
        "design_note": (
            "15 configs = 5 core k-aware clusterers x 3 spatial weights {0.0,0.3,0.8}. "
            "Cross-platform datasets subsampled to 6000 cells (fixed seed 0) to match "
            "Visium scale and fit the methods' brute-force spatial KNN."
        ),
        "limitations": [
            "Only DLPFC has expert manual spatial-domain truth; MERFISH/Slide-seqV2/Xenium "
            "use proxy cell-type or Leiden-cluster labels, so their ARI measures cell-type "
            "structure recovery, not histological domains.",
            "Cross-platform datasets subsampled to 6000 cells -> not full-resolution.",
            "Leave-one-dataset-out over 7-8 datasets is still a small recommendation sample.",
            "Spatial-aware set is a param sweep of sklearn-family clusterers, not dedicated "
            "GNN/HMRF methods (BANKSY etc. require containers unavailable here).",
        ],
    }
    with open(OUT / "figure_data.json", "w") as f:
        json.dump(master, f, indent=2)

    manifest = {"protocol": PROTOCOL, "artifacts": []}
    for p in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json")):
        if p.name == "manifest.json":
            continue
        manifest["artifacts"].append({"path": p.name, "sha256": _sha(p), "bytes": p.stat().st_size})
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log("\n=== DONE ===")
    _log(
        f"datasets={len(DATASETS_ORDER)} "
        f"LOOCV top1={summary['top1_accuracy']:.2f} top3={summary['top3_accuracy']:.2f} "
        f"mean_regret={summary['mean_selection_regret']:.3f} "
        f"(random={summary['random_expected_mean_regret']:.3f}, "
        f"global_best={summary['global_best_mean_regret']:.3f})"
    )


def leave_one_out(datasets, landscape, perf, order):
    from histoweave.benchmark.figure3 import _subset_landscape

    rows = []
    for held in order:
        train_names = [s for s in order if s != held]
        training = _subset_landscape(landscape, train_names)
        rec = MethodRecommender(training, k_neighbours=min(2, len(train_names))).recommend(
            datasets[held], dataset_name=held
        )
        recommended = [it.method for it in rec.ranked_methods]
        held_scores = {m: v for m, v in perf[held].items() if np.isfinite(v)}
        oracle = max(held_scores.values())
        oracle_methods = sorted(m for m, v in held_scores.items() if np.isclose(v, oracle))
        sel = recommended[0] if recommended else None
        sel_score = held_scores.get(sel, np.nan)
        train_means = {
            cfg: float(
                np.mean(
                    [perf[n][cfg] for n in train_names if np.isfinite(perf[n].get(cfg, np.nan))]
                )
            )
            for cfg in CONFIGS
            if any(np.isfinite(perf[n].get(cfg, np.nan)) for n in train_names)
        }
        gbest = min(train_means, key=lambda m: (-train_means[m], m))
        gscore = held_scores.get(gbest, np.nan)
        rand_exp = float(np.mean(list(held_scores.values())))
        rows.append(
            {
                "held_out_dataset": held,
                "held_out_platform": PLATFORM[held],
                "training_datasets": train_names,
                "oracle_methods": oracle_methods,
                "oracle_score": _f(oracle),
                "recommended_methods": recommended[:3],
                "recommended_method": sel,
                "recommended_score": _f(sel_score),
                "top1_hit": sel in oracle_methods,
                "top3_hit": bool(set(recommended[:3]) & set(oracle_methods)),
                "selection_regret": _f(oracle - sel_score),
                "global_best_baseline": gbest,
                "global_best_score": _f(gscore),
                "global_best_regret": _f(oracle - gscore),
                "random_expected_score": rand_exp,
                "random_expected_regret": _f(oracle - rand_exp),
            }
        )
    return rows


def rec_summary(rows):
    reg = [float(r["selection_regret"]) for r in rows]
    greg = [float(r["global_best_regret"]) for r in rows]
    rreg = [float(r["random_expected_regret"]) for r in rows]
    mr, gm, rm = float(np.mean(reg)), float(np.mean(greg)), float(np.mean(rreg))
    return {
        "n_queries": len(rows),
        "top1_accuracy": float(np.mean([r["top1_hit"] for r in rows])),
        "top3_accuracy": float(np.mean([r["top3_hit"] for r in rows])),
        "mean_selection_regret": mr,
        "median_selection_regret": float(np.median(reg)),
        "global_best_mean_regret": gm,
        "random_expected_mean_regret": rm,
        "regret_reduction_vs_global_best": (1 - mr / gm) if gm > 0 else None,
        "regret_reduction_vs_random": (1 - mr / rm) if rm > 0 else None,
    }


def _f(v):
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _js(v):
    if isinstance(v, dict):
        return {str(k): _js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _sha(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
