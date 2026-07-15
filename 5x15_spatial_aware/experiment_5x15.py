"""HistoWeave 5-dataset x 15-config spatial-aware domain-detection landscape on real DLPFC data.

Spatial-aware upgrade of the 5x10 DLPFC benchmark. Every HistoWeave domain-detection
method exposes a ``spatial_weight`` knob in [0, 1] that blends the expression embedding
with the spatial-neighbourhood term (0 = expression-only, 1 = space-only). Here we sweep
5 core, deterministic, k-aware clusterers across 3 spatial weights to produce 15 distinct
spatial-aware *configurations*:

  * 5 core methods x 3 spatial weights {0.0, 0.3, 0.8} = 15 configs
  * 5 DLPFC slices (spatialLIBD manual layer ground truth), same difficulty gradient as 5x10
  * metric = ARI vs manual layers; per-dataset n_domains = true layer count
  * >=3 random seeds per cell -> mean +/- sd
  * leave-one-slice-out recommendation validation with regret vs global-best / random

The landscape harness keys results by *method name* and only forwards params a method
declares, so we run ``run_task_landscape`` once per spatial weight and relabel each method
column as ``<method>@sw{w}`` before merging into a single 15-config performance matrix.

Consumes the pre-built .h5ad slices from prepare_dlpfc.py. Outputs land next to this
script by default.
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
DATA_DIR = Path(os.environ.get("HISTOWEAVE_DLPFC_DATA", BASE_DIR / "data"))
OUT = Path(os.environ.get("HISTOWEAVE_BENCHMARK_OUT", BASE_DIR))
OUT.mkdir(parents=True, exist_ok=True)

SLICES = ["151673", "151674", "151507", "151669", "151670"]
# 5 core deterministic, k-aware clusterers (accept n_domains + spatial_weight).
CORE_METHODS = ["kmeans", "gaussian_mixture", "agglomerative", "spectral", "birch"]
SPATIAL_WEIGHTS = [0.0, 0.3, 0.8]
# 15 configs = CORE_METHODS x SPATIAL_WEIGHTS, keyed "<method>@sw{w}".
CONFIGS = [f"{m}@sw{w}" for w in SPATIAL_WEIGHTS for m in CORE_METHODS]
SEEDS = [42, 1, 2]
PROTOCOL = "histoweave.landscape.dlpfc_spatial_aware.v1"


def _cfg(method: str, w: float) -> str:
    return f"{method}@sw{w}"


def load_slice(sid: str) -> tuple[SpatialTable, int]:
    """Load a prepared slice as a SpatialTable with raw counts in X.

    The landscape harness re-normalizes internally (log1p_cp10k), so we hand it
    raw counts to avoid double normalization.
    """
    a = sc.read_h5ad(DATA_DIR / f"{sid}.h5ad")
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
    n_domains = int(obs["domain_truth"].nunique())
    return tab, n_domains


def main() -> None:
    datasets: dict[str, SpatialTable] = {}
    n_domains_map: dict[str, int] = {}
    for sid in SLICES:
        tab, k = load_slice(sid)
        datasets[sid] = tab
        n_domains_map[sid] = k
        _log(f"[load] {sid}: {tab.n_obs} spots, {tab.n_vars} genes, n_domains={k}")

    # --- Multi-seed x multi-spatial-weight landscape ---
    # per_seed_perf[seed][dataset][config] = ARI
    per_seed_perf: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_time: dict[int, dict[str, dict[str, float]]] = {}
    landscape_ref = None
    for seed in SEEDS:
        _log(f"\n=== seed {seed} ===")
        merged_perf: dict[str, dict[str, float]] = {ds: {} for ds in SLICES}
        merged_time: dict[str, dict[str, float]] = {ds: {} for ds in SLICES}
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
            # Relabel <method> -> <method>@sw{w} and merge.
            for ds in SLICES:
                for m in CORE_METHODS:
                    cfg = _cfg(m, w)
                    merged_perf[ds][cfg] = ls.performance[ds].get(m, float("nan"))
                    merged_time[ds][cfg] = ls.timings[ds].get(m)
            _log(
                f"  [sw={w}] "
                + " ".join(
                    f"{ds}:{ls.performance[ds].get(ls.best_method.get(ds), float('nan')):.3f}"
                    for ds in SLICES
                )
            )

        per_seed_perf[seed] = merged_perf
        per_seed_time[seed] = merged_time
        if seed == SEEDS[0]:
            # Reuse the first seed's landscape object (features/embedding are
            # seed-independent) but overwrite its performance with the merged
            # 15-config matrix so downstream niches/recommender see all configs.
            landscape_ref = ref_for_seed
            landscape_ref.performance = merged_perf
            landscape_ref.best_method, landscape_ref.niches = _compute_niches(merged_perf)

    # --- Aggregate mean +/- sd across seeds ---
    long_rows = []
    for ds in SLICES:
        for cfg in CONFIGS:
            for s in SEEDS:
                long_rows.append(
                    {
                        "dataset": ds,
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

    # mean / std matrices (datasets x 15 configs)
    agg = long_df.groupby(["dataset", "config"])["ari"].agg(["mean", "std", "count"]).reset_index()
    mean_mat = agg.pivot(index="dataset", columns="config", values="mean").reindex(
        index=SLICES, columns=CONFIGS
    )
    std_mat = agg.pivot(index="dataset", columns="config", values="std").reindex(
        index=SLICES, columns=CONFIGS
    )
    mean_mat.to_csv(OUT / "performance_matrix_mean.csv")
    std_mat.to_csv(OUT / "performance_matrix_std.csv")

    # timings mean
    time_agg = (
        long_df.dropna(subset=["seconds"])
        .groupby(["dataset", "config"])["seconds"]
        .mean()
        .reset_index()
    )
    time_agg.to_csv(OUT / "timings_mean.csv", index=False)

    # --- Build a mean-score landscape for recommendation (seed-averaged) ---
    mean_perf = {
        ds: {
            cfg: float(mean_mat.loc[ds, cfg])
            if np.isfinite(mean_mat.loc[ds, cfg])
            else float("nan")
            for cfg in CONFIGS
        }
        for ds in SLICES
    }
    landscape_ref.performance = mean_perf
    landscape_ref.best_method, landscape_ref.niches = _compute_niches(mean_perf)

    # dataset features
    feat_rows = []
    for ds in SLICES:
        fv = landscape_ref.features[ds]
        row = {"dataset": ds}
        row.update({f: _f(fv[i]) for i, f in enumerate(landscape_ref.feature_order)})
        feat_rows.append(row)
    pd.DataFrame(feat_rows).to_csv(OUT / "dataset_features.csv", index=False)

    # save knowledge base
    MethodRecommender(landscape_ref, k_neighbours=2).save_knowledge_base(OUT / "landscape.json")

    # --- Leave-one-slice-out recommendation validation ---
    rec_rows = leave_one_out(datasets, landscape_ref, mean_perf)
    rec_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("neighbours", "feature_order")} for r in rec_rows]
    )
    for col in ("training_datasets", "oracle_methods", "recommended_methods"):
        rec_df[col] = rec_df[col].apply(lambda x: "|".join(map(str, x)))
    rec_df.to_csv(OUT / "recommendation_loocv.csv", index=False)

    summary = rec_summary(rec_rows)
    with open(OUT / "recommendation_loocv.json", "w") as f:
        json.dump({"protocol": PROTOCOL, "rows": _js(rec_rows), "summary": summary}, f, indent=2)

    # --- master JSON + manifest ---
    master = {
        "protocol": PROTOCOL,
        "task": "domain_detection_spatial_aware",
        "metric": "ARI",
        "higher_is_better": True,
        "datasets": SLICES,
        "configs": CONFIGS,
        "core_methods": CORE_METHODS,
        "spatial_weights": SPATIAL_WEIGHTS,
        "seeds": SEEDS,
        "n_domains_truth": n_domains_map,
        "performance_matrix_mean": _js(mean_perf),
        "best_method_seedmean": _js(landscape_ref.best_method),
        "recommendation": {"rows": _js(rec_rows), "summary": summary},
        "design_note": (
            "The '15' axis is a spatial-weight sweep: 5 core k-aware clusterers "
            "(kmeans, gaussian_mixture, agglomerative, spectral, birch) x 3 spatial "
            "weights {0.0, 0.3, 0.8}. spatial_weight blends the expression embedding "
            "with the spatial-neighbourhood term (0=expression-only, 1=space-only). "
            "BANKSY (the one distinct spatial-aware method in the registry) is "
            "container/R-only and was not runnable in this environment."
        ),
        "limitations": [
            "All 5 slices are from one study (Maynard 2021 human DLPFC) -> "
            "within-study validation.",
            "Only 5 leave-one-out queries -> imprecise recommendation-accuracy estimate.",
            "Spatial-aware set is a param sweep of sklearn-family clusterers, not "
            "dedicated GNN/HMRF spatial methods (BANKSY, SpaGCN, etc. require containers).",
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
        f"LOOCV top1={summary['top1_accuracy']:.2f} top3={summary['top3_accuracy']:.2f} "
        f"mean_regret={summary['mean_selection_regret']:.3f} "
        f"(random={summary['random_expected_mean_regret']:.3f}, "
        f"global_best={summary['global_best_mean_regret']:.3f})"
    )


def leave_one_out(datasets, landscape, perf):
    from histoweave.benchmark.figure3 import _subset_landscape

    rows = []
    for held in SLICES:
        train_names = [s for s in SLICES if s != held]
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
