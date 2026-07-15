"""HistoWeave 5-dataset x 10-method spatial-domain performance landscape on real DLPFC data.

Real-data upgrade of the bundled `figure3` synthetic protocol:
  * 5 DLPFC slices (spatialLIBD manual layer ground truth), difficulty gradient
  * 10 sklearn-family domain-detection methods (same set as figure3)
  * metric = ARI vs manual layers; per-dataset n_domains = true layer count
  * >=3 random seeds per cell -> mean +/- sd (figure3 was single-seed)
  * leave-one-slice-out recommendation validation with regret vs global-best / random

Bypasses the registry checksum bug by consuming the pre-built .h5ad slices from
prepare_dlpfc.py. Outputs land next to this script by default.
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

from histoweave.benchmark.landscape import run_task_landscape
from histoweave.benchmark.recommend import MethodRecommender
from histoweave.data import SpatialTable
from histoweave.plugins import MethodCategory

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("HISTOWEAVE_DLPFC_DATA", BASE_DIR / "data"))
OUT = Path(os.environ.get("HISTOWEAVE_BENCHMARK_OUT", BASE_DIR))
OUT.mkdir(parents=True, exist_ok=True)

SLICES = ["151673", "151674", "151507", "151669", "151670"]
METHODS = [
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "dbscan",
    "gaussian_mixture",
    "kmeans",
    "mean_shift",
    "minibatch_kmeans",
    "optics",
    "spectral",
]
SEEDS = [42, 1, 2]
PROTOCOL = "histoweave.landscape.dlpfc_real.v1"


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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets: dict[str, SpatialTable] = {}
    n_domains_map: dict[str, int] = {}
    for sid in SLICES:
        tab, k = load_slice(sid)
        datasets[sid] = tab
        n_domains_map[sid] = k
        _LOGGER.info("[load] %s: %s spots, %s genes, n_domains=%s", sid, tab.n_obs, tab.n_vars, k)

    # --- Multi-seed landscape ---
    per_seed_perf: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_time: dict[int, dict[str, dict[str, float]]] = {}
    landscape_ref = None
    for seed in SEEDS:
        _LOGGER.info("\n=== seed %s ===", seed)

        def factory(data: SpatialTable, _seed=seed) -> dict:
            return {"n_domains": n_domains_map[data.uns["slice_id"]], "random_state": _seed}

        ls = run_task_landscape(
            datasets,
            category=MethodCategory.DOMAIN_DETECTION,
            methods=list(METHODS),
            extra_params_factory=factory,
        )
        ls.metric = "ARI"
        ls.higher_is_better = True
        per_seed_perf[seed] = ls.performance
        per_seed_time[seed] = ls.timings
        if seed == SEEDS[0]:
            landscape_ref = ls
        for ds in SLICES:
            best = ls.best_method.get(ds)
            _LOGGER.info(
                "  %s: best=%s ARI=%.3f", ds, best, ls.performance[ds].get(best, float("nan"))
            )

    # --- Aggregate mean +/- sd across seeds ---
    long_rows = []
    for ds in SLICES:
        for m in METHODS:
            for s in SEEDS:
                long_rows.append(
                    {
                        "dataset": ds,
                        "method": m,
                        "seed": s,
                        "ari": _f(per_seed_perf[s][ds].get(m, np.nan)),
                        "seconds": per_seed_time[s][ds].get(m),
                        "n_domains_truth": n_domains_map[ds],
                    }
                )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "benchmark_long.csv", index=False)

    # mean matrix
    agg = long_df.groupby(["dataset", "method"])["ari"].agg(["mean", "std", "count"]).reset_index()
    mean_mat = agg.pivot(index="dataset", columns="method", values="mean").reindex(
        index=SLICES, columns=METHODS
    )
    std_mat = agg.pivot(index="dataset", columns="method", values="std").reindex(
        index=SLICES, columns=METHODS
    )
    mean_mat.to_csv(OUT / "performance_matrix_mean.csv")
    std_mat.to_csv(OUT / "performance_matrix_std.csv")

    # timings mean
    time_agg = (
        long_df.dropna(subset=["seconds"])
        .groupby(["dataset", "method"])["seconds"]
        .mean()
        .reset_index()
    )
    time_agg.to_csv(OUT / "timings_mean.csv", index=False)

    # --- Build a mean-score landscape for recommendation (seed-averaged) ---
    mean_perf = {
        ds: {
            m: float(mean_mat.loc[ds, m]) if np.isfinite(mean_mat.loc[ds, m]) else float("nan")
            for m in METHODS
        }
        for ds in SLICES
    }
    landscape_ref.performance = mean_perf
    from histoweave.benchmark.landscape import _compute_niches

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
        "task": "domain_detection",
        "metric": "ARI",
        "higher_is_better": True,
        "datasets": SLICES,
        "methods": METHODS,
        "seeds": SEEDS,
        "n_domains_truth": n_domains_map,
        "performance_matrix_mean": _js(mean_perf),
        "best_method_seedmean": _js(landscape_ref.best_method),
        "recommendation": {"rows": _js(rec_rows), "summary": summary},
        "limitations": [
            "All 5 slices are from one study (Maynard 2021 human DLPFC) -> "
            "within-study validation.",
            "Only 5 leave-one-out queries -> imprecise recommendation-accuracy estimate.",
            "Cross-tissue / cross-platform generalization requires study-grouped "
            "multi-source data.",
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

    _LOGGER.info("\n=== DONE ===")
    _LOGGER.info(
        "LOOCV top1=%.2f top3=%.2f mean_regret=%.3f (random=%.3f, global_best=%.3f)",
        summary["top1_accuracy"],
        summary["top3_accuracy"],
        summary["mean_selection_regret"],
        summary["random_expected_mean_regret"],
        summary["global_best_mean_regret"],
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
            m: float(
                np.mean([perf[n][m] for n in train_names if np.isfinite(perf[n].get(m, np.nan))])
            )
            for m in METHODS
            if any(np.isfinite(perf[n].get(m, np.nan)) for n in train_names)
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
