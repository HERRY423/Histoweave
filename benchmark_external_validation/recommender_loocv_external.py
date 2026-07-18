"""Leave-one-dataset-out recommender validation on the external landscape.

This is the **key generalization test** for HistoWeave's method-recommendation
engine. The within-study 5x10/5x15 DLPFC benchmarks showed the recommender
cannot beat the trivial "always pick kmeans" global baseline (top-1 = 0,
mean selection regret 0.075 vs global-best 0.055) because all 5 slices come
from one study with near-identical feature profiles. Here we retrain the
recommender on the 5 external validation datasets — which span 4 platforms,
2 organisms, 4 tissues, and 4 independent studies — and ask the same
question: does similarity-weighted method selection now beat the global-best
baseline?

Protocol (mirrors ``5x15_spatial_aware/experiment_5x15.py`` LOOCV):
  * For each held-out dataset, train ``MethodRecommender`` on the other 4.
  * Recommend a ranked method list for the held-out dataset.
  * Compare the top-1 pick's ARI to the oracle-best method on that dataset.
  * Report top-1 / top-3 accuracy, mean / median selection regret, and both
    baselines (global-best = "always pick the method with the best mean ARI
    on the training datasets"; random = expected regret from a uniform pick).

Consumes the mean-ARI performance matrix + dataset features produced by
``experiment_5x_external.py``. Outputs:
  * recommendation_loocv.csv  — per held-out dataset row
  * recommendation_loocv.json — rows + summary metrics
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_5X15 = _HERE.parent / "5x15_spatial_aware"
for p in (str(_HERE), str(_5X15)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiment_5x_external import DATASET_IDS, METHODS, load_dataset  # noqa: E402

from histoweave.benchmark.figure3 import _subset_landscape  # noqa: E402
from histoweave.benchmark.landscape import LandscapeResult, _compute_niches  # noqa: E402
from histoweave.benchmark.recommend import MethodRecommender  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402

_LOGGER = logging.getLogger(__name__)
OUT = Path(os.environ.get("HISTOWEAVE_EXT_OUT", _HERE))
PROTOCOL = "histoweave.external_validation.recommender_loocv.v1"


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


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


def _build_landscape(
    perf: dict[str, dict[str, float]],
    features: dict[str, np.ndarray],
    timings: dict[str, dict[str, float | None]] | None = None,
) -> LandscapeResult:
    best_method, niches = _compute_niches(perf)
    from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER
    from histoweave.benchmark.landscape import _embed_datasets

    embedding = _embed_datasets(features)
    return LandscapeResult(
        performance=perf,
        features=features,
        embedding=embedding,
        best_method=best_method,
        niches=niches,
        timings=timings or {ds: {} for ds in DATASET_IDS},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=len(METHODS),
        dataset_count=len(DATASET_IDS),
        task="domain_detection",
        metric="ARI",
        higher_is_better=True,
    )


def main() -> None:
    # --- load mean performance matrix + features from the experiment outputs ---
    mean_csv = OUT / "performance_matrix_mean.csv"
    if not mean_csv.exists():
        raise FileNotFoundError(f"{mean_csv} not found; run experiment_5x_external.py first")
    mean_mat = pd.read_csv(mean_csv, index_col="dataset").reindex(
        index=DATASET_IDS, columns=METHODS
    )
    perf = {
        ds: {
            m: (float(mean_mat.loc[ds, m]) if np.isfinite(mean_mat.loc[ds, m]) else float("nan"))
            for m in METHODS
        }
        for ds in DATASET_IDS
    }

    # --- extract dataset features (target-free) for every dataset ---
    datasets: dict[str, SpatialTable] = {}
    n_domains_map: dict[str, int] = {}
    features: dict[str, np.ndarray] = {}
    for ds in DATASET_IDS:
        tab, k, _ = load_dataset(ds, seed=42)
        datasets[ds] = tab
        n_domains_map[ds] = k
        # Feature extraction happens inside run_task_landscape; to avoid a full
        # re-run, extract directly via the same helper the harness uses.
        from histoweave.benchmark.features import (
            RECOMMENDATION_FEATURE_ORDER,
            extract_features,
            feature_vector,
        )

        feats = extract_features(tab, include_domain=False)
        features[ds] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
        _log(f"[features] {ds}: {tab.n_obs} cells, {len(features[ds])}-dim vector")

    # --- load timings from the long-format benchmark output ---
    timings: dict[str, dict[str, float | None]] = {ds: {} for ds in DATASET_IDS}
    long_csv = OUT / "benchmark_long.csv"
    if long_csv.exists():
        long_df = pd.read_csv(long_csv)
        for _, row in long_df.iterrows():
            ds = row["dataset"]
            m = row["method"]
            if ds in timings:
                timings[ds][m] = float(row["seconds"]) if pd.notna(row.get("seconds")) else None

    landscape = _build_landscape(perf, features, timings=timings)

    # --- leave-one-dataset-out ---
    rows = []
    for held in DATASET_IDS:
        train_names = [d for d in DATASET_IDS if d != held]
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
        _log(
            f"[LOOCV] {held}: oracle={oracle_methods} rec={recommended[:3]} "
            f"top1={sel in oracle_methods} regret={_f(oracle - sel_score)}"
        )

    # --- summary ---
    reg = [float(r["selection_regret"]) for r in rows]
    greg = [float(r["global_best_regret"]) for r in rows]
    rreg = [float(r["random_expected_regret"]) for r in rows]
    mr, gm, rm = float(np.mean(reg)), float(np.mean(greg)), float(np.mean(rreg))
    summary = {
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

    rec_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("neighbours", "feature_order")} for r in rows]
    )
    for col in ("training_datasets", "oracle_methods", "recommended_methods"):
        rec_df[col] = rec_df[col].apply(lambda x: "|".join(map(str, x)))
    rec_df.to_csv(OUT / "recommendation_loocv.csv", index=False)

    with open(OUT / "recommendation_loocv.json", "w") as f:
        json.dump({"protocol": PROTOCOL, "rows": _js(rows), "summary": summary}, f, indent=2)

    _log("\n===== RECOMMENDER LOOCV DONE =====")
    _log(
        f"top1={summary['top1_accuracy']:.2f} top3={summary['top3_accuracy']:.2f} "
        f"mean_regret={summary['mean_selection_regret']:.3f} "
        f"(random={summary['random_expected_mean_regret']:.3f}, "
        f"global_best={summary['global_best_mean_regret']:.3f})"
    )
    _log(
        f"regret reduction vs global_best: {summary['regret_reduction_vs_global_best']}, "
        f"vs random: {summary['regret_reduction_vs_random']}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
