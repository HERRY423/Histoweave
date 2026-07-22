"""Figures for the 5-dataset external-validation landscape (SVG + PNG, Phylo palette).

Produces five figures from the experiment + recommender outputs:
  * fig1_performance_heatmap  -> 5 datasets x 15 methods, mean ARI
  * fig2_method_boxplot       -> ARI distribution per method across datasets x seeds
  * fig3_landscape_embedding  -> 2D dataset-feature landscape, coloured by best method
  * fig4_recommender_regret   -> selection regret vs global-best / random baselines
  * selective_regret_coverage  - selective regret versus personalisation coverage
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    _LOGGER.info("%s", message)


BASE_DIR = Path(__file__).resolve().parent
SRC = Path(os.environ.get("HISTOWEAVE_EXT_OUT", BASE_DIR))
FIG = SRC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATASET_IDS = [
    "visium_hd_crc",
    "xenium_lung_cancer",
    "xenium_ovarian_cancer",
    "visium_mouse_brain",
    "allen_merfish_brain_section",
]
DATASET_LABELS = {
    "visium_hd_crc": "Visium HD\nCRC",
    "xenium_lung_cancer": "Xenium\nLung",
    "xenium_ovarian_cancer": "Xenium Prime\nOvarian",
    "visium_mouse_brain": "Visium\nMouse brain",
    "allen_merfish_brain_section": "MERFISH\nMouse brain",
}
# Phylo palette
CB = ["#0279EE", "#FF9400", "#75A025", "#FD9BED", "#000000", "#E9ED4C"]


def save(fig, name):
    for ext in ("svg", "png"):
        out = FIG / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=400)
        if ext == "svg":
            lines = out.read_text(encoding="utf-8").splitlines()
            out.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
            )
    plt.close(fig)


def heatmap():
    mean = pd.read_csv(SRC / "performance_matrix_mean.csv", index_col="dataset")
    mean = mean.reindex(index=DATASET_IDS)
    methods = list(mean.columns)
    M = mean.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    vmax = float(np.nanmax(M)) if np.isfinite(np.nanmax(M)) else 1.0
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(DATASET_IDS)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASET_IDS], fontsize=9)
    for i in range(len(DATASET_IDS)):
        for j in range(len(methods)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < vmax * 0.6 else "black",
                    fontsize=7,
                )
    ax.set_title("External-validation landscape: mean ARI (5 datasets 脳 15 methods)")
    fig.colorbar(im, ax=ax, label="mean ARI", shrink=0.8)
    save(fig, "fig1_performance_heatmap")
    _log("wrote fig1_performance_heatmap")


def method_boxplot():
    long = pd.read_csv(SRC / "benchmark_long.csv")
    long = long[long["ari"].notna()]
    order = long.groupby("method")["ari"].mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(13, 5.0))
    data = [long[long["method"] == m]["ari"].values for m in order]
    bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], CB * 4, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("ARI vs region ground truth")
    ax.set_xlabel("method")
    ax.set_title("Method ARI distribution across 5 external datasets 脳 3 seeds")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "fig2_method_boxplot")
    _log("wrote fig2_method_boxplot")


def landscape_embedding():
    """2D PCA embedding of dataset feature vectors, coloured by best method."""
    # Re-extract features via the experiment module to get the embedding.
    import sys

    sys.path.insert(0, str(BASE_DIR))
    from recommender_loocv_external import _build_landscape  # noqa: E402

    # Rebuild a landscape from the mean perf + features (features cached on import).
    # Simpler: read the recommender's landscape if present, else recompute.
    mean = pd.read_csv(SRC / "performance_matrix_mean.csv", index_col="dataset").reindex(
        index=DATASET_IDS
    )
    methods = list(mean.columns)
    perf = {
        ds: {
            m: (float(mean.loc[ds, m]) if np.isfinite(mean.loc[ds, m]) else float("nan"))
            for m in methods
        }
        for ds in DATASET_IDS
    }
    # features: extract fresh
    from experiment_5x_external import load_dataset

    from histoweave.benchmark.features import (
        RECOMMENDATION_FEATURE_ORDER,
        extract_features,
        feature_vector,
    )

    features = {}
    for ds in DATASET_IDS:
        tab, _, _ = load_dataset(ds, seed=42)
        feats = extract_features(tab, include_domain=False)
        features[ds] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
    landscape = _build_landscape(perf, features)
    emb = landscape.embedding
    best = landscape.best_method

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    method_colors = {m: CB[i % len(CB)] for i, m in enumerate(sorted(set(best.values())))}
    for ds in DATASET_IDS:
        x, y = emb[ds]
        ax.scatter(x, y, s=180, c=method_colors[best[ds]], edgecolor="black", zorder=3)
        ax.annotate(
            DATASET_LABELS[ds].replace("\n", " "),
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    handles = [
        plt.scatter([], [], c=c, s=80, edgecolor="black", label=m) for m, c in method_colors.items()
    ]
    ax.legend(handles=handles, title="best method", fontsize=8, loc="best")
    ax.set_xlabel("PC1 (dataset-feature space)")
    ax.set_ylabel("PC2")
    ax.set_title("External-validation dataset landscape (target-free features)")
    save(fig, "fig3_landscape_embedding")
    _log("wrote fig3_landscape_embedding")


def recommender_regret():
    rec = json.loads((SRC / "recommendation_loocv.json").read_text())
    rows = rec["rows"]
    ds = [r["held_out_dataset"] for r in rows]
    labels = [DATASET_LABELS[d].replace("\n", " ") for d in ds]
    sel = [float(r["selection_regret"]) for r in rows]
    gbr = [float(r["global_best_regret"]) for r in rows]
    rnd = [float(r["random_expected_regret"]) for r in rows]

    x = np.arange(len(ds))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.bar(x - w, rnd, w, label="random baseline", color="#CCCCCC", edgecolor="black")
    ax.bar(x, gbr, w, label="global-best baseline", color="#75A025", edgecolor="black")
    ax.bar(x + w, sel, w, label="recommender (top-1)", color="#0279EE", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("selection regret (oracle ARI 鈭?chosen ARI)")
    ax.set_title("Recommender regret vs baselines (leave-one-dataset-out)")
    ax.legend()
    save(fig, "fig4_recommender_regret")
    _log("wrote fig4_recommender_regret")


def selective_regret_coverage():
    """Plot the selective policy curve from the frozen n=20 endpoint."""
    source = SRC / "selective_regret_coverage.json"
    if not source.exists():
        source = BASE_DIR.parent / "protocol_endpoints_results" / source.name
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["curve"]
    if not rows:
        raise ValueError(f"empty selective-regret curve: {source}")

    rows = sorted(rows, key=lambda row: float(row["coverage"]))
    coverage = np.asarray([float(row["coverage"]) for row in rows])
    personalised = np.asarray(
        [float(row["mean_regret_always_personalised"]) for row in rows]
    )
    global_default = np.asarray(
        [float(row["mean_regret_always_global"]) for row in rows]
    )
    hybrid = np.asarray(
        [float(row["mean_regret_abstain_as_global"]) for row in rows]
    )

    positive = np.unique(coverage[coverage > 0])
    region_boundary = float(positive.min() / 2.0) if positive.size else 0.5

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvspan(-0.02, region_boundary, color="#D9D9D9", alpha=0.55, zorder=0)
    ax.axvspan(region_boundary, 1.02, color="#F2F2F2", alpha=0.65, zorder=0)
    ax.plot(
        coverage,
        personalised,
        color="#D55E00",
        linestyle="--",
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Always personalised",
        zorder=3,
    )
    ax.plot(
        coverage,
        global_default,
        color="#0072B2",
        linestyle="-",
        marker="s",
        markersize=4.5,
        linewidth=1.8,
        label="Always global",
        zorder=3,
    )
    ax.plot(
        coverage,
        hybrid,
        color="#009E73",
        linestyle="-.",
        marker="^",
        markersize=5,
        linewidth=1.8,
        label="Abstain as global",
        zorder=4,
    )

    delta = float(personalised[0] - global_default[0])
    ax.annotate(
        f"Global improves regret by {delta:.3f} ARI",
        xy=(0.98, global_default[-1]),
        xytext=(0.58, 0.0325),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.8},
        fontsize=8,
        ha="center",
    )
    ax.text(
        region_boundary / 2.0,
        0.0535,
        "Maintain global default\n(full abstention)",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    ax.text(
        (region_boundary + 1.0) / 2.0,
        0.0535,
        "Personalisation retained\n(no observed regret benefit)",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.02, 0.055)
    ax.set_xlabel("Personalisation coverage (fraction of queries)")
    ax.set_ylabel("Mean selection regret (ARI; lower is better)")
    ax.set_title(
        "Selective regret-coverage: abstention prevents harmful personalisation"
    )
    ax.legend(frameon=False, loc="lower center", ncol=3)
    ax.grid(axis="y", color="#FFFFFF", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.0,
        -0.22,
        f"n={payload['n_queries']} independent queries; points are observed means. "
        "Repeated thresholds overlap at identical coverage.",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    save(fig, "selective_regret_coverage")
    _log(f"wrote selective_regret_coverage from {source}")

def main() -> None:
    heatmap()
    method_boxplot()
    landscape_embedding()
    recommender_regret()
    selective_regret_coverage()
    _log("\n===== FIGURES DONE =====")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
