"""Figures for the 5x10 DLPFC landscape (SVG per user preference)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
SRC = Path(os.environ.get("HISTOWEAVE_BENCHMARK_OUT", BASE_DIR))
FIG = SRC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SLICES = ["151673", "151674", "151507", "151669", "151670"]
CB = ["#0279EE", "#FF9400", "#75A025", "#FD9BED", "#000000"]


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def heatmap():
    mean = pd.read_csv(SRC / "performance_matrix_mean.csv", index_col=0)
    mean.index = mean.index.astype(str)
    mean = mean.reindex(SLICES)
    methods = list(mean.columns)
    M = mean.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    vmax = float(np.nanmax(M))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(len(SLICES)))
    ax.set_yticklabels(SLICES)
    for i in range(len(SLICES)):
        for j in range(len(methods)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < np.nanmax(M) * 0.55 else "black",
                    fontsize=8,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean ARI (3 seeds)")
    ax.set_title("5 DLPFC slices x 10 domain-detection methods — ARI vs manual layers")
    save(fig, "fig1_performance_heatmap")


def boxplot():
    long = pd.read_csv(SRC / "benchmark_long.csv")
    order = long.groupby("method")["ari"].mean().sort_values(ascending=False).index.tolist()
    data = [long[long.method == m]["ari"].dropna().to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, tick_labels=order)
    for patch in bp["boxes"]:
        patch.set_facecolor("#ECE9E2")
        patch.set_edgecolor("#0279EE")
    for i, m in enumerate(order):
        y = long[long.method == m]["ari"].dropna().to_numpy()
        ax.scatter(
            np.full_like(y, i + 1) + np.random.uniform(-0.12, 0.12, len(y)),
            y,
            s=14,
            color="#FF9400",
            alpha=0.7,
            zorder=3,
        )
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("ARI (across 5 slices x 3 seeds)")
    ax.set_title("Method ARI distribution across DLPFC slices")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    save(fig, "fig2_method_boxplot")


def landscape_embed():
    with open(SRC / "landscape.json") as f:
        kb = json.load(f)
    best = kb.get("best_method", {})
    # Recompute 2D embedding from stored dataset feature vectors.
    from histoweave.benchmark.landscape import _embed_datasets

    feats = {ds: np.asarray(v, dtype=float) for ds, v in kb["features"].items()}
    emb = _embed_datasets(feats)
    methods_seen = sorted(set(best.values()))
    cmap = {m: CB[i % len(CB)] for i, m in enumerate(methods_seen)}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for ds, (x, y) in emb.items():
        bm = best.get(ds, "?")
        ax.scatter(x, y, s=180, color=cmap.get(bm, "grey"), edgecolor="black", zorder=3)
        ax.annotate(ds, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markersize=10,
            markerfacecolor=cmap[m],
            markeredgecolor="black",
            label=m,
        )
        for m in methods_seen
    ]
    ax.legend(handles=handles, title="Best method", loc="best", fontsize=8)
    ax.set_xlabel("Landscape dim 1")
    ax.set_ylabel("Landscape dim 2")
    ax.set_title("Dataset feature landscape (coloured by best method)")
    save(fig, "fig3_landscape_embedding")


def timing():
    t = pd.read_csv(SRC / "timings_mean.csv")
    order = t.groupby("method")["seconds"].mean().sort_values().index.tolist()
    means = t.groupby("method")["seconds"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(order)), means.to_numpy(), color="#75A025", edgecolor="black")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Mean runtime per slice (s)")
    ax.set_title("Domain-detection method runtime (mean across 5 slices x 3 seeds)")
    for i, v in enumerate(means.to_numpy()):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    save(fig, "fig4_runtime")


if __name__ == "__main__":
    heatmap()
    boxplot()
    landscape_embed()
    timing()
    print("figures written to", FIG)
