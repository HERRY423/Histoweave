"""Figures for the 5x15 spatial-aware DLPFC landscape (SVG + PNG, Phylo palette)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


BASE_DIR = Path(__file__).resolve().parent
SRC = Path(os.environ.get("HISTOWEAVE_BENCHMARK_OUT", BASE_DIR))
FIG = SRC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SLICES = ["151673", "151674", "151507", "151669", "151670"]
CORE = ["kmeans", "gaussian_mixture", "agglomerative", "spectral", "birch"]
WEIGHTS = [0.0, 0.3, 0.8]
CB = ["#0279EE", "#FF9400", "#75A025", "#FD9BED", "#000000"]


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def heatmap():
    mean = pd.read_csv(SRC / "performance_matrix_mean.csv", index_col=0)
    mean.index = mean.index.astype(str)
    mean = mean.reindex(SLICES)
    configs = list(mean.columns)
    M = mean.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    vmax = float(np.nanmax(M))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(SLICES)))
    ax.set_yticklabels(SLICES)
    # vertical separators between spatial-weight blocks (every 5 configs)
    for b in range(1, len(WEIGHTS)):
        ax.axvline(b * len(CORE) - 0.5, color="white", lw=2)
    for i in range(len(SLICES)):
        for j in range(len(configs)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < vmax * 0.55 else "black",
                    fontsize=7,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Mean ARI (3 seeds)")
    ax.set_title(
        "5 DLPFC slices x 15 spatial-aware configs (5 methods x spatial_weight {0.0,0.3,0.8}) — ARI vs manual layers"  # noqa: E501
    )
    save(fig, "heatmap_5x15")


def spatial_weight_effect():
    """Mean ARI vs spatial_weight per core method (does spatial awareness help?)."""
    long = pd.read_csv(SRC / "benchmark_long.csv")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, m in enumerate(CORE):
        sub = long[long.method == m]
        means = [sub[np.isclose(sub.spatial_weight, w)]["ari"].mean() for w in WEIGHTS]
        ax.plot(WEIGHTS, means, marker="o", color=CB[i % len(CB)], label=m, lw=2)
    ax.set_xlabel("spatial_weight (0 = expression only, 1 = space only)")
    ax.set_ylabel("Mean ARI (5 slices x 3 seeds)")
    ax.set_xticks(WEIGHTS)
    ax.set_title("Effect of spatial weighting on domain-detection ARI (DLPFC)")
    ax.legend(fontsize=8, title="method")
    ax.grid(alpha=0.3)
    save(fig, "spatial_weight_effect_5x15")


if __name__ == "__main__":
    heatmap()
    spatial_weight_effect()
    _log(f"figures written to {FIG}")
