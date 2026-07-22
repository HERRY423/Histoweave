"""Generate ``heatmap_5x19.svg`` from ``performance_matrix_mean.csv``.

Rows = slices (biological order), columns = methods sorted by mean ARI
descending. Cell colour = ARI (0 → paper, mid → lime, high → orange), text
label = 2-decimal ARI. Row family bands (sklearn vs spatial-aware) are
indicated with a thin coloured stripe on the right edge of the column
header.

The figure follows the Phylo palette convention (paper #FAF9F3, ink #000,
accents lime / orange / green / blue). SVG is kept text-editable
(``svg.fonttype = 'none'``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_HERE = Path(__file__).resolve().parent
SKLEARN = {
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
}

# Data cells use a perceptually-uniform, colorblind-safe sequential map so
# low/near-zero and negative ARI values stay distinguishable (the Phylo
# paper->lime->orange gradient washed out at the low end). Phylo accent
# colours are retained for the method-family band + legend below.
CMAP = plt.get_cmap("cividis")


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def build(csv: Path, out_svg: Path, out_png: Path, benchmark_json: Path | None = None) -> None:
    mat = pd.read_csv(csv, index_col=0)
    # Column sort: descending by column mean (ignoring NaN)
    order = mat.mean(axis=0, skipna=True).sort_values(ascending=False).index
    mat = mat[order]

    n_rows, n_cols = mat.shape
    fig_w = max(6.5, 0.9 + n_cols * 0.65)
    fig_h = max(3.0, 0.9 + n_rows * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    vmax = float(np.nanmax(mat.values)) if mat.values.size else 1.0
    vmax = max(0.1, vmax)
    vmin = float(np.nanmin(mat.values)) if mat.values.size else 0.0
    vmin = min(0.0, vmin)  # include any negative ARI in the scale
    im = ax.imshow(mat.values, cmap=CMAP, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(mat.index, fontsize=10)

    # Numeric labels — adaptive text colour for legibility on cividis
    # (dark cells get white text, bright cells get black text).
    span = (vmax - vmin) or 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat.iloc[i, j]
            if pd.isna(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#bbb")
            else:
                frac = (float(v) - vmin) / span
                txt_col = "#000000" if frac > 0.55 else "#ffffff"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=txt_col)

    # Family stripe below the column headers (green = spatial-aware).
    fam_h = 0.20
    for j, m in enumerate(mat.columns):
        colour = "#75A025" if m not in SKLEARN else "#ECE9E2"
        rect = Rectangle(
            (j - 0.5, -0.5 - fam_h),
            1,
            fam_h,
            transform=ax.transData,
            clip_on=False,
            facecolor=colour,
            edgecolor="none",
        )
        ax.add_patch(rect)
    ax.text(
        -0.6,
        -0.5 - fam_h / 2,
        "family",
        ha="right",
        va="center",
        fontsize=9,
        color="#666",
    )

    # Family legend — placed as a FIGURE-level legend at the very bottom so it
    # never collides with the rotated x-axis tick labels.
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#75A025", edgecolor="#000", label="spatial-aware"),
        Rectangle((0, 0), 1, 1, facecolor="#ECE9E2", edgecolor="#000", label="sklearn"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, fraction=0.03)
    cbar.set_label("ARI (mean over seeds)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"HistoWeave {n_rows}×{n_cols} — DLPFC domain-detection ARI",
        fontsize=12,
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"[write] {out_svg}")
    _log(f"[write] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(_HERE / "performance_matrix_mean.csv"))
    ap.add_argument("--svg", default=str(_HERE / "heatmap_5x19.svg"))
    ap.add_argument("--png", default=str(_HERE / "heatmap_5x19.png"))
    args = ap.parse_args()
    build(Path(args.csv), Path(args.svg), Path(args.png))


if __name__ == "__main__":
    main()
