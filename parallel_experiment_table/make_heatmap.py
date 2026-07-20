"""Render the parallel-experiment heatmap: all 33 method configs across the 5
shared DLPFC slices, grouped by family (sklearn | spatial_aware | sota).

Reads the matrix + method_meta produced by build_parallel_table.py (run that
first). Saves SVG (editable text) + PNG.

Usage (from the Histoweave repo root):
  python parallel_experiment_table/make_heatmap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

# Editable SVG text, Liberation Sans (Arial-metric) per project convention.
rcParams["svg.fonttype"] = "none"
rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Phylo palette
INK = "#000000"
FAMILY_COLOR = {
    "sklearn": "#0279EE",      # accent blue
    "spatial_aware": "#75A025", # accent green
    "sota": "#FF9400",          # accent orange
}

# True domain count per slice (for y-axis labels).
SLICE_NDOM = {
    "151673": 7, "151674": 7, "151507": 7, "151669": 8, "151670": 5,
}

matrix = pd.read_csv(HERE / "parallel_experiment_matrix.csv",
                     index_col="dataset", dtype={"dataset": str})
with open(HERE / "method_meta.json") as f:
    mm = json.load(f)
order = mm["order"]
meta = mm["meta"]
matrix = matrix[order]

vals = matrix.values.astype(float)
n_slice, n_method = vals.shape

cmap = plt.get_cmap("RdYlGn")
norm = plt.Normalize(vmin=-0.1, vmax=0.45)

fig, ax = plt.subplots(figsize=(n_method * 0.42 + 3.2, n_slice * 0.55 + 2.2))
im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

for i in range(n_slice):
    for j in range(n_method):
        v = vals[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center", color="#888888", fontsize=6.5)
            continue
        txt_color = INK if -0.1 <= v <= 0.25 else "#FFFFFF"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=txt_color, fontsize=6.5)

ax.set_xticks(range(n_method))
ax.set_xticklabels([c.replace("@sw", "\nsw").replace(" (", "\n(") for c in order],
                   rotation=0, fontsize=6.5, ha="center")
ax.set_yticks(range(n_slice))
slice_labels = [f"{s}\n({SLICE_NDOM[s]})" for s in matrix.index]
ax.set_yticklabels(slice_labels, fontsize=9)

# Family band on top.
band_y = -0.9
runs = []
start = 0
for i in range(1, n_method + 1):
    fam_now = meta[order[i - 1]]["family"] if i <= n_method else None
    fam_prev = meta[order[start]]["family"]
    if i == n_method or fam_now != fam_prev:
        runs.append((start, i - 1, fam_prev))
        start = i
for s, e, fam in runs:
    ax.add_patch(plt.Rectangle((s - 0.5, band_y), e - s + 1, 0.35,
                               facecolor=FAMILY_COLOR[fam], edgecolor="none",
                               clip_on=False, transform=ax.transData))
    ax.text((s + e) / 2, band_y + 0.17, fam, ha="center", va="center",
            color="#FFFFFF", fontsize=8, fontweight="bold", clip_on=False)

for s, e, fam in runs[:-1]:
    ax.axvline(x=e + 0.5, color=INK, linewidth=1.2)

ax.set_xlim(-0.5, n_method - 0.5)
ax.set_ylim(n_slice - 0.5, band_y - 0.1)
ax.tick_params(axis="both", which="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("Parallel experiment: spatial-domain ARI on the shared 5-slice DLPFC panel",
             fontsize=12, fontweight="bold", color=INK, pad=26, loc="left")
fig.text(0.02, 0.965,
         "Same task (domain detection, ARI vs Maynard 2021 layers) x same data (5 DLPFC Visium slices). "
         "sklearn & spatial-aware: 3 seeds, oracle-K. SOTA: 1 seed (42), oracle-K + 3 blind estimate-K.",
         fontsize=7.5, color="#444444", ha="left")

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[-0.1, 0, 0.15, 0.3, 0.45])
cbar.set_label("mean ARI", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.text(0.02, 0.02,
         "Y-labels: slice (true #domains).  sw = spatial_weight.  "
         "est-K = blind K estimate (silhouette / spatial_sil / ensemble).",
         fontsize=7, color="#666666", ha="left")

fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(FIG / "parallel_heatmap.svg", format="svg", bbox_inches="tight")
fig.savefig(FIG / "parallel_heatmap.png", format="png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"[write] {FIG/'parallel_heatmap.svg'}")
print(f"[write] {FIG/'parallel_heatmap.png'}")
