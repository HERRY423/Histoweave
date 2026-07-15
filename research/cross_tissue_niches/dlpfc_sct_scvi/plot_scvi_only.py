"""Render the scVI-only DLPFC pilot while SCT completes independently."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = ("astro_ion", "oligo_myelin", "vascular_barrier", "GEI")


def load_helper():
    path = Path(
        r"C:\Users\13264\.agents\skills\scientific-figure-pro\scripts\scientific_figure_pro.py"
    )
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    out = Path(__file__).resolve().parent / "results" / "scvi_only"
    overall = pd.read_csv(out / "overall_effects.csv").set_index("module").loc[list(PRIMARY)]
    donor = pd.read_csv(out / "donor_effects.csv")
    lodo = pd.read_csv(out / "leave_one_donor_out_prediction.csv")
    spots = pd.read_csv(out / "spot_module_scores.csv.gz")

    sfp = load_helper()
    sfp.apply_publication_style(sfp.FigureStyle(font_size=11, axes_linewidth=1.4))
    fig, axes = sfp.create_subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0]
    x = np.arange(len(PRIMARY))
    mean = overall["mean_beta_entropy"].to_numpy()
    low = mean - overall["ci95_low_donor_t"].to_numpy()
    high = overall["ci95_high_donor_t"].to_numpy() - mean
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([low, high]),
        fmt="o",
        color="#2166AC",
        ecolor="#555555",
        capsize=4,
        lw=1.7,
        ms=7,
    )
    ax.axhline(0, color="#B2182B", ls="--", lw=1)
    ax.set_xticks(x, PRIMARY, rotation=20, ha="right")
    ax.set_ylabel("Partial beta: layer-neighborhood entropy")
    ax.set_title("A  scVI effects (95% donor-t CI; n=3)")
    for index, module in enumerate(PRIMARY):
        row = overall.loc[module]
        ax.text(
            index,
            row["ci95_high_donor_t"] + 0.006,
            f"q={row['spatial_shift_q_bh']:.3f}",
            ha="center",
            fontsize=8,
        )

    ax = axes[1]
    matrix = (
        donor.loc[donor["module"].isin(PRIMARY)]
        .pivot(index="donor", columns="module", values="beta_entropy")
        .reindex(columns=PRIMARY)
    )
    limit = max(0.05, float(np.max(np.abs(matrix.to_numpy()))))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(PRIMARY)), PRIMARY, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_title("B  Direction in each biological donor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column, row, f"{matrix.iloc[row, column]:.3f}", ha="center", va="center", fontsize=8
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="partial beta")

    ax = axes[2]
    delta = (
        lodo.loc[lodo["module"].isin(PRIMARY)]
        .pivot(index="held_out_donor", columns="module", values="delta_r2_heldout")
        .reindex(columns=PRIMARY)
    )
    dlimit = max(0.001, float(np.max(np.abs(delta.to_numpy()))))
    image = ax.imshow(delta.to_numpy(), cmap="PiYG", vmin=-dlimit, vmax=dlimit, aspect="auto")
    ax.set_xticks(np.arange(len(PRIMARY)), PRIMARY, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(delta)), delta.index)
    ax.set_title("C  Leave-one-donor-out incremental R2")
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            ax.text(
                column, row, f"{delta.iloc[row, column]:.4f}", ha="center", va="center", fontsize=8
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="delta R2")

    ax = axes[3]
    spatial = spots.loc[spots["section"].astype(str) == "151673"]
    scatter = ax.scatter(
        spatial["x"],
        spatial["y"],
        c=spatial["scVI__vascular_barrier"],
        s=7,
        cmap="coolwarm",
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("D  151673: scVI vascular/barrier score")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "DLPFC raw-count scVI pilot: weak vascular-interface candidate",
        y=1.01,
        fontsize=14,
    )
    sfp.finalize_figure(fig, out / "figure_scvi_only", formats=["png", "pdf", "svg"], dpi=600)


if __name__ == "__main__":
    main()
