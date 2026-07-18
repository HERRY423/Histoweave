"""Figures for the failure-boundary mapping study.

1. Per-axis score-vs-parameter curves (mean +/- std over seeds), with the
   acceptability threshold tau drawn and the failure point x* marked.
2. A cross-method summary heatmap of the safe-operating margin per (method, axis).

Saves both .svg (editable text) and .png.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["figure.dpi"] = 110

import matplotlib.pyplot as plt  # noqa: E402


def _log(message: object) -> None:
    """Emit a line through standard logging (repo logging contract)."""
    logging.getLogger(__name__).info("%s", message)


HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly (Okabe-Ito) palette, cycled across methods.
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#999999",
    "#8C564B",
    "#117733",
    "#882255",
    "#44AA99",
    "#DDCC77",
    "#AA4499",
]


def _load() -> tuple[dict, pd.DataFrame]:
    with open(RES / "safe_operating_cards.json") as fh:
        bundle = json.load(fh)
    cards = pd.read_csv(RES / "safe_operating_cards.csv")
    return bundle, cards


def plot_axis_curves(bundle: dict) -> list[Path]:
    """One figure per (task, axis): score vs parameter for every method."""
    curves = bundle["curves"]
    tau = bundle["tau"]
    # group curve keys by (task, param)
    groups: dict[tuple[str, str], list[str]] = {}
    for key, c in curves.items():
        groups.setdefault((c["task"], c["param"]), []).append(key)

    written: list[Path] = []
    for (task, param), keys in sorted(groups.items()):
        keys = sorted(keys, key=lambda k: curves[k]["method"])
        fig, ax = plt.subplots(figsize=(9, 5.6))
        label = curves[keys[0]]["label"] or param
        unit = curves[keys[0]]["unit"]
        direction = curves[keys[0]]["degrade_direction"]

        line_styles = ["-", "--", "-.", ":"]
        for i, key in enumerate(keys):
            c = curves[key]
            x = np.array(c["values"], dtype=float)
            m = np.array(c["mean"], dtype=float)
            s = np.array(c["std"], dtype=float)
            order = np.argsort(x)
            x, m, s = x[order], m[order], s[order]
            color = OKABE_ITO[i % len(OKABE_ITO)]
            # cycle line style every full pass through the palette so same-hued
            # curves remain distinguishable (colorblind-safety aid)
            ls = line_styles[(i // len(OKABE_ITO)) % len(line_styles)]
            ax.plot(
                x, m, marker="o", ms=3, lw=1.4, ls=ls, color=color, label=c["method"], alpha=0.9
            )
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.06, linewidth=0)
            # mark x* if a boundary was found
            xs = c.get("x_star")
            if xs is not None and not (isinstance(xs, float) and np.isnan(xs)):
                ax.plot(
                    [xs],
                    [tau],
                    marker="v",
                    ms=7,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    zorder=5,
                )

        ax.axhline(tau, color="crimson", ls="--", lw=1.3, zorder=1)
        ax.text(
            0.995,
            tau + 0.015,
            f"acceptability tau = {tau}",
            ha="right",
            va="bottom",
            color="crimson",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )
        arrow = "→ harder" if direction == "increasing" else "← harder"
        ax.set_xlabel(f"{label}" + (f" [{unit}]" if unit else "") + f"   ({arrow})")
        ax.set_ylabel("Benchmark score (mean ± std over seeds)")
        ax.set_title(f"Failure-boundary sweep — {task} · {param}", fontsize=12, weight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25, lw=0.5)
        if direction == "decreasing":
            ax.invert_xaxis()  # show easy(left) -> hard(right)
        # legend outside
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7.5,
            frameon=False,
            title="method",
            title_fontsize=8,
        )
        fig.tight_layout()
        stem = FIG / f"curve_{task}_{param}"
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
        written.append(stem.with_suffix(".png"))
    return written


def plot_boundary_heatmap(cards: pd.DataFrame, bundle: dict) -> Path:
    """Heatmap of a normalized 'safe margin' per (method, task:axis).

    Cell value encodes how much of the tested difficulty range each method
    survives (0 = fails immediately / never acceptable, 1 = robust across the
    whole range). Direction-aware so 'more is better' consistently.
    """
    axes_meta = {(a["task"], a["param"]): a for a in bundle["axes"]}

    # Build a robustness fraction in [0,1] for every card.
    def robust_fraction(row) -> float:
        meta = axes_meta[(row["task"], row["param"])]
        vals = np.array(meta["values"], dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        span = vmax - vmin if vmax > vmin else 1.0
        if row["verdict"] == "always_acceptable":
            return 1.0
        if row["verdict"] in ("never_acceptable", "no_runs", "non_monotone"):
            return 0.0
        xs = row["x_star"]
        if xs is None or (isinstance(xs, float) and np.isnan(xs)):
            return 0.0
        # fraction of the difficulty axis (easy->hard) that is safe
        if meta["degrade_direction"] == "increasing":
            frac = (float(xs) - vmin) / span  # safe from vmin up to x*
        else:
            frac = (vmax - float(xs)) / span  # safe from vmax down to x*
        return float(np.clip(frac, 0.0, 1.0))

    cards = cards.copy()
    cards["robust_frac"] = cards.apply(robust_fraction, axis=1)
    cards["col"] = cards["task"].str.replace("domain_detection", "domain") + "\n" + cards["param"]

    pivot = cards.pivot_table(index="method", columns="col", values="robust_frac", aggfunc="mean")
    # order columns by task then param, methods by mean robustness
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * pivot.shape[1] + 3), 0.42 * pivot.shape[0] + 2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(
        "Safe-operating margin by method and stress axis\n"
        "(fraction of tested difficulty range with score ≥ tau; green = robust, red = fails early)",
        fontsize=11,
        weight="bold",
    )
    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                txt, col = "n/a", "black"
            else:
                txt = f"{v:.2f}"
                # high-contrast text: white on saturated red/green extremes
                col = "white" if (v <= 0.28 or v >= 0.82) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=col, weight="medium")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("robust fraction of difficulty range", fontsize=8)
    fig.tight_layout()
    stem = FIG / "summary_boundary_heatmap"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle, cards = _load()
    curve_pngs = plot_axis_curves(bundle)
    heat_png = plot_boundary_heatmap(cards, bundle)
    _log("Wrote figures:")
    for p in curve_pngs:
        _log(f"  {p.name}")
    _log(f"  {heat_png.name}")


if __name__ == "__main__":
    main()
