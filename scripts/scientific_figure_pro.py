"""Small project-local subset of the scientific-figure-pro plotting helpers.

Research scripts import this module instead of carrying one-off matplotlib styles.
It deliberately stays dependency-light and is not part of the public SDK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np

PALETTE: Final[dict[str, str]] = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


@dataclass(frozen=True)
class FigureStyle:
    """Global matplotlib style configuration."""

    font_size: int = 12
    axes_linewidth: float = 1.5
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Apply publication-focused, vector-safe matplotlib defaults."""

    cfg = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(cfg.font_family),
            "font.size": cfg.font_size,
            "axes.labelsize": cfg.font_size,
            "axes.titlesize": cfg.font_size + 1,
            "axes.linewidth": cfg.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "legend.fontsize": max(8, cfg.font_size - 2),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure and return a flat axes array."""

    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(np.asarray(axes, dtype=object)).ravel()


def make_trend(
    ax: plt.Axes,
    x: Sequence[float],
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Draw deterministic publication-style trend lines."""

    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 1 or not len(x_values):
        raise ValueError("x must be a non-empty 1D sequence")
    if len(y_series) != len(labels):
        raise ValueError("y_series and labels must have the same length")
    color_values = list(colors or (PALETTE["blue_main"], PALETTE["green_3"]))
    for index, values in enumerate(y_series):
        y_values = np.asarray(values, dtype=float)
        if y_values.shape != x_values.shape:
            raise ValueError(f"series {index} does not match x")
        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=color_values[index % len(color_values)],
            label=labels[index],
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()


def finalize_figure(
    fig: plt.Figure,
    out_path: str | Path,
    *,
    formats: Sequence[str] = ("svg", "png"),
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
) -> list[Path]:
    """Export a figure consistently and without a volatile SVG timestamp."""

    base = Path(out_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"svg", "png", "pdf"}:
            raise ValueError(f"unsupported figure format: {normalized}")
        target = base.with_suffix(f".{normalized}")
        options: dict[str, Any] = {
            "format": normalized,
            "bbox_inches": "tight",
            "pad_inches": pad,
            "metadata": {"Date": None},
        }
        if normalized == "png":
            options["dpi"] = dpi
        fig.savefig(target, **options)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved
