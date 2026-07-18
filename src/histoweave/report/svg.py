"""Tiny pure-Python SVG scatter plotting.

Reports must be *self-contained* (no external assets) and the scaffold must run without
a plotting stack, so spatial scatter plots are emitted as inline SVG. A real deployment
delegates interactive exploration to Vitessce / napari-spatialdata, as the plan states;
this is the static, embeddable fallback used inside the HTML report.
"""

from __future__ import annotations

import colorsys
import html

import numpy as np

# A colour-blind-friendly-ish qualitative palette; extended procedurally if needed.
_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def _colors_for(labels: list[str]) -> dict[str, str]:
    uniq = sorted(dict.fromkeys(labels))
    mapping = {}
    for i, lab in enumerate(uniq):
        if i < len(_PALETTE):
            mapping[lab] = _PALETTE[i]
        else:  # generate additional distinct hues past the base palette
            h = (i * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.85)
            mapping[lab] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return mapping


def spatial_scatter_svg(
    coords: np.ndarray,
    labels: list[str],
    *,
    width: int = 460,
    height: int = 380,
    radius: float = 2.6,
    title: str = "",
) -> str:
    """Return an inline SVG string: points at ``coords`` coloured by ``labels``."""
    coords = np.asarray(coords, dtype=float)
    labels = [str(x) for x in labels]
    if len(labels) != len(coords):
        raise ValueError(
            f"coords and labels must be the same length, got {len(coords)} points "
            f"and {len(labels)} labels"
        )
    color_map = _colors_for(labels)

    pad = 34
    xs, ys = coords[:, 0], coords[:, 1]
    xmin: float = float(xs.min())
    xmax: float = float(xs.max())
    ymin: float = float(ys.min())
    ymax: float = float(ys.max())
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def sx(x: float) -> float:
        return pad + (x - xmin) / xr * (width - 2 * pad)

    def sy(y: float) -> float:
        # Flip y so the plot reads like an image.
        return height - pad - (y - ymin) / yr * (height - 2 * pad)

    points = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" '
        f'fill="{color_map[lab]}" fill-opacity="0.85"/>'
        for x, y, lab in zip(xs, ys, labels, strict=True)
    )

    # Legend
    legend_items = []
    for i, (lab, col) in enumerate(color_map.items()):
        ly = pad + i * 18
        legend_items.append(
            f'<rect x="{width - pad + 6}" y="{ly}" width="11" height="11" fill="{col}"/>'
            f'<text x="{width - pad + 22}" y="{ly + 10}" font-size="11" '
            f'fill="currentColor">{html.escape(lab)}</text>'
        )
    legend = "".join(legend_items)
    legend_width = 120

    title_el = (
        f'<text x="{pad}" y="20" font-size="13" font-weight="600" '
        f'fill="currentColor">{html.escape(title)}</text>'
        if title
        else ""
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width + legend_width} {height}" '
        f'width="{width + legend_width}" height="{height}" '
        f'font-family="system-ui, sans-serif">'
        f"{title_el}{points}{legend}</svg>"
    )


def continuous_scatter_svg(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    width: int = 460,
    height: int = 380,
    radius: float = 2.6,
    title: str = "",
) -> str:
    """Inline SVG scatter coloured by a continuous scalar (e.g. uncertainty)."""
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coords and values must have the same length")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("values must contain at least one finite entry")
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = (vmax - vmin) or 1.0

    def _colour(v: float) -> str:
        if not np.isfinite(v):
            return "#BAB0AC"
        t = (v - vmin) / span
        # Pale yellow → accent orange (colour-blind friendlier than rainbow).
        r = int(255 * (0.95 - 0.05 * t))
        g = int(255 * (0.92 - 0.55 * t))
        b = int(255 * (0.70 - 0.55 * t))
        return f"#{r:02x}{g:02x}{b:02x}"

    pad = 34
    xs, ys = coords[:, 0], coords[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def sx(x: float) -> float:
        return pad + (x - xmin) / xr * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - ymin) / yr * (height - 2 * pad)

    points = "".join(
        f'<circle cx="{sx(float(x)):.1f}" cy="{sy(float(y)):.1f}" r="{radius}" '
        f'fill="{_colour(float(v))}" fill-opacity="0.9"/>'
        for x, y, v in zip(xs, ys, values, strict=True)
    )
    title_el = (
        f'<text x="{pad}" y="20" font-size="13" font-weight="600" '
        f'fill="currentColor">{html.escape(title)}</text>'
        if title
        else ""
    )
    legend = (
        f'<text x="{width - 100}" y="{pad + 10}" font-size="11" fill="currentColor">'
        f"{html.escape(f'{vmin:.2f}')} → {html.escape(f'{vmax:.2f}')}</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width + 40} {height}" '
        f'width="{width + 40}" height="{height}" '
        f'font-family="system-ui, sans-serif">'
        f"{title_el}{points}{legend}</svg>"
    )
