"""Unit tests for static SVG report helpers."""

import numpy as np

from histoweave.report.svg import continuous_scatter_svg, spatial_scatter_svg


def test_spatial_scatter_svg_contains_circles() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    svg = spatial_scatter_svg(coords, ["a", "b", "a"], title="demo")
    assert "<svg" in svg
    assert "circle" in svg
    assert "demo" in svg


def test_continuous_scatter_svg_contains_legend_range() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    values = np.array([0.1, 0.5, 0.9])
    svg = continuous_scatter_svg(coords, values, title="uncertainty")
    assert "<svg" in svg
    assert "uncertainty" in svg
