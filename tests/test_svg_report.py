"""Tests for report/svg.py — spatial scatter SVG generation."""

import numpy as np
import pytest

from histoweave.report.svg import _colors_for, spatial_scatter_svg


def test_colors_for_returns_mapping_for_unique_labels():
    labels = ["domain_0", "domain_1", "domain_2"]
    mapping = _colors_for(labels)
    assert set(mapping) == set(labels)
    assert all(c.startswith("#") for c in mapping.values())


def test_colors_for_extends_past_base_palette():
    """When labels exceed the base palette size, colors are still generated."""
    labels = [f"label_{i}" for i in range(20)]
    mapping = _colors_for(labels)
    assert len(mapping) == 20
    assert all(c.startswith("#") for c in mapping.values())


def test_spatial_scatter_svg_basic():
    coords = np.array([[0, 0], [10, 10], [20, 0]])
    labels = ["a", "b", "a"]
    svg = spatial_scatter_svg(coords, labels)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert 'fill="#4C78A8"' in svg or 'fill="#F58518"' in svg
    # Three points + legend entries
    assert svg.count("<circle") == 3


def test_spatial_scatter_svg_with_title():
    coords = np.array([[0, 0], [10, 10]])
    labels = ["x", "y"]
    svg = spatial_scatter_svg(coords, labels, title="Test Title")
    assert "Test Title" in svg


def test_spatial_scatter_svg_custom_dimensions():
    coords = np.array([[0, 0]])
    labels = ["a"]
    svg = spatial_scatter_svg(coords, labels, width=600, height=500, radius=5.0)
    assert 'width="720"' in svg  # width + legend_width (600 + 120)
    assert 'height="500"' in svg
    assert 'r="5.0"' in svg


def test_spatial_scatter_svg_mismatched_lengths_raises():
    coords = np.array([[0, 0], [10, 10]])
    labels = ["a"]
    with pytest.raises(ValueError, match="coords and labels must be the same length"):
        spatial_scatter_svg(coords, labels)


def test_spatial_scatter_svg_empty_labels_raises():
    """Zero-length arrays raise ValueError on reduction operation."""
    coords = np.empty((0, 2))
    labels: list[str] = []
    with pytest.raises(ValueError):
        spatial_scatter_svg(coords, labels)


def test_spatial_scatter_svg_single_point():
    coords = np.array([[50.0, 50.0]])
    labels = ["only"]
    svg = spatial_scatter_svg(coords, labels)
    assert "<circle" in svg
    assert "only" in svg  # legend entry
