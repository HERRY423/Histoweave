"""Pathology-polygon → cell-domain assignment, shared across Xenium preparers.

Several HistoWeave benchmark preparers ingest 10x Xenium bundles whose only
genuine *spatial-domain* ground truth is a pathologist's polygon annotation
(QuPath-exported GeoJSON). Cells are assigned to a domain by point-in-polygon
test on their centroid; cells outside every polygon, or inside overlapping
polygons carrying different labels, are excluded. Predicted cell-type labels
are never used as spatial-domain truth (consistent with the cross-tissue
benchmark policy).

This module factors the assignment logic out of
``benchmark_cross_tissue/prepare_human_lymph_node.py`` so the lung / ovarian
/ other Xenium-pathology preparers import a single tested implementation
instead of duplicating it.

Requires the ``spatial`` extra (shapely)::

    pip install "histoweave[spatial]"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _property(properties: dict[str, Any], dotted_name: str) -> Any:
    """Resolve a dotted property path (e.g. ``classification.name``)."""
    value: Any = properties
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def feature_label(feature: dict[str, Any], label_property: str | None) -> str | None:
    """Return the human-readable label of a GeoJSON feature.

    Tries the explicit ``label_property`` first, then the common QuPath /
    Xenium Explorer conventions (``classification.name``, ``name``, ``label``,
    ``type``). Returns ``None`` when no non-empty label is found.

    The fallback chain is always consulted even when ``label_property`` is
    set, because real-world QuPath / Xenium Explorer exports mix conventions
    within a single file (some features carry ``classification.name``, others
    only ``name``).
    """
    properties = feature.get("properties") or {}
    # Always try the explicit property first, then fall back to the standard
    # conventions so mixed-property GeoJSON files resolve every feature.
    candidates: tuple[str, ...]
    if label_property:
        candidates = (label_property, "classification.name", "name", "label", "type")
        # De-duplicate while preserving order.
        candidates = tuple(dict.fromkeys(candidates))
    else:
        candidates = ("classification.name", "name", "label", "type")
    for candidate in candidates:
        value = _property(properties, candidate)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def assign_pathology_domains(
    coordinates: np.ndarray,
    geojson: dict[str, Any],
    *,
    label_property: str | None = None,
    prefer_smaller: bool = True,
) -> pd.Series:
    """Assign unambiguous polygon labels to x/y coordinates.

    Parameters
    ----------
    coordinates
        Array of shape ``(n_cells, 2)`` with the x/y centroid of each cell,
        already transformed into the GeoJSON coordinate frame (apply any
        scale / offset before calling).
    geojson
        Parsed GeoJSON ``FeatureCollection`` (the contents of a QuPath
        ``Export objects as GeoJSON`` file).
    label_property
        Optional dotted property name overriding the auto-detected label
        conventions.

    Returns
    -------
    pd.Series
        String labels of length ``n_cells``. Cells inside no polygon are
        ``NA``; cells inside overlapping polygons with conflicting labels
        are ``"ambiguous"``.

    prefer_smaller : bool, default True
        When a cell falls inside multiple labelled polygons, assign it to the
        smallest-area polygon (the most specific annotation) instead of
        marking it ``"ambiguous"``. Pathology annotations commonly nest
        small specific regions (e.g. lymphoid aggregates) inside large
        background regions (e.g. tumor), so preferring the smaller polygon
        recovers the finer-grained label.

    Raises
    ------
    ImportError
        If shapely (the ``spatial`` extra) is not installed.
    ValueError
        If no labelled polygon intersects any cell (usually a coordinate
        scale/offset mismatch or a wrong ``label_property``).
    """
    try:
        from shapely import contains_xy
        from shapely.geometry import shape
    except ImportError as exc:
        raise ImportError(
            "Pathology polygon assignment requires the 'spatial' extra: "
            "pip install 'histoweave[spatial]'"
        ) from exc

    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_cells, 2)")
    labels = np.full(xy.shape[0], None, dtype=object)
    areas = np.full(xy.shape[0], np.inf, dtype=float)
    features = geojson.get("features", [])
    used = 0
    for feature in features:
        label = feature_label(feature, label_property)
        geometry = feature.get("geometry")
        if label is None or not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        polygon = shape(geometry)
        inside = np.asarray(contains_xy(polygon, xy[:, 0], xy[:, 1]), dtype=bool)
        if not inside.any():
            continue
        used += 1
        area = polygon.area
        empty = inside & pd.isna(labels)
        labels[empty] = label
        areas[empty] = area
        if prefer_smaller:
            # A cell already has a label; only overwrite if this polygon is
            # smaller (more specific annotation).
            overwrite = inside & ~pd.isna(labels) & (labels != "ambiguous") & (area < areas)
            labels[overwrite] = label
            areas[overwrite] = area
        else:
            conflict = inside & ~pd.isna(labels) & (labels != label)
            labels[conflict] = "ambiguous"
    if used == 0:
        raise ValueError(
            "no labelled pathology polygon intersected any cell; check the GeoJSON "
            "label property and coordinate scale/offset"
        )
    return pd.Series(labels, dtype="string")


def stratified_indices(labels: pd.Series, limit: int, seed: int) -> np.ndarray:
    """Return sorted indices of a stratified-by-label subsample of size ``limit``.

    Each label present in ``labels`` receives a proportional quota (minimum 1);
    the result is trimmed or topped up to exactly ``limit`` rows. When
    ``len(labels) <= limit`` every index is returned unchanged.
    """
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups = labels.groupby(labels, observed=True).indices
    selected: list[np.ndarray] = []
    for indices in groups.values():
        quota = max(1, round(len(indices) / len(labels) * limit))
        selected.append(rng.choice(indices, min(quota, len(indices)), replace=False))
    merged = np.unique(np.concatenate(selected))
    if len(merged) > limit:
        merged = rng.choice(merged, limit, replace=False)
    elif len(merged) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), merged, assume_unique=False)
        merged = np.concatenate([merged, rng.choice(remaining, limit - len(merged), False)])
    return np.sort(merged)


__all__ = ["assign_pathology_domains", "feature_label", "stratified_indices"]
