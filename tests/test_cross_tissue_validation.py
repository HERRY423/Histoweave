"""Contracts for the two non-DLPFC real-data validation datasets."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pathology_polygons_supply_truth_without_cell_types():
    pytest.importorskip("shapely")
    prepare = _load(
        "prepare_human_lymph_node",
        ROOT / "benchmark_cross_tissue" / "prepare_human_lymph_node.py",
    )
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "B-cell follicle"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"classification": {"name": "T-cell zone"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[3, 0], [5, 0], [5, 2], [3, 2], [3, 0]]],
                },
            },
        ],
    }
    labels = prepare.assign_pathology_domains(np.array([[1, 1], [4, 1], [8, 8]]), geojson)
    assert labels.tolist()[:2] == ["B-cell follicle", "T-cell zone"]
    assert labels.isna().iloc[2]


def test_seven_by_nineteen_protocol_and_explicit_unsupported_cell(monkeypatch):
    original_import = builtins.__import__

    def import_without_scanpy(name, *args, **kwargs):
        if name == "scanpy":
            raise ModuleNotFoundError("scanpy is intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_scanpy)
    experiment = _load(
        "experiment_7x19",
        ROOT / "benchmark_cross_tissue" / "experiment_7x19.py",
    )
    assert len(experiment.DATASETS) == 7
    assert len(experiment.METHODS) == 19
    assert "xenium_human_lymph_node" in experiment.DATASETS
    assert "merfish_mouse_brain" in experiment.DATASETS
    assert experiment.unsupported_reason("bayesspace", "xenium_human_lymph_node")
    assert experiment.unsupported_reason("bayesspace", "merfish_mouse_brain")
    assert experiment.unsupported_reason("spagcn", "merfish_mouse_brain") is None


def test_readme_names_official_truth_sources():
    text = (ROOT / "benchmark_cross_tissue" / "README.md").read_text(encoding="utf-8")
    assert "pathology annotation polygons" in text
    assert "Allen CCF anatomical" in text
    assert "cell-type predictions" in text.lower()
