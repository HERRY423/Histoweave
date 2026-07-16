"""Anatomical-truth safeguards for Allen mouse-brain preparation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_allen_mouse_brain",
    ROOT / "benchmark_cross_tissue" / "prepare_allen_mouse_brain.py",
)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def _args(**overrides):
    values = {
        "region_column": None,
        "allow_cell_class_fallback": False,
        "label_column": "subclass",
        "mapping": ROOT / "src/histoweave/datasets/domain_mappings.json",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_prefers_coarse_allen_ccf_anatomy():
    data = SimpleNamespace(
        obs=pd.DataFrame(
            {
                "parcellation_division": ["Isocortex", "Hippocampal formation"],
                "subclass": ["IT", "Oligo"],
            }
        )
    )
    labels, source, column = PREPARE.resolve_truth(data, _args())
    assert labels.tolist() == ["Isocortex", "Hippocampal formation"]
    assert source == "allen_ccf_anatomical"
    assert column == "parcellation_division"


def test_rejects_cell_class_as_primary_truth():
    data = SimpleNamespace(obs=pd.DataFrame({"subclass": ["IT", "Oligo"]}))
    with pytest.raises(ValueError, match="not accepted as primary"):
        PREPARE.resolve_truth(data, _args())


def test_cell_class_fallback_is_explicitly_tagged_sensitivity_only():
    data = SimpleNamespace(obs=pd.DataFrame({"subclass": ["IT", "Oligo"]}))
    _, source, column = PREPARE.resolve_truth(data, _args(allow_cell_class_fallback=True))
    assert source == "cell_class_sensitivity_fallback"
    assert column == "subclass"
