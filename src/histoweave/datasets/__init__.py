"""Canonical datasets. Real deployments register versioned public reference data here;
this scaffold ships a deterministic synthetic generator for tests and tutorials."""

from __future__ import annotations

from .real import DatasetEntry, get_dataset, list_datasets
from .synthetic import (
    make_developmental_gradient,
    make_mixture_synthetic,
    make_synthetic,
    make_tumor_microenvironment,
)
from .vendor import write_visium_fixture, write_xenium_fixture

__all__ = [
    "DatasetEntry",
    "get_dataset",
    "list_datasets",
    "make_developmental_gradient",
    "make_mixture_synthetic",
    "make_synthetic",
    "make_tumor_microenvironment",
    "write_visium_fixture",
    "write_xenium_fixture",
]
