"""Canonical datasets. Real deployments register versioned public reference data here;
this scaffold ships a deterministic synthetic generator for tests and tutorials."""

from __future__ import annotations

from .phenomenology import (
    PHENOMENOLOGY_SCHEMA_VERSION,
    ConditionSpec,
    ObservationCondition,
    PhenomenonSpec,
    ScenarioManifest,
    SpatialPhenomenon,
    default_scenario_manifest,
    make_phenomenology_scenario,
    make_phenomenology_suite,
)
from .real import DatasetEntry, get_dataset, list_datasets, registry_summary
from .scale_contract import (
    SCALE_CONTRACTS,
    ScaleContract,
    registry_scale_table,
    scale_contract_for_assay,
)
from .digital_twin import (
    DIGITAL_TWIN_SCHEMA_VERSION,
    TWIN_MATCH_FEATURES,
    DigitalTwinResult,
    FeatureMatchReport,
    make_digital_twin,
)
from .synthetic import (
    make_developmental_gradient,
    make_mixture_synthetic,
    make_synthetic,
    make_tumor_microenvironment,
)
from .vendor import write_visium_fixture, write_xenium_fixture

__all__ = [
    "PHENOMENOLOGY_SCHEMA_VERSION",
    "ConditionSpec",
    "ObservationCondition",
    "PhenomenonSpec",
    "ScenarioManifest",
    "SpatialPhenomenon",
    "default_scenario_manifest",
    "make_phenomenology_scenario",
    "make_phenomenology_suite",
    "DatasetEntry",
    "get_dataset",
    "list_datasets",
    "registry_summary",
    "SCALE_CONTRACTS",
    "ScaleContract",
    "registry_scale_table",
    "scale_contract_for_assay",
    "DIGITAL_TWIN_SCHEMA_VERSION",
    "TWIN_MATCH_FEATURES",
    "DigitalTwinResult",
    "FeatureMatchReport",
    "make_digital_twin",
    "make_developmental_gradient",
    "make_mixture_synthetic",
    "make_synthetic",
    "make_tumor_microenvironment",
    "write_visium_fixture",
    "write_xenium_fixture",
]

from .synthetic import make_scalable_synthetic as make_scalable_synthetic

__all__.append("make_scalable_synthetic")
