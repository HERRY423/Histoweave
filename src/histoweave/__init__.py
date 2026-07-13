"""HistoWeave — an open-source orchestration & evaluation platform for spatial transcriptomics.

HistoWeave is *not* a new method zoo. It is the connective tissue the field lacks: a
unified data substrate, scalable/reproducible pipelines, a plugin interface that wraps
existing R & Python methods behind stable APIs, and a continuous benchmarking harness
that turns method proliferation into guided method selection.

Quick start
-----------
>>> import histoweave as ts
>>> data = ts.datasets.make_synthetic(seed=0)      # tiny canonical dataset
>>> result = ts.run_pipeline(data)                 # ingest→QC→norm→domains→annotate
>>> ts.build_report(result, "report.html")         # self-contained HTML report
"""

from __future__ import annotations

__version__ = "0.1.0b1"
__version_info__ = (0, 1, 0, "beta", 1)

from . import datasets
from .data import Provenance, SpatialTable
from .plugins import MethodCategory, MethodMaturity, get_method, list_methods, register
from .report import build_report
from .workflow import (
    PipelineExecutionError,
    PipelineStep,
    PipelineStepError,
    RunManifest,
    default_pipeline,
    run_pipeline,
)

__all__ = [
    "__version__",
    "__version_info__",
    "SpatialTable",
    "Provenance",
    "MethodCategory",
    "MethodMaturity",
    "register",
    "get_method",
    "list_methods",
    "PipelineStep",
    "PipelineStepError",
    "PipelineExecutionError",
    "RunManifest",
    "run_pipeline",
    "default_pipeline",
    "build_report",
    "datasets",
]
