"""HistoWeave — orchestration & evaluation for reproducible spatial transcriptomics.

HistoWeave is *not* a new method zoo. It quantifies and reduces
**method × spatial-context selection uncertainty**: a unified data substrate,
containerized pipelines, typed plugins over existing R/Python methods, task-bound
benchmarks, and a recommendation engine that reports when strong defaults beat
personalisation.

Quick start
-----------
>>> import histoweave as ts
>>> data = ts.datasets.make_synthetic(seed=0)      # tiny canonical dataset
>>> result = ts.run_pipeline(data)                 # ingest→QC→norm→domains→annotate
>>> ts.build_report(result, "report.html")         # self-contained HTML report
"""

from __future__ import annotations

__version__ = "0.1.0"
__version_info__ = (0, 1, 0, "final", 0)

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
