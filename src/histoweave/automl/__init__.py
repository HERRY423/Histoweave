"""Spatial AutoML compiler — landscape recommendation + multi-method execution.

Combines the natural-language compiler (:mod:`histoweave.compiler`) with the
landscape recommender (:class:`~histoweave.benchmark.recommend.MethodRecommender`)
into an automated loop:

1. Extract target-free dataset features.
2. Retrieve nearest reference datasets from a knowledge base.
3. Run the recommended top-*k* methods on the user sample.
4. Compare results with spatial-coherence / consensus proxies.
5. Rank methods on a Pareto front and emit a full HTML report.
"""

from __future__ import annotations

from .compiler import (
    AUTOML_SCHEMA_VERSION,
    AutoMLResult,
    MethodRunResult,
    ParetoPoint,
    compute_pareto_front,
    run_spatial_automl,
    write_automl_artifacts,
)
from .report import build_automl_report

__all__ = [
    "AUTOML_SCHEMA_VERSION",
    "AutoMLResult",
    "MethodRunResult",
    "ParetoPoint",
    "build_automl_report",
    "compute_pareto_front",
    "run_spatial_automl",
    "write_automl_artifacts",
]
