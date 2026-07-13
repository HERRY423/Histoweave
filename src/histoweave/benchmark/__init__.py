"""Benchmarking & evaluation layer."""

from __future__ import annotations

from .features import (
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_dataframe,
    feature_vector,
)
from .figure3 import (
    FIGURE3_DATASETS,
    FIGURE3_METHODS,
    Figure3Result,
    run_figure3_experiment,
)
from .harness import (
    BenchmarkResult,
    Task,
    deconvolution_task,
    domain_detection_task,
    get_task,
    run_benchmark,
    svg_task,
)
from .landscape import (
    LandscapeResult,
    MultiLandscapeResult,
    landscape_svg,
    run_landscape,
    run_multi_landscape,
    run_task_landscape,
)
from .recommend import MethodRecommender, MethodScore, Recommendation

__all__ = [
    # harness
    "Task", "BenchmarkResult", "run_benchmark",
    "domain_detection_task", "deconvolution_task", "svg_task", "get_task",
    # Figure 3 experiment
    "FIGURE3_DATASETS", "FIGURE3_METHODS", "Figure3Result",
    "run_figure3_experiment",
    # landscape
    "LandscapeResult", "MultiLandscapeResult",
    "run_landscape", "run_task_landscape", "run_multi_landscape",
    "landscape_svg",
    # recommend
    "MethodRecommender", "MethodScore", "Recommendation",
    # features
    "RECOMMENDATION_FEATURE_ORDER",
    "extract_features", "feature_vector", "feature_dataframe",
]
