"""Compatibility shim.

The failure-boundary engine has moved into the installed package at
:mod:`histoweave.benchmark.failure_boundary` so it is importable everywhere and
reachable from the ``histoweave benchmark-boundary`` CLI. This module re-exports
the public API so scripts that did ``from failure_boundary import ...`` keep
working from a checkout.
"""

from __future__ import annotations

from histoweave.benchmark.failure_boundary import (  # noqa: F401
    DEFAULT_TAU,
    Boundary,
    BoundaryStudyResult,
    SweepAxis,
    SweepPoint,
    build_axes,
    detect_boundary,
    make_svg_task_fixed,
    probe_runnable,
    run_boundary_study,
    run_sweep,
    write_cards_md,
    write_study_outputs,
)

__all__ = [
    "DEFAULT_TAU",
    "Boundary",
    "BoundaryStudyResult",
    "SweepAxis",
    "SweepPoint",
    "build_axes",
    "detect_boundary",
    "make_svg_task_fixed",
    "probe_runnable",
    "run_boundary_study",
    "run_sweep",
    "write_cards_md",
    "write_study_outputs",
]
