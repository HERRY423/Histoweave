#!/usr/bin/env python3
"""Compatibility entry point for :mod:`run_scvi_smoke`.

scvi-tools 1.3.3 leaves ``model.history`` as ``None`` when Lightning logging is
disabled. The primary script intentionally disables external experiment logs;
this wrapper supplies an empty local history table so all scientific outputs
remain reproducible without changing the trained model or any QC calculation.
"""

from __future__ import annotations

import pandas as pd
import run_scvi_smoke as workflow

original_flatten_history = workflow.flatten_history


def flatten_history(history):
    if history is None:
        return pd.DataFrame({"record": pd.Series(dtype=int)})
    return original_flatten_history(history)


workflow.flatten_history = flatten_history


if __name__ == "__main__":
    raise SystemExit(workflow.main())
