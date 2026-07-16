"""Python-to-R adapter for the official Bioconductor BayesSpace package."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ._sota_common import make_adata

_HERE = Path(__file__).resolve().parent
_R_SCRIPT = _HERE.parent / "run_bayesspace.R"
_DEFAULT_R_LIB = "/workspace/histoweave_work/r_libs"


def run(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    seed: int,
    n_domains: int,
    array_coords: np.ndarray | None = None,
    nrep: int = 10000,
) -> np.ndarray:
    """Run BayesSpace spatialPreprocess + spatialCluster at known q."""
    if array_coords is None:
        raise ValueError("BayesSpace requires Visium array_row/array_col coordinates")
    if not _R_SCRIPT.exists():
        raise FileNotFoundError(f"BayesSpace R driver missing at {_R_SCRIPT}")

    adata = make_adata(
        X_counts,
        spatial,
        gene_names,
        n_genes=2000,
        array_coords=array_coords,
    )
    r_lib = os.environ.get("HISTOWEAVE_R_LIB", _DEFAULT_R_LIB)
    tmp = Path(tempfile.mkdtemp(prefix="bayesspace_"))
    try:
        in_h5 = tmp / "in.h5ad"
        out_csv = tmp / "labels.csv"
        adata.write_h5ad(in_h5)
        env = os.environ.copy()
        env["R_LIBS_USER"] = r_lib
        proc = subprocess.run(
            [
                "Rscript",
                "--vanilla",
                str(_R_SCRIPT),
                str(in_h5),
                str(out_csv),
                str(int(seed)),
                str(int(n_domains)),
                str(int(nrep)),
                r_lib,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"BayesSpace R driver failed (exit {proc.returncode})\n"
                f"stdout: {proc.stdout[-2000:]}\n"
                f"stderr: {proc.stderr[-2000:]}"
            )
        result = pd.read_csv(out_csv)
        if result.columns.tolist() != ["spot_id", "label"]:
            raise RuntimeError(f"unexpected BayesSpace output columns: {result.columns.tolist()}")
        if result["spot_id"].astype(str).tolist() != adata.obs_names.tolist():
            raise RuntimeError("BayesSpace output spot order does not match input")
        return result["label"].to_numpy(dtype=int)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
