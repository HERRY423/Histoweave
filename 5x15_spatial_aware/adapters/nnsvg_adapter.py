"""nnSVG (Weber 2023, Bioconductor) → top-N → PCA → KMeans pipeline.

nnSVG is R-only, so this adapter shells out to :file:`run_nnsvg.R` — a Bash
call that reads a temporary h5ad + spatial coord CSV, ranks genes on the R side
(nnSVG likelihood-ratio), and writes back a CSV with columns ``gene, rank``.

Requires:
  * ``R >= 4.4``
  * ``nnSVG`` installed to ``/workspace/histoweave_work/r_libs`` (see the
    ``scripts/install_nnsvg.R`` one-shot Bioconductor install).

Environment variable ``HISTOWEAVE_R_LIB`` overrides the R library path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from svg_domain_pipeline import cluster_from_svg_ranking  # noqa: E402

_DEFAULT_R_LIB = "/workspace/histoweave_work/r_libs"
_R_SCRIPT = _HERE.parent / "run_nnsvg.R"


def _to_csr(x):
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def _rank_by_nnsvg(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    n_candidates: int = 2000,
    n_threads: int = 4,
) -> list[str]:
    """Write a filtered h5ad to /tmp and call the R nnSVG driver."""
    if not _R_SCRIPT.exists():
        raise FileNotFoundError(f"run_nnsvg.R missing at {_R_SCRIPT}; task-2 adapters are broken")
    r_lib = os.environ.get("HISTOWEAVE_R_LIB", _DEFAULT_R_LIB)

    # dispersion-based prefilter to make nnSVG feasible on Visium slices
    X = _to_csr(X_counts).astype(float)
    Xdense = np.asarray(X.todense())
    var = Xdense.var(axis=0)
    keep_idx = np.argsort(-var)[:n_candidates]
    keep_idx = np.asarray([j for j in keep_idx if var[j] > 0], dtype=int)
    Xk = sparse.csr_matrix(Xdense[:, keep_idx])
    gnames = [str(gene_names[j]) for j in keep_idx]

    a = ad.AnnData(
        X=Xk,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(Xk.shape[0])]),
        var=pd.DataFrame(index=gnames),
        obsm={"spatial": np.asarray(spatial, dtype=float)},
    )

    tmp = Path(tempfile.mkdtemp(prefix="nnsvg_"))
    try:
        in_h5 = tmp / "in.h5ad"
        out_csv = tmp / "nnsvg_rank.csv"
        # Stage to /workspace to avoid random-access h5ad writes to S3 mounts
        a.write_h5ad(in_h5)
        cmd = [
            "Rscript",
            "--vanilla",
            str(_R_SCRIPT),
            str(in_h5),
            str(out_csv),
            str(n_threads),
            r_lib,
        ]
        env = os.environ.copy()
        env["R_LIBS_USER"] = r_lib
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(
                f"nnSVG R driver failed (exit {proc.returncode})\n"
                f"stdout: {proc.stdout[-2000:]}\n"
                f"stderr: {proc.stderr[-2000:]}"
            )
        df = pd.read_csv(out_csv)
        if "gene" not in df.columns:
            raise RuntimeError(f"nnSVG output missing 'gene' column: {df.columns.tolist()}")
        return df["gene"].astype(str).tolist()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    seed: int,
    n_domains: int,
    n_top: int = 500,
) -> np.ndarray:
    ranked = _rank_by_nnsvg(X_counts, spatial, gene_names)
    return cluster_from_svg_ranking(
        X_counts,
        ranked_genes=ranked,
        all_genes=list(gene_names),
        n_domains=n_domains,
        seed=seed,
        n_top=n_top,
    )
