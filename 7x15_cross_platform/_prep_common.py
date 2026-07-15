"""Shared helpers for cross-platform dataset preparation (Xenium / MERFISH / Slide-seqV2).

Each ``prep_*.py`` downloads one real public spatial dataset via squidpy, subsamples to
``MAX_CELLS`` with a fixed seed, derives a proxy ``domain_truth`` label from the platform's
own cell-type / cluster annotation, builds a ``counts`` layer, HVG-subsets, and writes a
cached ``.h5ad``. The 7x15 experiment consumes these caches alongside the 5 DLPFC slices.

Proxy-label caveat: these platforms have no expert *spatial-domain* ground truth like the
spatialLIBD manual layers. We use each dataset's published cell-type / transcriptomic
cluster label as a domain proxy, so cross-platform ARI reflects recovery of the annotated
cell-type structure, not histological domains. This is documented in the report.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


# The domain methods' spatial term uses a brute-force n-x-n KNN (histoweave._math.
# knn_indices), which is O(n^2) memory. To keep the exact same code path as the DLPFC
# slices (~3.6k spots each) and stay well within memory, cross-platform datasets are
# subsampled to MAX_CELLS, matching the Visium slice scale for fair cross-platform ARI.
MAX_CELLS = 6_000
SUBSAMPLE_SEED = 0
N_HVG = 2000

# Local scratch cache (random-access h5 writes must NOT go to S3-backed mounts),
# then mirrored to the shared cache for the experiment step.
LOCAL_CACHE = Path(os.environ.get("HISTOWEAVE_XPLAT_LOCAL", "/workspace/xplat_cache"))
SHARED_CACHE = Path(
    os.environ.get("HISTOWEAVE_XPLAT_SHARED", "/mnt/shared-workspace/shared/xplat_cache")
)
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
SHARED_CACHE.mkdir(parents=True, exist_ok=True)


def _dense(X) -> np.ndarray:
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def build_counts_layer(a: sc.AnnData) -> sc.AnnData:
    """Populate a['counts'] with raw-like counts.

    - If X is (near-)integer count-like (>=95% integer OR max>50 with high integer
      fraction), treat X directly as counts.
    - Otherwise X is already log-normalized (e.g. Slide-seqV2), so recover pseudo-counts
      via expm1 so the harness's log1p_cp10k re-normalization is well-behaved.
    """
    X = _dense(a.X).astype(float)
    int_frac = float(np.mean(np.isclose(X, np.round(X))))
    if int_frac >= 0.95 or (X.max() > 50 and int_frac >= 0.55):
        counts = np.rint(np.clip(X, 0, None))
    else:
        counts = np.clip(np.expm1(X), 0, None)
    a.layers["counts"] = counts
    a.uns["counts_source"] = "X_as_counts" if int_frac >= 0.55 else "expm1_X_pseudocounts"
    return a


def finalize(
    a: sc.AnnData,
    *,
    dataset_id: str,
    platform: str,
    label_col: str,
    drop_labels: tuple[str, ...] = (),
) -> sc.AnnData:
    """Common finalize: subsample, set domain_truth, QC, HVG, cache."""
    a.var_names_make_unique()

    # --- proxy domain label ---
    if label_col not in a.obs.columns:
        raise KeyError(
            f"{dataset_id}: label column {label_col!r} missing; have {list(a.obs.columns)}"
        )
    lab = a.obs[label_col].astype(str)
    keep = ~lab.isin(set(drop_labels) | {"nan", "NA", ""})
    a = a[keep.to_numpy()].copy()
    a.obs["domain_truth"] = a.obs[label_col].astype(str).values

    # --- subsample to MAX_CELLS (stratified-agnostic random, fixed seed) ---
    if a.n_obs > MAX_CELLS:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        idx = np.sort(rng.choice(a.n_obs, size=MAX_CELLS, replace=False))
        a = a[idx].copy()

    # --- spatial coords ---
    if "spatial" not in a.obsm:
        raise KeyError(f"{dataset_id}: no obsm['spatial']")
    a.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)[:, :2]

    # --- counts layer (before any normalization overwrites X) ---
    a = build_counts_layer(a)

    # --- QC: drop empty genes / cells ---
    sc.pp.filter_genes(a, min_cells=3)
    a.obs["n_counts"] = np.asarray(a.layers["counts"].sum(axis=1)).ravel()
    a = a[a.obs["n_counts"] > 0].copy()

    # --- normalize a copy of X for HVG selection (counts layer preserved raw) ---
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(N_HVG, a.n_vars - 1))
    a = a[:, a.var["highly_variable"]].copy()

    a.uns["slice_id"] = dataset_id
    a.uns["platform"] = platform
    a.uns["label_col"] = label_col
    a.uns["n_domains_truth"] = int(pd.Series(a.obs["domain_truth"]).nunique())

    # --- write to local scratch, then mirror to shared ---
    local = LOCAL_CACHE / f"{dataset_id}.h5ad"
    a.write_h5ad(local)
    shutil.copy(local, SHARED_CACHE / f"{dataset_id}.h5ad")

    _log(
        f"[{dataset_id}] platform={platform} cells={a.n_obs} hvg={a.n_vars} "
        f"domains={a.uns['n_domains_truth']} counts_src={a.uns['counts_source']}"
    )
    return a
