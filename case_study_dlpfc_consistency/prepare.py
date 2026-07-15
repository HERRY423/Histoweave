"""Prepare one DLPFC slice (151673) for the cross-method consistency case study.

Reuses the proven ingestion logic from ``5x10_dlpfc_benchmark/prepare_dlpfc.py``
but targets a single slice (151673, the cleanest 6-layer + WM section) and keeps
both raw counts (``layers['counts']``) and log-normalized ``X`` so downstream
methods can pick what they need. The manual layer annotation (spatialLIBD,
Maynard et al. 2021) is stored in ``obs['domain_truth']``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SLICE_ID = "151673"
N_HVG = 2000


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def build_anndata(n_hvg: int = N_HVG):
    h5 = DATA_DIR / f"{SLICE_ID}.h5"
    lab = DATA_DIR / f"{SLICE_ID}_layers.csv"
    pos = DATA_DIR / f"{SLICE_ID}_positions.txt"

    a = sc.read_10x_h5(str(h5))
    a.var_names_make_unique()

    labels = pd.read_csv(lab)
    labels["spot_name"] = labels["spot_name"].astype(str)
    lab_map = (
        labels.dropna(subset=["layer"])
        .groupby("spot_name")["layer"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    positions = pd.read_csv(
        pos,
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
    )
    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.drop_duplicates(subset="barcode")
    pos_map = positions.set_index("barcode")[["pxl_row", "pxl_col"]]

    a.obs["domain_truth"] = a.obs_names.map(lab_map)
    n_total = a.n_obs
    a = a[a.obs["domain_truth"].notna()].copy()
    a = a[~a.obs["domain_truth"].isin(["NA", "nan", ""])].copy()

    coords = pos_map.reindex(a.obs_names)
    a = a[coords.notna().all(axis=1).to_numpy()].copy()
    coords = pos_map.reindex(a.obs_names)
    a.obsm["spatial"] = coords[["pxl_row", "pxl_col"]].to_numpy(dtype=float)
    n_annotated = a.n_obs

    sc.pp.filter_genes(a, min_cells=3)
    a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    a = a[a.obs["n_counts"] > 0].copy()

    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    sc.pp.highly_variable_genes(a, n_top_genes=min(n_hvg, a.n_vars - 1))
    a = a[:, a.var["highly_variable"]].copy()

    a.uns["slice_id"] = SLICE_ID
    a.uns["n_total_spots"] = int(n_total)
    a.uns["n_annotated_spots"] = int(n_annotated)
    a.uns["n_domains_truth"] = int(pd.Series(a.obs["domain_truth"]).nunique())
    return a


if __name__ == "__main__":
    a = build_anndata()
    out = DATA_DIR / f"{SLICE_ID}.h5ad"
    a.write_h5ad(out)
    layers = sorted(pd.Series(a.obs["domain_truth"]).unique().tolist())
    _log(
        f"[{SLICE_ID}] spots={a.n_obs} hvg={a.n_vars} "
        f"domains={a.uns['n_domains_truth']} layers={layers}"
    )
    _log(f"wrote {out}")
