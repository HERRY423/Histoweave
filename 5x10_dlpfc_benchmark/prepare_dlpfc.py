"""Download 5 DLPFC slices + spatialLIBD manual-layer labels, build SpatialTables.

The HistoWeave real-data registry declares ground_truth={"domain_truth": "spatialLIBD_layer"}
for the DLPFC entries, but (a) the .h5 files on S3 contain ONLY the count matrix (no layer
labels) and (b) the registry sha256 fields are placeholders, so DatasetEntry.load() fails a
checksum. We therefore fetch counts + labels directly, join by barcode, and construct
SpatialTable objects with obs['domain_truth'] so the landscape harness can score ARI.

Labels: Maynard et al. 2021 (spatialLIBD) manual layer annotation, from the LieberInstitute
HumanPilot repo (Analysis/Layer_Guesses).
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

BASE_DIR = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("HISTOWEAVE_DLPFC_DATA", BASE_DIR / "data"))
CACHE.mkdir(parents=True, exist_ok=True)

# 5 slices spanning a difficulty gradient (layer completeness / clarity).
# All are in the HumanPilot "First_Round" label folder.
SLICES = {
    "151673": "First_Round/spatialLIBD_layerGuesses_2019-12-19 17_14_24_151673.csv",
    "151674": "First_Round/spatialLIBD_layerGuesses_2019-12-19 19_42_45_151674.csv",
    "151507": "First_Round/spatialLIBD_layerGuesses_2019-12-19 15_55_14_151507.csv",
    "151669": "First_Round/spatialLIBD_layerGuesses_2019-12-19 16_10_07_151669.csv",
    "151670": "First_Round/spatialLIBD_layerGuesses_2019-12-19 19_09_31_151670.csv",
}

H5_URL = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{s}_filtered_feature_bc_matrix.h5"
LABEL_BASE = (
    "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/Analysis/Layer_Guesses/"
)
POS_URL = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{s}/tissue_positions_list.txt"
N_HVG = 2000


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


def fetch_slice(slice_id: str, label_rel: str) -> tuple[Path, Path]:
    h5 = _download(H5_URL.format(s=slice_id), CACHE / f"{slice_id}.h5")
    # URL-encode the spaces/colons in the label filename.
    parts = label_rel.split("/", 1)
    url = LABEL_BASE + parts[0] + "/" + urllib.parse.quote(parts[1])
    lab = _download(url, CACHE / f"{slice_id}_layers.csv")
    pos = _download(POS_URL.format(s=slice_id), CACHE / f"{slice_id}_positions.txt")
    return h5, lab, pos


def build_anndata(slice_id: str, h5: Path, lab: Path, pos: Path, n_hvg: int = N_HVG):
    """Load counts, join manual layers + Visium spatial coords, QC + normalize + HVG."""
    a = sc.read_10x_h5(str(h5))
    a.var_names_make_unique()

    labels = pd.read_csv(lab)
    labels["spot_name"] = labels["spot_name"].astype(str)
    # Some label files contain duplicate barcodes (multiple annotator rounds).
    # Collapse to one label per barcode by majority vote (first on ties).
    lab_map = (
        labels.dropna(subset=["layer"])
        .groupby("spot_name")["layer"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    # Standard Visium tissue_positions_list.txt (no header):
    # barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    positions = pd.read_csv(
        pos,
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
    )
    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.drop_duplicates(subset="barcode")
    pos_map = positions.set_index("barcode")[["pxl_row", "pxl_col"]]

    # Join by barcode; keep only annotated spots with a real layer label.
    a.obs["domain_truth"] = a.obs_names.map(lab_map)
    n_total = a.n_obs
    a = a[a.obs["domain_truth"].notna()].copy()
    a = a[~a.obs["domain_truth"].isin(["NA", "nan", ""])].copy()

    # Attach real spatial pixel coordinates for the neighbourhood term.
    coords = pos_map.reindex(a.obs_names)
    a = a[coords.notna().all(axis=1).to_numpy()].copy()
    coords = pos_map.reindex(a.obs_names)
    a.obsm["spatial"] = coords[["pxl_row", "pxl_col"]].to_numpy(dtype=float)
    n_annotated = a.n_obs

    # --- QC ---
    sc.pp.filter_genes(a, min_cells=3)
    a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    a = a[a.obs["n_counts"] > 0].copy()

    # --- normalize ---
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    # --- HVG subset (consistent across slices) ---
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_hvg, a.n_vars - 1))
    a = a[:, a.var["highly_variable"]].copy()

    a.uns["slice_id"] = slice_id
    a.uns["n_total_spots"] = int(n_total)
    a.uns["n_annotated_spots"] = int(n_annotated)
    a.uns["n_domains_truth"] = int(pd.Series(a.obs["domain_truth"]).nunique())
    return a


if __name__ == "__main__":
    summary = []
    for sid, lab_rel in SLICES.items():
        h5, lab, pos = fetch_slice(sid, lab_rel)
        a = build_anndata(sid, h5, lab, pos)
        out = CACHE / f"{sid}.h5ad"
        a.write_h5ad(out)
        summary.append(
            dict(
                slice=sid,
                n_spots=a.n_obs,
                n_hvg=a.n_vars,
                n_domains=a.uns["n_domains_truth"],
                annot_frac=round(a.uns["n_annotated_spots"] / a.uns["n_total_spots"], 3),
                layers=sorted(pd.Series(a.obs["domain_truth"]).unique().tolist()),
            )
        )
        print(
            f"[{sid}] spots={a.n_obs} hvg={a.n_vars} domains={a.uns['n_domains_truth']} "
            f"annot={summary[-1]['annot_frac']}"
        )
    pd.DataFrame(summary).to_csv(CACHE / "slice_summary.csv", index=False)
    print("\nWrote slice_summary.csv")
