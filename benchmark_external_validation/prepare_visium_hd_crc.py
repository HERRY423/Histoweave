"""Prepare a Visium HD Human Colorectal Cancer bundle with pathologist ground truth.

Ground truth comes ONLY from the pathologist annotation released on Zenodo
(Kiessling et al., SpaceHack2023, CC0), which labels each 16 µm Visium HD bin
with one of: Neoplasm, Non-neoplastic Epithelium, Connective Tissue, Smooth
Muscle (and a few rarer labels). These are histological region labels — never
cell-type predictions — so they satisfy HistoWeave's strict spatial-domain
ground-truth policy.

Inputs (downloaded automatically unless --no-download is set):
  * 10x Visium HD CRC binned count matrix (16 µm), from the 10x dataset page.
  * Zenodo pathologist annotation CSV (16um_squares_annotation.csv).

The script joins the annotation to the matrix by barcode, attaches the bin
x/y as ``obsm['spatial']``, sets ``obs['domain_truth']``, builds a raw
``counts`` layer, QC-filters, HVG-subsets, and writes a checksummed ``.h5ad``
bundle plus a ``.json`` receipt.

Source / citation
-----------------
10x Genomics, "Visium HD Spatial Gene Expression Library, Human Colorectal
Cancer (FFPE)" — https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc
Oliveira et al., Nature Genetics 2025 — https://doi.org/10.1038/s41588-025-02193-3
Pathologist annotation: Kiessling, El-Heliebi, Ishaque — Zenodo record 11077886 (CC0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc"
ANNOTATION_URL = "https://zenodo.org/records/11077886/files/16um_squares_annotation.csv"
# The 10x Visium HD CRC release ships a tar.gz of the binned outputs. The
# 16um filtered feature-barcode matrix lives under
# ``binned_outputs/square_016um/filtered_feature_bc_matrix.h5`` once extracted.
# 10x exposes the Space Ranger output tarball via the dataset page's
# "Download" links; the canonical CF URL is:
MATRIX_TAR_URL = (
    "https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/"
    "Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz"
)

_LOGGER = logging.getLogger(__name__)

# Labels considered invalid as spatial-domain truth.
INVALID_LABELS = {"", "nan", "none", "unknown", "unannotated", "na", "NA"}
# Default cap for tractability (matches the 7x15 N_MAX policy).
DEFAULT_MAX_BINS = 15_000
DEFAULT_N_HVG = 2000


def _log(message: object) -> None:
    _LOGGER.info("%s", message)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"downloading {url} -> {dest}")
    # Use curl (with a browser-like User-Agent) because 10x's CloudFront
    # returns 403 to urllib's default Python User-Agent.
    import subprocess

    subprocess.run(
        ["curl", "-sS", "-L", "-o", str(dest), url],
        check=True,
    )
    return dest


def _extract_tar(tar_path: Path, dest: Path) -> Path:
    """Extract the 10x binned-outputs tarball and return the 16um matrix dir."""
    import tarfile

    if not (dest / "binned_outputs").exists():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest)
    # Layout: <dest>/binned_outputs/square_016um/filtered_feature_bc_matrix.h5
    candidates = list(dest.rglob("square_016um/filtered_feature_bc_matrix.h5"))
    if not candidates:
        # Some releases nest under a top-level dir; search more broadly.
        candidates = list(dest.rglob("filtered_feature_bc_matrix.h5"))
        candidates = [p for p in candidates if "square_016um" in str(p) or "016um" in str(p)]
    if not candidates:
        raise FileNotFoundError(
            f"no 16um filtered_feature_bc_matrix.h5 found under {dest}; check the tarball layout"
        )
    return candidates[0].parent


def _load_matrix(matrix_dir: Path):
    """Read the 10x filtered feature-barcode matrix (h5) as AnnData."""
    import scanpy as sc

    h5 = matrix_dir / "filtered_feature_bc_matrix.h5"
    a = sc.read_10x_h5(str(h5))
    a.var_names_make_unique()
    return a


def _load_spatial_coords(matrix_dir: Path, barcodes: pd.Index) -> pd.DataFrame:
    """Load Visium HD bin x/y coords from tissue_positions.parquet/h5.

    Visium HD Space Ranger v3+ writes ``tissue_positions.parquet`` (or
    ``tissue_positions.h5``) under the matrix folder. Columns:
    barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres.
    We use the full-res pixel coords as the spatial embedding (same convention
    as the DLPFC preparer).
    """
    # Visium HD stores tissue_positions under a ``spatial/`` subdirectory of
    # the bin-size folder (e.g. square_016um/spatial/tissue_positions.parquet),
    # not next to the h5 matrix. Search both locations.
    search_dirs = [matrix_dir, matrix_dir / "spatial"]
    for directory in search_dirs:
        for name in ("tissue_positions.parquet", "tissue_positions.h5"):
            p = directory / name
            if p.exists():
                if p.suffix == ".parquet":
                    pos = pd.read_parquet(p)
                else:
                    pos = pd.read_hdf(p, key="data")
                pos = pos.drop_duplicates(subset=pos.columns[0])
                pos = pos.set_index(pos.columns[0])
                pos.index = pos.index.astype(str)
                return pos.reindex(barcodes.astype(str))
    raise FileNotFoundError(
        f"no tissue_positions.parquet/h5 under {matrix_dir} or {matrix_dir / 'spatial'}; "
        "Visium HD coordinate file not found"
    )


def _load_annotation(csv_path: Path) -> pd.DataFrame:
    """Load the Zenodo pathologist annotation CSV.

    The Zenodo file (record 11077886) is a headerless tab-separated file with
    two columns: barcode (e.g. ``s_016um_00186_00418-1``) and pathology label.
    Some versions may carry a header; detect and handle both.
    """
    # Peek at the first line to decide whether a header is present.
    with csv_path.open("r") as f:
        first_line = f.readline().strip()
    parts = first_line.split("\t") if "\t" in first_line else first_line.split(",")
    has_header = not parts[0].startswith("s_016um_")
    sep = "\t" if "\t" in first_line else ","
    ann = pd.read_csv(csv_path, sep=sep, header=0 if has_header else None)
    if not has_header:
        ann.columns = ["barcode", "pathology_label"]
    else:
        ann.columns = [c.strip() for c in ann.columns]
        barcode_col = next(
            (c for c in ann.columns if c.lower() in {"barcode", "barcodes"}), ann.columns[0]
        )
        ann = ann.rename(columns={barcode_col: "barcode"})
        label_col = next(c for c in ann.columns if c != "barcode")
        ann = ann.rename(columns={label_col: "pathology_label"})
    ann["barcode"] = ann["barcode"].astype(str)
    return ann[["barcode", "pathology_label"]]


def _stratified_indices(labels: pd.Series, limit: int, seed: int) -> np.ndarray:
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups = labels.groupby(labels, observed=True).indices
    selected: list[np.ndarray] = []
    for indices in groups.values():
        quota = max(1, round(len(indices) / len(labels) * limit))
        selected.append(rng.choice(indices, min(quota, len(indices)), replace=False))
    merged = np.unique(np.concatenate(selected))
    if len(merged) > limit:
        merged = rng.choice(merged, limit, replace=False)
    elif len(merged) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), merged, assume_unique=False)
        merged = np.concatenate([merged, rng.choice(remaining, limit - len(merged), False)])
    return np.sort(merged)


def build(args: argparse.Namespace) -> dict[str, object]:
    import scanpy as sc

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # --- 1. counts matrix ---
    if args.matrix_dir:
        matrix_dir = Path(args.matrix_dir)
    else:
        tar = _download(MATRIX_TAR_URL, cache / "binned_outputs.tar.gz")
        matrix_dir = _extract_tar(tar, cache / "extracted")

    adata = _load_matrix(matrix_dir)
    n_original = int(adata.n_obs)
    _log(f"matrix: {adata.n_obs} bins x {adata.n_vars} genes")

    # --- 2. spatial coords ---
    pos = _load_spatial_coords(matrix_dir, adata.obs_names)
    coord_ok = pos.notna().all(axis=1).to_numpy()
    adata = adata[coord_ok].copy()
    pos = pos.loc[adata.obs_names]
    # pxl_row / pxl_col (full-res) — match DLPFC convention.
    pxl_cols = [c for c in pos.columns if "pxl" in c.lower()]
    if len(pxl_cols) >= 2:
        adata.obsm["spatial"] = pos[pxl_cols[:2]].to_numpy(dtype=float)
    else:
        # Fall back to array_row / array_col if pixel coords absent.
        adata.obsm["spatial"] = pos[["array_row", "array_col"]].to_numpy(dtype=float)

    # --- 3. pathologist annotation ---
    if args.annotation:
        ann_path = Path(args.annotation)
    else:
        ann_path = _download(ANNOTATION_URL, cache / "16um_squares_annotation.csv")
    ann = _load_annotation(ann_path)
    _log(f"annotation: {len(ann)} labelled bins, labels={sorted(ann['pathology_label'].unique())}")

    adata.obs["barcode"] = adata.obs_names.astype(str)
    adata.obs = adata.obs.merge(ann, left_on="barcode", right_on="barcode", how="left")
    adata.obs["domain_truth_raw"] = adata.obs["pathology_label"].astype("string")

    # --- 4. filter to valid, labelled bins ---
    raw = adata.obs["domain_truth_raw"].fillna("")
    valid = ~raw.str.strip().str.casefold().isin(INVALID_LABELS)
    adata = adata[valid.to_numpy()].copy()
    if adata.n_obs == 0:
        raise ValueError("no bins retained a valid pathologist label")

    # --- 5. stratified subsample for tractability ---
    labels = adata.obs["domain_truth_raw"].astype(str).reset_index(drop=True)
    idx = _stratified_indices(labels, args.max_bins, args.seed)
    adata = adata[idx].copy()

    adata.obs["domain_truth"] = pd.Categorical(adata.obs["domain_truth_raw"].astype(str).to_numpy())
    adata.obs["truth_source"] = "pathologist_annotation_zenodo_11077886"
    adata.obs_names = adata.obs["barcode"].astype(str).to_numpy()
    del adata.obs["barcode"], adata.obs["pathology_label"], adata.obs["domain_truth_raw"]

    # --- 6. counts layer + QC + HVG ---
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    adata.obs["n_counts"] = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    adata = adata[adata.obs["n_counts"] > 0].copy()
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(args.n_hvg, adata.n_vars - 1))
    adata = adata[:, adata.var["highly_variable"]].copy()

    adata.uns.update(
        {
            "schema_version": "histoweave.visium_hd.crc.bundle.v1",
            "source": "10x Visium HD Human Colorectal Cancer (FFPE)",
            "source_url": SOURCE_URL,
            "annotation_source": "Zenodo record 11077886 (SpaceHack2023, CC0)",
            "annotation_url": ANNOTATION_URL,
            "license": "10x Genomics EULA (data) + CC0 (annotation)",
            "n_original": n_original,
            "ground_truth_definition": (
                "pathologist histological region labels; unannotated bins excluded"
            ),
            "bin_size_um": 16,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "visium_hd_crc",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_domains": int(adata.obs["domain_truth"].nunique()),
        "domains": sorted(adata.obs["domain_truth"].astype(str).unique().tolist()),
        "truth_source": "pathologist_annotation_zenodo_11077886",
        "bin_size_um": 16,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    _log(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        help="Pre-extracted 16um filtered_feature_bc_matrix directory (skips download)",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        help="Zenodo 16um_squares_annotation.csv (skips download)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            os.environ.get("HISTOWEAVE_EXT_CACHE", root / "datasets_cache" / "visium_hd_crc")
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets_cache" / "visium_hd_crc" / "visium_hd_crc.h5ad",
    )
    parser.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS)
    parser.add_argument("--n-hvg", type=int, default=DEFAULT_N_HVG)
    parser.add_argument("--seed", type=int, default=42)
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
