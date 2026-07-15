"""Build the 12 checksummed DLPFC h5ad bundles used by the registry.

For each of the 12 spatialLIBD DLPFC slices (Maynard et al. 2021) this script:

  1. Downloads the filtered count matrix (``*_filtered_feature_bc_matrix.h5``) from
     the ``spatial-dlpfc`` S3 bucket.
  2. Downloads the manual layer annotation (``spatialLIBD_layerGuesses_*.csv``) from
     LieberInstitute/HumanPilot on GitHub.
  3. Downloads the Visium ``tissue_positions_list.txt`` from HumanPilot.
  4. Joins by barcode, majority-votes duplicate labels, filters spots without a
     label, and builds an ``AnnData`` with:

        * ``X``          — raw counts (sparse CSR),
        * ``obs['spatialLIBD_layer']``  — canonical layer label (baked in),
        * ``obs['domain_truth']``       — alias of ``spatialLIBD_layer`` for the
          benchmark harness,
        * ``obsm['spatial']``           — (pxl_col, pxl_row) coordinates,
        * ``uns['dlpfc_bundle_version']`` — schema version tag.

  5. Writes the artefact to ``<out>/dlpfc_<sid>.h5ad``, computes its SHA-256, and
     records ``{name, url, sha256, bytes, n_obs, n_vars, n_domains}`` in
     ``<out>/checksums.json``.

The script is idempotent: it skips slices whose on-disk SHA-256 matches the
recorded value.  It also emits a small ``registry_snippet.py`` fragment
containing the 12 DatasetEntry declarations with baked hashes and URLs, which
Task 1b copies into ``datasets/real.py``.

Usage
-----
    python scripts/build_dlpfc_bundles.py --out datasets_cache/dlpfc

Requires: ``scanpy``, ``anndata``, ``pandas``, ``numpy`` and ~250 MB of disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy import sparse

SLICES: dict[str, str] = {
    "151507": "First_Round/spatialLIBD_layerGuesses_2019-12-19 15_55_14_151507.csv",
    "151508": "First_Round/spatialLIBD_layerGuesses_2019-12-19 18_18_25_151508.csv",
    "151509": "Second_Round/spatialLIBD_layerGuesses_2019-12-30 14_51_11_151509.csv",
    "151510": "Second_Round/spatialLIBD_layerGuesses_2019-12-30 15_12_27_151510.csv",
    "151669": "First_Round/spatialLIBD_layerGuesses_2019-12-19 16_10_07_151669.csv",
    "151670": "First_Round/spatialLIBD_layerGuesses_2019-12-19 19_09_31_151670.csv",
    "151671": "Second_Round/spatialLIBD_layerGuesses_2019-12-30 15_29_00_151671.csv",
    "151672": "Second_Round/spatialLIBD_layerGuesses_2019-12-30 16_10_00_151672.csv",
    "151673": "First_Round/spatialLIBD_layerGuesses_2019-12-19 17_14_24_151673.csv",
    "151674": "First_Round/spatialLIBD_layerGuesses_2019-12-19 19_42_45_151674.csv",
    "151675": "Second_Round/spatialLIBD_layerGuesses_2019-12-30 16_59_28_151675.csv",
    "151676": "Second_Round/spatialLIBD_layerGuesses_2019-12-30 17_25_53_151676.csv",
}

H5_URL_TEMPLATE = (
    "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{sid}_filtered_feature_bc_matrix.h5"
)
LABEL_BASE = (
    "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/Analysis/Layer_Guesses/"
)
POS_URL_TEMPLATE = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{sid}/tissue_positions_list.txt"

SCHEMA_VERSION = "histoweave.dlpfc.bundle.v1"
LICENSE = "CC-BY 4.0"
PAPER_DOI = "10.1038/s41593-020-00787-0"


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _sha256(path: Path, chunk: int = 128 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            _log(f"    retry {attempt + 1}/{retries}: {exc}")
            time.sleep(2 * (attempt + 1))
    return dest


def _majority_layer(series: pd.Series) -> str:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return "NA"
    return counts.idxmax()


def build_bundle(sid: str, downloads: Path, out: Path, staging: Path | None = None) -> dict:
    """Assemble one DLPFC slice into a checksummed h5ad bundle.

    HDF5 (and therefore h5ad) requires random-access writes, which are not
    supported on the S3-backed ``/mnt/results/`` FUSE mount.  If ``out`` is
    below such a mount, the file is written to ``staging`` (default:
    ``/workspace/histoweave_work/staging``) first and then copied.
    """
    _log(f"[{sid}] downloading + assembling ...")
    h5_url = H5_URL_TEMPLATE.format(sid=sid)
    label_rel = SLICES[sid]
    subfolder, fname = label_rel.split("/", 1)
    label_url = LABEL_BASE + subfolder + "/" + urllib.parse.quote(fname)
    pos_url = POS_URL_TEMPLATE.format(sid=sid)

    h5_path = _download(h5_url, downloads / f"{sid}.h5")
    label_path = _download(label_url, downloads / f"{sid}_layers.csv")
    pos_path = _download(pos_url, downloads / f"{sid}_positions.txt")

    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    labels = pd.read_csv(label_path)
    labels["spot_name"] = labels["spot_name"].astype(str)
    lab_series = labels.dropna(subset=["layer"]).groupby("spot_name")["layer"].agg(_majority_layer)

    positions = pd.read_csv(
        pos_path,
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    )
    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.set_index("barcode")

    common = adata.obs_names.intersection(lab_series.index).intersection(positions.index)
    if len(common) == 0:
        raise RuntimeError(f"[{sid}] barcode intersection empty — schema drift?")
    adata = adata[common].copy()
    adata.obs["spatialLIBD_layer"] = pd.Categorical(lab_series.reindex(common))
    adata.obs["domain_truth"] = adata.obs["spatialLIBD_layer"]
    adata.obs["array_row"] = positions.reindex(common)["array_row"].values
    adata.obs["array_col"] = positions.reindex(common)["array_col"].values
    coords = positions.reindex(common)[["pxl_col_in_fullres", "pxl_row_in_fullres"]]
    adata.obsm["spatial"] = coords.to_numpy(dtype=float)

    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()

    adata.uns["dlpfc_bundle_version"] = SCHEMA_VERSION
    adata.uns["dlpfc_slice_id"] = sid
    adata.uns["dlpfc_source_urls"] = {
        "counts": h5_url,
        "labels": label_url,
        "positions": pos_url,
    }
    adata.uns["dlpfc_license"] = LICENSE
    adata.uns["dlpfc_paper_doi"] = PAPER_DOI

    out_path = out / f"dlpfc_{sid}.h5ad"
    # Write via /workspace staging when the destination is an S3-backed mount.
    stage_root = staging or Path("/workspace/histoweave_work/staging")
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_path = stage_root / f"dlpfc_{sid}.h5ad"
    adata.write_h5ad(stage_path, compression="gzip")
    import shutil as _shutil

    # copyfile (no metadata) — S3 FUSE forbids utime/chmod that copy2 does.
    _shutil.copyfile(stage_path, out_path)
    digest = _sha256(stage_path)
    nbytes = stage_path.stat().st_size

    n_layers = int(adata.obs["domain_truth"].nunique())
    return {
        "name": f"dlpfc_{sid}",
        "slice_id": sid,
        "path": f"datasets_cache/dlpfc/dlpfc_{sid}.h5ad",
        "url": f"local://datasets_cache/dlpfc/dlpfc_{sid}.h5ad",
        "sha256": digest,
        "bytes": nbytes,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_domains": n_layers,
        "layer_counts": {
            str(k): int(v) for k, v in adata.obs["domain_truth"].value_counts().items()
        },
    }


def write_registry_snippet(records: list[dict], dest: Path) -> None:
    lines = [
        "# ---- auto-generated by scripts/build_dlpfc_bundles.py ----",
        "# Do not edit by hand; regenerate to refresh checksums.",
        "# Each entry loads a bundled h5ad whose obs['spatialLIBD_layer'] and",
        "# obsm['spatial'] are baked in (schema v1).",
        "",
        "_DLPFC_SLICE_ENTRIES = [",
    ]
    for rec in records:
        lines += [
            "    DatasetEntry(",
            f'        name="{rec["name"]}",',
            (
                f'        description="DLPFC slice {rec["slice_id"]} — '
                "human dorsolateral prefrontal cortex "
                '(Maynard et al. 2021, spatialLIBD)",'
            ),
            f'        url="{rec["url"]}",',
            f'        sha256="{rec["sha256"]}",',
            '        assay="visium",',
            '        tissue="brain",',
            '        species="human",',
            f"        n_obs={rec['n_obs']},",
            f"        n_vars={rec['n_vars']},",
            ('        ground_truth={"domain_truth": "obs[\'spatialLIBD_layer\']"},'),
            f'        license="{LICENSE}",',
            f'        paper_doi="{PAPER_DOI}",',
            "    ),",
        ]
    lines += ["]"]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets_cache/dlpfc"))
    ap.add_argument(
        "--downloads",
        type=Path,
        default=Path("/workspace/histoweave_work/downloads/dlpfc"),
    )
    ap.add_argument("--slices", nargs="*", default=list(SLICES))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.downloads.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    checksum_path = args.out / "checksums.json"

    existing: dict[str, dict] = {}
    if checksum_path.exists():
        existing = {rec["name"]: rec for rec in json.loads(checksum_path.read_text())}

    for sid in args.slices:
        name = f"dlpfc_{sid}"
        out_path = args.out / f"dlpfc_{sid}.h5ad"
        prior = existing.get(name)
        if prior and out_path.exists() and _sha256(out_path) == prior["sha256"]:
            _log(f"[{sid}] cached (sha256 match) — skipping")
            records.append(prior)
            continue
        rec = build_bundle(sid, args.downloads, args.out)
        records.append(rec)

    checksum_path.write_text(json.dumps(records, indent=2))
    write_registry_snippet(records, args.out / "registry_snippet.py")
    _log(f"\n[done] wrote {len(records)} bundles → {args.out}")
    _log(f"[done] checksums → {checksum_path}")


if __name__ == "__main__":
    main()
