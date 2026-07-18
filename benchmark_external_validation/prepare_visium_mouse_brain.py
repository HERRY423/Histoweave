"""Prepare the squidpy Visium mouse-brain bundle with anatomical ground truth.

Source: 10x Genomics V1 Adult Mouse Brain (Visium H&E), accessed via
``squidpy.datasets.visium_hne_adata()``. The dataset ships 15 manually
annotated anatomical regions (five cortical layers, hippocampus, two
pyramidal layers, two thalamic regions, two hypothalamic regions, fibre
tract, striatum, lateral ventricle) derived from the Allen Brain Atlas and
the Linnarsson lab mouse brain gene-expression atlas. These are genuine
anatomical region labels — not cell-type predictions — so they satisfy
HistoWeave's strict spatial-domain ground-truth policy.

The squidpy AnnData already carries ``obs['cluster']`` (the 15 region labels)
and ``obsm['spatial']``; this script renames ``cluster`` → ``domain_truth``,
ensures a raw ``counts`` layer is present, QC-filters, HVG-subsets, and writes
a checksummed ``.h5ad`` bundle plus a ``.json`` receipt.

Requires the ``scanpy``/``squidpy`` extras::

    pip install "histoweave[scanpy,spatial]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://squidpy.readthedocs.io/en/stable/api/squidpy.datasets.visium_hne_adata.html"
# 15 anatomical region labels (Allen Brain Atlas reference). The squidpy
# dataset's obs['cluster'] column carries exactly these.
ANATOMICAL_REGIONS = (
    "Fiber_tract",
    "Hippocampus",
    "Hypothalamus_1",
    "Hypothalamus_2",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "Pyramidal_layer",
    "Pyramidal_layer_dentate_gyrus",
    "Striatum",
    "Thalamus_1",
    "Thalamus_2",
)
INVALID_LABELS = {"", "nan", "none", "unknown", "na", "NA"}
DEFAULT_N_HVG = 2000

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    _LOGGER.info("%s", message)


def _load_squidpy_visium_hne():
    """Load the squidpy Visium H&E mouse brain AnnData.

    The squidpy dataset is pre-processed (normalized + clustered). We recover
    raw-like counts for the harness's re-normalization: squidpy stores the
    raw counts in ``adata.raw`` when present, otherwise we treat X as already
    log-normalized and recover pseudo-counts via expm1.
    """
    import squidpy as sq

    a = sq.datasets.visium_hne_adata()
    # squidpy ships a.raw with the raw counts.
    if a.raw is not None:
        raw = a.raw.to_adata()
        raw.obs = a.obs.copy()
        raw.obsm = dict(a.obsm)
        raw.uns = dict(a.uns)
        a = raw
    a.var_names_make_unique()
    return a


def build(args: argparse.Namespace) -> dict[str, object]:
    import scanpy as sc

    adata = _load_squidpy_visium_hne()
    n_original = int(adata.n_obs)
    _log(f"loaded squidpy visium_hne: {adata.n_obs} spots x {adata.n_vars} genes")

    if "cluster" not in adata.obs.columns:
        raise ValueError(
            "squidpy visium_hne_adata is missing obs['cluster'] (the 15 anatomical "
            "region labels); check your squidpy version"
        )
    if "spatial" not in adata.obsm:
        raise ValueError("squidpy visium_hne_adata is missing obsm['spatial']")

    labels = adata.obs["cluster"].astype("string")
    valid = labels.notna().to_numpy()
    valid = valid & ~labels.fillna("").str.strip().str.casefold().isin(INVALID_LABELS).to_numpy()
    adata = adata[valid].copy()
    if adata.n_obs == 0:
        raise ValueError("no spots retained a valid anatomical region label")

    adata.obs["domain_truth"] = pd.Categorical(adata.obs["cluster"].astype(str).to_numpy())
    adata.obs["truth_source"] = "allen_brain_atlas_anatomical_regions"
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]

    # counts layer: prefer an existing 'counts' layer, else use X if integer-like,
    # else recover pseudo-counts from log-normalized X.
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
    else:
        X = adata.X
        Xdense = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
        int_frac = float(np.mean(np.isclose(Xdense, np.round(Xdense))))
        if int_frac >= 0.95:
            counts = np.clip(Xdense, 0, None)
        else:
            counts = np.clip(np.expm1(Xdense), 0, None)
    adata.layers["counts"] = counts

    # QC + HVG (on a normalized copy so counts stay raw).
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
            "schema_version": "histoweave.visium.mouse_brain.bundle.v1",
            "source": "10x Visium V1 Adult Mouse Brain (H&E) via squidpy",
            "source_url": SOURCE_URL,
            "license": "10x Genomics EULA",
            "n_original": n_original,
            "ground_truth_definition": "15 Allen-reference anatomical regions (obs['cluster'])",
            "annotation_references": [
                "Allen Brain Atlas",
                "Linnarsson lab Mouse Brain gene-expression atlas",
            ],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "visium_mouse_brain",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_domains": int(adata.obs["domain_truth"].nunique()),
        "domains": sorted(adata.obs["domain_truth"].astype(str).unique().tolist()),
        "truth_source": "allen_brain_atlas_anatomical_regions",
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    _log(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets_cache" / "visium" / "visium_mouse_brain.h5ad",
    )
    parser.add_argument("--n-hvg", type=int, default=DEFAULT_N_HVG)
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
