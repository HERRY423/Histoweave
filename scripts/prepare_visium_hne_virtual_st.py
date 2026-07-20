#!/usr/bin/env python3
"""Prepare a virtual_st-ready Visium H&E bundle with registered histology.

Builds ``datasets_cache/visium/visium_mouse_brain_hne.h5ad`` from the public
10x Visium Adult Mouse Brain H&E slide (via squidpy or a local cache of
``visium_hne_adata.h5ad``). The bundle keeps:

* measured expression (HVG-subset counts) as ``X`` / ``layers['counts']``
* ``obsm['spatial']``
* lowres (and optionally hires) H&E under AnnData ``uns['spatial'][…]['images']``
  so :func:`histoweave.datasets.load_visium_hne_paired` and the registry loader
  both attach ``SpatialTable.images['image']``

Usage
-----
::

    python scripts/prepare_visium_hne_virtual_st.py
    python scripts/prepare_visium_hne_virtual_st.py --source data/anndata/visium_hne_adata.h5ad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def _log(message: object) -> None:
    _LOGGER.info("%s", message)


def build(args: argparse.Namespace) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from histoweave.datasets.histology import load_visium_hne_paired

    table = load_visium_hne_paired(
        source=args.source,
        prefer=args.prefer,
        n_hvg=args.n_hvg,
        min_cells=args.min_cells,
    )
    _log(
        f"loaded virtual_st table: {table.n_obs} spots x {table.n_vars} genes; "
        f"images={list(table.images)}"
    )

    # Write as AnnData so the existing h5ad registry path can load it.
    import anndata as ad
    import numpy as np

    x = np.asarray(table.X, dtype=np.float32)
    layers = {
        str(k): np.asarray(v, dtype=np.float32)
        for k, v in table.layers.items()
        if k is not None and str(k) not in {"", "None"}
    }
    # Avoid AnnData rejecting layers that shadow X under a null key.
    layers = {k: v for k, v in layers.items() if k != "X"}
    adata = ad.AnnData(
        X=x,
        obs=table.obs.copy(),
        var=table.var.copy(),
        obsm={"spatial": np.asarray(table.spatial, dtype=float)},
        layers=layers,
        uns={
        k: v
        for k, v in dict(table.uns).items()
        if k is not None and str(k) not in {"", "None", "provenance"}
    },
    )
    # Ensure uns.spatial images are present for _load_h5ad_bundle extraction.
    if "spatial" not in adata.uns or not isinstance(adata.uns["spatial"], dict):
        adata.uns["spatial"] = {
            "V1_Adult_Mouse_Brain": {
                "images": {},
                "scalefactors": {},
            }
        }
    lib = next(iter(adata.uns["spatial"]))
    images = dict(adata.uns["spatial"][lib].get("images") or {})
    if "image_lowres" in table.images or "image" in table.images:
        images["lowres"] = np.asarray(
            table.images.get("image_lowres", table.images["image"])
        )
    if "image_hires" in table.images:
        images["hires"] = np.asarray(table.images["image_hires"])
    if "image" in table.images and "lowres" not in images and "hires" not in images:
        images["lowres"] = np.asarray(table.images["image"])
    adata.uns["spatial"][lib]["images"] = images
    adata.uns["histoweave_virtual_st"] = {
        "schema_version": "histoweave.virtual_st.visium_hne.v1",
        "analysis_task": "virtual_st",
        "ground_truth_kind": "measured_expression",
        "image_keys": list(table.images.keys()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "visium_mouse_brain_hne",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "image_keys": list(images.keys()),
        "image_shapes": {k: list(np.asarray(v).shape) for k, v in images.items()},
        "analysis_task": "virtual_st",
        "ground_truth_kind": "measured_expression",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    _log(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional path to visium_hne_adata.h5ad (else squidpy / repo cache).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets_cache" / "visium" / "visium_mouse_brain_hne.h5ad",
    )
    parser.add_argument("--prefer", choices=("lowres", "hires"), default="lowres")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--min-cells", type=int, default=3)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
