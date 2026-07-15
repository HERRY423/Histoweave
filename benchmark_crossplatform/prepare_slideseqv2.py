"""Build a Slide-seq V2 mouse hippocampus bundle for Histoweave cross-platform benchmark.

Source: `sq.datasets.slideseqv2()` -> Stickels et al 2021 Slide-seq V2 mouse
hippocampus (https://doi.org/10.1038/s41587-020-0739-1). 41786 cells × 4000
genes with pre-computed leiden clusters + biological annotation.

Domain-truth strategy: collapse the 14 fine-grained cluster labels into 6
tissue compartments defined in ``src/histoweave/datasets/domain_mappings.json``:

    hippocampus_pyramidal, neuron_interneuron, glia_astrocyte,
    glia_oligodendrocyte, vascular, other_glia

Outputs:
  /mnt/results/histoweave_upgrade/datasets_cache/slideseqv2/slideseqv2_mouse_hippocampus.h5ad
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SCHEMA_VERSION = "histoweave.slideseqv2.bundle.v1"
PAPER_DOI = "10.1038/s41587-020-0739-1"

OUT_DIR = Path(
    os.environ.get(
        "HISTOWEAVE_SLIDESEQ_OUT",
        "/mnt/results/histoweave_upgrade/datasets_cache/slideseqv2",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING = Path(os.environ.get("HISTOWEAVE_STAGING", "/workspace/histoweave_work/staging"))
STAGING.mkdir(parents=True, exist_ok=True)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _load_domain_map() -> dict:
    p = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "histoweave"
        / "datasets"
        / "domain_mappings.json"
    )
    return json.loads(p.read_text())["slideseqv2_mouse_hippocampus"]["compartments"]


def _collapse_labels(clusters: pd.Series, compartments: dict) -> pd.Series:
    inverse: dict[str, str] = {}
    for comp, members in compartments.items():
        for m in members:
            inverse[m] = comp
    out = clusters.astype(str).map(inverse).fillna("other")
    return out.astype("category")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build() -> dict:
    import squidpy as sq

    adata = sq.datasets.slideseqv2(path="/workspace/histoweave_work/squidpy_cache/slideseqv2.h5ad")
    _log(f"[load] slideseqv2: {adata.n_obs} beads x {adata.n_vars} genes")
    _log(f"       cluster:\n{adata.obs['cluster'].value_counts().to_string()}")

    domain_map = _load_domain_map()
    truth = _collapse_labels(adata.obs["cluster"], domain_map)
    adata.obs["domain_truth"] = truth
    _log(f"[collapse] domain_truth:\n{truth.value_counts().to_string()}")

    # Guarantee raw-count layer
    if "counts" not in adata.layers:
        X = adata.X
        if sparse.issparse(X):
            adata.layers["counts"] = X.copy()
        else:
            adata.layers["counts"] = sparse.csr_matrix(np.maximum(X, 0))

    # Sanity-check spatial coords
    if "spatial" not in adata.obsm:
        if all(c in adata.obs for c in ("x", "y")):
            adata.obsm["spatial"] = np.column_stack(
                [adata.obs["x"].astype(float), adata.obs["y"].astype(float)]
            )
        else:
            raise KeyError("no spatial coords available for SlideseqV2 bundle")

    adata.uns["schema_version"] = SCHEMA_VERSION
    adata.uns["source"] = "squidpy.datasets.slideseqv2"
    adata.uns["paper_doi"] = PAPER_DOI
    adata.uns["license"] = "CC-BY 4.0"
    adata.uns["preparation_script"] = "benchmark_crossplatform/prepare_slideseqv2.py"

    staged = STAGING / "slideseqv2_mouse_hippocampus.h5ad"
    adata.write_h5ad(staged)
    final = OUT_DIR / "slideseqv2_mouse_hippocampus.h5ad"
    shutil.copyfile(staged, final)
    sha = _sha256(final)
    _log(f"[write] {final} ({final.stat().st_size / 1e6:.1f} MB) sha256={sha[:12]}...")
    return {
        "name": "slideseqv2_mouse_hippocampus",
        "path": str(final),
        "sha256": sha,
        "bytes": final.stat().st_size,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_domains": int(truth.nunique()),
        "domain_counts": truth.value_counts().to_dict(),
    }


if __name__ == "__main__":
    info = build()
    _log(json.dumps(info, indent=2, default=str))
