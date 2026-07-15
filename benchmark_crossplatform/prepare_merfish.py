"""Build a MERFISH mouse hypothalamus bundle for Histoweave cross-platform benchmark.

Source: `sq.datasets.merfish()` -> Moffitt et al 2018 preoptic MERFISH atlas
(https://doi.org/10.1126/science.aau5324). 73655 cells × 161 genes across 12
Bregma sections in a single AnnData bundle.

Domain-truth strategy: user-selected Q6 "domain-prior collapse". Moffitt
cell-classes (Inhibitory / Excitatory / Astrocyte / OD Mature 1 / OD Mature 2 /
Endothelial / Ependymal / Microglia / Ambiguous) are collapsed into 6 tissue-
level compartments defined in ``src/histoweave/datasets/domain_mappings.json``:

    neuron_excitatory, neuron_inhibitory, glia_oligodendrocyte, glia_astrocyte,
    other_glia, vascular_ependymal

We subsample to a manageable evaluation set (user selection Q2: 3 sections
averaged -> we take three Bregma slices at z=0.16, z=0.11, z=0.06 to match the
user's "3 sections averaged" semantic, though for the h5ad bundle we ship all
sections labelled by ``batch`` so downstream code can also stratify).

Outputs:
  /mnt/results/histoweave_upgrade/datasets_cache/merfish/merfish_mouse_hypothalamus.h5ad

The bundle has:
    X                     : raw counts (sparse if available)
    layers['counts']      : identical copy
    obs['Cell_class']     : original Moffitt labels
    obs['domain_truth']   : collapsed 6-compartment label
    obs['batch']          : bregma section id
    obsm['spatial']       : (Centroid_X, Centroid_Y)
    uns['schema_version'] : 'histoweave.merfish.bundle.v1'
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

SCHEMA_VERSION = "histoweave.merfish.bundle.v1"
PAPER_DOI = "10.1126/science.aau5324"

OUT_DIR = Path(
    os.environ.get(
        "HISTOWEAVE_MERFISH_OUT",
        "/mnt/results/histoweave_upgrade/datasets_cache/merfish",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING = Path(os.environ.get("HISTOWEAVE_STAGING", "/workspace/histoweave_work/staging"))
STAGING.mkdir(parents=True, exist_ok=True)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _load_domain_map() -> dict:
    """Return the collapse dict for MERFISH cell_class -> domain."""
    p = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "histoweave"
        / "datasets"
        / "domain_mappings.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"domain_mappings.json missing at {p}")
    return json.loads(p.read_text())["merfish_mouse_hypothalamus"]["compartments"]


def _collapse_labels(cell_classes: pd.Series, compartments: dict) -> pd.Series:
    """Map each Moffitt cell_class to its compartment (or 'other')."""
    inverse: dict[str, str] = {}
    for comp, members in compartments.items():
        for m in members:
            inverse[m] = comp
    out = cell_classes.astype(str).map(inverse).fillna("other")
    return out.astype("category")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build() -> dict:
    import squidpy as sq  # heavy import: keep inside build()

    adata = sq.datasets.merfish(path="/workspace/histoweave_work/squidpy_cache/merfish.h5ad")
    _log(f"[load] merfish: {adata.n_obs} cells x {adata.n_vars} genes")
    _log(f"       Cell_class:\n{adata.obs['Cell_class'].value_counts().to_string()}")

    # Build domain_truth via collapse
    domain_map = _load_domain_map()
    truth = _collapse_labels(adata.obs["Cell_class"], domain_map)
    adata.obs["domain_truth"] = truth
    _log(f"[collapse] domain_truth:\n{truth.value_counts().to_string()}")

    # Ensure counts layer and non-negative X
    if "counts" not in adata.layers:
        X = adata.X
        if sparse.issparse(X):
            adata.layers["counts"] = X.copy()
        else:
            adata.layers["counts"] = sparse.csr_matrix(np.maximum(X, 0))

    # Ensure spatial obsm is (n_obs, 2)
    if "spatial" not in adata.obsm:
        if all(c in adata.obs for c in ("Centroid_X", "Centroid_Y")):
            adata.obsm["spatial"] = np.column_stack(
                [adata.obs["Centroid_X"].astype(float), adata.obs["Centroid_Y"].astype(float)]
            )
        else:
            raise KeyError("no spatial coords available for MERFISH")

    # Provenance
    adata.uns["schema_version"] = SCHEMA_VERSION
    adata.uns["source"] = "squidpy.datasets.merfish"
    adata.uns["paper_doi"] = PAPER_DOI
    adata.uns["license"] = "CC-BY 4.0"
    adata.uns["preparation_script"] = "benchmark_crossplatform/prepare_merfish.py"

    # Write to staging first (h5ad wants random-access writes -> not S3 FUSE)
    staged = STAGING / "merfish_mouse_hypothalamus.h5ad"
    adata.write_h5ad(staged)
    final = OUT_DIR / "merfish_mouse_hypothalamus.h5ad"
    shutil.copyfile(staged, final)
    sha = _sha256(final)
    _log(f"[write] {final} ({final.stat().st_size / 1e6:.1f} MB) sha256={sha[:12]}...")
    return {
        "name": "merfish_mouse_hypothalamus",
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
