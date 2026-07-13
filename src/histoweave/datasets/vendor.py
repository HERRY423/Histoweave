"""Format-faithful vendor fixtures for exercising the IO readers offline.

Real Visium/Xenium bundles are hundreds of MB to many GB and require a network
download, which makes them unusable in CI. These writers fabricate *tiny* datasets on
disk in the **exact** Space Ranger / Xenium directory layout the readers parse, so the
ingestion path can be tested end-to-end deterministically — the same "tiny canonical
dataset" philosophy the synthetic generator follows, extended to the vendor formats.

The counts/coordinates come from :func:`make_synthetic`, so a fixture round-tripped
through a reader still carries ground-truth domains in ``obs['domain_truth']`` and can
feed the benchmarking harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io._tenx import write_10x_h5
from .synthetic import make_synthetic


def _gene_ids(gene_names: list[str]) -> list[str]:
    """Fabricate stable ENSEMBL-style ids paired with the symbols."""
    return [f"ENSG{i:08d}" for i in range(len(gene_names))]


def write_visium_fixture(
    out_dir: str | Path,
    *,
    n_spots: int = 64,
    n_genes: int = 24,
    n_domains: int = 3,
    seed: int = 0,
) -> Path:
    """Write a minimal Space Ranger output tree under ``out_dir`` and return its path.

    Layout::

        out_dir/
            filtered_feature_bc_matrix.h5
            spatial/tissue_positions.csv
            spatial/scalefactors_json.json
    """
    out_dir = Path(out_dir)
    (out_dir / "spatial").mkdir(parents=True, exist_ok=True)

    table = make_synthetic(n_cells=n_spots, n_genes=n_genes, n_domains=n_domains, seed=seed)
    gene_names = list(table.var_names)
    barcodes = [f"{i:016d}-1" for i in range(n_spots)]

    write_10x_h5(
        str(out_dir / "filtered_feature_bc_matrix.h5"),
        table.X,
        feature_ids=_gene_ids(gene_names),
        feature_names=gene_names,
        barcodes=barcodes,
        genome="synthetic",
    )

    # Map the synthetic [0, 100] coordinates onto full-resolution pixel space.
    coords = table.obsm["spatial"]
    scale = 10.0
    pxl_col = np.rint(coords[:, 0] * scale).astype(int)
    pxl_row = np.rint(coords[:, 1] * scale).astype(int)
    positions = pd.DataFrame(
        {
            "barcode": barcodes,
            "in_tissue": 1,
            "array_row": np.rint(coords[:, 1]).astype(int),
            "array_col": np.rint(coords[:, 0]).astype(int),
            "pxl_row_in_fullres": pxl_row,
            "pxl_col_in_fullres": pxl_col,
        }
    )
    positions.to_csv(out_dir / "spatial" / "tissue_positions.csv", index=False)

    scalefactors = {
        "spot_diameter_fullres": 89.0,
        "tissue_hires_scalef": 0.15,
        "tissue_lowres_scalef": 0.045,
        "fiducial_diameter_fullres": 144.0,
    }
    (out_dir / "spatial" / "scalefactors_json.json").write_text(
        json.dumps(scalefactors, indent=2), encoding="utf-8"
    )
    return out_dir


def write_xenium_fixture(
    out_dir: str | Path,
    *,
    n_cells: int = 64,
    n_genes: int = 24,
    n_domains: int = 3,
    seed: int = 0,
    with_controls: bool = True,
) -> Path:
    """Write a minimal Xenium output bundle under ``out_dir`` and return its path.

    Layout::

        out_dir/
            cell_feature_matrix.h5
            cells.parquet
            experiment.xenium

    When ``with_controls`` is set, a ``Negative Control Probe`` feature is appended so
    the reader's gene-expression filtering is exercised.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = make_synthetic(n_cells=n_cells, n_genes=n_genes, n_domains=n_domains, seed=seed)
    gene_names = list(table.var_names)
    feature_ids = _gene_ids(gene_names)
    feature_types = ["Gene Expression"] * n_genes
    X = table.X

    if with_controls:
        rng = np.random.default_rng(seed)
        control = rng.poisson(0.2, size=(n_cells, 1)).astype(float)
        X = np.hstack([X, control])
        gene_names = [*gene_names, "NegControlProbe_00001"]
        feature_ids = [*feature_ids, "NegControlProbe_00001"]
        feature_types = [*feature_types, "Negative Control Probe"]

    cell_ids = [f"cell_{i:05d}" for i in range(n_cells)]
    write_10x_h5(
        str(out_dir / "cell_feature_matrix.h5"),
        X,
        feature_ids=feature_ids,
        feature_names=gene_names,
        barcodes=cell_ids,
        feature_types=feature_types,
        genome="xenium_panel",
    )

    coords = table.obsm["spatial"]
    cells = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_centroid": coords[:, 0],
            "y_centroid": coords[:, 1],
            "transcript_counts": table.X.sum(axis=1).astype(int),
            "cell_area": np.full(n_cells, 25.0),
            "nucleus_area": np.full(n_cells, 12.0),
        }
    )
    cells.to_parquet(out_dir / "cells.parquet", index=False)

    experiment = {
        "run_name": "histoweave_fixture",
        "panel_name": "synthetic_panel",
        "region_name": "region_1",
        "num_cells": n_cells,
        "instrument": "fixture",
    }
    (out_dir / "experiment.xenium").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    return out_dir
