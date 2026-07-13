"""Build a self-contained Vitessce view config + inline data from a SpatialTable.

Vitessce v3 renders interactive spatial scatterplots, heatmaps, and genome
browser views directly in the browser.  By bundling the view config and all
cell-level data as JSON inside the HTML report we avoid a Python dependency
on vitessce (it loads from CDN) and keep the report fully self-contained.

Data-size strategy
------------------
For interactive exploration the spatial scatterplot needs coordinates and
colour-by columns; the heatmap shows the top *n* spatially variable genes.
Both subsets are small enough (≤ 500 spots × ≤ 50 genes) to embed inline
without bloating the HTML past a few hundred KB.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..data import SpatialTable


def build_vitessce_view_config(
    data: SpatialTable,
    *,
    top_genes: int = 30,
    max_spots: int = 2000,
) -> dict[str, Any]:
    """Return a Vitessce view config with inline data for *data*.

    The returned dict has two top-level keys:

    ``"config"``
        The Vitessce view config (version, layout, coordination scopes).
    ``"data"``
        A flat dict of keys → values that will be converted to a JS
        ``File` object` at render time.

    Parameters
    ----------
    data : SpatialTable
        A processed table (domains / annotations already computed).
    top_genes : int
        Number of top spatially variable genes to include in the heatmap.
    max_spots : int
        If ``data.n_obs > max_spots``, randomly subsample for the
        scatterplot to keep the HTML size reasonable.
    """
    coords = data.spatial
    if coords is None:
        raise ValueError("Vitessce scatterplot requires obsm['spatial']")

    # --- subsample for size control -----------------------------------------
    rng = np.random.default_rng(0)
    n = coords.shape[0]
    if n > max_spots:
        sampled_indices: np.ndarray = rng.choice(n, size=max_spots, replace=False)
        sampled_indices.sort()
    else:
        sampled_indices = np.arange(n)

    # Build cells.json content (array of { xy, mappings })
    cells = []
    for i in sampled_indices:
        di = int(i)
        cell: dict[str, Any] = {
            "x": float(coords[di, 0]),
            "y": float(coords[di, 1]),
        }
        # Colour-by columns: add every categorical-ish obs column
        for col in data.obs.columns:
            val = data.obs.iloc[di][col]
            if hasattr(val, "item"):
                val = val.item()
            cell.setdefault("mappings", {})[col] = str(val) if val is not None else "NA"
        cells.append(cell)

    # Build expression matrix for heatmap (genes x cells = features x samples)
    expr_cols: list[str] = []
    expr_data: list[list[float]] = []
    if "svg" in data.uns and "top_genes" in data.uns["svg"]:
        top = data.uns["svg"]["top_genes"][:top_genes]
    else:
        # Fallback: use the first top_genes genes from var
        top = [{"gene": g} for g in data.var.index[:top_genes]]

    X = data.X
    for entry in top:
        gene = entry["gene"]
        if gene in data.var.index:
            j = data.var.index.get_loc(gene)
            col_vals = X[:, j]
            if hasattr(col_vals, "toarray"):
                col_vals = col_vals.toarray().flatten()
            expr_data.append([float(v) for v in np.atleast_1d(col_vals)[:n]])
            expr_cols.append(gene)

    # --- view config --------------------------------------------------------
    config: dict[str, Any] = {
        "version": "1.0.16",
        "name": f"HistoWeave · {data.uns.get('assay', 'spatial')}",
        "description": (
            f"{data.n_obs} observations × {data.n_vars} genes. "
            f"Provenance: {len(data.provenance)} steps."
        ),
        "layout": [
            {
                "component": "scatterplot",
                "coordinationScopes": {"dataset": "histoweave_scatterplot"},
                "x": 0, "y": 0, "w": 7, "h": 6,
                "props": {
                    "mapping": "UMAP",  # use as colour-by target
                    "mappingSelect": "domain",
                },
            },
            {
                "component": "description",
                "coordinationScopes": {"dataset": "histoweave_scatterplot"},
                "x": 7, "y": 0, "w": 5, "h": 6,
                "props": {"title": "Domains"},
            },
            {
                "component": "heatmap",
                "coordinationScopes": {"dataset": "histoweave_heatmap"},
                "x": 0, "y": 6, "w": 12, "h": 5,
                "props": {"transpose": True},
            },
        ],
        "datasets": [
            {
                "uid": "histoweave_scatterplot",
                "name": f"{data.uns.get('assay', 'spatial')} — Scatterplot",
                "files": [{"type": "cells", "fileType": "cells.json"}],
            },
            {
                "uid": "histoweave_heatmap",
                "name": f"{data.uns.get('assay', 'spatial')} — Expression",
                "files": [{"type": "cells", "fileType": "cells.json"}],
            },
        ],
        "coordinationSpace": {"dataset": {"A": "histoweave_scatterplot"}},
        "initStrategy": "auto",
    }

    # Build the cells.json payload for the second dataset
    heatmap_cells = []
    for i in sampled_indices:
        di = int(i)
        hc: dict[str, Any] = {"x": float(coords[di, 0]), "y": float(coords[di, 1])}
        hc["mappings"] = {
            gene: expr_data[gi][di] if gi < len(expr_data) and di < len(expr_data[gi]) else 0.0
            for gi, gene in enumerate(expr_cols)
        }
        heatmap_cells.append(hc)

    data_files: dict[str, str] = {
        "cells.json": json.dumps(cells, allow_nan=False),
        "heatmap_cells.json": json.dumps(heatmap_cells, allow_nan=False),
        "genes.json": json.dumps(expr_cols, allow_nan=False),
    }

    return {"config": config, "data": data_files, "gene_names": expr_cols}


def vitessce_data_json(data: SpatialTable, top_genes: int = 30) -> str:
    """Return a JSON-safe payload that the Jinja2 template can embed directly.

    The returned JSON is a single string wrapping the ``build_vitessce_view_config``
    output so the template can do ``{{ vitessce_data|safe }}``.
    """
    return json.dumps(
        build_vitessce_view_config(data, top_genes=top_genes),
        allow_nan=False,
        default=str,
    )


__all__ = ["build_vitessce_view_config", "vitessce_data_json"]
