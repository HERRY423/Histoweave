"""Build a self-contained Vitessce v3 config and inline CSV data.

Bidirectional linking (task 3)
------------------------------
The scatterplot, obsSets, and heatmap views share coordination scopes so that
interactions in one view drive the others:

* **obsSet selection is shared** across scatterplot ↔ obsSets ↔ heatmap. Selecting
  (or brushing) a cluster in any view highlights the same cells everywhere.
* **A per-cluster top-marker table** (``uns['cluster_top_markers']``) is embedded in
  the payload so the report can, on selection of a cluster, push that cluster's top
  marker genes into the heatmap's ``featureSelection`` — "select cluster → show its
  top markers → update heatmap".
* **featureSelection / obsHighlight** scopes are declared so gene selection and cell
  hovering are also linked between the heatmap and scatterplot.

The Python side is responsible for (a) wiring the shared coordination scopes and
(b) computing the per-cluster marker map; the report template's JS listens for
obsSet selection changes and applies the marker map to the heatmap featureSelection.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..data import SpatialTable


def _csv(frame: pd.DataFrame) -> str:
    """Serialize a Vitessce CSV payload with deterministic Unix newlines."""
    return frame.to_csv(index=False, lineterminator="\n")


def _display_name(column: str) -> str:
    return column.replace("_", " ").title()


def _cluster_top_markers(
    matrix: np.ndarray,
    gene_names: list[str],
    labels: np.ndarray,
    top_n: int = 10,
) -> dict[str, list[str]]:
    """Rank genes per cluster by mean-difference (cluster mean − rest mean).

    Returns ``{cluster_label: [gene, ...]}`` — a lightweight one-vs-rest marker
    score used to drive the "select cluster → show top markers" heatmap update.
    """
    markers: dict[str, list[str]] = {}
    genes = np.asarray(gene_names)
    uniq = pd.unique(labels)
    global_mean = matrix.mean(axis=0)
    for lab in uniq:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        in_mean = matrix[mask].mean(axis=0)
        # one-vs-rest difference; falls back to in-cluster mean if rest is empty
        rest = ~mask
        rest_mean = matrix[rest].mean(axis=0) if rest.any() else global_mean
        score = in_mean - rest_mean
        order = np.argsort(score)[::-1][:top_n]
        markers[str(lab)] = [str(g) for g in genes[order]]
    return markers


def build_vitessce_view_config(
    data: SpatialTable,
    *,
    top_genes: int = 30,
    max_spots: int = 2000,
    markers_per_cluster: int = 10,
) -> dict[str, Any]:
    """Return a Vitessce v3 config plus self-contained inline CSV payloads.

    Spatial coordinates are exposed through obsEmbedding.csv and obsSpots.csv.
    Categorical annotations become obsSets.csv, and an observation-by-gene
    slice becomes obsFeatureMatrix.csv. The views are linked through shared
    coordination scopes and a per-cluster top-marker map (see module docstring).
    """
    coords = data.spatial
    if coords is None:
        raise ValueError("Vitessce scatterplot requires obsm['spatial']")
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("obsm['spatial'] must contain at least two coordinate columns")

    rng = np.random.default_rng(0)
    n_obs = coords.shape[0]
    if n_obs > max_spots:
        sampled_indices = np.sort(rng.choice(n_obs, size=max_spots, replace=False))
    else:
        sampled_indices = np.arange(n_obs)

    obs_ids = data.obs.index.astype(str).to_numpy()[sampled_indices]
    sampled_coords = coords[sampled_indices, :2].astype(float, copy=False)

    embedding = pd.DataFrame(
        {"obs_id": obs_ids, "e0": sampled_coords[:, 0], "e1": sampled_coords[:, 1]}
    )
    spots = pd.DataFrame(
        {"obs_id": obs_ids, "x": sampled_coords[:, 0], "y": sampled_coords[:, 1]}
    )

    preferred_labels = ["domain", "cell_type", "domain_truth"]
    label_columns = [column for column in preferred_labels if column in data.obs.columns]
    for column in data.obs.columns:
        if column in label_columns:
            continue
        series = data.obs[column]
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            label_columns.append(str(column))

    obs_sets = pd.DataFrame({"obs_id": obs_ids})
    label_specs: list[dict[str, str]] = []
    for column in label_columns:
        values = data.obs.iloc[sampled_indices][column]
        obs_sets[column] = values.astype("string").fillna("NA").to_numpy()
        label_specs.append({"name": _display_name(column), "column": column})

    if "svg" in data.uns and isinstance(data.uns["svg"], dict):
        candidates = data.uns["svg"].get("top_genes", [])
    else:
        candidates = []
    selected_genes: list[str] = []
    for entry in candidates:
        gene = entry.get("gene") if isinstance(entry, dict) else entry
        if gene is not None and str(gene) in data.var.index and str(gene) not in selected_genes:
            selected_genes.append(str(gene))
        if len(selected_genes) >= top_genes:
            break
    if not selected_genes:
        selected_genes = [str(gene) for gene in data.var.index[:top_genes]]

    gene_indices = [int(data.var.index.get_loc(gene)) for gene in selected_genes]
    matrix = data.X[sampled_indices][:, gene_indices]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix_array = np.asarray(matrix, dtype=float)
    matrix_array = np.nan_to_num(matrix_array, nan=0.0, posinf=0.0, neginf=0.0)
    feature_matrix = pd.DataFrame(matrix_array, columns=selected_genes)
    feature_matrix.insert(0, "obs_id", obs_ids)

    # --- Per-cluster top markers for the "select cluster -> update heatmap" link ---
    # Computed over the FULL selected-gene matrix using the primary label column so the
    # heatmap can jump to a cluster's markers when it is selected in the scatterplot.
    cluster_top_markers: dict[str, dict[str, list[str]]] = {}
    if label_specs:
        for spec in label_specs:
            col = spec["column"]
            labels = data.obs.iloc[sampled_indices][col].astype("string").fillna("NA").to_numpy()
            cluster_top_markers[spec["name"]] = _cluster_top_markers(
                matrix_array, selected_genes, labels, top_n=markers_per_cluster
            )

    files: list[dict[str, Any]] = [
        {
            "fileType": "obsEmbedding.csv",
            "url": "inline:obs_embedding.csv",
            "coordinationValues": {"obsType": "spot", "embeddingType": "Spatial"},
            "options": {"obsIndex": "obs_id", "obsEmbedding": ["e0", "e1"]},
        },
        {
            "fileType": "obsSpots.csv",
            "url": "inline:obs_spots.csv",
            "coordinationValues": {"obsType": "spot"},
            "options": {"obsIndex": "obs_id", "obsSpots": ["x", "y"]},
        },
        {
            "fileType": "obsFeatureMatrix.csv",
            "url": "inline:obs_matrix.csv",
            "coordinationValues": {
                "obsType": "spot",
                "featureType": "gene",
                "featureValueType": "expression",
            },
            # Vitessce treats the first matrix column as the observation index.
            "options": None,
        },
    ]
    if label_specs:
        files.append(
            {
                "fileType": "obsSets.csv",
                "url": "inline:obs_sets.csv",
                "coordinationValues": {"obsType": "spot"},
                "options": {"obsIndex": "obs_id", "obsSets": label_specs},
            }
        )

    # ---------------------------------------------------------------------
    # Coordination space: shared scopes make the three views bidirectionally
    # linked. All views point at the SAME scope for obsType/featureType/etc.,
    # plus shared selection + highlight + feature scopes so brushing/linking
    # propagates between scatterplot, obsSets, and heatmap.
    # ---------------------------------------------------------------------
    default_feature = selected_genes[0] if selected_genes else None
    coordination_space: dict[str, Any] = {
        "dataset": {"A": "histoweave"},
        "embeddingType": {"A": "Spatial"},
        "obsType": {"A": "spot"},
        "featureType": {"A": "gene"},
        "featureValueType": {"A": "expression"},
        "obsColorEncoding": {"A": "cellSetSelection"},
        # shared interaction scopes (task 3: bidirectional linking)
        "obsSetSelection": {"A": None},
        "obsSetColor": {"A": None},
        "obsHighlight": {"A": None},
        "featureSelection": {"A": [default_feature] if default_feature else None},
        "featureValueColormap": {"A": "plasma"},
        "featureValueColormapRange": {"A": [0.0, 1.0]},
    }
    if label_specs:
        coordination_space["obsSetSelection"] = {"A": [[label_specs[0]["name"]]]}

    # Every view shares the same scope letter "A" for these coordination types,
    # which is what makes selection/color/highlight/feature changes propagate.
    scopes = {
        "dataset": "A",
        "embeddingType": "A",
        "obsType": "A",
        "featureType": "A",
        "featureValueType": "A",
        "obsColorEncoding": "A",
        "obsSetSelection": "A",
        "obsSetColor": "A",
        "obsHighlight": "A",
        "featureSelection": "A",
        "featureValueColormap": "A",
        "featureValueColormapRange": "A",
    }
    layout: list[dict[str, Any]] = [
        {
            "component": "scatterplot",
            "coordinationScopes": scopes,
            "x": 0,
            "y": 0,
            "w": 8,
            "h": 6,
        }
    ]
    if label_specs:
        layout.append(
            {
                "component": "obsSets",
                "coordinationScopes": scopes,
                "x": 8,
                "y": 0,
                "w": 4,
                "h": 6,
            }
        )
    else:
        layout.append(
            {
                "component": "description",
                "coordinationScopes": scopes,
                "x": 8,
                "y": 0,
                "w": 4,
                "h": 6,
            }
        )
    layout.append(
        {
            "component": "heatmap",
            "coordinationScopes": scopes,
            "x": 0,
            "y": 6,
            "w": 12,
            "h": 5,
            "props": {"transpose": True},
        }
    )

    assay = data.uns.get("assay", "spatial")
    config: dict[str, Any] = {
        "version": "1.0.16",
        "name": f"HistoWeave · {assay}",
        "description": (
            f"{data.n_obs} observations × {data.n_vars} genes. "
            f"Provenance: {len(data.provenance)} steps."
        ),
        "initStrategy": "auto",
        "datasets": [
            {"uid": "histoweave", "name": f"{assay} — spatial", "files": files}
        ],
        "coordinationSpace": coordination_space,
        "layout": layout,
    }
    data_files = {
        "obs_spots.csv": _csv(spots),
        "obs_embedding.csv": _csv(embedding),
        "obs_sets.csv": _csv(obs_sets),
        "obs_matrix.csv": _csv(feature_matrix),
    }
    return {
        "config": config,
        "data": data_files,
        "gene_names": selected_genes,
        # linking metadata consumed by the report template's JS
        "cluster_top_markers": cluster_top_markers,
        "primary_label": label_specs[0]["name"] if label_specs else None,
    }


def vitessce_data_json(data: SpatialTable, top_genes: int = 30) -> str:
    """Return a JSON-safe payload for the report application/json script tag."""
    return json.dumps(
        build_vitessce_view_config(data, top_genes=top_genes),
        allow_nan=False,
        default=str,
    )


__all__ = ["build_vitessce_view_config", "vitessce_data_json"]
