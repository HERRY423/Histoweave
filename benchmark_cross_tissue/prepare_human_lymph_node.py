"""Build a labelled Human Lymph Node Xenium bundle from official 10x files.

Ground truth comes only from the pathology annotation polygons supplied with
the Xenium Prime preview dataset.  Cells outside polygons or in conflicting
overlapping polygons are excluded; predicted cell-type labels are never used
as spatial-domain truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression"


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


def _metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _property(properties: dict[str, Any], dotted_name: str) -> Any:
    value: Any = properties
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _feature_label(feature: dict[str, Any], label_property: str | None) -> str | None:
    properties = feature.get("properties") or {}
    candidates = (
        (label_property,) if label_property else ("classification.name", "name", "label", "type")
    )
    for candidate in candidates:
        value = _property(properties, candidate)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def assign_pathology_domains(
    coordinates: np.ndarray,
    geojson: dict[str, Any],
    *,
    label_property: str | None = None,
) -> pd.Series:
    """Assign unambiguous polygon labels to x/y coordinates.

    A cell intersecting polygons carrying different labels is marked
    ``ambiguous``.  Repeated polygons with the same label are harmless.
    """
    try:
        from shapely import contains_xy
        from shapely.geometry import shape
    except ImportError as exc:
        raise ImportError(
            "Pathology polygon assignment requires the 'spatial' extra: "
            "pip install 'histoweave[spatial]'"
        ) from exc

    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_cells, 2)")
    labels = np.full(xy.shape[0], None, dtype=object)
    features = geojson.get("features", [])
    used = 0
    for feature in features:
        label = _feature_label(feature, label_property)
        geometry = feature.get("geometry")
        if label is None or not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        polygon = shape(geometry)
        inside = np.asarray(contains_xy(polygon, xy[:, 0], xy[:, 1]), dtype=bool)
        if not inside.any():
            continue
        used += 1
        empty = inside & pd.isna(labels)
        conflict = inside & ~pd.isna(labels) & (labels != label)
        labels[empty] = label
        labels[conflict] = "ambiguous"
    if used == 0:
        raise ValueError(
            "no labelled pathology polygon intersected any cell; check the GeoJSON "
            "label property and coordinate scale/offset"
        )
    return pd.Series(labels, dtype="string")


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

    adata = sc.read_10x_h5(args.matrix)
    n_original = int(adata.n_obs)
    meta = _metadata(args.metadata)
    id_column = args.id_column or next(
        (name for name in ("cell_id", "barcode", "CellID") if name in meta), None
    )
    if id_column is None:
        raise ValueError("metadata needs --id-column (for example cell_id)")
    meta[id_column] = meta[id_column].astype(str)
    meta = meta.set_index(id_column).reindex(adata.obs_names.astype(str))
    for coordinate in (args.x_column, args.y_column):
        if coordinate not in meta:
            raise ValueError(f"metadata is missing coordinate column {coordinate!r}")

    cell_xy = meta[[args.x_column, args.y_column]].to_numpy(dtype=float)
    annotation_xy = cell_xy.copy()
    annotation_xy[:, 0] = annotation_xy[:, 0] * args.geojson_scale + args.geojson_offset_x
    annotation_xy[:, 1] = annotation_xy[:, 1] * args.geojson_scale + args.geojson_offset_y
    geojson = json.loads(args.pathology_geojson.read_text(encoding="utf-8"))
    domains = assign_pathology_domains(annotation_xy, geojson, label_property=args.label_property)

    excluded = {value.casefold() for value in args.exclude_label}
    valid = meta[[args.x_column, args.y_column]].notna().all(axis=1).to_numpy()
    valid &= domains.notna().to_numpy()
    valid &= domains.ne("ambiguous").to_numpy()
    valid &= ~domains.fillna("").str.casefold().isin(excluded).to_numpy()
    counts = domains[valid].value_counts()
    retained = set(counts[counts >= args.min_cells_per_domain].index.astype(str))
    valid &= domains.fillna("").isin(retained).to_numpy()
    if not valid.any():
        raise ValueError("no unambiguous pathology-labelled cells remained after filtering")

    adata = adata[valid].copy()
    meta = meta.iloc[np.flatnonzero(valid)]
    domains = domains.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    idx = _stratified_indices(domains, args.max_cells, args.seed)
    adata = adata[idx].copy()
    meta = meta.iloc[idx]
    domains = domains.iloc[idx]
    adata.obs["domain_truth"] = pd.Categorical(domains.to_numpy())
    adata.obs["truth_source"] = "10x_pathology_annotation"
    adata.obsm["spatial"] = meta[[args.x_column, args.y_column]].to_numpy(dtype=np.float32)
    adata.layers["counts"] = adata.X.copy()
    adata.uns.update(
        {
            "schema_version": "histoweave.xenium.lymph_node.bundle.v1",
            "source": "10x Xenium Prime Human Lymph Node preview",
            "source_url": SOURCE_URL,
            "license": "CC-BY-4.0",
            "n_original": n_original,
            "ground_truth_definition": "pathology polygons; unannotated/ambiguous cells excluded",
            "pathology_geojson": str(args.pathology_geojson),
            "geojson_transform": {
                "scale": args.geojson_scale,
                "offset_x": args.geojson_offset_x,
                "offset_y": args.geojson_offset_y,
            },
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "xenium_human_lymph_node",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_domains": int(adata.obs["domain_truth"].nunique()),
        "domains": sorted(adata.obs["domain_truth"].astype(str).unique().tolist()),
        "truth_source": "10x pathology annotation polygons",
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path, required=True, help="Official cell_feature_matrix.h5"
    )
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Official cells CSV(.gz)/Parquet"
    )
    parser.add_argument("--pathology-geojson", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets_cache/xenium/xenium_human_lymph_node.h5ad",
    )
    parser.add_argument("--id-column")
    parser.add_argument("--x-column", default="x_centroid")
    parser.add_argument("--y-column", default="y_centroid")
    parser.add_argument(
        "--label-property", help="Dotted GeoJSON property, e.g. classification.name"
    )
    parser.add_argument("--geojson-scale", type=float, default=1.0)
    parser.add_argument("--geojson-offset-x", type=float, default=0.0)
    parser.add_argument("--geojson-offset-y", type=float, default=0.0)
    parser.add_argument("--exclude-label", action="append", default=["unannotated", "unknown"])
    parser.add_argument("--min-cells-per-domain", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
