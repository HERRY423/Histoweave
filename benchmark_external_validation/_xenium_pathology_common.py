"""Shared builder for Xenium pathology-labelled bundles (lung / ovarian / ...).

Each 10x Xenium public dataset that ships a pathologist GeoJSON annotation is
prepared identically: read the official ``cell_feature_matrix.h5`` + cells
metadata, assign each cell's centroid to a pathology polygon via point-in-
polygon (the shared ``histoweave.datasets.pathology_domains`` helper), exclude
unannotated / ambiguous cells, stratified-subsample to a tractable size, and
write a checksummed ``.h5ad`` bundle with ``obs['domain_truth']``,
``obsm['spatial']``, ``layers['counts']``.

This module exists so the per-dataset preparers are thin wrappers that only
carry the dataset-specific metadata (name, source URL, license, default
GeoJSON label property). The ground-truth policy is strict: only pathology
polygon labels count as spatial-domain truth; predicted cell types are never
used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from histoweave.datasets.pathology_domains import (  # noqa: E402
    assign_pathology_domains,
    stratified_indices,
)

_LOGGER = logging.getLogger(__name__)

INVALID_LABELS = {"", "nan", "none", "unknown", "unannotated", "na", "NA", "unmapped"}


@dataclass
class XeniumDatasetSpec:
    """Per-dataset metadata for a Xenium pathology bundle."""

    name: str
    source: str
    source_url: str
    license: str
    schema_version: str
    # Default GeoJSON label property (QuPath convention). None => auto-detect.
    default_label_property: str | None = "classification.name"
    # Labels to exclude even if present in the GeoJSON.
    default_exclude_labels: tuple[str, ...] = ("unannotated", "unknown", "unmapped")


def _log(message: object) -> None:
    _LOGGER.info("%s", message)


def _metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def build_xenium_pathology_bundle(
    args: argparse.Namespace,
    spec: XeniumDatasetSpec,
) -> dict[str, object]:
    """Build a checksumed Xenium pathology ``.h5ad`` bundle.

    Expected args fields: matrix, metadata, pathology_geojson, output,
    id_column, x_column, y_column, label_property, geojson_scale,
    geojson_offset_x, geojson_offset_y, exclude_label, min_cells_per_domain,
    max_cells, seed.
    """
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
    # The GeoJSON pathology polygons may be in a different coordinate frame
    # than the cell centroids. By default they share the same micron frame
    # (scale=1, offset=0). For QuPath / H&E-aligned annotations in H&E image
    # pixel space, supply a 10x ``he_imagealignment.csv`` via
    # ``--alignment-matrix``: the 3×3 affine maps H&E pixels → Xenium pixels,
    # and we multiply by ``--pixel-size`` (default 0.2125) to land in Xenium
    # microns. The polygons are transformed INTO the cell frame (not the other
    # way around) so the cell centroids stay in their native micron space.
    geojson = json.loads(args.pathology_geojson.read_text(encoding="utf-8"))
    if getattr(args, "alignment_matrix", None):
        from shapely.affinity import affine_transform
        from shapely.geometry import shape

        lines = Path(args.alignment_matrix).read_text().strip().split("\n")
        M = np.array([[float(v) for v in line.split(",")] for line in lines[:2]])
        ps = getattr(args, "pixel_size", 0.2125)
        # affine_transform params: [a, b, d, e, xoff, yoff] for
        # x' = a*x + b*y + xoff ; y' = d*x + e*y + yoff
        params = [
            M[0, 0] * ps,
            M[0, 1] * ps,
            M[1, 0] * ps,
            M[1, 1] * ps,
            M[0, 2] * ps,
            M[1, 2] * ps,
        ]
        transformed_features = []
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry")
            if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                transformed_features.append(feature)
                continue
            poly = shape(geometry)
            poly = affine_transform(poly, params)
            transformed_features.append(
                {**feature, "geometry": json.loads(json.dumps(poly.__geo_interface__))}
            )
        geojson = {"type": "FeatureCollection", "features": transformed_features}
        annotation_xy = cell_xy  # cells stay in native micron frame
    else:
        annotation_xy = cell_xy.copy()
        annotation_xy[:, 0] = annotation_xy[:, 0] * args.geojson_scale + args.geojson_offset_x
        annotation_xy[:, 1] = annotation_xy[:, 1] * args.geojson_scale + args.geojson_offset_y

    label_property = args.label_property or spec.default_label_property
    domains = assign_pathology_domains(annotation_xy, geojson, label_property=label_property)

    excluded = {value.casefold() for value in (args.exclude_label or spec.default_exclude_labels)}
    valid = meta[[args.x_column, args.y_column]].notna().all(axis=1).to_numpy()
    valid = valid & domains.notna().to_numpy()
    # Compare against a string-coerced view so NA entries don't produce an
    # object-dtype Series that breaks the boolean & assignment.
    domains_str = domains.astype("string").fillna("")
    valid = valid & (domains_str != "ambiguous").to_numpy()
    valid = valid & ~domains_str.str.casefold().isin(excluded).to_numpy()
    counts = domains[valid].value_counts()
    retained = set(counts[counts >= args.min_cells_per_domain].index.astype(str))
    valid = valid & domains_str.isin(retained).to_numpy()
    if not valid.any():
        raise ValueError("no unambiguous pathology-labelled cells remained after filtering")

    adata = adata[valid].copy()
    meta = meta.iloc[np.flatnonzero(valid)]
    domains = domains.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    idx = stratified_indices(domains, args.max_cells, args.seed)
    adata = adata[idx].copy()
    meta = meta.iloc[idx]
    domains = domains.iloc[idx]

    adata.obs["domain_truth"] = pd.Categorical(domains.to_numpy())
    adata.obs["truth_source"] = "10x_pathology_annotation"
    adata.obsm["spatial"] = meta[[args.x_column, args.y_column]].to_numpy(dtype=np.float32)
    adata.layers["counts"] = adata.X.copy()
    adata.uns.update(
        {
            "schema_version": spec.schema_version,
            "source": spec.source,
            "source_url": spec.source_url,
            "license": spec.license,
            "n_original": n_original,
            "ground_truth_definition": "pathology polygons; unannotated/ambiguous cells excluded",
            "pathology_geojson": str(args.pathology_geojson),
            "geojson_transform": {
                "scale": args.geojson_scale,
                "offset_x": args.geojson_offset_x,
                "offset_y": args.geojson_offset_y,
            },
            "label_property": label_property,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": spec.name,
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
    _log(json.dumps(receipt, indent=2))
    return receipt


def add_common_xenium_args(parser: argparse.ArgumentParser, *, default_output: Path) -> None:
    """Add the shared CLI args every Xenium pathology preparer needs."""
    parser.add_argument(
        "--matrix", type=Path, required=True, help="Official cell_feature_matrix.h5"
    )
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Official cells CSV(.gz)/Parquet"
    )
    parser.add_argument("--pathology-geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--id-column")
    parser.add_argument("--x-column", default="x_centroid")
    parser.add_argument("--y-column", default="y_centroid")
    parser.add_argument(
        "--label-property", help="Dotted GeoJSON property, e.g. classification.name"
    )
    parser.add_argument("--geojson-scale", type=float, default=1.0)
    parser.add_argument("--geojson-offset-x", type=float, default=0.0)
    parser.add_argument("--geojson-offset-y", type=float, default=0.0)
    parser.add_argument(
        "--alignment-matrix",
        type=Path,
        default=None,
        help="2x3 affine transform CSV (he_imagealignment.csv) from 10x; "
        "transforms GeoJSON polygons from H&E pixel space into Xenium micron "
        "space (multiplied by --pixel-size)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.2125,
        help="Xenium pixel size in microns (default 0.2125); used with --alignment-matrix",
    )
    parser.add_argument("--exclude-label", action="append", default=None)
    parser.add_argument("--min-cells-per-domain", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
