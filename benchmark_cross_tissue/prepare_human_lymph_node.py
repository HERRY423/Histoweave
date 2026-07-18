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
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse one tested pathology-polygon implementation across Xenium preparers.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from histoweave.datasets.pathology_domains import (  # noqa: E402
    assign_pathology_domains,
    feature_label,
)

SOURCE_URL = "https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression"


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


def _metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _stratified_indices(
    labels: pd.Series,
    limit: int,
    seed: int,
    *,
    min_per_domain: int = 50,
) -> np.ndarray:
    """Stratified subsample with a floor per domain so rare GC polygons survive.

    Plain proportional quotas map a 0.1% domain onto ~1 cell at limit=12k,
    which destroys GC/aggregate signal needed for deep-dives.
    """
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups = {
        str(k): np.asarray(v, dtype=int)
        for k, v in labels.groupby(labels, observed=True).indices.items()
    }
    n_groups = max(len(groups), 1)
    floor = min(min_per_domain, max(1, limit // (n_groups * 2)))
    # First pass: guaranteed floor (or all cells if domain smaller).
    selected: list[np.ndarray] = []
    remaining_budget = limit
    leftovers: dict[str, np.ndarray] = {}
    for key, indices in groups.items():
        take = min(len(indices), floor, remaining_budget)
        chosen = rng.choice(indices, take, replace=False)
        selected.append(chosen)
        remaining_budget -= take
        rest = np.setdiff1d(indices, chosen, assume_unique=False)
        if len(rest):
            leftovers[key] = rest
    # Second pass: fill proportionally from leftovers.
    if remaining_budget > 0 and leftovers:
        rest_sizes = {k: len(v) for k, v in leftovers.items()}
        rest_total = sum(rest_sizes.values()) or 1
        for key, rest in leftovers.items():
            if remaining_budget <= 0:
                break
            quota = max(0, round(rest_sizes[key] / rest_total * remaining_budget))
            quota = min(quota, len(rest), remaining_budget)
            if quota <= 0:
                continue
            selected.append(rng.choice(rest, quota, replace=False))
            remaining_budget -= quota
            leftovers[key] = np.setdiff1d(rest, selected[-1], assume_unique=False)
    # Third pass: dump any leftover budget into largest remaining pools.
    if remaining_budget > 0:
        pool = (
            np.concatenate([v for v in leftovers.values() if len(v)])
            if leftovers
            else np.array([], dtype=int)
        )
        if len(pool):
            selected.append(rng.choice(pool, min(remaining_budget, len(pool)), replace=False))
    merged = np.unique(np.concatenate(selected)) if selected else np.arange(0)
    if len(merged) > limit:
        # Never drop below floor for any domain if possible.
        keep: list[np.ndarray] = []
        for _key, indices in groups.items():
            in_merged = np.intersect1d(indices, merged, assume_unique=False)
            if len(in_merged) <= floor:
                keep.append(in_merged)
            else:
                keep.append(
                    rng.choice(
                        in_merged,
                        max(floor, round(len(in_merged) / len(merged) * limit)),
                        replace=False,
                    )
                )
        merged = np.unique(np.concatenate(keep))
        if len(merged) > limit:
            # Trim from largest domains only.
            counts = {k: len(np.intersect1d(g, merged)) for k, g in groups.items()}
            merged_set = set(int(i) for i in merged)
            while len(merged_set) > limit:
                largest = max(counts, key=counts.get)
                candidates = [i for i in groups[largest] if i in merged_set]
                if len(candidates) <= floor:
                    counts[largest] = -1
                    if all(v < 0 for v in counts.values()):
                        break
                    continue
                drop = int(rng.choice(candidates))
                merged_set.remove(drop)
                counts[largest] -= 1
            merged = np.array(sorted(merged_set), dtype=int)
    return np.sort(merged)


def calibrate_geojson_transform(
    cell_xy: np.ndarray,
    geojson: dict[str, Any],
    *,
    label_property: str | None = None,
    max_sample: int = 40_000,
    seed: int = 0,
) -> tuple[float, float, float, dict[str, object]]:
    """Estimate scale (and small offset) mapping cell microns → GeoJSON pixels.

    Xenium ``cells.csv`` centroids are typically microns; pathology annotations
    exported from Xenium Explorer are often image-pixel coordinates. A pure
    identity transform frequently labels zero cells.
    """
    from shapely import contains_xy
    from shapely.geometry import shape

    xy = np.asarray(cell_xy, dtype=float)
    rng = np.random.default_rng(seed)
    if len(xy) > max_sample:
        idx = rng.choice(len(xy), max_sample, replace=False)
        sample = xy[idx]
    else:
        sample = xy

    polys: list[tuple[str, Any]] = []
    for feature in geojson.get("features", []):
        label = feature_label(feature, label_property)
        geometry = feature.get("geometry")
        if label is None or not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        polys.append((label, shape(geometry)))
    if not polys:
        raise ValueError("GeoJSON has no labelled polygons for calibration")

    def _count(scale: float, ox: float = 0.0, oy: float = 0.0) -> tuple[int, int, int]:
        sx = sample[:, 0] * scale + ox
        sy = sample[:, 1] * scale + oy
        labels = np.full(len(sample), None, dtype=object)
        for lab, poly in polys:
            inside = np.asarray(contains_xy(poly, sx, sy), dtype=bool)
            empty = inside & pd.isna(labels)
            conflict = inside & ~pd.isna(labels) & (labels != lab)
            labels[empty] = lab
            labels[conflict] = "ambiguous"
        labelled = int((~pd.isna(labels)).sum())
        ambiguous = int((labels == "ambiguous").sum())
        good = labelled - ambiguous
        n_domains = int(pd.Series(labels).dropna().loc[lambda s: s != "ambiguous"].nunique())
        return good, ambiguous, n_domains

    # Coarse scale sweep (covers ~0.2125 µm/px ↔ ~pixel/micron inverses).
    candidates: list[tuple[tuple[int, int, int], float, float, float]] = []
    for scale in np.linspace(2.0, 5.5, 36):
        good, amb, n_dom = _count(float(scale))
        candidates.append(((good, -amb, n_dom), float(scale), 0.0, 0.0))
    # Prefer physically common 1/0.2125 if competitive.
    for scale in (1.0 / 0.2125, 1.0 / 0.2125 * 0.5, 0.2125, 1.0):
        good, amb, n_dom = _count(float(scale))
        candidates.append(((good, -amb, n_dom), float(scale), 0.0, 0.0))

    candidates.sort(key=lambda row: row[0], reverse=True)
    best_key, best_scale, best_ox, best_oy = candidates[0]

    # Local offset refine around best scale (small grid).
    for ox in np.linspace(-1500, 1500, 7):
        for oy in np.linspace(-1500, 1500, 7):
            good, amb, n_dom = _count(best_scale, float(ox), float(oy))
            key = (good, -amb, n_dom)
            if key > best_key:
                best_key, best_ox, best_oy = key, float(ox), float(oy)

    # Fine scale refine with fixed best offset.
    for scale in np.linspace(best_scale * 0.9, best_scale * 1.1, 21):
        good, amb, n_dom = _count(float(scale), best_ox, best_oy)
        key = (good, -amb, n_dom)
        if key > best_key:
            best_key, best_scale = key, float(scale)

    good, amb, n_dom = best_key[0], -best_key[1], best_key[2]
    meta = {
        "good_labelled_sample": good,
        "ambiguous_sample": amb,
        "n_domains_sample": n_dom,
        "sample_size": int(len(sample)),
        "label_fraction_sample": good / max(len(sample), 1),
    }
    if good < max(50, int(0.05 * len(sample))):
        raise ValueError(
            f"geojson calibration failed: only {good}/{len(sample)} sample cells labelled "
            f"(scale={best_scale}, offset=({best_ox},{best_oy}))"
        )
    return best_scale, best_ox, best_oy, meta


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
    geojson = json.loads(args.pathology_geojson.read_text(encoding="utf-8"))

    scale = float(args.geojson_scale)
    ox = float(args.geojson_offset_x)
    oy = float(args.geojson_offset_y)
    calib_meta: dict[str, object] | None = None
    if getattr(args, "auto_calibrate_geojson", False) or (
        scale == 1.0
        and ox == 0.0
        and oy == 0.0
        and getattr(args, "auto_calibrate_geojson", True) is not False
    ):
        # Default: auto-calibrate when user left identity transform (common failure mode).
        try:
            scale, ox, oy, calib_meta = calibrate_geojson_transform(
                cell_xy,
                geojson,
                label_property=args.label_property or "name",
                seed=int(args.seed),
            )
            _log(
                f"geojson auto-calibrate: scale={scale:.4f} offset=({ox:.1f},{oy:.1f}) "
                f"sample_label_frac={calib_meta.get('label_fraction_sample')}"
            )
        except Exception as exc:
            _log(f"geojson auto-calibrate skipped/failed: {exc}")

    annotation_xy = cell_xy.copy()
    annotation_xy[:, 0] = annotation_xy[:, 0] * scale + ox
    annotation_xy[:, 1] = annotation_xy[:, 1] * scale + oy
    domains = assign_pathology_domains(
        annotation_xy, geojson, label_property=args.label_property or "name"
    )

    excluded = {value.casefold() for value in args.exclude_label}
    # Force pure bool masks — pandas StringDtype can yield object arrays that
    # break numpy ``&=`` (Cannot cast ufunc 'bitwise_and' output from dtype('O')).
    xy_ok = meta[[args.x_column, args.y_column]].notna().all(axis=1).to_numpy(dtype=bool)
    dom_ok = domains.notna().to_numpy(dtype=bool)
    not_ambiguous = (domains.fillna("").astype(str) != "ambiguous").to_numpy(dtype=bool)
    not_excluded = ~domains.fillna("").astype(str).str.casefold().isin(excluded).to_numpy(
        dtype=bool
    )
    valid = xy_ok & dom_ok & not_ambiguous & not_excluded
    counts = domains[valid].value_counts()
    retained = set(counts[counts >= args.min_cells_per_domain].index.astype(str))
    in_retained = domains.fillna("").astype(str).isin(retained).to_numpy(dtype=bool)
    valid = valid & in_retained
    if not valid.any():
        raise ValueError("no unambiguous pathology-labelled cells remained after filtering")

    adata = adata[valid].copy()
    meta = meta.iloc[np.flatnonzero(valid)]
    domains = domains.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    idx = _stratified_indices(
        domains,
        args.max_cells,
        args.seed,
        min_per_domain=max(50, int(getattr(args, "min_cells_per_domain", 50))),
    )
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
            "expression_source": "official_10x_cell_feature_matrix",
            "ground_truth_definition": "pathology polygons; unannotated/ambiguous cells excluded",
            "pathology_geojson": str(args.pathology_geojson),
            "geojson_transform": {
                "scale": scale,
                "offset_x": ox,
                "offset_y": oy,
                "auto_calibrate": calib_meta,
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
    parser.add_argument(
        "--auto-calibrate-geojson",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-estimate scale/offset when identity would label ~0 cells (default: on).",
    )
    parser.add_argument("--exclude-label", action="append", default=["unannotated", "unknown"])
    parser.add_argument("--min-cells-per-domain", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
