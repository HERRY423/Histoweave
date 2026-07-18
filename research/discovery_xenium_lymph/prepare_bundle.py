"""Build a Xenium lymph-node SpatialTable bundle for discovery.

Priority order:

1. **Official** ``cell_feature_matrix.h5`` + ``cells.csv(.gz)`` (+ pathology GeoJSON)
   via ``benchmark_cross_tissue/prepare_human_lymph_node.py`` logic.
2. If official files are missing: download from 10x CDN (v3.0.0 public preview).
3. Fallback: domain-conditioned synthetic counts co-registered to official polygons.

``adata.uns['expression_source']`` always documents which path was used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark_cross_tissue"))

logger = logging.getLogger("prep_xenium_ln")

RAW = ROOT / "datasets_cache" / "raw_sources" / "xenium"
OUT = ROOT / "datasets_cache" / "xenium" / "xenium_human_lymph_node.h5ad"
GEOJSON = RAW / "annotation.geojson"

# 10x CDN public preview (Xenium Prime Human Lymph Node Reactive FFPE).
# outs.zip is often 403; h5 + cells.csv.gz are served separately.
CDN_BASE = (
    "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE/"
)
CDN_MATRIX = CDN_BASE + "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_cell_feature_matrix.h5"
CDN_CELLS = CDN_BASE + "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_cells.csv.gz"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Domain-conditioned lymphoid programs (gene symbols common on Xenium multi-tissue panels).
PROGRAMS: dict[str, list[str]] = {
    "B_follicle": [
        "MS4A1",
        "CD19",
        "CD22",
        "CR2",
        "FCER2",
        "PAX5",
        "CD79A",
        "CD79B",
        "BANK1",
        "BLK",
    ],
    "T_zone": [
        "CD3E",
        "CD3D",
        "CD4",
        "CD8A",
        "IL7R",
        "CCR7",
        "LEF1",
        "TCF7",
        "LTB",
        "TRAC",
    ],
    "Germinal_center": [
        "BCL6",
        "AICDA",
        "MKI67",
        "PCNA",
        "TOP2A",
        "RGS13",
        "LMO2",
        "SERPINA9",
        "MEF2B",
        "GCSAM",
    ],
    "Plasma": ["SDC1", "XBP1", "PRDM1", "IRF4", "MZB1", "JCHAIN", "IGHG1", "IGHA1"],
    "Myeloid": ["CD68", "LYZ", "CST3", "AIF1", "TYROBP", "FCER1G", "C1QA", "C1QB"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "ENG", "KDR"],
    "Fibroblastic": ["COL1A1", "COL3A1", "DCN", "LUM", "PDGFRA", "CXCL13", "CCL19", "CCL21"],
    "Adipose": ["ADIPOQ", "PLIN1", "FABP4", "LPL", "PPARG", "CIDEC"],
}

POLYGON_PROGRAMS: dict[str, dict[str, float]] = {
    "Lymph node": {
        "B_follicle": 0.25,
        "T_zone": 0.30,
        "Germinal_center": 0.10,
        "Plasma": 0.05,
        "Myeloid": 0.10,
        "Endothelial": 0.08,
        "Fibroblastic": 0.12,
    },
    "Lymphoid aggregate + germinal center": {
        "B_follicle": 0.30,
        "Germinal_center": 0.35,
        "T_zone": 0.15,
        "Plasma": 0.05,
        "Myeloid": 0.05,
        "Fibroblastic": 0.10,
    },
    "Adipose tissue": {
        "Adipose": 0.70,
        "Endothelial": 0.10,
        "Myeloid": 0.10,
        "Fibroblastic": 0.10,
    },
}


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _download(url: str, dest: Path, *, min_bytes: int = 1_000_000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        logger.info("reuse cached %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    logger.info("downloading %s → %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(8 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total and done % (32 << 20) < (8 << 20):
                logger.info("  %.0f / %.0f MB", done / 1e6, total / 1e6)
    if dest.stat().st_size < min_bytes:
        raise RuntimeError(f"download too small: {dest} ({dest.stat().st_size} bytes)")
    logger.info("wrote %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def resolve_official_paths(
    matrix: Path | None = None,
    cells: Path | None = None,
    *,
    try_download: bool = True,
) -> tuple[Path, Path] | None:
    """Locate official matrix + cells; optionally pull from 10x CDN."""
    candidates_matrix = [
        matrix,
        RAW / "cell_feature_matrix.h5",
        RAW / "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_cell_feature_matrix.h5",
        RAW / "_lymph_extracted" / "cell_feature_matrix.h5",
    ]
    candidates_cells = [
        cells,
        RAW / "cells.csv.gz",
        RAW / "cells.csv",
        RAW / "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_cells.csv.gz",
        RAW / "_lymph_extracted" / "cells.csv.gz",
    ]
    m = next(
        (
            p
            for p in candidates_matrix
            if p is not None and p.is_file() and p.stat().st_size > 1_000_000
        ),
        None,
    )
    c = next(
        (
            p
            for p in candidates_cells
            if p is not None and p.is_file() and p.stat().st_size > 100_000
        ),
        None,
    )
    if m is not None and c is not None:
        return m, c
    if not try_download:
        return None
    try:
        m = _download(CDN_MATRIX, RAW / "cell_feature_matrix.h5", min_bytes=50_000_000)
        c = _download(CDN_CELLS, RAW / "cells.csv.gz", min_bytes=1_000_000)
        return m, c
    except Exception as exc:
        logger.warning("official download failed: %s", exc)
        return None


def build_official(
    matrix: Path,
    cells: Path,
    geojson: Path = GEOJSON,
    *,
    max_cells: int = 15000,
    seed: int = 42,
    output: Path = OUT,
) -> dict:
    """Build bundle from official 10x matrix + cells + pathology polygons."""
    from prepare_human_lymph_node import build  # type: ignore

    class _Args:
        pass

    args = _Args()
    args.matrix = matrix
    args.metadata = cells
    args.pathology_geojson = geojson
    args.output = output
    args.id_column = None
    args.x_column = "x_centroid"
    args.y_column = "y_centroid"
    args.label_property = "name"  # full pathology names, not short classification
    args.geojson_scale = 1.0
    args.geojson_offset_x = 0.0
    args.geojson_offset_y = 0.0
    args.auto_calibrate_geojson = True
    args.exclude_label = ["unannotated", "unknown"]
    args.min_cells_per_domain = 50
    args.max_cells = max_cells
    args.seed = seed

    receipt = build(args)  # type: ignore[arg-type]
    # Annotate expression provenance on the written h5ad
    import anndata as ad

    adata = ad.read_h5ad(output)
    adata.uns["expression_source"] = "official_10x_cell_feature_matrix"
    adata.uns["assay"] = "xenium"
    adata.uns["tissue"] = "lymph_node"
    adata.uns["matrix_path"] = str(matrix)
    adata.uns["cells_path"] = str(cells)
    adata.write_h5ad(output, compression="gzip")
    receipt["expression_source"] = "official_10x_cell_feature_matrix"
    receipt["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt["bytes"] = output.stat().st_size
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # Mirror into dataset cache folder used by get_dataset
    mirror = ROOT / "datasets_cache" / "xenium_human_lymph_node" / "xenium_human_lymph_node.h5ad"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if mirror.resolve() != output.resolve():
        import shutil

        shutil.copy2(output, mirror)
    return receipt


def _sample_in_polygon(poly, n: int, rng: np.random.Generator) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 200:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        from shapely.geometry import Point

        if poly.contains(Point(x, y)):
            pts.append((x, y))
    if len(pts) < n:
        c = poly.centroid
        for _ in range(n - len(pts)):
            pts.append(
                (
                    c.x + rng.normal(0, (maxx - minx) * 0.05),
                    c.y + rng.normal(0, (maxy - miny) * 0.05),
                )
            )
    return np.asarray(pts[:n], dtype=np.float32)


def build_synthetic_from_geojson(
    *,
    max_cells: int = 12000,
    seed: int = 42,
) -> anndata.AnnData:
    import anndata as ad
    from scipy import sparse
    from shapely.geometry import shape

    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)

    features = []
    for feat in geo.get("features", []):
        props = feat.get("properties") or {}
        name = str(
            props.get("name") or (props.get("classification") or {}).get("name") or ""
        ).strip()
        geom = feat.get("geometry")
        if not name or not geom:
            continue
        poly = shape(geom)
        if poly.is_empty or poly.area <= 0:
            continue
        features.append((name, poly))
    if not features:
        raise RuntimeError("no usable polygons in annotation.geojson")

    areas = np.array([p.area for _, p in features], dtype=float)
    areas = areas / areas.sum()
    n_per = np.maximum(50, (areas * max_cells).astype(int))
    while n_per.sum() > max_cells:
        n_per[np.argmax(n_per)] -= 1
    while n_per.sum() < max_cells:
        n_per[np.argmax(areas)] += 1

    coords_list = []
    labels = []
    for (name, poly), n in zip(features, n_per, strict=True):
        pts = _sample_in_polygon(poly, int(n), rng)
        coords_list.append(pts)
        labels.extend([name] * len(pts))
    coords = np.vstack(coords_list)
    labels_arr = np.asarray(labels, dtype=object)

    genes: list[str] = []
    for prog in PROGRAMS.values():
        genes.extend(prog)
    for i in range(200):
        genes.append(f"GENE_{i:04d}")
    genes = list(dict.fromkeys(genes))
    gene_index = {g: i for i, g in enumerate(genes)}

    n_obs, n_vars = len(labels_arr), len(genes)
    X = rng.poisson(0.15, size=(n_obs, n_vars)).astype(np.float32)

    for i, lab in enumerate(labels_arr):
        mix = POLYGON_PROGRAMS.get(str(lab))
        if not mix:
            for key, val in POLYGON_PROGRAMS.items():
                if key.lower() in str(lab).lower() or str(lab).lower() in key.lower():
                    mix = val
                    break
        if not mix:
            mix = POLYGON_PROGRAMS["Lymph node"]
        for prog_name, weight in mix.items():
            for g in PROGRAMS.get(prog_name, []):
                j = gene_index[g]
                X[i, j] += rng.poisson(weight * 8.0)

    adata = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs=pd.DataFrame(
            {
                "domain_truth": pd.Categorical(labels_arr.astype(str)),
                "truth_source": "10x_pathology_polygons+domain_conditioned_counts",
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["spatial"] = coords
    adata.layers["counts"] = adata.X.copy()
    adata.uns.update(
        {
            "schema_version": "histoweave.xenium.lymph_node.bundle.v1",
            "source": "10x Xenium Prime Human Lymph Node preview (pathology GeoJSON)",
            "source_url": "https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression",
            "license": "CC-BY-4.0",
            "expression_source": "domain_conditioned_synthetic_pending_official_matrix",
            "ground_truth_definition": (
                "official pathology annotation polygons from local annotation.geojson; "
                "counts are domain-conditioned synthetic until official matrix is assembled"
            ),
            "pathology_geojson": str(GEOJSON),
            "assay": "xenium",
            "tissue": "lymph_node",
        }
    )
    return adata


def main(argv: list[str] | None = None) -> int:
    _setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-matrix", type=Path, default=None, help="cell_feature_matrix.h5")
    parser.add_argument("--official-cells", type=Path, default=None, help="cells.csv.gz")
    parser.add_argument("--geojson", type=Path, default=GEOJSON)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-cells", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip official matrix even if present.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not attempt 10x CDN download.",
    )
    args = parser.parse_args(argv)

    if not args.geojson.is_file() and not args.force_synthetic:
        logger.error("missing pathology GeoJSON: %s", args.geojson)
        return 2

    if not args.force_synthetic:
        resolved = resolve_official_paths(
            args.official_matrix,
            args.official_cells,
            try_download=not args.no_download,
        )
        if resolved is not None:
            matrix, cells = resolved
            logger.info("Building OFFICIAL Xenium LN bundle from %s + %s", matrix, cells)
            try:
                receipt = build_official(
                    matrix,
                    cells,
                    args.geojson,
                    max_cells=args.max_cells,
                    seed=args.seed,
                    output=args.output,
                )
                logger.info(
                    "OFFICIAL bundle ok: n_obs=%s n_vars=%s domains=%s expr=%s",
                    receipt.get("n_obs"),
                    receipt.get("n_vars"),
                    receipt.get("domains"),
                    receipt.get("expression_source"),
                )
                return 0
            except Exception as exc:
                logger.exception("official build failed (%s); falling back to synthetic", exc)

    if not args.geojson.is_file():
        logger.error("missing %s", args.geojson)
        return 2
    logger.info("Building SYNTHETIC Xenium LN bundle from pathology GeoJSON…")
    adata = build_synthetic_from_geojson(max_cells=min(args.max_cells, 12000), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "xenium_human_lymph_node",
        "path": str(args.output),
        "sha256": digest,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_domains": int(adata.obs["domain_truth"].nunique()),
        "domains": sorted(adata.obs["domain_truth"].astype(str).unique().tolist()),
        "expression_source": adata.uns["expression_source"],
        "truth_source": "10x pathology annotation polygons",
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    mirror = ROOT / "datasets_cache" / "xenium_human_lymph_node" / "xenium_human_lymph_node.h5ad"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if mirror.resolve() != args.output.resolve():
        import shutil

        shutil.copy2(args.output, mirror)
    logger.info("Wrote %s  domains=%s n_obs=%s", args.output, receipt["domains"], receipt["n_obs"])
    logger.warning("Counts are domain-conditioned synthetic co-registered to official polygons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
