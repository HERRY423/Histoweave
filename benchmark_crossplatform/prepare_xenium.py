"""Prepare a checksumed Xenium breast-cancer bundle for the 7x15 benchmark.

The official Xenium matrix and cell metadata are intentionally supplied as inputs so
the script remains reproducible when 10x release URLs change. The metadata must contain
a cell id, x/y centroids, and a cell-type prediction column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _metadata(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _mapping(path: Path) -> tuple[str, dict[str, str]]:
    row = json.loads(path.read_text(encoding="utf-8"))["xenium_breast_cancer"]
    reverse = {
        str(source).casefold(): domain
        for domain, values in row["compartments"].items()
        for source in values
    }
    return row["input_column"], reverse


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
    meta = _metadata(args.metadata)
    id_column = args.id_column or next(
        (name for name in ("cell_id", "barcode", "CellID") if name in meta), None
    )
    if id_column is None:
        raise ValueError("metadata needs --id-column (for example cell_id)")
    meta[id_column] = meta[id_column].astype(str)
    meta = meta.set_index(id_column).reindex(adata.obs_names.astype(str))

    default_type, reverse = _mapping(args.mapping)
    type_column = args.cell_type_column or next(
        (
            name
            for name in (default_type, "cell_type_predicted", "cell_type", "cluster")
            if name in meta
        ),
        None,
    )
    if type_column is None:
        raise ValueError("metadata needs a cell-type column; pass --cell-type-column")
    for coordinate in (args.x_column, args.y_column):
        if coordinate not in meta:
            raise ValueError(f"metadata is missing coordinate column {coordinate!r}")

    source_labels = meta[type_column].astype("string")
    domains = source_labels.str.casefold().map(reverse).fillna("unmapped")
    keep = domains.ne("unmapped") & meta[[args.x_column, args.y_column]].notna().all(axis=1)
    adata = adata[keep.to_numpy()].copy()
    meta = meta.loc[keep]
    domains = domains.loc[keep]
    if adata.n_obs == 0:
        raise ValueError("no cells matched domain_mappings.json")

    idx = _stratified_indices(domains.reset_index(drop=True), args.max_cells, args.seed)
    adata = adata[idx].copy()
    meta = meta.iloc[idx]
    domains = domains.iloc[idx]
    adata.obs["cell_type"] = source_labels.loc[meta.index].astype(str).to_numpy()
    adata.obs["domain_truth"] = pd.Categorical(domains.to_numpy())
    adata.obsm["spatial"] = meta[[args.x_column, args.y_column]].to_numpy(dtype=np.float32)
    adata.layers["counts"] = adata.X.copy()
    adata.uns["schema_version"] = "histoweave.xenium.bundle.v1"
    adata.uns["source"] = "10x Xenium Human Breast Cancer Rep1"
    adata.uns["domain_mapping"] = str(args.mapping)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "xenium_breast_cancer",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_domains": int(adata.obs["domain_truth"].nunique()),
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
        "--metadata", type=Path, required=True, help="CSV(.gz) or Parquet cells table"
    )
    parser.add_argument(
        "--mapping", type=Path, default=root / "src/histoweave/datasets/domain_mappings.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "datasets_cache/xenium/xenium_breast_cancer.h5ad"
    )
    parser.add_argument("--id-column")
    parser.add_argument("--cell-type-column")
    parser.add_argument("--x-column", default="x_centroid")
    parser.add_argument("--y-column", default="y_centroid")
    parser.add_argument("--max-cells", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
