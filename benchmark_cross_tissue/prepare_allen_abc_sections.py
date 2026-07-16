"""Prepare an anatomical Allen MERFISH benchmark bundle from official ABC files.

This reader joins raw per-section MERFISH count h5ad files with the official
CCF parcellation table without loading the full 4-million-cell metadata table
into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://alleninstitute.github.io/abc_atlas_access/notebooks/merfish_tutorial_part_1.html"
)
INVALID_LABELS = {"", "nan", "none", "unknown", "unassigned", "unmapped", "fiber tracts"}


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


def _stratified_indices(labels: pd.Series, limit: int, seed: int) -> np.ndarray:
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for indices in labels.groupby(labels, observed=True).indices.values():
        quota = max(1, round(len(indices) / len(labels) * limit))
        selected.append(rng.choice(indices, min(quota, len(indices)), replace=False))
    merged = np.unique(np.concatenate(selected))
    if len(merged) > limit:
        merged = rng.choice(merged, limit, replace=False)
    elif len(merged) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), merged, assume_unique=False)
        merged = np.concatenate([merged, rng.choice(remaining, limit - len(merged), False)])
    return np.sort(merged)


def _collect_cell_ids(section_paths: list[Path]) -> set[str]:
    import anndata as ad

    identifiers: set[str] = set()
    for path in section_paths:
        data = ad.read_h5ad(path, backed="r")
        try:
            identifiers.update(data.obs_names.astype(str))
        finally:
            data.file.close()
    return identifiers


def _read_selected_metadata(
    path: Path,
    cell_ids: set[str],
    *,
    id_column: str,
    region_column: str,
    x_column: str,
    y_column: str,
    chunksize: int,
) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    required = [id_column, region_column, x_column, y_column]
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Allen CCF metadata is missing required columns: {missing}")
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=required, chunksize=chunksize):
        chunk[id_column] = chunk[id_column].astype(str)
        matched = chunk[chunk[id_column].isin(cell_ids)]
        if not matched.empty:
            selected.append(matched)
    if not selected:
        raise ValueError("no raw section cells were found in the CCF metadata")
    metadata = pd.concat(selected, ignore_index=True)
    if metadata[id_column].duplicated().any():
        raise ValueError("CCF metadata has duplicate cell identifiers")
    return metadata.set_index(id_column)


def build(args: argparse.Namespace) -> dict[str, object]:
    import anndata as ad

    cell_ids = _collect_cell_ids(args.section)
    metadata = _read_selected_metadata(
        args.metadata,
        cell_ids,
        id_column=args.id_column,
        region_column=args.region_column,
        x_column=args.x_column,
        y_column=args.y_column,
        chunksize=args.metadata_chunksize,
    )
    sections = []
    n_original = 0
    for number, path in enumerate(args.section):
        data = ad.read_h5ad(path)
        n_original += int(data.n_obs)
        matched = metadata.reindex(data.obs_names.astype(str))
        labels = matched[args.region_column].astype("string")
        valid = labels.notna().to_numpy()
        valid &= ~labels.fillna("").str.strip().str.casefold().isin(INVALID_LABELS).to_numpy()
        valid &= matched[[args.x_column, args.y_column]].notna().all(axis=1).to_numpy()
        data = data[valid].copy()
        matched = matched.iloc[np.flatnonzero(valid)]
        labels = labels.iloc[np.flatnonzero(valid)].reset_index(drop=True)
        if data.n_obs == 0:
            raise ValueError(f"{path} has no CCF-labelled cells after filtering")
        n_domains = int(labels.nunique())
        if n_domains < 2 or n_domains > args.max_domains:
            raise ValueError(
                f"{path}: {args.region_column!r} has {n_domains} domains; "
                f"expected 2..{args.max_domains}."
            )
        keep = _stratified_indices(labels, args.max_per_section, args.seed + number)
        data = data[keep].copy()
        matched = matched.iloc[keep]
        labels = labels.iloc[keep]
        data.obs["domain_truth"] = pd.Categorical(labels.to_numpy())
        data.obs["truth_source"] = "allen_ccf_anatomical"
        data.obs["source_section"] = path.stem
        data.obsm["spatial"] = matched[[args.x_column, args.y_column]].to_numpy(dtype=np.float32)
        data.layers["counts"] = data.layers.get("counts", data.X.copy())
        sections.append(data)

    combined = ad.concat(sections, join="inner", merge="same", index_unique="-")
    combined.uns.update(
        {
            "schema_version": "histoweave.allen.merfish.abc.bundle.v1",
            "source": "Allen Brain Cell Atlas MERFISH-C57BL6J-638850 raw counts",
            "source_url": SOURCE_URL,
            "license": "CC-BY-NC-4.0",
            "source_sections": [str(path) for path in args.section],
            "ccf_metadata": str(args.metadata),
            "truth_source": "allen_ccf_anatomical",
            "truth_column": args.region_column,
            "n_original": n_original,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "merfish_mouse_brain",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": combined.n_obs,
        "n_vars": combined.n_vars,
        "n_domains": int(combined.obs["domain_truth"].nunique()),
        "sections": len(sections),
        "truth_source": "allen_ccf_anatomical",
        "truth_column": args.region_column,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--id-column", default="cell_label")
    parser.add_argument("--region-column", default="parcellation_division")
    parser.add_argument("--x-column", default="x_section")
    parser.add_argument("--y-column", default="y_section")
    parser.add_argument("--metadata-chunksize", type=int, default=250_000)
    parser.add_argument("--max-domains", type=int, default=50)
    parser.add_argument("--max-per-section", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets_cache/merfish/merfish_mouse_brain.h5ad",
    )
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
