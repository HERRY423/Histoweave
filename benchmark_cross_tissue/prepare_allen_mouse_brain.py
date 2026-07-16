"""Build an Allen Brain Atlas MERFISH bundle with anatomical ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://alleninstitute.org/education-resources/database-guide-abc-atlas"
ANATOMICAL_COLUMNS = (
    "parcellation_division",
    "parcellation_structure",
    "parcellation_substructure",
    "CCF_region",
    "ccf_region",
    "region",
)
INVALID_LABELS = {"", "nan", "none", "unknown", "unassigned", "unmapped", "fiber tracts"}


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


def _collapse_cell_classes(labels: pd.Series, mapping_path: Path) -> pd.Series:
    row = json.loads(mapping_path.read_text(encoding="utf-8"))["merfish_mouse_brain"]
    patterns = [
        (str(pattern).casefold(), domain)
        for domain, values in row["compartments"].items()
        for pattern in values
    ]

    def one(value: object) -> str:
        text = str(value).casefold()
        return next((domain for pattern, domain in patterns if pattern in text), "unmapped")

    return labels.map(one).astype("string")


def resolve_truth(data, args: argparse.Namespace) -> tuple[pd.Series, str, str]:
    """Resolve labels, source type and source column without label leakage."""
    if args.region_column:
        if args.region_column not in data.obs:
            raise ValueError(f"requested region column {args.region_column!r} is missing")
        column = args.region_column
    else:
        column = next((name for name in ANATOMICAL_COLUMNS if name in data.obs), None)
    if column is not None:
        return data.obs[column].astype("string"), "allen_ccf_anatomical", column
    if not args.allow_cell_class_fallback:
        raise ValueError(
            "no Allen anatomical CCF column found; pass --region-column. "
            "Cell-class labels are not accepted as primary spatial-domain truth."
        )
    if args.label_column not in data.obs:
        raise ValueError(f"cell-class fallback column {args.label_column!r} is missing")
    return (
        _collapse_cell_classes(data.obs[args.label_column], args.mapping),
        "cell_class_sensitivity_fallback",
        args.label_column,
    )


def _valid_mask(labels: pd.Series) -> np.ndarray:
    return (
        labels.notna().to_numpy()
        & ~labels.fillna("").str.strip().str.casefold().isin(INVALID_LABELS).to_numpy()
    )


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


def build(args: argparse.Namespace) -> dict[str, object]:
    import anndata as ad

    sections = []
    truth_sources: set[str] = set()
    source_columns: set[str] = set()
    n_original = 0
    for number, path in enumerate(args.section):
        data = ad.read_h5ad(path)
        n_original += int(data.n_obs)
        if "spatial" not in data.obsm:
            raise ValueError(f"{path} is missing obsm['spatial']")
        labels, truth_source, source_column = resolve_truth(data, args)
        valid = _valid_mask(labels)
        data = data[valid].copy()
        labels = labels.iloc[np.flatnonzero(valid)].reset_index(drop=True)
        if data.n_obs == 0:
            raise ValueError(f"{path} has no valid anatomical labels")
        n_domains = int(labels.nunique())
        if n_domains < 2 or n_domains > args.max_domains:
            raise ValueError(
                f"{path}: {source_column!r} has {n_domains} domains; "
                f"expected 2..{args.max_domains}. "
                "Choose a coarser Allen CCF column such as parcellation_division."
            )
        keep = _stratified_indices(labels, args.max_per_section, args.seed + number)
        data = data[keep].copy()
        labels = labels.iloc[keep]
        data.obs["domain_truth"] = pd.Categorical(labels.to_numpy())
        data.obs["truth_source"] = truth_source
        data.obs["source_section"] = path.stem
        data.layers["counts"] = data.layers.get("counts", data.X.copy())
        sections.append(data)
        truth_sources.add(truth_source)
        source_columns.add(source_column)

    if len(truth_sources) != 1 or len(source_columns) != 1:
        raise ValueError("all sections must use the same truth source and anatomical column")
    combined = ad.concat(sections, join="inner", merge="same", index_unique="-")
    combined.uns.update(
        {
            "schema_version": "histoweave.allen.merfish.brain.bundle.v2",
            "source": "Allen Brain Cell Atlas whole mouse brain MERFISH",
            "source_url": SOURCE_URL,
            "license": "CC-BY-NC-4.0",
            "source_sections": [str(path) for path in args.section],
            "truth_source": next(iter(truth_sources)),
            "truth_column": next(iter(source_columns)),
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
        "domains": sorted(combined.obs["domain_truth"].astype(str).unique().tolist()),
        "sections": len(sections),
        "truth_source": next(iter(truth_sources)),
        "truth_column": next(iter(source_columns)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", type=Path, action="append", required=True, help="Repeat for each section h5ad"
    )
    parser.add_argument(
        "--region-column", help="Allen CCF anatomical column; auto-detected if omitted"
    )
    parser.add_argument("--max-domains", type=int, default=50)
    parser.add_argument("--max-per-section", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cell-class-fallback",
        action="store_true",
        help="Sensitivity analysis only; never treated as primary anatomical validation",
    )
    parser.add_argument("--label-column", default="subclass")
    parser.add_argument(
        "--mapping", type=Path, default=root / "src/histoweave/datasets/domain_mappings.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "datasets_cache/merfish/merfish_mouse_brain.h5ad"
    )
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
