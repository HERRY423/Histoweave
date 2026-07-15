"""Combine MERFISH brain sections into a benchmark-ready, labelled h5ad bundle."""

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


def _collapse(labels: pd.Series, mapping_path: Path) -> pd.Series:
    row = json.loads(mapping_path.read_text(encoding="utf-8"))["merfish_mouse_brain"]
    patterns = [
        (str(pattern).casefold(), domain)
        for domain, values in row["compartments"].items()
        for pattern in values
    ]

    def one(value: object) -> str:
        text = str(value).casefold()
        return next((domain for pattern, domain in patterns if pattern in text), "other")

    return labels.map(one)


def _subsample(adata, labels: pd.Series, limit: int, seed: int):
    if adata.n_obs <= limit:
        return adata, labels
    rng = np.random.default_rng(seed)
    selected = []
    for indices in labels.groupby(labels, observed=True).indices.values():
        quota = max(1, round(len(indices) / len(labels) * limit))
        selected.extend(rng.choice(indices, min(quota, len(indices)), False).tolist())
    selected = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected) > limit:
        selected = np.sort(rng.choice(selected, limit, False))
    return adata[selected].copy(), labels.iloc[selected]


def build(args: argparse.Namespace) -> dict[str, object]:
    import anndata as ad

    sections = []
    for number, path in enumerate(args.section):
        data = ad.read_h5ad(path)
        if "spatial" not in data.obsm:
            raise ValueError(f"{path} is missing obsm['spatial']")
        if args.region_column and args.region_column in data.obs:
            labels = data.obs[args.region_column].astype(str)
        elif args.label_column in data.obs:
            labels = _collapse(data.obs[args.label_column], args.mapping)
        else:
            raise ValueError(f"{path} is missing {args.label_column!r}")
        data, labels = _subsample(
            data, labels.reset_index(drop=True), args.max_per_section, args.seed + number
        )
        data.obs["domain_truth"] = pd.Categorical(labels.to_numpy())
        data.obs["source_section"] = path.stem
        data.layers["counts"] = data.layers.get("counts", data.X.copy())
        sections.append(data)
    combined = ad.concat(sections, join="inner", merge="same", index_unique="-")
    combined.uns["schema_version"] = "histoweave.merfish.brain.bundle.v1"
    combined.uns["source_sections"] = [str(path) for path in args.section]
    combined.uns["domain_mapping"] = str(args.mapping)
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
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each anterior-section h5ad",
    )
    parser.add_argument("--label-column", default="subclass")
    parser.add_argument(
        "--region-column", help="Use an anatomical region column directly when available"
    )
    parser.add_argument(
        "--mapping", type=Path, default=root / "src/histoweave/datasets/domain_mappings.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "datasets_cache/merfish/merfish_mouse_brain.h5ad"
    )
    parser.add_argument("--max-per-section", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
