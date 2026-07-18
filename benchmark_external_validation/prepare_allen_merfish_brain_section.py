"""Prepare a single Allen Brain Cell Atlas MERFISH section with anatomical ground truth.

Source: Allen Brain Cell Atlas — whole mouse brain MERFISH (Yao et al. 2023),
hosted as an AWS Public Dataset (``arn:aws:s3:::allen-brain-cell-atlas``).
Access is via the ``abc_atlas_access`` AbcProjectCache (no account needed).

Ground truth comes ONLY from the Allen CCFv3 anatomical parcellation
(``parcellation_division`` by default — the coarsest anatomical level, e.g.
Isocortex, Hippocampal formation, Olfactory areas, ...). These are anatomical
region labels registered to the Allen Common Coordinate Framework — never
cell-type predictions — so they satisfy HistoWeave's strict spatial-domain
ground-truth policy. This mirrors ``benchmark_cross_tissue/prepare_allen_mouse_brain.py``
but targets a *single* coronal section (rather than concatenating several) so
the external-validation landscape gets an independent mouse-brain entry whose
feature profile differs from the multi-section cross-tissue bundle.

The script downloads one MERFISH coronal section's cell metadata + expression
h5ad via AbcProjectCache, joins the anatomical parcellation column, attaches
the reconstructed ``obsm['spatial']`` (x/y in the CCF-registered frame),
stratified-subsamples to a tractable size, and writes a checksummed ``.h5ad``.

Note: not all sections have CCF parcellation annotations. Sections
``C57BL6J-638850.01``–``.04`` lack CCF mapping; the first annotated section is
``C57BL6J-638850.05``. Use ``--section-label C57BL6J-638850.05`` or
``--section-index 4`` to select a section with CCF ground truth.

Requires::

    pip install abc-atlas-access   # or: pip install "histoweave[spatial]"
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
# Anatomical CCF columns tried in order (matches the cross-tissue Allen preparer).
ANATOMICAL_COLUMNS = (
    "parcellation_division",
    "parcellation_structure",
    "parcellation_substructure",
    "CCF_region",
    "ccf_region",
    "region",
)
INVALID_LABELS = {"", "nan", "none", "unknown", "unassigned", "unmapped", "fiber tracts"}
# Default MERFISH-C57BL6J-638850 dataset (whole mouse brain, Yao et al. 2023).
DEFAULT_DATASET = "MERFISH-C57BL6J-638850"
DEFAULT_MAX_CELLS = 15_000

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    _LOGGER.info("%s", message)


def _resolve_truth(obs: pd.DataFrame, region_column: str | None) -> tuple[pd.Series, str]:
    """Resolve the anatomical truth column; reject cell-class fallbacks."""
    if region_column:
        if region_column not in obs.columns:
            raise ValueError(f"requested region column {region_column!r} is missing")
        column = region_column
    else:
        column = next((name for name in ANATOMICAL_COLUMNS if name in obs.columns), None)
    if column is None:
        raise ValueError(
            "no Allen anatomical CCF column found; pass --region-column. "
            "Cell-class labels are not accepted as primary spatial-domain truth."
        )
    return obs[column].astype("string"), column


def _valid_mask(labels: pd.Series) -> np.ndarray:
    return (
        labels.notna().to_numpy()
        & ~labels.fillna("").str.strip().str.casefold().isin(INVALID_LABELS).to_numpy()
    )


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


def _load_section(args: argparse.Namespace):
    """Download + load one MERFISH coronal section as AnnData with cell metadata.

    Uses AbcProjectCache to fetch the MERFISH-C57BL6J-638850 log2 expression
    matrix and the cell metadata (which carries the CCF parcellation columns
    and reconstructed x/y coordinates). A single section is selected by
    ``--section-index`` (default 0 = first coronal section in the metadata).
    """
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    except ImportError as exc:
        raise ImportError(
            "Allen MERFISH preparation requires abc-atlas-access: "
            "pip install abc_atlas_access @ git+https://github.com/AllenInstitute/abc_atlas_access.git"
        ) from exc

    cache = AbcProjectCache.from_cache_dir(args.cache_dir)
    # Ensure the MERFISH dataset + CCF metadata are downloaded.
    manifest = cache.current_manifest
    _log(f"abc_atlas manifest: {manifest}")

    # Cell metadata for the whole-brain MERFISH dataset (carries brain_section_label).
    cell_meta = cache.get_metadata_dataframe(
        directory=args.dataset, file_name="cell_metadata", dtype={"cell_label": str}
    )
    # The CCF parcellation + reconstructed coordinates live in the CCF metadata
    # dir. ``cell_metadata_with_parcellation_annotation`` is a superset that
    # already includes parcellation_division/structure/substructure AND
    # x_reconstructed/y_reconstructed/z_reconstructed — so a single merge from
    # that file gives us everything we need.
    ccf_dir = f"{args.dataset}-CCF"
    try:
        ccf_meta = cache.get_metadata_dataframe(
            directory=ccf_dir,
            file_name="cell_metadata_with_parcellation_annotation",
            dtype={"cell_label": str},
        )
    except Exception:
        # Older manifests fold the CCF columns into the main cell metadata.
        ccf_meta = None

    if ccf_meta is not None:
        # Merge parcellation columns + reconstructed coordinates from ccf_meta.
        # ccf_meta uses x_reconstructed/y_reconstructed; rename to x/y for the
        # downstream code. Avoid column-name collisions with cell_meta's own x/y.
        merge_cols = [
            c
            for c in (
                "parcellation_division",
                "parcellation_structure",
                "parcellation_substructure",
                "x_reconstructed",
                "y_reconstructed",
                "z_reconstructed",
            )
            if c in ccf_meta.columns
        ]
        ccf_subset = ccf_meta[["cell_label"] + merge_cols].copy()
        # Rename reconstructed coords to x/y/z (dropping cell_meta's raw x/y first).
        rename_map = {
            "x_reconstructed": "x",
            "y_reconstructed": "y",
            "z_reconstructed": "z",
        }
        ccf_subset = ccf_subset.rename(
            columns={k: v for k, v in rename_map.items() if k in ccf_subset.columns}
        )
        # Drop cell_meta's own x/y/z to avoid _x/_y suffixes; we want the CCF ones.
        drop_cols = [c for c in ("x", "y", "z") if c in cell_meta.columns]
        if drop_cols:
            cell_meta = cell_meta.drop(columns=drop_cols)
        cell_meta = cell_meta.merge(ccf_subset, on="cell_label", how="left")

    # Pick one coronal section. The metadata has a ``brain_section_label`` column.
    if "brain_section_label" not in cell_meta.columns:
        raise ValueError("cell metadata missing 'brain_section_label'; cannot pick a section")
    sections = sorted(cell_meta["brain_section_label"].unique())
    if args.section_label:
        if args.section_label not in sections:
            raise ValueError(
                f"section {args.section_label!r} not found; available examples: {sections[:5]}"
            )
        section_label = args.section_label
    else:
        section_label = sections[args.section_index]
    section_cells = cell_meta[cell_meta["brain_section_label"] == section_label].copy()
    _log(f"section {section_label}: {len(section_cells)} cells")

    # Expression matrix (log2 h5ad). We read it backed and subset to this section.
    # The ABC Atlas expression_matrices key for MERFISH-C57BL6J-638850 is
    # "C57BL6J-638850" (the part after "MERFISH-"), with sub-key "log2".
    import anndata as ad

    matrix_name = args.dataset.split("MERFISH-", 1)[-1]
    expr_file = cache.get_data_path(directory=args.dataset, file_name=f"{matrix_name}/log2")
    adata_full = ad.read_h5ad(expr_file, backed="r")
    # Reindex the section cells into the full matrix's obs order.
    section_cells = section_cells.set_index("cell_label")
    section_cells = section_cells.reindex(adata_full.obs_names.astype(str))
    keep = section_cells.notna().all(axis=1).to_numpy()
    adata = adata_full[keep].to_memory()
    adata.obs = section_cells.loc[adata.obs_names].copy()
    adata_full.file.close()

    # Spatial coords: reconstructed x/y (CCF-registered frame).
    if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
        raise ValueError(
            "section cells missing x/y columns; the CCF metadata did not join. "
            "Check the abc_atlas manifest version."
        )
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=float)
    adata.uns["brain_section_label"] = section_label
    return adata


def build(args: argparse.Namespace) -> dict[str, object]:
    adata = _load_section(args)
    n_original = int(adata.n_obs)

    labels, truth_column = _resolve_truth(adata.obs, args.region_column)
    valid = _valid_mask(labels)
    adata = adata[valid].copy()
    labels = labels.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    if adata.n_obs == 0:
        raise ValueError("no cells retained a valid anatomical label")

    n_domains = int(labels.nunique())
    if n_domains < 2 or n_domains > args.max_domains:
        raise ValueError(
            f"{truth_column!r} has {n_domains} domains; expected 2..{args.max_domains}. "
            "Choose a coarser Allen CCF column such as parcellation_division."
        )

    idx = _stratified_indices(labels, args.max_cells, args.seed)
    adata = adata[idx].copy()
    labels = labels.iloc[idx]

    adata.obs["domain_truth"] = pd.Categorical(labels.to_numpy())
    adata.obs["truth_source"] = "allen_ccf_anatomical"
    adata.obs["truth_column"] = truth_column
    # counts layer: the ABC Atlas ships log2-normalized data; recover pseudo-counts.
    X = adata.X
    Xdense = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
    counts = np.clip(np.expm1(Xdense), 0, None)
    adata.layers["counts"] = counts
    adata.uns.update(
        {
            "schema_version": "histoweave.allen.merfish.brain_section.bundle.v1",
            "source": "Allen Brain Cell Atlas MERFISH whole mouse brain (single section)",
            "source_url": SOURCE_URL,
            "license": "CC-BY-NC 4.0",
            "paper_doi": "10.1038/s41586-023-06812-z",
            "n_original": n_original,
            "ground_truth_definition": f"Allen CCFv3 {truth_column} anatomical parcellation",
            "truth_column": truth_column,
            "brain_section_label": adata.uns.get("brain_section_label", ""),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "name": "allen_merfish_brain_section",
        "path": str(args.output),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_domains": int(adata.obs["domain_truth"].nunique()),
        "domains": sorted(adata.obs["domain_truth"].astype(str).unique().tolist()),
        "truth_source": "allen_ccf_anatomical",
        "truth_column": truth_column,
        "brain_section_label": adata.uns.get("brain_section_label", ""),
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    _log(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/abc_atlas"),
        help="AbcProjectCache download directory",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--section-index",
        type=int,
        default=0,
        help="Index into the sorted brain_section_label list (0 = first section)",
    )
    parser.add_argument(
        "--section-label",
        help="Explicit brain_section_label (overrides --section-index)",
    )
    parser.add_argument(
        "--region-column", help="Allen CCF anatomical column; auto-detected if omitted"
    )
    parser.add_argument("--max-domains", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets_cache" / "merfish" / "allen_merfish_brain_section.h5ad",
    )
    _log(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
