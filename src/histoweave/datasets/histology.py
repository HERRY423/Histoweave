"""Histology (H&E) helpers for virtual spatial transcriptomics.

Real Visium slides store registered H&E as:

* AnnData: ``uns['spatial'][<library_id>]['images']['hires'|'lowres']``
* Space Ranger folder: ``spatial/tissue_hires_image.png`` (+ lowres)

This module extracts those arrays into :class:`~histoweave.data.SpatialTable.images`
under the canonical keys ``image`` (preferred resolution) and optional
``image_hires`` / ``image_lowres``, so :mod:`histoweave.plugins.builtin.virtual_st`
methods can run without assay-specific code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import SpatialTable

# Canonical image keys written onto SpatialTable.images
IMAGE_KEY = "image"
IMAGE_HIRES_KEY = "image_hires"
IMAGE_LOWRES_KEY = "image_lowres"

_VISIUM_PNG_CANDIDATES = (
    ("hires", "tissue_hires_image.png"),
    ("lowres", "tissue_lowres_image.png"),
)


def extract_images_from_anndata_uns(
    uns: dict[str, Any],
    *,
    prefer: str = "lowres",
) -> dict[str, np.ndarray]:
    """Pull Visium-style H&E arrays out of ``AnnData.uns['spatial']``.

    Parameters
    ----------
    uns
        AnnData / SpatialTable unstructured annotations.
    prefer
        Preferred resolution for the canonical ``image`` key
        (``"lowres"`` or ``"hires"``). Falls back to the other resolution
        when the preferred one is absent.
    """
    spatial = uns.get("spatial")
    if not isinstance(spatial, dict) or not spatial:
        return {}

    collected: dict[str, np.ndarray] = {}
    for _library_id, payload in spatial.items():
        if not isinstance(payload, dict):
            continue
        images = payload.get("images")
        if not isinstance(images, dict):
            continue
        for res_name, arr in images.items():
            key = str(res_name).lower()
            if key not in {"hires", "lowres"}:
                continue
            array = _coerce_image_array(arr)
            if array is None:
                continue
            collected[key] = array
        if collected:
            break  # first library with usable images

    if not collected:
        return {}

    out: dict[str, np.ndarray] = {}
    if "hires" in collected:
        out[IMAGE_HIRES_KEY] = collected["hires"]
    if "lowres" in collected:
        out[IMAGE_LOWRES_KEY] = collected["lowres"]
    preferred = prefer.lower()
    if preferred in collected:
        out[IMAGE_KEY] = collected[preferred]
    elif "lowres" in collected:
        out[IMAGE_KEY] = collected["lowres"]
    else:
        out[IMAGE_KEY] = next(iter(collected.values()))
    return out


def load_visium_spatial_folder_images(
    spatial_dir: str | Path,
    *,
    prefer: str = "lowres",
) -> dict[str, np.ndarray]:
    """Load ``tissue_{hires,lowres}_image.png`` from a Space Ranger ``spatial/`` folder."""
    root = Path(spatial_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"spatial folder not found: {root}")

    collected: dict[str, np.ndarray] = {}
    for res_name, filename in _VISIUM_PNG_CANDIDATES:
        path = root / filename
        if not path.exists():
            continue
        array = _read_image_file(path)
        if array is not None:
            collected[res_name] = array

    if not collected:
        return {}

    out: dict[str, np.ndarray] = {}
    if "hires" in collected:
        out[IMAGE_HIRES_KEY] = collected["hires"]
    if "lowres" in collected:
        out[IMAGE_LOWRES_KEY] = collected["lowres"]
    preferred = prefer.lower()
    if preferred in collected:
        out[IMAGE_KEY] = collected[preferred]
    else:
        out[IMAGE_KEY] = next(iter(collected.values()))
    return out


def attach_histology_images(
    data: SpatialTable,
    images: dict[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> SpatialTable:
    """Return a copy of ``data`` with histology arrays merged into ``.images``."""
    if not images:
        return data
    result = data.copy()
    merged = dict(result.images)
    for key, array in images.items():
        if key in merged and not overwrite:
            continue
        merged[key] = np.asarray(array)
    result.images = merged
    meta = dict(result.uns.get("histology", {}))
    meta.update(
        {
            "keys": sorted(merged.keys()),
            "canonical_image_key": IMAGE_KEY if IMAGE_KEY in merged else next(iter(merged)),
            "shapes": {k: list(np.asarray(v).shape) for k, v in merged.items()},
        }
    )
    result.uns["histology"] = meta
    return result


def ensure_histology(
    data: SpatialTable,
    *,
    prefer: str = "lowres",
    spatial_dir: str | Path | None = None,
) -> SpatialTable:
    """Ensure ``data.images['image']`` exists for virtual ST.

    Resolution order
    ----------------
    1. Already-present ``images['image']`` (or hires/lowres aliases).
    2. ``uns['spatial']`` Visium library images.
    3. Optional Space Ranger ``spatial/`` directory (``spatial_dir``).
    """
    if IMAGE_KEY in data.images:
        return data
    # Promote existing hires/lowres aliases.
    for alias in (
        IMAGE_LOWRES_KEY if prefer == "lowres" else IMAGE_HIRES_KEY,
        IMAGE_HIRES_KEY,
        IMAGE_LOWRES_KEY,
    ):
        if alias in data.images:
            result = data.copy()
            result.images = {**result.images, IMAGE_KEY: np.asarray(result.images[alias])}
            return result

    extracted = extract_images_from_anndata_uns(dict(data.uns), prefer=prefer)
    if extracted:
        return attach_histology_images(data, extracted)

    if spatial_dir is not None:
        folder_images = load_visium_spatial_folder_images(spatial_dir, prefer=prefer)
        if folder_images:
            return attach_histology_images(data, folder_images)

    raise KeyError(
        "no histology image found: expected images['image'], "
        "uns['spatial'][<lib>]['images'], or a Space Ranger spatial/ folder"
    )


def load_visium_hne_paired(
    *,
    source: str | Path | None = None,
    prefer: str = "lowres",
    n_hvg: int = 2000,
    min_cells: int = 3,
    seed: int = 0,
) -> SpatialTable:
    """Load the public 10x Visium Adult Mouse Brain H&E slide with expression + image.

    This is the primary **real** paired H&E→ST dataset for the virtual_st task.

    Parameters
    ----------
    source
        Optional path to a cached ``visium_hne_adata.h5ad`` (squidpy export).
        When omitted, uses squidpy's ``visium_hne_adata()`` download (requires
        ``squidpy``) or the repo path ``data/anndata/visium_hne_adata.h5ad`` if present.
    prefer
        Image resolution for ``images['image']`` (``lowres`` recommended for CI).
    n_hvg
        Highly variable gene cap for a manageable virtual_st head.
    min_cells
        Drop genes present in fewer than this many spots.
    seed
        Reserved for future deterministic HVG fallbacks.
    """
    del seed  # reserved
    adata = _load_visium_hne_anndata(source)
    return spatial_table_from_visium_hne(
        adata,
        prefer=prefer,
        n_hvg=n_hvg,
        min_cells=min_cells,
    )


def spatial_table_from_visium_hne(
    adata: Any,
    *,
    prefer: str = "lowres",
    n_hvg: int = 2000,
    min_cells: int = 3,
) -> SpatialTable:
    """Convert a squidpy / scanpy Visium H&E AnnData into a virtual-ST-ready table."""
    import numpy as np

    if "spatial" not in adata.obsm:
        raise ValueError("Visium H&E AnnData is missing obsm['spatial']")

    # Recover raw-like counts when possible.
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.obs = adata.obs.copy()
        raw.obsm = dict(adata.obsm)
        raw.uns = dict(adata.uns)
        if "spatial" in adata.uns and "spatial" not in raw.uns:
            raw.uns["spatial"] = adata.uns["spatial"]
        adata = raw

    obs = pd.DataFrame(adata.obs).copy()
    var = pd.DataFrame(adata.var).copy()
    X = adata.X
    X_dense_probe = np.asarray(
        X[: min(50, X.shape[0])].todense() if hasattr(X, "todense") else X[: min(50, X.shape[0])]
    )
    int_frac = float(np.mean(np.isclose(X_dense_probe, np.round(X_dense_probe))))
    if int_frac < 0.9:
        # Likely log-normalised — store expm1 pseudo-counts in layers later.
        counts = np.clip(
            np.expm1(np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)),
            0.0,
            None,
        )
    else:
        counts = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
        counts = np.clip(counts, 0.0, None)

    # Anatomical labels when present (domain GT — not used as virtual_st primary metric).
    if "cluster" in obs.columns and "domain_truth" not in obs.columns:
        obs["domain_truth"] = pd.Categorical(obs["cluster"].astype(str))

    # Lightweight gene filter without requiring scanpy.
    present = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep_genes = present >= int(min_cells)
    if keep_genes.sum() < 10:
        keep_genes = np.ones(counts.shape[1], dtype=bool)
    counts = counts[:, keep_genes]
    var = var.iloc[np.flatnonzero(keep_genes)].copy()

    if n_hvg is not None and n_hvg > 0 and counts.shape[1] > n_hvg:
        variances = counts.var(axis=0)
        order = np.argsort(variances)[::-1][: int(n_hvg)]
        counts = counts[:, order]
        var = var.iloc[order].copy()

    images = extract_images_from_anndata_uns(dict(adata.uns), prefer=prefer)
    if not images:
        raise ValueError(
            "Visium H&E AnnData has no uns['spatial'][…]['images']; cannot build a virtual_st table"
        )

    uns: dict[str, Any] = {
        "assay": "visium",
        "platform": "visium",
        "technology": "visium",
        "analysis_task": "virtual_st",
        "ground_truth_kind": "measured_expression",
        "source": "10x Visium V1 Adult Mouse Brain (H&E)",
        "source_via": "squidpy.datasets.visium_hne_adata",
        "schema_version": "histoweave.virtual_st.visium_hne.v1",
    }
    # Preserve library spatial metadata (scalefactors) without duplicating huge images in uns.
    spatial_meta = adata.uns.get("spatial")
    if isinstance(spatial_meta, dict):
        slim: dict[str, Any] = {}
        for lib, payload in spatial_meta.items():
            if not isinstance(payload, dict):
                continue
            slim[lib] = {k: v for k, v in payload.items() if k != "images"}
            # Keep a tiny marker so extract_images_from_anndata_uns can still work
            # if images are re-attached later.
            if "images" in payload:
                slim[lib]["images"] = {
                    name: np.asarray(arr)
                    for name, arr in payload["images"].items()
                    if str(name).lower() in {"hires", "lowres"}
                }
        uns["spatial"] = slim

    table = SpatialTable(
        X=counts,
        obs=obs,
        var=var,
        obsm={"spatial": np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]},
        layers={"counts": counts.copy()},
        images=images,
        uns=uns,
    )
    table.uns["histology"] = {
        "keys": sorted(images.keys()),
        "canonical_image_key": IMAGE_KEY,
        "shapes": {k: list(np.asarray(v).shape) for k, v in images.items()},
        "prefer": prefer,
    }
    return table


def prepare_virtual_st_table(
    data: SpatialTable,
    *,
    image_key: str = IMAGE_KEY,
    prefer: str = "lowres",
    spatial_dir: str | Path | None = None,
    expression_layer: str | None = "counts",
) -> SpatialTable:
    """Normalise an arbitrary SpatialTable for the virtual_st task contract.

    Ensures histology is attached, marks analysis metadata, and prefers a raw
    count layer for paired H&E→expression supervision.
    """
    data = ensure_histology(data, prefer=prefer, spatial_dir=spatial_dir)
    if image_key not in data.images:
        # ensure_histology guarantees IMAGE_KEY; allow caller override via alias.
        if IMAGE_KEY in data.images:
            result = data.copy()
            result.images = {**result.images, image_key: result.images[IMAGE_KEY]}
            data = result
        else:
            raise KeyError(f"image key {image_key!r} missing after ensure_histology")

    result = data.copy()
    result.uns = dict(result.uns)
    result.uns["analysis_task"] = "virtual_st"
    result.uns["ground_truth_kind"] = "measured_expression"
    result.uns.setdefault("platform", result.uns.get("assay", "histology"))
    if expression_layer and expression_layer in result.layers:
        # Keep measured expression in X for paired virtual_st methods by default.
        measured = result.layers[expression_layer]
        result.X = np.asarray(
            measured.todense() if hasattr(measured, "todense") else measured, dtype=float
        )
    if result.spatial is None:
        raise ValueError("virtual_st requires obsm['spatial']")
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_visium_hne_anndata(source: str | Path | None):
    import anndata as ad

    candidates: list[Path] = []
    if source is not None:
        candidates.append(Path(source))
    else:
        repo = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                repo / "data" / "anndata" / "visium_hne_adata.h5ad",
                repo / "datasets_cache" / "visium" / "visium_hne_adata.h5ad",
                Path.home() / ".cache" / "squidpy" / "visium_hne_adata.h5ad",
            ]
        )

    for path in candidates:
        if path.exists():
            return ad.read_h5ad(path)

    try:
        import squidpy as sq
    except ImportError as exc:
        raise ImportError(
            "load_visium_hne_paired requires either a local visium_hne_adata.h5ad "
            "or squidpy (pip install squidpy)"
        ) from exc
    return sq.datasets.visium_hne_adata()


def _coerce_image_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        return None
    # Channel-first small channel dim → channel-last.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    # Integer histology (uint8 PNG / Visium) is valid; only reject non-finite floats.
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        return None
    return arr


def _read_image_file(path: Path) -> np.ndarray | None:
    # Prefer imageio / PIL when available; fall back to matplotlib.
    try:
        from imageio.v3 import imread

        return _coerce_image_array(imread(path))
    except Exception:
        pass
    try:
        from PIL import Image

        return _coerce_image_array(np.asarray(Image.open(path)))
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt

        return _coerce_image_array(plt.imread(path))
    except Exception:
        return None


__all__ = [
    "IMAGE_KEY",
    "IMAGE_HIRES_KEY",
    "IMAGE_LOWRES_KEY",
    "attach_histology_images",
    "ensure_histology",
    "extract_images_from_anndata_uns",
    "load_visium_hne_paired",
    "load_visium_spatial_folder_images",
    "prepare_virtual_st_table",
    "spatial_table_from_visium_hne",
]
