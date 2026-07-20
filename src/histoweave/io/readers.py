"""Per-assay readers: vendor output -> canonical :class:`~histoweave.data.SpatialTable`.

Two engines are offered per reader:

* ``engine="native"`` (default) parses the vendor layout directly with ``h5py`` +
  ``pandas`` (the ``.h5`` count matrix, the spatial/cells tables). It has no dependency
  on the heavy ``spatialdata`` stack, so ingestion works out of the box.
* ``engine="spatialdata"`` delegates to ``spatialdata-io`` (installed via the
  ``spatial`` extra) for full-fidelity ingestion — multiscale images, shapes, and the
  canonical SpatialData object — then converts its table to a ``SpatialTable``.

Phase-1 assay scope is Visium / Visium HD, Xenium, and one sequencing assay
(Stereo-seq). Stereo-seq's binary GEF format has no native parser yet and still routes
through the ``spatialdata`` engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import Provenance, SpatialTable
from ._tenx import read_10x_h5
from .base import Reader

_TISSUE_POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


class VisiumReader(Reader):
    assay = "visium"
    platform = "10x Visium / Visium HD"

    def read(self, path: str, *, engine: str = "native") -> SpatialTable:
        if engine == "spatialdata":
            sdata_io = self._require_spatialdata_io()
            sdata = sdata_io.visium(path)
            return _from_spatialdata(sdata, assay=self.assay, source=path)
        if engine != "native":
            raise ValueError(f"unknown engine {engine!r}; use 'native' or 'spatialdata'")
        return _read_visium_native(path)


class XeniumReader(Reader):
    assay = "xenium"
    platform = "10x Xenium"

    def read(
        self, path: str, *, engine: str = "native", gene_expression_only: bool = True
    ) -> SpatialTable:
        if engine == "spatialdata":
            sdata_io = self._require_spatialdata_io()
            sdata = sdata_io.xenium(path)
            return _from_spatialdata(sdata, assay=self.assay, source=path)
        if engine != "native":
            raise ValueError(f"unknown engine {engine!r}; use 'native' or 'spatialdata'")
        return _read_xenium_native(path, gene_expression_only=gene_expression_only)


class StereoSeqReader(Reader):
    assay = "stereo_seq"
    platform = "Stereo-seq"

    def read(self, path: str, *, engine: str = "spatialdata") -> SpatialTable:
        # The .gef binary format has no native parser yet; route through spatialdata-io.
        sdata_io = self._require_spatialdata_io()
        sdata = sdata_io.stereoseq(path)
        return _from_spatialdata(sdata, assay=self.assay, source=path)


# ---------------------------------------------------------------------------
# Native parsers
# ---------------------------------------------------------------------------
def _read_visium_native(path: str) -> SpatialTable:
    root = Path(path)
    h5 = _first_existing(
        root, ["filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"]
    )
    mat = read_10x_h5(str(h5))

    positions = _read_tissue_positions(root / "spatial")
    positions = positions.set_index("barcode").reindex(mat.barcodes)
    if positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].isna().any().any():
        raise ValueError("tissue_positions is missing rows for some matrix barcodes")

    obs = pd.DataFrame(index=pd.Index(mat.barcodes, name="barcode"))
    for col in ("in_tissue", "array_row", "array_col"):
        obs[col] = positions[col].to_numpy()
    var = pd.DataFrame(
        {"feature_name": mat.feature_names, "feature_type": mat.feature_types},
        index=pd.Index(mat.feature_ids, name="feature_id"),
    )
    coords = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)

    uns: dict = {"assay": "visium", "genome": mat.genome}
    scalefactors = _read_json(root / "spatial" / "scalefactors_json.json")
    if scalefactors is not None:
        uns["spatial"] = {"scalefactors": scalefactors}

    table = _finalize(mat.X, obs, var, coords, uns, "visium_reader", str(root))
    # Attach registered H&E when Space Ranger wrote tissue_*_image.png.
    try:
        from ..datasets.histology import attach_histology_images, load_visium_spatial_folder_images

        images = load_visium_spatial_folder_images(root / "spatial", prefer="lowres")
        if images:
            table = attach_histology_images(table, images)
    except Exception:
        # Image loading is best-effort; expression + coordinates remain valid.
        pass
    return table


def _read_xenium_native(path: str, *, gene_expression_only: bool = True) -> SpatialTable:
    root = Path(path)
    h5 = root / "cell_feature_matrix.h5"
    if not h5.exists():
        raise FileNotFoundError(
            f"{h5} not found. The native Xenium reader needs 'cell_feature_matrix.h5'; "
            "for the zarr-only export use engine='spatialdata'."
        )
    mat = read_10x_h5(str(h5))

    X = mat.X
    ids, names, types = mat.feature_ids, mat.feature_names, mat.feature_types
    if gene_expression_only and any(t != "Gene Expression" for t in types):
        keep = np.array([t == "Gene Expression" for t in types])
        X = X[:, keep]
        ids = [i for i, k in zip(ids, keep, strict=True) if k]
        names = [n for n, k in zip(names, keep, strict=True) if k]
        types = [t for t, k in zip(types, keep, strict=True) if k]

    cells = _read_cells_table(root)
    cells = cells.set_index("cell_id").reindex(mat.barcodes)
    obs = pd.DataFrame(index=pd.Index(mat.barcodes, name="cell_id"))
    for col in ("transcript_counts", "cell_area", "nucleus_area"):
        if col in cells.columns:
            obs[col] = cells[col].to_numpy()
    var = pd.DataFrame(
        {"feature_name": names, "feature_type": types},
        index=pd.Index(ids, name="feature_id"),
    )
    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)

    uns: dict = {"assay": "xenium", "genome": mat.genome}
    experiment = _read_json(root / "experiment.xenium")
    if experiment is not None:
        uns["xenium"] = experiment

    return _finalize(X, obs, var, coords, uns, "xenium_reader", str(root))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _finalize(
    X: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    coords: np.ndarray,
    uns: dict,
    method: str,
    source: str,
) -> SpatialTable:
    table = SpatialTable(X=X, obs=obs, var=var, obsm={"spatial": coords}, uns=uns)
    table.record(
        Provenance(
            step="ingestion",
            method=method,
            method_version="0.1.0",
            params={"path": source, "engine": "native"},
        )
    )
    return table


def _first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"none of {names} found under {root}")


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    parquet = spatial_dir / "tissue_positions.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    modern = spatial_dir / "tissue_positions.csv"
    if modern.exists():
        return pd.read_csv(modern)
    legacy = spatial_dir / "tissue_positions_list.csv"  # Space Ranger < 2.0: no header
    if legacy.exists():
        return pd.read_csv(legacy, header=None, names=_TISSUE_POSITION_COLUMNS)
    raise FileNotFoundError(f"no tissue_positions file under {spatial_dir}")


def _read_cells_table(root: Path) -> pd.DataFrame:
    for name in ("cells.parquet", "cells.csv.gz", "cells.csv"):
        p = root / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"no cells table (cells.parquet/csv) under {root}")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _from_spatialdata(sdata, *, assay: str, source: str) -> SpatialTable:
    """Convert a ``spatialdata.SpatialData`` object to a SpatialTable.

    Uses :meth:`SpatialTable.from_spatialdata` which preserves the full
    SpatialData — including images and shapes — rather than extracting
    only the AnnData table.
    """
    out = SpatialTable.from_spatialdata(sdata)
    out.uns.setdefault("assay", assay)
    out.record(
        Provenance(
            step="ingestion",
            method=f"{assay}_reader",
            method_version="0.1.0",
            params={"path": source, "engine": "spatialdata"},
        )
    )
    return out


class MerfishReader(Reader):
    """Vizgen MERFISH — spatialdata-io backend."""

    assay = "merfish"
    platform = "Vizgen MERFISH"

    def read(self, path: str, *, engine: str = "spatialdata") -> SpatialTable:
        sdata_io = self._require_spatialdata_io()
        sdata = sdata_io.merfish(path)
        return _from_spatialdata(sdata, assay=self.assay, source=path)


class CosMxReader(Reader):
    """NanoString CosMx SMI — spatialdata-io backend."""

    assay = "cosmx"
    platform = "NanoString CosMx SMI"

    def read(self, path: str, *, engine: str = "spatialdata") -> SpatialTable:
        sdata_io = self._require_spatialdata_io()
        sdata = sdata_io.cosmx(path)
        return _from_spatialdata(sdata, assay=self.assay, source=path)


class MerscopeReader(Reader):
    """Vizgen MERSCOPE — spatialdata-io backend."""

    assay = "merscope"
    platform = "Vizgen MERSCOPE"

    def read(self, path: str, *, engine: str = "spatialdata") -> SpatialTable:
        sdata_io = self._require_spatialdata_io()
        sdata = sdata_io.merscope(path)
        return _from_spatialdata(sdata, assay=self.assay, source=path)


class SlideSeqReader(Reader):
    """Slide-seq V2 — spatialdata-io backend."""

    assay = "slideseq"
    platform = "Slide-seq V2"

    def read(self, path: str, *, engine: str = "spatialdata") -> SpatialTable:
        sdata_io = self._require_spatialdata_io()
        sdata = sdata_io.slideseq(path)
        return _from_spatialdata(sdata, assay=self.assay, source=path)


READERS: dict[str, type[Reader]] = {
    VisiumReader.assay: VisiumReader,
    XeniumReader.assay: XeniumReader,
    StereoSeqReader.assay: StereoSeqReader,
    MerfishReader.assay: MerfishReader,
    CosMxReader.assay: CosMxReader,
    MerscopeReader.assay: MerscopeReader,
    SlideSeqReader.assay: SlideSeqReader,
}


def get_reader(assay: str) -> Reader:
    """Instantiate the reader registered for ``assay``."""
    if assay not in READERS:
        raise KeyError(f"No reader for assay '{assay}'. Available: {sorted(READERS)}")
    return READERS[assay]()


def read(assay: str, path: str, **kwargs) -> SpatialTable:
    """Convenience: read ``path`` for a given ``assay`` into a SpatialTable."""
    return get_reader(assay).read(path, **kwargs)
