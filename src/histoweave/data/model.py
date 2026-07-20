"""Canonical in-platform data model.

The platform standardizes on **SpatialData / OME-Zarr** (with AnnData/MuData for the
tabular molecular layer) as the canonical representation.  :class:`SpatialTable` is now
a thin wrapper around :class:`spatialdata.SpatialData`, delegating the molecular layer
(``X`` / ``obs`` / ``var`` / ``obsm`` / ``layers`` / ``uns``) to the internal AnnData
table while preserving the same public API that downstream code was written against.
"""

from __future__ import annotations

import copy
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData
    from spatialdata import SpatialData as SpatialDataClass

Matrix: TypeAlias = np.ndarray | spmatrix


def _matrix_copy(value: Any) -> Matrix:
    """Copy dense/sparse matrices without relying on AnnData's broad union types."""
    if value is None:
        raise TypeError("cannot copy a missing matrix")
    copy_fn = getattr(value, "copy", None)
    if callable(copy_fn):
        return cast(Matrix, copy_fn())
    return np.asarray(value).copy()


def _axis_array_copy(value: Any) -> np.ndarray:
    return np.asarray(value).copy()


SPATIAL_KEY = "spatial"

# OME-Zarr / SpatialData enforces alphanumeric + underscore/dot/hyphen key names.
_SANITIZE_KEY_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


def _sanitize_mapping_keys(
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """Replace characters that are illegal in OME-Zarr key names with underscores."""
    sanitized: dict[str, Any] = {}
    for key, value in mapping.items():
        clean = _SANITIZE_KEY_RE.sub("_", key)
        sanitized[clean] = value
    return sanitized


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance:
    """Structured provenance for a single processing step.

    Every object carries the chain of steps that produced it (source assay,
    method, version, parameters, container digest) so any result is reproducible
    and auditable — a first-class requirement in the plan, not an afterthought.
    """

    step: str
    method: str
    method_version: str
    params: dict[str, Any]
    histoweave_version: str
    container_digest: str | None
    code_revision: str | None
    executor: str | None
    timestamp: str

    def __init__(
        self,
        step: str,
        method: str,
        method_version: str,
        params: dict[str, Any] | None = None,
        histoweave_version: str = "",
        container_digest: str | None = None,
        code_revision: str | None = None,
        executor: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.step = step
        self.method = method
        self.method_version = method_version
        self.params = dict(params) if params is not None else {}
        self.histoweave_version = histoweave_version
        self.container_digest = container_digest
        self.code_revision = code_revision
        self.executor = executor
        self.timestamp = timestamp if timestamp is not None else datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "method": self.method,
            "method_version": self.method_version,
            "params": self.params,
            "histoweave_version": self.histoweave_version,
            "container_digest": self.container_digest,
            "code_revision": self.code_revision,
            "executor": self.executor,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# SpatialTable — SpatialData-backed spatial container
# ---------------------------------------------------------------------------


class SpatialTable:
    """A SpatialData-backed spatial container.

    Parameters
    ----------
    X
        Cell/spot x gene matrix (dense ``np.ndarray`` or sparse ``scipy.sparse.spmatrix``).
    obs
        Per-observation (cell/spot) metadata, indexed by observation id.
    var
        Per-gene metadata, indexed by gene id.
    obsm
        Multi-dimensional per-observation arrays, e.g. ``"spatial"`` coordinates
        or learned embeddings (``"X_pca"``).
    layers
        Alternative cell x gene matrices sharing ``X``'s shape, keyed by name
        (AnnData's ``.layers`` concept). This is where transformations stash the
        matrix they replaced — e.g. normalization moves the raw counts to
        ``layers["counts"]`` so count-based methods can recover them.
    images
        Raster **spatial layers** keyed by name (SpatialData's ``images`` element),
        e.g. an H&E or DAPI image of the tissue.
    shapes
        Vector **spatial layers** keyed by name (SpatialData's ``shapes`` element),
        e.g. cell/nucleus boundary polygons or Visium spot circles.
    uns
        Unstructured annotations, including the ``"provenance"`` chain.

    Notes
    -----
    The molecular/tabular layer (``X``, ``obs``, ``var``, ``obsm``, ``layers``,
    ``uns``) is stored in an :class:`anndata.AnnData` table inside a
    :class:`spatialdata.SpatialData`.  Spatial layers (``images``, ``shapes``)
    are kept as plain dicts for API compatibility and merged into the SpatialData
    on :meth:`to_spatialdata`.
    """

    def __init__(
        self,
        X: Matrix,
        obs: pd.DataFrame,
        var: pd.DataFrame,
        obsm: dict[str, np.ndarray] | None = None,
        layers: dict[str, Matrix] | None = None,
        images: dict[str, np.ndarray] | None = None,
        shapes: dict[str, Any] | None = None,
        uns: dict[str, Any] | None = None,
    ) -> None:
        # -- coerce & validate X -------------------------------------------------
        X = _coerce_matrix(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (cells x genes), got shape {X.shape}")
        n_obs, n_var = X.shape

        # -- defaults ------------------------------------------------------------
        obsm = dict(obsm) if obsm is not None else {}
        layers = dict(layers) if layers is not None else {}
        images = dict(images) if images is not None else {}
        shapes = dict(shapes) if shapes is not None else {}
        uns = dict(uns) if uns is not None else {}

        # -- validate obs --------------------------------------------------------
        if len(obs) != n_obs:
            raise ValueError(f"obs has {len(obs)} rows but X has {n_obs} observations")
        if not obs.index.is_unique:
            raise ValueError("obs index must contain unique observation identifiers")
        if obs.index.hasnans:
            raise ValueError("obs index must not contain missing identifiers")

        # -- validate var --------------------------------------------------------
        if len(var) != n_var:
            raise ValueError(f"var has {len(var)} rows but X has {n_var} variables")
        if not var.index.is_unique:
            raise ValueError("var index must contain unique feature identifiers")
        if var.index.hasnans:
            raise ValueError("var index must not contain missing identifiers")

        # -- validate obsm -------------------------------------------------------
        for name, value in obsm.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != n_obs:
                raise ValueError(
                    f"obsm {name!r} has shape {array.shape}; first dimension must be n_obs={n_obs}"
                )
            if name == SPATIAL_KEY and (array.ndim != 2 or array.shape[1] not in (2, 3)):
                raise ValueError(
                    f"obsm[{SPATIAL_KEY!r}] must have shape (n_obs, 2 or 3), got {array.shape}"
                )
            obsm[name] = array

        # -- validate layers -----------------------------------------------------
        for name, layer in layers.items():
            layer = _coerce_matrix(layer)
            if layer.shape != X.shape:
                raise ValueError(
                    f"layer {name!r} has shape {layer.shape}, expected {X.shape} to match X"
                )
            layers[name] = layer

        uns.setdefault("provenance", [])

        # -- sanitize keys for SpatialData compliance ----------------------------
        obsm = _sanitize_mapping_keys(obsm)
        layers = _sanitize_mapping_keys(layers)

        # -- build backing SpatialData -------------------------------------------
        from anndata import AnnData
        from spatialdata import SpatialData as SD

        # AnnData stubs type obsm/layers more narrowly than runtime accepts.
        adata = AnnData(
            X=X,
            obs=obs,
            var=var,
            obsm=cast(Any, obsm),
            layers=cast(Any, layers),
            uns=uns,
        )
        self._sdata: SpatialDataClass = SD(tables={"table": adata})
        self._images = images
        self._shapes = shapes

    # -- internal helper --------------------------------------------------------
    @property
    def _table(self) -> AnnData:
        """The first (and typically only) AnnData table in the backing SpatialData."""
        return cast(Any, list(self._sdata.tables.values())[0])

    # -- tabular properties -----------------------------------------------------
    @property
    def X(self) -> Matrix:
        return self._table.X

    @X.setter
    def X(self, value: Matrix) -> None:
        self._table.X = _coerce_matrix(value)

    @property
    def obs(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._table.obs)

    @obs.setter
    def obs(self, value: pd.DataFrame) -> None:
        if len(value) != self.n_obs:
            raise ValueError(f"obs has {len(value)} rows but X has {self.n_obs} observations")
        self._table.obs = value

    @property
    def var(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._table.var)

    @var.setter
    def var(self, value: pd.DataFrame) -> None:
        if len(value) != self.n_vars:
            raise ValueError(f"var has {len(value)} rows but X has {self.n_vars} variables")
        self._table.var = value

    @property
    def obsm(self) -> Any:  # AxisArrays (MutableMapping), not plain dict
        return self._table.obsm

    @obsm.setter
    def obsm(self, value: dict[str, np.ndarray]) -> None:
        for name, arr in value.items():
            array = np.asarray(arr)
            if array.ndim == 0 or array.shape[0] != self.n_obs:
                raise ValueError(
                    f"obsm {name!r} has shape {array.shape}; "
                    f"first dimension must be n_obs={self.n_obs}"
                )
            if name == SPATIAL_KEY and (array.ndim != 2 or array.shape[1] not in (2, 3)):
                raise ValueError(
                    f"obsm[{SPATIAL_KEY!r}] must have shape (n_obs, 2 or 3), got {array.shape}"
                )
        self._table.obsm = {k: np.asarray(v) for k, v in value.items()}

    @property
    def layers(self) -> Any:  # Layers (MutableMapping), not plain dict
        return self._table.layers

    @layers.setter
    def layers(self, value: dict[str, Matrix]) -> None:
        coerced: dict[str, Matrix] = {}
        for name, layer in value.items():
            matrix = _coerce_matrix(layer)
            if matrix.shape != self.shape:
                raise ValueError(
                    f"layer {name!r} has shape {matrix.shape}, expected {self.shape} to match X"
                )
            coerced[name] = matrix
        self._table.layers = cast(Any, coerced)

    @property
    def uns(self) -> dict[str, Any]:
        # AnnData types uns as a MutableMapping; expose a plain dict view.
        return cast(dict[str, Any], self._table.uns)

    @uns.setter
    def uns(self, value: dict[str, Any]) -> None:
        self._table.uns = dict(value)

    # -- spatial-layer properties -----------------------------------------------
    @property
    def images(self) -> dict[str, np.ndarray]:
        return self._images

    @images.setter
    def images(self, value: dict[str, np.ndarray]) -> None:
        self._images = dict(value)

    @property
    def shapes(self) -> dict[str, Any]:
        return self._shapes

    @shapes.setter
    def shapes(self, value: dict[str, Any]) -> None:
        self._shapes = dict(value)

    # -- shape helpers ---------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return self._table.n_obs

    @property
    def n_vars(self) -> int:
        return self._table.n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_vars)

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    @property
    def spatial(self) -> np.ndarray | None:
        """Convenience accessor for the canonical spatial coordinates."""
        value = dict(self._table.obsm).get(SPATIAL_KEY)
        return None if value is None else np.asarray(value)

    # -- provenance ------------------------------------------------------------
    def record(self, prov: Provenance) -> None:
        """Append a provenance entry describing how this object was transformed."""
        chain = self.uns.get("provenance", [])
        # AnnData h5ad round-trips may restore empty provenance as ndarray.
        if not isinstance(chain, list):
            chain = list(chain) if getattr(chain, "dtype", None) == object else []
        chain.append(prov.to_dict())
        self.uns["provenance"] = chain

    @property
    def provenance(self) -> list[dict[str, Any]]:
        chain = self.uns.get("provenance", [])
        if not isinstance(chain, list):
            return list(chain) if getattr(chain, "dtype", None) == object else []
        return chain

    def copy(self) -> SpatialTable:
        return SpatialTable(
            X=_matrix_copy(self._table.X),
            obs=self.obs.copy(),
            var=self.var.copy(),
            obsm={str(k): _axis_array_copy(v) for k, v in dict(self._table.obsm).items()},
            layers={
                str(k): _matrix_copy(v)
                for k, v in dict(self._table.layers).items()
                if k is not None
            },
            images=copy.deepcopy(self._images),
            shapes=copy.deepcopy(self._shapes),
            uns=copy.deepcopy(dict(self._table.uns)),
        )

    # -- subsetting ------------------------------------------------------------
    def subset_obs(self, mask: np.ndarray) -> SpatialTable:
        """Return a new table keeping only observations where ``mask`` is True.

        Obs-aligned data (``X``, ``obs``, ``obsm``, ``layers``) is masked; the spatial
        layers (``images``/``shapes``) live in a coordinate system rather than on the
        observation axis and so are carried through unchanged.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != self.n_obs:
            raise ValueError("mask must be one-dimensional with length n_obs")
        obsm = {str(k): np.asarray(v)[mask] for k, v in dict(self._table.obsm).items()}
        layers = {
            str(k): cast(Matrix, np.asarray(v)[mask] if not hasattr(v, "tocsr") else v[mask])
            for k, v in dict(self._table.layers).items()
            if k is not None
        }
        return SpatialTable(
            X=cast(Matrix, self.X[mask]),
            obs=self.obs.loc[mask].copy(),
            var=self.var.copy(),
            obsm=obsm,
            layers=layers,
            images=copy.deepcopy(self._images),
            shapes=copy.deepcopy(self._shapes),
            uns=copy.deepcopy(dict(self.uns)),
        )

    # -- AnnData bridge --------------------------------------------------------
    def to_anndata(self) -> AnnData:
        """Convert to :class:`anndata.AnnData` (requires the ``spatial`` extra).

        AnnData models only the molecular/tabular layer, so the spatial layers
        (``images``/``shapes``) are **not** carried over — use
        :meth:`to_spatialdata` when those must be preserved.
        """
        try:
            from anndata import AnnData
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                "anndata is required for the AnnData bridge. "
                "Install it with: pip install 'histoweave-spatial[spatial]'"
            ) from exc
        # Convert categorical columns to string — AnnData auto-converts
        # ``object``-dtype categoricals to h5py-incompatible HDF5 objects
        # (gh anndata#…).  Pandas ``string`` dtype avoids this.
        obs = self.obs.copy()
        for col in obs.columns:
            if isinstance(obs[col].dtype, pd.CategoricalDtype):
                obs[col] = obs[col].astype("string")
        var = self.var.copy()
        for col in var.columns:
            if isinstance(var[col].dtype, pd.CategoricalDtype):
                var[col] = var[col].astype("string")

        layers = {
            str(k): _matrix_copy(v) for k, v in dict(self._table.layers).items() if k is not None
        }
        return AnnData(
            X=self.X,
            obs=obs,
            var=var,
            obsm=cast(Any, {str(k): np.asarray(v) for k, v in dict(self._table.obsm).items()}),
            layers=cast(Any, layers),
            uns=_sanitize_uns_for_h5ad(dict(self.uns)),
        )

    @classmethod
    def from_anndata(cls, adata: AnnData) -> SpatialTable:
        """Build a table without densifying sparse AnnData matrices."""
        if adata.X is None:
            raise ValueError("AnnData.X is required to build a SpatialTable")
        layers = {str(k): _matrix_copy(v) for k, v in dict(adata.layers).items() if k is not None}
        return cls(
            X=_matrix_copy(adata.X),
            obs=cast(pd.DataFrame, adata.obs).copy(),
            var=cast(pd.DataFrame, adata.var).copy(),
            obsm={str(k): np.asarray(v) for k, v in dict(adata.obsm).items()},
            layers=layers,
            uns=dict(adata.uns),
        )

    # -- SpatialData bridge ----------------------------------------------------
    def to_spatialdata(
        self,
        *,
        table_name: str = "table",
    ) -> SpatialDataClass:
        """Convert to a :class:`spatialdata.SpatialData` object (``spatial`` extra).

        Unlike :meth:`to_anndata`, this bridge is **lossless for the spatial layers**:
        the molecular/tabular part becomes a SpatialData *table* element (an
        :class:`~anndata.AnnData` wrapped with :class:`spatialdata.models.TableModel`),
        each entry of :attr:`images` becomes an ``Image2DModel`` element, and each
        entry of :attr:`shapes` becomes a ``ShapesModel`` element.

        Parameters
        ----------
        table_name
            Key used for the table element inside the returned ``SpatialData``.
        Notes
        -----
        The molecular table is intentionally non-annotating: HistoWeave does not assume
        that arbitrary image or shape layers have a one-to-one relationship with rows.
        Images are channel-last and are transposed to SpatialData's channel-first
        convention on export.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            from spatialdata import SpatialData
            from spatialdata.models import Image2DModel, ShapesModel, TableModel
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                "spatialdata (and geopandas/shapely) are required for the SpatialData "
                "bridge. Install with: pip install 'histoweave-spatial[spatial]'"
            ) from exc

        table = TableModel.parse(self.to_anndata())

        images: dict[str, Any] = {}
        for key, img in self.images.items():
            arr = np.asarray(img)
            if arr.ndim == 2:  # (y, x) -> (c=1, y, x)
                parsed = Image2DModel.parse(arr[np.newaxis, ...], dims=("c", "y", "x"))
            elif arr.ndim == 3:  # (y, x, c) channel-last -> (c, y, x)
                parsed = Image2DModel.parse(np.transpose(arr, (2, 0, 1)), dims=("c", "y", "x"))
            else:  # pragma: no cover - defensive
                raise ValueError(f"image '{key}' must be 2D or 3D, got shape {arr.shape}")
            images[key] = parsed

        shapes: dict[str, Any] = {}
        for key, geom in self.shapes.items():
            if isinstance(geom, gpd.GeoDataFrame):
                shapes[key] = ShapesModel.parse(geom)
            else:
                # Fall back to treating the object as an (n, 2) coordinate array of
                # circle/point centroids — the minimal Visium-spot representation.
                coords = np.asarray(geom, dtype=float)
                gdf = gpd.GeoDataFrame({"geometry": [Point(float(x), float(y)) for x, y in coords]})
                gdf["radius"] = 1.0
                shapes[key] = ShapesModel.parse(gdf)

        return SpatialData(images=images, shapes=shapes, tables={table_name: table})

    @classmethod
    def from_spatialdata(
        cls,
        sdata: SpatialDataClass,
        *,
        table_name: str | None = None,
    ) -> SpatialTable:
        """Build a :class:`SpatialTable` from a :class:`spatialdata.SpatialData`.

        The chosen table element supplies ``X``/``obs``/``var``/``obsm``/``layers``
        (via :meth:`from_anndata`); every ``images`` element is carried back as a
        channel-last array in :attr:`images`; every ``shapes`` element is carried
        back verbatim (as a ``geopandas.GeoDataFrame``) in :attr:`shapes`.

        Parameters
        ----------
        table_name
            Which table element to use. Defaults to the sole table when there is
            exactly one; raises if ambiguous.
        """
        try:
            import spatialdata  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                "spatialdata is required for the SpatialData bridge. "
                "Install with: pip install 'histoweave-spatial[spatial]'"
            ) from exc

        tables = dict(sdata.tables)
        if not tables:
            raise ValueError("SpatialData object has no table element to convert")
        if table_name is None:
            if len(tables) != 1:
                raise ValueError(f"multiple tables {list(tables)}; pass table_name explicitly")
            table_name = next(iter(tables))
        table = tables[table_name]

        st = cls.from_anndata(table)

        for key, img in dict(sdata.images).items():
            arr = np.asarray(img.data if hasattr(img, "data") else img)
            # SpatialData images are channel-first (c, y, x) -> channel-last.
            if arr.ndim == 3:
                arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 1:
                    arr = arr[..., 0]
            st.images[key] = arr

        for key, geom in dict(sdata.shapes).items():
            st.shapes[key] = geom

        return st

    def __repr__(self) -> str:
        obsm_keys = ", ".join(str(k) for k in dict(self._table.obsm)) or "-"
        layer_keys = ", ".join(str(k) for k in dict(self._table.layers) if k is not None) or "-"
        spatial_keys = ", ".join([*self._images, *self._shapes]) or "-"
        return (
            f"SpatialTable(n_obs={self.n_obs}, n_vars={self.n_vars}, "
            f"obsm=[{obsm_keys}], layers=[{layer_keys}], spatial=[{spatial_keys}], "
            f"provenance_steps={len(self.provenance)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_uns_for_h5ad(uns: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``uns``, converting provenance to a JSON-safe form.

    AnnData/h5py cannot round-trip a list of heterogeneous dicts through the
    HDF5 VLEN-string path.  We serialise provenance to a list of JSON strings
    instead; the reader on the R side can parse them back.
    """
    import dataclasses
    import json

    safe: dict[str, Any] = {}
    for key, value in uns.items():
        if key == "provenance" and isinstance(value, list):
            safe[key] = [
                json.dumps(
                    dataclasses.asdict(entry)
                    if dataclasses.is_dataclass(entry) and not isinstance(entry, type)
                    else entry,
                    default=str,
                )
                for entry in value
            ]
        else:
            safe[key] = copy.deepcopy(value)
    return safe


def _coerce_matrix(value: Matrix) -> Matrix:
    """Coerce dense inputs while preserving sparse matrices as CSR."""
    try:
        from scipy.sparse import issparse
    except ModuleNotFoundError:  # pragma: no cover - sparse inputs require SciPy
        return np.asarray(value)

    if issparse(value):
        return cast(Any, value).tocsr()
    return np.asarray(value)
