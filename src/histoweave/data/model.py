"""Canonical in-platform data model.

The platform standardizes on **SpatialData / OME-Zarr** (with AnnData/MuData for the
tabular molecular layer) as the canonical representation.  :class:`SpatialTable` is now
a thin wrapper around :class:`spatialdata.SpatialData`, delegating the molecular layer
(``X`` / ``obs`` / ``var`` / ``obsm`` / ``layers`` / ``uns``) to the internal AnnData
table while preserving the same public API that downstream code was written against.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData
    from spatialdata import SpatialData as SpatialDataClass

Matrix: TypeAlias = np.ndarray | spmatrix

SPATIAL_KEY = "spatial"

# OME-Zarr / SpatialData enforces alphanumeric + underscore/dot/hyphen key names.
import re as _re
_SANITIZE_KEY_RE = _re.compile(r"[^a-zA-Z0-9_.\-]")
del _re


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
                    f"obsm {name!r} has shape {array.shape}; "
                    f"first dimension must be n_obs={n_obs}"
                )
            if name == SPATIAL_KEY and (array.ndim != 2 or array.shape[1] not in (2, 3)):
                raise ValueError(
                    f"obsm[{SPATIAL_KEY!r}] must have shape (n_obs, 2 or 3), "
                    f"got {array.shape}"
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

        adata = AnnData(X=X, obs=obs, var=var, obsm=obsm, layers=layers, uns=uns)
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
        return self._table.obs

    @obs.setter
    def obs(self, value: pd.DataFrame) -> None:
        if len(value) != self.n_obs:
            raise ValueError(
                f"obs has {len(value)} rows but X has {self.n_obs} observations"
            )
        self._table.obs = value

    @property
    def var(self) -> pd.DataFrame:
        return self._table.var

    @var.setter
    def var(self, value: pd.DataFrame) -> None:
        if len(value) != self.n_vars:
            raise ValueError(
                f"var has {len(value)} rows but X has {self.n_vars} variables"
            )
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
                    f"obsm[{SPATIAL_KEY!r}] must have shape (n_obs, 2 or 3), "
                    f"got {array.shape}"
                )
        self._table.obsm = {k: np.asarray(v) for k, v in value.items()}

    @property
    def layers(self) -> Any:  # Layers (MutableMapping), not plain dict
        return self._table.layers

    @layers.setter
    def layers(self, value: dict[str, Matrix]) -> None:
        for name, layer in value.items():
            layer = _coerce_matrix(layer)
            if layer.shape != self.shape:
                raise ValueError(
                    f"layer {name!r} has shape {layer.shape}, expected {self.shape} to match X"
                )
        self._table.layers = {k: _coerce_matrix(v) for k, v in value.items()}

    @property
    def uns(self) -> dict[str, Any]:
        return self._table.uns

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
        return self._table.obsm.get(SPATIAL_KEY)

    # -- provenance ------------------------------------------------------------
    def record(self, prov: Provenance) -> None:
        """Append a provenance entry describing how this object was transformed."""
        self.uns.setdefault("provenance", []).append(prov.to_dict())

    @property
    def provenance(self) -> list[dict[str, Any]]:
        return self.uns.get("provenance", [])

    def copy(self) -> SpatialTable:
        return SpatialTable(
            X=self._table.X.copy(),
            obs=self.obs.copy(),
            var=self.var.copy(),
            obsm={k: v.copy() for k, v in self._table.obsm.items()},
            layers={k: v.copy() for k, v in self._table.layers.items() if k is not None},
            images=copy.deepcopy(self._images),
            shapes=copy.deepcopy(self._shapes),
            uns=copy.deepcopy(self._table.uns),
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
        return SpatialTable(
            X=self.X[mask],
            obs=self.obs.loc[mask].copy(),
            var=self.var.copy(),
            obsm={k: v[mask] for k, v in self._table.obsm.items()},
            layers={k: v[mask] for k, v in self._table.layers.items() if k is not None},
            images=copy.deepcopy(self._images),
            shapes=copy.deepcopy(self._shapes),
            uns=copy.deepcopy(self.uns),
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

        return AnnData(
            X=self.X,
            obs=obs,
            var=var,
            obsm=dict(self._table.obsm),
            layers={k: v.copy() for k, v in self._table.layers.items() if k is not None},
            uns=_sanitize_uns_for_h5ad(self.uns),
        )

    @classmethod
    def from_anndata(cls, adata: AnnData) -> SpatialTable:
        """Build a table without densifying sparse AnnData matrices."""
        return cls(
            X=adata.X.copy(),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            obsm={k: np.asarray(v) for k, v in adata.obsm.items()},
            layers={k: v.copy() for k, v in adata.layers.items() if k is not None},
            uns=dict(adata.uns),
        )

    # -- SpatialData bridge ----------------------------------------------------
    def to_spatialdata(self):
        """Return the backing :class:`spatialdata.SpatialData` object.

        Images and shapes stored on this SpatialTable are merged into the
        SpatialData before returning, so the result carries the complete data
        (molecular + spatial layers).
        """
        # Merge images into the SpatialData (best-effort; SpatialImage wrapping
        # may be needed for full fidelity).
        for key, img in self._images.items():
            if key not in self._sdata.images:
                try:
                    self._sdata.images[key] = img
                except Exception:
                    pass
        # Merge shapes into the SpatialData.
        for key, shp in self._shapes.items():
            if key not in self._sdata.shapes:
                try:
                    self._sdata.shapes[key] = shp
                except Exception:
                    pass
        return self._sdata

    @classmethod
    def from_spatialdata(cls, sdata) -> SpatialTable:
        """Build a SpatialTable from a :class:`spatialdata.SpatialData`, preserving
        images and shapes.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            The SpatialData object to wrap.

        Returns
        -------
        SpatialTable
        """
        # Extract the first table (AnnData).
        if hasattr(sdata, "tables") and sdata.tables:
            table = next(iter(sdata.tables.values()))
        elif hasattr(sdata, "table"):
            table = sdata.table
        else:
            raise ValueError("SpatialData has no tables; cannot build SpatialTable")

        # Extract images as numpy arrays (best-effort).
        images: dict[str, np.ndarray] = {}
        if hasattr(sdata, "images"):
            for key in sdata.images:
                try:
                    images[key] = np.asarray(sdata.images[key])
                except Exception:
                    pass

        # Extract shapes.
        shapes: dict[str, Any] = {}
        if hasattr(sdata, "shapes"):
            for key in sdata.shapes:
                try:
                    shapes[key] = sdata.shapes[key]
                except Exception:
                    pass

        return cls(
            X=table.X.copy(),
            obs=table.obs.copy(),
            var=table.var.copy(),
            obsm={k: np.asarray(v) for k, v in table.obsm.items()},
            layers={k: v.copy() for k, v in table.layers.items()},
            images=images,
            shapes=shapes,
            uns=dict(table.uns),
        )

    def __repr__(self) -> str:
        obsm_keys = ", ".join(self._table.obsm) or "-"
        layer_keys = ", ".join(k for k in self._table.layers if k is not None) or "-"
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
