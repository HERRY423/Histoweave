"""Canonical in-platform data model.

The plan standardizes on **SpatialData / OME-Zarr** (with AnnData/MuData for the
tabular molecular layer) as the canonical representation. Pulling that full stack in
is a Phase-1 task and a heavy dependency, so this scaffold ships a light,
AnnData-shaped container — :class:`SpatialTable` — that mirrors the same mental model
(``X`` / ``obs`` / ``var`` / ``obsm`` / ``uns``) and provides bridges to and from
AnnData when it is installed.

Deliberately, everything downstream (plugins, pipeline, benchmarking, reporting) is
written against :class:`SpatialTable`, so swapping in a real SpatialData-backed
implementation later is an internal change, not an API break.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData
    from scipy.sparse import spmatrix

    Matrix: TypeAlias = np.ndarray | spmatrix
else:
    Matrix: TypeAlias = Any

SPATIAL_KEY = "spatial"


@dataclass
class Provenance:
    """Structured provenance for a single processing step.

    Every object carries the chain of steps that produced it (source assay,
    method, version, parameters, container digest) so any result is reproducible
    and auditable — a first-class requirement in the plan, not an afterthought.
    """

    step: str
    method: str
    method_version: str
    params: dict[str, Any] = field(default_factory=dict)
    histoweave_version: str = ""
    container_digest: str | None = None
    code_revision: str | None = None
    executor: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

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


@dataclass
class SpatialTable:
    """A minimal, AnnData-shaped spatial container.

    Parameters
    ----------
    X
        Cell/spot x gene matrix (dense ``np.ndarray`` in this scaffold; a real
        deployment keeps this lazy/chunked via Dask-backed Zarr).
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
        ``layers["counts"]`` so count-based methods can recover them. Because
        layers are shape-aligned to ``X``, they are subset alongside it and never
        drift out of alignment (unlike an array parked in ``uns``).
    images
        Raster **spatial layers** keyed by name (SpatialData's ``images`` element),
        e.g. an H&E or DAPI image of the tissue. These live in a physical coordinate
        system, *not* on the observation axis, so — unlike ``obsm``/``layers`` — they
        are carried through :meth:`subset_obs` unchanged (dropping cells does not crop
        the tissue image).
    shapes
        Vector **spatial layers** keyed by name (SpatialData's ``shapes`` element),
        e.g. cell/nucleus boundary polygons or Visium spot circles. Held as opaque
        geometry objects (a ``geopandas.GeoDataFrame`` in a full deployment); like
        ``images`` they are coordinate-system aligned rather than obs-aligned.
    uns
        Unstructured annotations, including the ``"provenance"`` chain.

    Notes
    -----
    ``layers`` are *molecular* layers (obs x var); ``images``/``shapes`` are *spatial*
    layers (coordinate-system aligned). Carrying both is the prerequisite for a
    lossless bridge to **SpatialData**; the :meth:`to_anndata` bridge only round-trips
    the molecular/tabular part and therefore drops ``images``/``shapes``.
    """

    X: Matrix
    obs: pd.DataFrame
    var: pd.DataFrame
    obsm: dict[str, np.ndarray] = field(default_factory=dict)
    layers: dict[str, Matrix] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    shapes: dict[str, Any] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = _coerce_matrix(self.X)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D (cells x genes), got shape {self.X.shape}")
        n_obs, n_var = self.X.shape
        if len(self.obs) != n_obs:
            raise ValueError(f"obs has {len(self.obs)} rows but X has {n_obs} observations")
        if len(self.var) != n_var:
            raise ValueError(f"var has {len(self.var)} rows but X has {n_var} variables")
        if not self.obs.index.is_unique:
            raise ValueError("obs index must contain unique observation identifiers")
        if not self.var.index.is_unique:
            raise ValueError("var index must contain unique feature identifiers")
        if self.obs.index.hasnans:
            raise ValueError("obs index must not contain missing identifiers")
        if self.var.index.hasnans:
            raise ValueError("var index must not contain missing identifiers")
        for name, value in self.obsm.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != n_obs:
                raise ValueError(
                    f"obsm {name!r} has shape {array.shape}; first dimension must be n_obs={n_obs}"
                )
            if name == SPATIAL_KEY and (array.ndim != 2 or array.shape[1] not in (2, 3)):
                raise ValueError(
                    f"obsm[{SPATIAL_KEY!r}] must have shape (n_obs, 2 or 3), got {array.shape}"
                )
            self.obsm[name] = array
        for name, layer in self.layers.items():
            layer = _coerce_matrix(layer)
            if layer.shape != self.X.shape:
                raise ValueError(
                    f"layer {name!r} has shape {layer.shape}, expected {self.X.shape} to match X"
                )
            self.layers[name] = layer
        self.uns.setdefault("provenance", [])

    # -- shape helpers ---------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]

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
        return self.obsm.get(SPATIAL_KEY)

    # -- provenance ------------------------------------------------------------
    def record(self, prov: Provenance) -> None:
        """Append a provenance entry describing how this object was transformed."""
        self.uns.setdefault("provenance", []).append(prov.to_dict())

    @property
    def provenance(self) -> list[dict[str, Any]]:
        return self.uns.get("provenance", [])

    def copy(self) -> SpatialTable:
        return SpatialTable(
            X=self.X.copy(),
            obs=self.obs.copy(),
            var=self.var.copy(),
            obsm={k: v.copy() for k, v in self.obsm.items()},
            layers={k: v.copy() for k, v in self.layers.items()},
            images=copy.deepcopy(self.images),
            shapes=copy.deepcopy(self.shapes),
            uns=copy.deepcopy(self.uns),
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
            obsm={k: v[mask] for k, v in self.obsm.items()},
            layers={k: v[mask] for k, v in self.layers.items()},
            images=copy.deepcopy(self.images),
            shapes=copy.deepcopy(self.shapes),
            uns=copy.deepcopy(self.uns),
        )

    # -- AnnData bridge --------------------------------------------------------
    def to_anndata(self) -> AnnData:
        """Convert to :class:`anndata.AnnData` (requires the ``spatial`` extra).

        AnnData models only the molecular/tabular layer, so the spatial layers
        (``images``/``shapes``) are **not** carried over — use a SpatialData bridge
        when those must be preserved.
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
            obsm=dict(self.obsm),
            layers={k: v.copy() for k, v in self.layers.items()},
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
            layers={k: v.copy() for k, v in adata.layers.items()},
            uns=dict(adata.uns),
        )

    def __repr__(self) -> str:
        obsm_keys = ", ".join(self.obsm) or "-"
        layer_keys = ", ".join(self.layers) or "-"
        spatial_keys = ", ".join([*self.images, *self.shapes]) or "-"
        return (
            f"SpatialTable(n_obs={self.n_obs}, n_vars={self.n_vars}, "
            f"obsm=[{obsm_keys}], layers=[{layer_keys}], spatial=[{spatial_keys}], "
            f"provenance_steps={len(self.provenance)})"
        )


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
