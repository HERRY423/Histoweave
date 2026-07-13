"""Native reader/writer for the 10x Genomics feature-barcode HDF5 matrix.

Both Space Ranger (Visium) and Xenium emit their count matrix in the same
CellRanger-style ``.h5`` layout::

    /matrix
        data       (nnz,)      CSC non-zero values
        indices    (nnz,)      CSC row (feature) indices
        indptr     (C+1,)      CSC column (barcode) pointers
        shape      (2,)        [n_features, n_barcodes]
        barcodes   (C,)        cell/spot ids (bytes)
        features/
            id            (G,)  stable feature ids  (e.g. ENSEMBL / Xenium gene id)
            name          (G,)  human-readable gene symbol
            feature_type  (G,)  "Gene Expression", "Negative Control Probe", ...
            genome        (G,)  reference genome label

This module reads/writes that layout with nothing heavier than ``h5py`` + ``numpy``,
so the ingestion path works without the full ``spatialdata``/``anndata`` stack. The
matrix is materialized dense — fine for the small canonical datasets this scaffold
targets; a production reader keeps it sparse/chunked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TenxMatrix:
    """A parsed 10x matrix: dense counts plus per-feature / per-barcode metadata."""

    X: np.ndarray  # (n_barcodes, n_features) — cells/spots x genes
    feature_ids: list[str]
    feature_names: list[str]
    feature_types: list[str]
    barcodes: list[str]
    genome: str = "unknown"


def _decode(values) -> list[str]:
    """Coerce an h5py string dataset (bytes or str) to a plain ``list[str]``."""
    out = []
    for v in np.asarray(values).tolist():
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
        raise ModuleNotFoundError(
            "Reading/writing 10x .h5 matrices requires h5py. "
            "Install the io extra with: pip install 'histoweave-spatial[io]'"
        ) from exc
    return h5py


def read_10x_h5(path: str) -> TenxMatrix:
    """Parse a CellRanger/Xenium ``.h5`` feature-barcode matrix into a :class:`TenxMatrix`."""
    h5py = _import_h5py()

    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise ValueError(
                f"{path!r} is not a 10x feature-barcode HDF5 file (no '/matrix' group)."
            )
        grp = f["matrix"]
        data = np.asarray(grp["data"][:])
        indices = np.asarray(grp["indices"][:])
        indptr = np.asarray(grp["indptr"][:])
        n_features, n_barcodes = (int(x) for x in grp["shape"][:])
        barcodes = _decode(grp["barcodes"][:])

        feats = grp["features"]
        feature_ids = _decode(feats["id"][:])
        feature_names = _decode(feats["name"][:]) if "name" in feats else list(feature_ids)
        feature_types = (
            _decode(feats["feature_type"][:])
            if "feature_type" in feats
            else ["Gene Expression"] * n_features
        )
        genome = _decode(feats["genome"][:])[0] if "genome" in feats else "unknown"

    # Reconstruct the dense (features x barcodes) CSC matrix, then transpose to the
    # canonical cells x genes orientation.  When scipy is available the sparse CSC is
    # assembled in C (one call, no Python loop); otherwise a numpy fallback handles it.
    try:
        from scipy.sparse import csc_matrix

        sparse = csc_matrix(
            (data, indices, indptr), shape=(n_features, n_barcodes)
        )
        dense = sparse.toarray()
    except ModuleNotFoundError:
        dense = np.zeros(
            (n_features, n_barcodes), dtype=data.dtype if data.size else float
        )
        for col in range(n_barcodes):
            start, stop = int(indptr[col]), int(indptr[col + 1])
            dense[indices[start:stop], col] = data[start:stop]

    return TenxMatrix(
        X=dense.T.astype(float),
        feature_ids=feature_ids,
        feature_names=feature_names,
        feature_types=feature_types,
        barcodes=barcodes,
        genome=genome,
    )


def write_10x_h5(
    path: str,
    X: np.ndarray,
    feature_ids: list[str],
    feature_names: list[str],
    barcodes: list[str],
    *,
    feature_types: list[str] | None = None,
    genome: str = "synthetic",
) -> None:
    """Write a cells x genes matrix to the 10x feature-barcode HDF5 layout.

    Used to fabricate format-faithful fixtures; the inverse of :func:`read_10x_h5`.
    """
    h5py = _import_h5py()

    X = np.asarray(X)
    n_barcodes, n_features = X.shape
    if len(feature_ids) != n_features or len(barcodes) != n_barcodes:
        raise ValueError("feature_ids/barcodes lengths must match X's shape")
    feature_types = feature_types or ["Gene Expression"] * n_features

    # Build CSC over the (features x barcodes) transpose: one column per barcode.
    features_by_cells = X.T
    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for col in range(n_barcodes):
        column = features_by_cells[:, col]
        nz = np.nonzero(column)[0]
        indices.extend(int(i) for i in nz)
        data.extend(float(v) for v in column[nz])
        indptr.append(len(data))

    def _bytes(strings: list[str]) -> np.ndarray:
        return np.array([s.encode() for s in strings])

    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("data", data=np.array(data, dtype=np.int32))
        m.create_dataset("indices", data=np.array(indices, dtype=np.int64))
        m.create_dataset("indptr", data=np.array(indptr, dtype=np.int64))
        m.create_dataset("shape", data=np.array([n_features, n_barcodes], dtype=np.int64))
        m.create_dataset("barcodes", data=_bytes(barcodes))
        feats = m.create_group("features")
        feats.create_dataset("id", data=_bytes(feature_ids))
        feats.create_dataset("name", data=_bytes(feature_names))
        feats.create_dataset("feature_type", data=_bytes(feature_types))
        feats.create_dataset("genome", data=_bytes([genome] * n_features))
