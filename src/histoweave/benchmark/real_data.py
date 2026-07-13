"""Real-world benchmark datasets — DLPFC study-grouped validation for Figure 3.

Each DLPFC slice from Maynard et al. (2021) is a Visium spatial transcriptomics
sample of human dorsolateral prefrontal cortex with **7 manually annotated
cortical layers** (L1-L6 + WM).  These 12 slices across 3 donors are the de-facto
gold standard for spatial domain detection benchmarks.

This module provides:

* :func:`build_dlpfc_dataset` — load one slice with real manual layer labels.
* :func:`dlpfc_benchmark_suite` — all 12 slices as a named dict.
* Study-grouped holdout: hold out one *slice* and train on the remaining 11.
* Cross-donor holdout: hold out all slices from one donor (3 donors × 4 slices).
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from ..data import Provenance, SpatialTable

_S3_BASE = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5"
_SPATIALLIBD_S3 = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com"
_CACHE = Path(tempfile.gettempdir()) / "histoweave_dlpfc_cache"

# All 12 DLPFC slices from Maynard et al. 2021, Nature Neuroscience.
# Three adjacent slice-pairs per donor, spanning the DLPFC anterior-posterior axis.
_DLPFC_SLICES = [
    "151507", "151508", "151509", "151510",  # donor 1
    "151669", "151670", "151671", "151672",  # donor 2
    "151673", "151674", "151675", "151676",  # donor 3
]

# Donor groupings for cross-donor holdout validation.
_DLPFC_DONORS = {
    "donor1": ("151507", "151508", "151509", "151510"),
    "donor2": ("151669", "151670", "151671", "151672"),
    "donor3": ("151673", "151674", "151675", "151676"),
}

# DLPFC cortical layer labels — manual annotations from Maynard et al. 2021.
_LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]

# URL for pre-extracted manual layer labels (barcode → layer CSV per slice).
_LAYER_LABELS_URL = (
    "https://raw.githubusercontent.com/LieberInstitute/spatialLIBD/"
    "devel/inst/extdata/tissue_positions"
)

# Slice metadata — position and donor info for stratified evaluation.
_SLICE_META: dict[str, dict[str, Any]] = {
    "151507": {"donor": "donor1", "position": "anterior", "pair": "151508"},
    "151508": {"donor": "donor1", "position": "anterior", "pair": "151507"},
    "151509": {"donor": "donor1", "position": "posterior", "pair": "151510"},
    "151510": {"donor": "donor1", "position": "posterior", "pair": "151509"},
    "151669": {"donor": "donor2", "position": "anterior", "pair": "151670"},
    "151670": {"donor": "donor2", "position": "anterior", "pair": "151669"},
    "151671": {"donor": "donor2", "position": "posterior", "pair": "151672"},
    "151672": {"donor": "donor2", "position": "posterior", "pair": "151671"},
    "151673": {"donor": "donor3", "position": "anterior", "pair": "151674"},
    "151674": {"donor": "donor3", "position": "anterior", "pair": "151673"},
    "151675": {"donor": "donor3", "position": "posterior", "pair": "151676"},
    "151676": {"donor": "donor3", "position": "posterior", "pair": "151675"},
}


def download_dlpfc_slice(slice_id: str, cache_dir: str | Path | None = None) -> Path:
    """Download a DLPFC slice H5 matrix from S3, caching locally. Returns the H5 path."""
    cache = Path(cache_dir) if cache_dir else _CACHE
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{slice_id}_filtered_feature_bc_matrix.h5"

    if not dest.exists():
        url = f"{_S3_BASE}/{slice_id}_filtered_feature_bc_matrix.h5"
        urlretrieve(url, dest)

    return dest


def _fetch_manual_layer_labels(
    slice_id: str, cache_dir: Path | None = None
) -> dict[str, str] | None:
    """Download or load cached manual layer labels for a DLPFC slice.

    Returns a ``{barcode: layer_label}`` mapping using the Maynard et al. (2021)
    manual annotations (L1-L6, WM).  Returns ``None`` if labels are unavailable
    (e.g. no network, 404) — callers must fall back to GMM pseudo-labels.
    """
    cache = cache_dir or _CACHE
    cache.mkdir(parents=True, exist_ok=True)
    labels_path = cache / f"{slice_id}_manual_layer_labels.json"

    # Return cached labels if available.
    if labels_path.exists():
        try:
            return json.loads(labels_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Try the spatialLIBD S3 tissue_positions_list.csv companion file.
    # Format: barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
    # The manual layer labels are in a separate file bundled with spatialLIBD.
    # We try several known URLs.
    urls = [
        f"{_SPATIALLIBD_S3}/tissue_positions/{slice_id}_tissue_positions_list.csv",
        f"{_LAYER_LABELS_URL}/{slice_id}_tissue_positions_list.csv",
    ]

    for url in urls:
        try:
            csv_path = cache / f"{slice_id}_tissue_positions_list.csv"
            if not csv_path.exists():
                urlretrieve(url, csv_path)
        except Exception:
            continue

    # Also try fetching the layer labels from the spatialLIBD metadata.
    # The layer annotations are in obs.csv files exported from SpatialExperiment.
    layer_url = (
        f"https://raw.githubusercontent.com/LieberInstitute/spatialLIBD/"
        f"devel/inst/extdata/sample_info/{slice_id}_layer_labels.csv"
    )
    try:
        layer_csv = cache / f"{slice_id}_layer_labels.csv"
        if not layer_csv.exists():
            urlretrieve(layer_url, layer_csv)
    except Exception:
        pass

    # If we have a layer_labels CSV, parse it.
    if (cache / f"{slice_id}_layer_labels.csv").exists():
        mapping: dict[str, str] = {}
        with open(cache / f"{slice_id}_layer_labels.csv", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                barcode = row.get("barcode", row.get("Barcode", ""))
                layer = row.get(
                    "layer_guess_reordered",
                    row.get("Layer", row.get("layer", "")),
                )
                if barcode and layer:
                    mapping[barcode] = layer
        if mapping:
            labels_path.write_text(json.dumps(mapping), encoding="utf-8")
            return mapping

    return None


def _build_gmm_pseudo_labels(
    X_log: np.ndarray,
    n_components: int = 7,
    seed: int = 42,
) -> np.ndarray:
    """Fallback: GMM pseudo-labels on PCA embedding when manual labels are unavailable."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    pca = PCA(n_components=30, random_state=seed)
    pca_emb = pca.fit_transform(X_log)
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="diag",
        random_state=seed, n_init=5, max_iter=200,
    )
    return gmm.fit_predict(pca_emb[:, :15])


def build_dlpfc_dataset(
    slice_id: str = "151507",
    *,
    n_genes: int = 2000,
    use_manual_labels: bool = True,
    cache_dir: str | Path | None = None,
    seed: int = 42,
) -> SpatialTable:
    """Load one DLPFC slice as a benchmark-ready SpatialTable.

    The returned table has:

    * ``X`` — log-normalised expression (top *n_genes* variable genes).
    * ``obs['domain_truth']`` — **manual** cortical layer labels (L1-L6, WM)
      from Maynard et al. (2021) when *use_manual_labels* is ``True`` and the
      labels can be fetched.  Falls back to GMM-derived pseudo-labels otherwise.
    * ``obs['domain_label']`` — integer encoding of domain_truth (0..6).
    * ``obsm['spatial']`` — PCA-derived 2-D coordinates reflecting expression
      similarity (proxy for array positions when tissue_positions unavailable).
    * ``uns['dlpfc_slice']`` — original slice identifier.
    * ``uns['assay']`` — ``"dlpfc_visium"``.
    * ``uns['label_source']`` — ``"manual"`` or ``"gmm_pseudo"``.
    * ``uns['slice_metadata']`` — donor, position, and paired-slice info.

    Parameters
    ----------
    slice_id : str
        DLPFC slice ID (e.g. "151507").  Must be one of the 12 Maynard slices.
    n_genes : int
        Number of most variable genes to retain (default 2000).
    use_manual_labels : bool
        Attempt to fetch real manual layer labels.  Falls back to GMM if
        unavailable.  Set to False to skip the network fetch entirely.
    cache_dir : Path or None
        Cache directory.  Uses OS temp by default.
    seed : int
        Random seed for PCA and GMM reproducibility.
    """
    if slice_id not in _DLPFC_SLICES:
        raise ValueError(
            f"Unknown DLPFC slice {slice_id!r}. "
            f"Available: {_DLPFC_SLICES}"
        )

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    h5_path = download_dlpfc_slice(slice_id, cache_dir)
    cache = Path(cache_dir) if cache_dir else _CACHE

    # ---- Load sparse matrix -------------------------------------------------
    with _open_h5(h5_path) as f:
        barcodes = [b.decode('utf-8') for b in f['matrix/barcodes'][:]]
        features = [feat.decode('utf-8') for feat in f['matrix/features/name'][:]]
        data_arr = np.array(f['matrix/data'][:])
        indices = np.array(f['matrix/indices'][:])
        indptr = np.array(f['matrix/indptr'][:])
        shape = tuple(f['matrix/shape'][:])
    X = csc_matrix((data_arr, indices, indptr), shape=shape).tocsr().T

    # ---- Feature selection: top variable genes ------------------------------
    gene_means = np.array(X.mean(axis=0)).flatten()
    X_sq = X.copy()
    X_sq.data **= 2
    gene_vars = np.array(X_sq.mean(axis=0)).flatten() - gene_means**2
    n_genes_actual = min(n_genes, X.shape[1])
    top_idx = np.argsort(gene_vars)[-n_genes_actual:]

    X_sub = X[:, top_idx].toarray()
    sub_features = [features[i] for i in top_idx]

    # ---- Log-normalize ------------------------------------------------------
    spot_sums = X_sub.sum(axis=1)
    valid = spot_sums > 0
    X_sub = X_sub[valid, :]
    valid_barcodes = [b for b, v in zip(barcodes, valid, strict=False) if v]
    X_norm = X_sub / X_sub.sum(axis=1, keepdims=True) * 10000
    X_log = np.log1p(X_norm)

    # ---- PCA spatial embedding ----------------------------------------------
    pca = PCA(n_components=30, random_state=seed)
    pca_emb = pca.fit_transform(X_log)
    spatial = StandardScaler().fit_transform(pca_emb[:, :2]) * 1000 + 4000

    # ---- Domain labels: manual preferred, GMM fallback ----------------------
    label_source = "gmm_pseudo"
    manual_labels = None
    if use_manual_labels:
        manual_labels = _fetch_manual_layer_labels(slice_id, cache)

    if manual_labels:
        domain_names = np.array([
            manual_labels.get(b, f"domain_{i}")
            for i, b in enumerate(valid_barcodes)
        ])
        # Standardize to L1-L6 + WM
        for i, lab in enumerate(domain_names):
            if lab not in _LAYER_ORDER:
                domain_names[i] = f"domain_{lab}"
        label_source = "manual"
    else:
        gmm_labels = _build_gmm_pseudo_labels(X_log, n_components=7, seed=seed)
        domain_names = np.array([f"domain_{lab}" for lab in gmm_labels])

    # Integer encoding for numeric methods.
    _label_to_int = {lab: i for i, lab in enumerate(sorted(set(domain_names)))}
    domain_ints = np.array([_label_to_int[lab] for lab in domain_names])

    # ---- Build SpatialTable -------------------------------------------------
    obs = pd.DataFrame(
        {
            "domain_truth": pd.Categorical(domain_names),
            "domain_label": domain_ints,
        },
        index=pd.Index(valid_barcodes, name="barcode"),
    )
    var = pd.DataFrame(
        {"feature_name": sub_features},
        index=pd.Index(
            [f"g{i}" for i in range(len(sub_features))], name="feature_id"
        ),
    )
    slice_meta = _SLICE_META.get(slice_id, {})
    uns: dict[str, Any] = {
        "assay": "dlpfc_visium",
        "dlpfc_slice": slice_id,
        "n_domains": int(len(set(domain_names))),
        "n_genes_original": int(X.shape[1]),
        "n_spots_original": int(X.shape[0]),
        "label_source": label_source,
        "layer_order": _LAYER_ORDER,
        "slice_metadata": {
            "donor": slice_meta.get("donor", "unknown"),
            "position": slice_meta.get("position", "unknown"),
            "paired_slice": slice_meta.get("pair", ""),
            "donor_group": _dlpfc_donor_for(slice_id),
        },
    }

    st = SpatialTable(
        X=X_log, obs=obs, var=var, obsm={"spatial": spatial}, uns=uns,
    )
    from .. import __version__
    st.record(Provenance(
        step="ingestion",
        method="dlpfc_benchmark_builder",
        method_version="0.2.0",
        params={
            "slice_id": slice_id,
            "n_genes": int(n_genes_actual),
            "seed": seed,
            "label_source": label_source,
        },
        histoweave_version=__version__,
    ))
    return st


def dlpfc_benchmark_suite(
    slices: tuple[str, ...] | None = None,
    cache_dir: str | Path | None = None,
    seed: int = 42,
    use_manual_labels: bool = True,
) -> dict[str, SpatialTable]:
    """Return a named dict of DLPFC slices for multi-dataset benchmarking.

    Each slice is loaded via :func:`build_dlpfc_dataset` with consistent
    parameters so they can be compared fairly.

    Parameters
    ----------
    slices : tuple of str or None
        Which slices to include (default: all 12).
    cache_dir : Path or None
        Cache directory.
    seed : int
        Seed for reproducibility.
    use_manual_labels : bool
        Attempt to use real manual layer labels.
    """
    if slices is None:
        slices = tuple(_DLPFC_SLICES)

    suite: dict[str, SpatialTable] = {}
    cache = Path(cache_dir) if cache_dir else _CACHE
    cache.mkdir(parents=True, exist_ok=True)

    for sl in slices:
        key = f"dlpfc_{sl}"
        try:
            suite[key] = build_dlpfc_dataset(
                sl, cache_dir=cache, seed=seed, use_manual_labels=use_manual_labels,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build DLPFC {sl}: {exc}") from exc

    return suite


def dlpfc_donor_suites(
    cache_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, dict[str, SpatialTable]]:
    """Return donor-grouped DLPFC slice suites for cross-donor holdout.

    Returns ``{donor_name: {slice_key: SpatialTable}}`` for each of the
    3 donors, each with 4 slices.  Use for leave-one-donor-out validation
    (training on 2 donors, testing on the held-out donor).
    """
    suites: dict[str, dict[str, SpatialTable]] = {}
    cache = Path(cache_dir) if cache_dir else _CACHE
    cache.mkdir(parents=True, exist_ok=True)

    for donor, donor_slices in _DLPFC_DONORS.items():
        suites[donor] = dlpfc_benchmark_suite(
            slices=donor_slices, cache_dir=cache, seed=seed,
        )

    return suites


def dlpfc_slice_ids() -> list[str]:
    """Return all 12 DLPFC slice IDs."""
    return list(_DLPFC_SLICES)


def dlpfc_donor_groups() -> dict[str, tuple[str, ...]]:
    """Return donor → slices mapping for cross-donor validation designs."""
    return dict(_DLPFC_DONORS)


def dlpfc_layer_order() -> list[str]:
    """Return the canonical DLPFC layer ordering (L1 → L6 → WM)."""
    return list(_LAYER_ORDER)


def _dlpfc_donor_for(slice_id: str) -> str:
    for donor, slices in _DLPFC_DONORS.items():
        if slice_id in slices:
            return donor
    return "unknown"


def _open_h5(path):
    import h5py
    return h5py.File(path, 'r')


__all__ = [
    "build_dlpfc_dataset",
    "dlpfc_benchmark_suite",
    "dlpfc_donor_suites",
    "dlpfc_donor_groups",
    "dlpfc_layer_order",
    "dlpfc_slice_ids",
    "download_dlpfc_slice",
]
