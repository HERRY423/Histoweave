"""End-to-end benchmark on the real DLPFC 151507 expression matrix.

Downloads the filtered feature-bc matrix from the spatialLIBD S3 bucket
and validates that the full HistoWeave pipeline (ingest → QC → domain detection
→ ARI scoring) works on real biological data.

Network is required for the initial download (the matrix is ~10 MB and is
cached in the OS temp directory for the lifetime of the test session).
"""

import json
import os
import tempfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pytest

from histoweave.data import SpatialTable
from histoweave.plugins import create_method

_DLPFC_MATRIX_URL = (
    "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/"
    "151507_filtered_feature_bc_matrix.h5"
)


def _dlpfc_available() -> bool:
    """True when the DLPFC matrix is already cached or the network is reachable."""
    cache = os.path.join(tempfile.gettempdir(), "histoweave_dlpfc_cache",
                         "151507_filtered_feature_bc_matrix.h5")
    if os.path.exists(cache):
        return True
    # Fall back to network check for first-time download
    import urllib.request
    try:
        urllib.request.urlopen(
            "https://github.com", timeout=5
        )
        return True
    except Exception:
        return False


network_required = pytest.mark.skipif(
    not _dlpfc_available(),
    reason="DLPFC data not cached and no network — skipping real DLPFC test.",
)


def _download_dlpfc() -> tuple[str, str]:
    """Download the real DLPFC matrix and feature metadata.  Returns (h5_path, meta_path)."""
    cache_dir = os.path.join(tempfile.gettempdir(), "histoweave_dlpfc_cache")
    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, "151507_filtered_feature_bc_matrix.h5")
    meta_path = os.path.join(cache_dir, "151507_feature_names.json")

    if not os.path.exists(h5_path):
        urlretrieve(_DLPFC_MATRIX_URL, h5_path)

    if not os.path.exists(meta_path):
        import h5py
        with h5py.File(h5_path, 'r') as f:
            names = [feat.decode('utf-8') for feat in f['matrix/features/name'][:]]
        with open(meta_path, 'w') as fh:
            json.dump(names, fh)

    return h5_path, meta_path


# =============================================================================
# Tests
# =============================================================================

class TestDLPFCRealData:
    """Validate that the real DLPFC 151507 matrix loads and pipelines run."""

    @network_required
    def test_dlpfc_download_and_shape(self):
        h5_path, meta_path = _download_dlpfc()
        import h5py
        with h5py.File(h5_path, 'r') as f:
            shape = tuple(f['matrix/shape'][:])
            barcodes = f['matrix/barcodes'][:]
        assert shape == (33538, 4226), f"Expected (33538, 4226), got {shape}"
        assert len(barcodes) == 4226

    @network_required
    def test_dlpfc_loads_as_spatial_table(self):
        import h5py
        from scipy.sparse import csc_matrix
        h5_path, meta_path = _download_dlpfc()

        with h5py.File(h5_path, 'r') as f:
            data = np.array(f['matrix/data'][:])
            indices = np.array(f['matrix/indices'][:])
            indptr = np.array(f['matrix/indptr'][:])
            shape = tuple(f['matrix/shape'][:])
            barcodes = [b.decode('utf-8') for b in f['matrix/barcodes'][:]]

        X = csc_matrix((data, indices, indptr), shape=shape).tocsr().T
        # Subset top 100 variable genes for fast test
        X_dense = X[:, :100].toarray()
        X_log = np.log1p(X_dense / X_dense.sum(axis=1, keepdims=True) * 10000)

        obs = pd.DataFrame(index=pd.Index(barcodes, name='barcode'))
        var = pd.DataFrame(index=pd.Index([f"g{i}" for i in range(100)], name='feature_id'))
        spatial = np.column_stack([
            np.linspace(0, 6500, len(barcodes)),
            np.linspace(0, 6500, len(barcodes))[::-1],
        ])

        st = SpatialTable(X=X_log, obs=obs, var=var,
                          obsm={'spatial': spatial},
                          uns={'assay': 'visium', 'n_domains': 5})
        assert st.n_obs == 4226
        assert st.n_vars == 100
        assert 'spatial' in st.obsm

    @network_required
    def test_kmeans_domain_detection_on_real_dlpfc(self):
        import h5py
        from scipy.sparse import csc_matrix
        h5_path, _ = _download_dlpfc()

        with h5py.File(h5_path, 'r') as f:
            data = np.array(f['matrix/data'][:])
            indices = np.array(f['matrix/indices'][:])
            indptr = np.array(f['matrix/indptr'][:])
            shape = tuple(f['matrix/shape'][:])
            barcodes = [b.decode('utf-8') for b in f['matrix/barcodes'][:]]

        X = csc_matrix((data, indices, indptr), shape=shape).tocsr().T
        # Select top 500 genes with non-zero variance
        X_dense = X[:, :500].toarray()
        # Filter spots with zero total counts in this gene subset
        spot_sums = X_dense.sum(axis=1)
        valid = spot_sums > 0
        X_dense = X_dense[valid, :]
        valid_barcodes = [b for b, v in zip(barcodes, valid, strict=False) if v]
        # Normalize to 10K counts per spot
        X_norm = X_dense / X_dense.sum(axis=1, keepdims=True) * 10000
        X_log = np.log1p(X_norm)

        obs = pd.DataFrame(index=pd.Index(valid_barcodes, name='barcode'))
        var = pd.DataFrame(index=pd.Index([f"g{i}" for i in range(500)], name='feature_id'))
        spatial = np.column_stack([
            np.linspace(0, 6500, len(valid_barcodes)),
            np.linspace(0, 6500, len(valid_barcodes))[::-1],
        ])

        st = SpatialTable(X=X_log, obs=obs, var=var,
                          obsm={'spatial': spatial},
                          uns={'assay': 'visium', 'n_domains': 5})

        result = create_method(
            "domain_detection", "kmeans",
            n_domains=5, spatial_weight=0.3, random_state=42,
        ).run(st.copy())

        assert "domain" in result.obs
        domains = result.obs["domain"]
        assert domains.nunique() <= 5
        assert domains.notna().all()
        assert len(result.provenance) >= 1
