"""Tests for the Harmony embedding-space batch-integration plugin.

The real-integration tests are skipped when ``harmonypy``/``scikit-learn`` (the
``harmony`` extra) are unavailable; the registration and error-handling tests
run without them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import histoweave  # noqa: F401  (triggers builtin registration)
from histoweave.data import SpatialTable
from histoweave.plugins import get_method


def _two_batch_table(seed: int = 0, shift: float = 6.0) -> SpatialTable:
    """Two batches sharing three cell-type clusters plus a per-batch offset."""
    rng = np.random.default_rng(seed)
    n_genes = 30
    centers = [rng.normal(0, 5, size=n_genes) for _ in range(3)]

    def make(offset: float) -> np.ndarray:
        blocks = [rng.normal(c, 1.0, size=(30, n_genes)) for c in centers]
        return np.vstack(blocks) + offset

    X = np.vstack([make(0.0), make(shift)])
    n = X.shape[0]
    obs = pd.DataFrame(
        {
            "batch": ["A"] * 90 + ["B"] * 90,
            "cell_type": (["t0"] * 30 + ["t1"] * 30 + ["t2"] * 30) * 2,
        },
        index=[f"cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    return SpatialTable(X=X, obs=obs, var=var)


def test_harmony_is_registered() -> None:
    cls = get_method("integration", "harmony")
    assert cls.spec.name == "harmony"
    assert cls.spec.category == "integration"


def test_harmony_missing_batch_key_raises() -> None:
    cls = get_method("integration", "harmony")
    st = _two_batch_table()
    with pytest.raises(ValueError, match="not found in obs"):
        cls(batch_key="nonexistent").run(st)


def test_harmony_reduces_batch_separation() -> None:
    pytest.importorskip("harmonypy")
    pytest.importorskip("sklearn")
    cls = get_method("integration", "harmony")
    st = _two_batch_table()
    out = cls(batch_key="batch", n_pcs=15, max_iter_harmony=20, seed=0).run(st)

    assert "X_pca_harmony" in out.obsm
    assert out.obsm["X_pca_harmony"].shape == (st.n_obs, 15)
    assert out.uns["integration"]["method"] == "harmony"
    assert out.uns["integration"]["n_batches"] == 2
    assert out.provenance[-1]["method"] == "harmony"

    is_a = (out.obs["batch"] == "A").to_numpy()

    def batch_centroid_dist(emb: np.ndarray) -> float:
        return float(np.linalg.norm(emb[is_a].mean(0) - emb[~is_a].mean(0)))

    raw = out.obsm["X_pca"]
    corrected = out.obsm["X_pca_harmony"]
    # Harmony should pull the two batch centroids closer together.
    assert batch_centroid_dist(corrected) < batch_centroid_dist(raw)


def test_harmony_single_batch_passes_through() -> None:
    pytest.importorskip("harmonypy")
    pytest.importorskip("sklearn")
    cls = get_method("integration", "harmony")
    st = _two_batch_table()
    st.obs["batch"] = "only"
    out = cls(batch_key="batch", n_pcs=10).run(st)
    assert "X_pca_harmony" in out.obsm
    assert out.uns["integration"]["n_batches"] == 1
    # X_pca is built and passed through unchanged.
    np.testing.assert_allclose(out.obsm["X_pca_harmony"], out.obsm["X_pca"])


def test_harmony_uses_existing_embedding() -> None:
    pytest.importorskip("harmonypy")
    cls = get_method("integration", "harmony")
    st = _two_batch_table()
    rng = np.random.default_rng(1)
    st.obsm["X_pca"] = rng.normal(size=(st.n_obs, 12))
    out = cls(batch_key="batch", use_rep="X_pca", max_iter_harmony=5).run(st)
    assert out.obsm["X_pca_harmony"].shape == (st.n_obs, 12)
