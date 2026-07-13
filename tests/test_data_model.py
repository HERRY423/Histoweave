from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from histoweave.data import Provenance, SpatialTable


def _tiny():
    return SpatialTable(
        X=np.arange(12, dtype=float).reshape(4, 3),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        obsm={"spatial": np.random.default_rng(0).random((4, 2))},
    )


def test_shape_and_accessors():
    t = _tiny()
    assert t.shape == (4, 3)
    assert t.n_obs == 4 and t.n_vars == 3
    assert list(t.var_names) == ["g0", "g1", "g2"]
    assert t.spatial.shape == (4, 2)


def test_validation_rejects_mismatched_obs():
    with pytest.raises(ValueError):
        SpatialTable(
            X=np.zeros((4, 3)),
            obs=pd.DataFrame(index=["a", "b"]),  # wrong length
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )


def test_validation_rejects_duplicate_identifiers():
    with pytest.raises(ValueError, match="obs index"):
        SpatialTable(
            X=np.zeros((2, 2)),
            obs=pd.DataFrame(index=["cell", "cell"]),
            var=pd.DataFrame(index=["g0", "g1"]),
        )


def test_validation_rejects_misaligned_or_invalid_spatial_coordinates():
    with pytest.raises(ValueError, match="first dimension"):
        SpatialTable(
            X=np.zeros((2, 2)),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
            obsm={"embedding": np.zeros((1, 3))},
        )
    with pytest.raises(ValueError, match="2 or 3"):
        SpatialTable(
            X=np.zeros((2, 2)),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
            obsm={"spatial": np.zeros((2, 1))},
        )


def test_subset_obs_keeps_alignment():
    t = _tiny()
    sub = t.subset_obs(np.array([True, False, True, False]))
    assert sub.n_obs == 2
    assert list(sub.obs_names) == ["c0", "c2"]
    assert sub.obsm["spatial"].shape == (2, 2)


def test_provenance_records_appended():
    t = _tiny()
    assert t.provenance == []
    t.record(Provenance(step="qc", method="basic_qc", method_version="0.1"))
    assert len(t.provenance) == 1
    assert t.provenance[0]["method"] == "basic_qc"


def test_copy_is_deep():
    t = _tiny()
    t2 = t.copy()
    t2.X[0, 0] = 999
    assert t.X[0, 0] == 0


def test_layers_must_match_x_shape():
    with pytest.raises(ValueError, match="layer"):
        SpatialTable(
            X=np.zeros((4, 3)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
            layers={"counts": np.zeros((4, 2))},  # wrong width
        )



def test_sparse_x_and_layers_stay_sparse_through_copy_and_subset():
    sparse = pytest.importorskip("scipy.sparse")
    X = sparse.csc_matrix(np.arange(12, dtype=float).reshape(4, 3))
    counts = sparse.coo_matrix(X)
    table = SpatialTable(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        layers={"counts": counts},
    )

    assert sparse.isspmatrix_csr(table.X)
    assert sparse.isspmatrix_csr(table.layers["counts"])

    copied = table.copy()
    copied.X[0, 1] = 999
    assert table.X[0, 1] == 1

    subset = table.subset_obs(np.array([True, False, True, False]))
    assert sparse.isspmatrix_csr(subset.X)
    assert sparse.isspmatrix_csr(subset.layers["counts"])
    assert subset.shape == (2, 3)
    assert list(subset.obs_names) == ["c0", "c2"]


def test_from_anndata_preserves_sparse_x_and_layers():
    sparse = pytest.importorskip("scipy.sparse")
    adata = SimpleNamespace(
        X=sparse.csc_matrix(np.eye(3)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
        obsm={"spatial": np.zeros((3, 2))},
        layers={"counts": sparse.coo_matrix(np.eye(3))},
        uns={"source": "test"},
    )

    table = SpatialTable.from_anndata(adata)
    assert sparse.isspmatrix_csr(table.X)
    assert sparse.isspmatrix_csr(table.layers["counts"])
    assert table.uns["source"] == "test"



def test_spatial_layers_survive_subset_unchanged():
    # images/shapes are coordinate-aligned, not obs-aligned: subsetting cells must not
    # crop the tissue image or drop geometries.
    t = _tiny()
    t.images["he"] = np.arange(2 * 5 * 5).reshape(2, 5, 5)  # (c, y, x), not obs-sized
    t.shapes["cells"] = {"kind": "polygons", "n": 4}
    sub = t.subset_obs(np.array([True, False, True, False]))
    assert sub.n_obs == 2
    assert sub.images["he"].shape == (2, 5, 5)
    assert np.array_equal(sub.images["he"], t.images["he"])
    assert sub.shapes["cells"] == {"kind": "polygons", "n": 4}


def test_spatial_layers_copy_is_deep():
    t = _tiny()
    t.images["he"] = np.ones((1, 3, 3))
    t.shapes["cells"] = {"n": 4}
    t2 = t.copy()
    t2.images["he"][0, 0, 0] = 999
    t2.shapes["cells"]["n"] = 99
    assert t.images["he"][0, 0, 0] == 1
    assert t.shapes["cells"]["n"] == 4


def test_repr_lists_spatial_layers():
    t = _tiny()
    t.images["he"] = np.zeros((1, 2, 2))
    assert "spatial=[he]" in repr(t)
