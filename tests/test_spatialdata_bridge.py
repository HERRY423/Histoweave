"""Round-trip tests for the SpatialTable <-> SpatialData bridge.

These exercise the additive, lossless-for-spatial-layers bridge added in the
SpatialData backend migration. They are skipped when the optional ``spatial``
extra (spatialdata + geopandas + shapely) is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from histoweave.data import SpatialTable

spatialdata = pytest.importorskip("spatialdata")


def _make_table(seed: int = 0) -> SpatialTable:
    rng = np.random.default_rng(seed)
    n, g = 24, 6
    X = rng.poisson(2, size=(n, g)).astype(float)
    obs = pd.DataFrame(
        {"batch": ["s1"] * 12 + ["s2"] * 12},
        index=[f"cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(g)])
    st = SpatialTable(X=X, obs=obs, var=var)
    st.obsm["spatial"] = rng.random((n, 2)) * 100.0
    st.obsm["X_pca"] = rng.random((n, 4))
    st.layers["counts"] = X.copy()
    st.images["he"] = (rng.random((16, 20, 3)) * 255).astype(np.uint8)
    st.shapes["spots"] = st.obsm["spatial"].copy()
    return st


def test_to_spatialdata_builds_valid_elements() -> None:
    st = _make_table()
    sdata = st.to_spatialdata()
    assert "table" in sdata.tables
    assert "he" in sdata.images
    assert "spots" in sdata.shapes
    # table preserves obs/var dimensions
    table = sdata.tables["table"]
    assert table.n_obs == st.n_obs
    assert table.n_vars == st.n_vars


def test_round_trip_preserves_molecular_and_spatial_layers() -> None:
    st = _make_table()
    back = SpatialTable.from_spatialdata(st.to_spatialdata())

    assert back.shape == st.shape
    np.testing.assert_allclose(np.asarray(back.X), np.asarray(st.X))
    np.testing.assert_allclose(back.obsm["spatial"], st.obsm["spatial"])
    np.testing.assert_allclose(back.obsm["X_pca"], st.obsm["X_pca"])
    # 3-channel H&E image survives the channel-last -> channel-first -> channel-last trip
    assert back.images["he"].shape == (16, 20, 3)
    assert "spots" in back.shapes


def test_two_dim_image_round_trips_to_2d() -> None:
    st = _make_table()
    st.images.clear()
    rng = np.random.default_rng(1)
    st.images["dapi"] = (rng.random((12, 14)) * 255).astype(np.uint8)
    back = SpatialTable.from_spatialdata(st.to_spatialdata())
    assert back.images["dapi"].shape == (12, 14)


def test_from_spatialdata_requires_table() -> None:
    from spatialdata import SpatialData

    empty = SpatialData()
    with pytest.raises(ValueError, match="no table element"):
        SpatialTable.from_spatialdata(empty)


def test_from_spatialdata_ambiguous_tables_raise() -> None:
    st = _make_table()
    sdata = st.to_spatialdata()
    sdata.tables["table2"] = st.to_spatialdata().tables["table"]
    with pytest.raises(ValueError, match="multiple tables"):
        SpatialTable.from_spatialdata(sdata)
    # explicit selection resolves the ambiguity
    back = SpatialTable.from_spatialdata(sdata, table_name="table")
    assert back.shape == st.shape
