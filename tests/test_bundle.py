"""Round-trip tests for the portable SpatialTable bundle (io.bundle)."""

import numpy as np
import pytest

from histoweave.datasets import make_synthetic
from histoweave.io import (
    BundleIntegrityError,
    BundleSerializationError,
    inspect_bundle,
    read_bundle,
    write_bundle,
)
from histoweave.plugins import create_method


def test_bundle_roundtrip_preserves_core(tmp_path):
    data = make_synthetic(n_cells=60, n_genes=20, seed=0)
    # Populate layers and spatial slots so the whole container is exercised.
    data = create_method("normalization", "log1p_cp10k").run(data)  # adds layers['counts']
    data.images["he"] = np.zeros((1, 4, 4))
    data.shapes["cells"] = {"kind": "polygons", "n": 4}

    bundle = write_bundle(data, tmp_path / "b.ttab")
    restored = read_bundle(bundle)

    assert (bundle / "bundle.json").is_file()
    assert inspect_bundle(bundle)["table"]["shape"] == [data.n_obs, data.n_vars]
    assert restored.shape == data.shape
    assert np.allclose(restored.X, data.X)
    assert list(restored.obs_names) == list(data.obs_names)
    assert list(restored.var_names) == list(data.var_names)
    assert np.allclose(restored.spatial, data.spatial)
    assert np.allclose(restored.layers["counts"], data.layers["counts"])
    assert restored.images["he"].shape == (1, 4, 4)
    assert restored.shapes["cells"] == {"kind": "polygons", "n": 4}
    assert restored.uns["marker_genes"] == data.uns["marker_genes"]


def test_bundle_preserves_categorical_and_numpy_uns(tmp_path):
    data = make_synthetic(n_cells=40, seed=2)
    data = create_method("normalization", "log1p_cp10k").run(data)
    data = create_method("domain_detection", "kmeans").run(data)

    write_bundle(data, tmp_path / "b.ttab")
    restored = read_bundle(tmp_path / "b.ttab")

    # Categorical domain labels survive the parquet round-trip.
    assert list(restored.obs["domain"]) == list(data.obs["domain"])
    # numpy-typed values in uns (n_domains is a plain int here) serialize cleanly.
    assert restored.uns["n_domains"] == data.uns["n_domains"]
    assert "X_pca" in restored.obsm


def test_bundle_detects_corrupt_artifact(tmp_path):
    bundle = write_bundle(make_synthetic(n_cells=20, n_genes=8), tmp_path / "b.ttab")
    x_path = bundle / "X.npy"
    x_path.write_bytes(x_path.read_bytes() + b"corrupt")

    with pytest.raises(BundleIntegrityError, match="mismatch"):
        read_bundle(bundle)


def test_bundle_overwrite_is_atomic_and_removes_stale_keys(tmp_path):
    bundle = tmp_path / "b.ttab"
    first = make_synthetic(n_cells=20, n_genes=8)
    first.obsm["stale"] = np.ones((first.n_obs, 2))
    write_bundle(first, bundle)

    second = make_synthetic(n_cells=20, n_genes=8, seed=2)
    write_bundle(second, bundle, overwrite=True)
    restored = read_bundle(bundle)

    assert "stale" not in restored.obsm
    assert np.array_equal(restored.X, second.X)


def test_bundle_encodes_mapping_keys_instead_of_treating_them_as_paths(tmp_path):
    data = make_synthetic(n_cells=20, n_genes=8)
    data.obsm["../outside"] = np.ones((data.n_obs, 2))
    bundle = write_bundle(data, tmp_path / "safe.ttab")

    assert "../outside" in read_bundle(bundle).obsm
    assert not (tmp_path / "outside.npy").exists()


def test_bundle_refuses_silent_shape_loss_and_leaves_no_partial_output(tmp_path):
    data = make_synthetic(n_cells=20, n_genes=8)
    data.shapes["cells"] = object()
    bundle = tmp_path / "bad.ttab"

    with pytest.raises(BundleSerializationError, match="refusing a lossy bundle"):
        write_bundle(data, bundle)
    assert not bundle.exists()


def test_bundle_requires_explicit_overwrite(tmp_path):
    bundle = write_bundle(make_synthetic(n_cells=20, n_genes=8), tmp_path / "b.ttab")
    with pytest.raises(FileExistsError, match="already exists"):
        write_bundle(make_synthetic(n_cells=20, n_genes=8, seed=3), bundle)
