"""Histology helpers + real H&E path (offline fixtures + optional squidpy cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from histoweave.data import SpatialTable
from histoweave.datasets.histology import (
    IMAGE_KEY,
    attach_histology_images,
    ensure_histology,
    extract_images_from_anndata_uns,
    prepare_virtual_st_table,
    spatial_table_from_visium_hne,
)
from histoweave.plugins import MethodCategory, create_method
from histoweave.plugins.builtin import register_all

REPO = Path(__file__).resolve().parents[1]
VISIUM_HNE_CACHE = REPO / "data" / "anndata" / "visium_hne_adata.h5ad"


@pytest.fixture(scope="module", autouse=True)
def _register() -> None:
    register_all()


def test_extract_images_from_uns_spatial() -> None:
    hires = np.random.default_rng(0).random((32, 30, 3))
    lowres = np.random.default_rng(1).random((16, 15, 3))
    uns = {
        "spatial": {
            "lib1": {
                "images": {"hires": hires, "lowres": lowres},
                "scalefactors": {"tissue_hires_scalef": 0.1},
            }
        }
    }
    images = extract_images_from_anndata_uns(uns, prefer="lowres")
    assert IMAGE_KEY in images
    assert images[IMAGE_KEY].shape == lowres.shape
    assert "image_hires" in images
    assert "image_lowres" in images


def test_ensure_histology_from_uns_and_prepare() -> None:
    rng = np.random.default_rng(2)
    n_obs, n_vars = 20, 8
    image = rng.random((40, 40, 3))
    table = SpatialTable(
        X=rng.random((n_obs, n_vars)),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
        obsm={"spatial": np.column_stack([np.arange(n_obs), np.zeros(n_obs)])},
        uns={
            "spatial": {
                "demo": {"images": {"lowres": image}, "scalefactors": {}},
            }
        },
    )
    with_he = ensure_histology(table, prefer="lowres")
    assert IMAGE_KEY in with_he.images
    ready = prepare_virtual_st_table(with_he)
    assert ready.uns["analysis_task"] == "virtual_st"
    assert ready.uns["ground_truth_kind"] == "measured_expression"


def test_attach_histology_images_overwrite_policy() -> None:
    rng = np.random.default_rng(3)
    table = SpatialTable(
        X=rng.random((4, 3)),
        obs=pd.DataFrame(index=list("abcd")),
        var=pd.DataFrame(index=list("xyz")),
        obsm={"spatial": np.zeros((4, 2))},
        images={"image": np.ones((8, 8, 3))},
    )
    other = np.zeros((8, 8, 3))
    kept = attach_histology_images(table, {"image": other}, overwrite=False)
    assert np.allclose(kept.images["image"], 1.0)
    replaced = attach_histology_images(table, {"image": other}, overwrite=True)
    assert np.allclose(replaced.images["image"], 0.0)


def test_h5ad_bundle_loader_extracts_images(tmp_path: Path) -> None:
    import anndata as ad

    from histoweave.datasets.real import _load_h5ad_bundle

    rng = np.random.default_rng(4)
    lowres = (rng.random((24, 20, 3)) * 255).astype(np.uint8)
    adata = ad.AnnData(
        X=rng.random((10, 5)),
        obs=pd.DataFrame(
            {"domain_truth": [f"d{i % 2}" for i in range(10)]},
            index=[f"c{i}" for i in range(10)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        obsm={"spatial": rng.random((10, 2))},
        uns={
            "spatial": {
                "lib": {"images": {"lowres": lowres}, "scalefactors": {}},
            }
        },
    )
    path = tmp_path / "with_he.h5ad"
    adata.write_h5ad(path)
    table = _load_h5ad_bundle(path)
    assert IMAGE_KEY in table.images
    assert table.images[IMAGE_KEY].shape[:2] == lowres.shape[:2]


def test_virtual_st_on_synthetic_histology_from_uns() -> None:
    rng = np.random.default_rng(5)
    n_obs, n_vars = 36, 12
    coords = np.column_stack([np.tile(np.arange(6), 6), np.repeat(np.arange(6), 6)]).astype(
        float
    )
    # Morphology-linked expression.
    unit = coords / np.maximum(coords.max(axis=0), 1.0)
    morph = np.column_stack([unit[:, 0], unit[:, 1], unit[:, 0] * unit[:, 1]])
    loadings = rng.normal(size=(3, n_vars))
    expr = np.clip(np.exp(morph @ loadings + rng.normal(0, 0.1, size=(n_obs, n_vars))), 0, None)
    yy, xx = np.mgrid[0:48, 0:48]
    image = np.stack(
        [
            0.3 + 0.5 * np.sin(xx / 6.0),
            0.3 + 0.5 * np.cos(yy / 5.0),
            0.4 * np.ones_like(xx),
        ],
        axis=-1,
    )
    table = SpatialTable(
        X=expr,
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
        obsm={"spatial": coords},
        layers={"counts": expr.copy()},
        uns={"spatial": {"lib": {"images": {"lowres": image}}}},
    )
    data = prepare_virtual_st_table(table)
    result = create_method(
        MethodCategory.VIRTUAL_ST,
        "virtual_st_storm",
        mode="paired",
        image_key="image",
        n_genes=12,
        seed=0,
    ).run(data)
    pearson = result.uns["virtual_st"]["virtual_st_storm"]["mean_gene_pearson"]
    assert pearson > 0.3
    assert result.layers["virtual_st"].shape == (n_obs, n_vars)


@pytest.mark.skipif(not VISIUM_HNE_CACHE.exists(), reason="visium_hne_adata.h5ad not cached")
def test_real_visium_hne_paired_load_and_virtual_st() -> None:
    from histoweave.datasets import load_visium_hne_paired

    data = load_visium_hne_paired(
        source=VISIUM_HNE_CACHE,
        prefer="lowres",
        n_hvg=200,
        min_cells=3,
    )
    assert data.n_obs > 1000
    assert IMAGE_KEY in data.images
    assert data.spatial is not None
    # Subsample for speed while keeping spatial + image alignment.
    idx = np.linspace(0, data.n_obs - 1, 80, dtype=int)
    mask = np.zeros(data.n_obs, dtype=bool)
    mask[idx] = True
    sub = data.subset_obs(mask)
    # Re-attach image (subset_obs keeps images).
    sub = prepare_virtual_st_table(sub)
    result = create_method(
        MethodCategory.VIRTUAL_ST,
        "virtual_st_morphology",
        mode="paired",
        image_key="image",
        n_genes=32,
        seed=0,
        patch_size=9,
    ).run(sub)
    meta = result.uns["virtual_st"]["virtual_st_morphology"]
    assert np.isfinite(meta["mean_gene_pearson"])
    assert result.layers["virtual_st"].shape[0] == sub.n_obs


@pytest.mark.skipif(not VISIUM_HNE_CACHE.exists(), reason="visium_hne_adata.h5ad not cached")
def test_spatial_table_from_visium_hne_anndata() -> None:
    import anndata as ad

    adata = ad.read_h5ad(VISIUM_HNE_CACHE)
    table = spatial_table_from_visium_hne(adata, prefer="lowres", n_hvg=100)
    assert IMAGE_KEY in table.images
    assert table.uns["analysis_task"] == "virtual_st"
    assert table.n_vars <= 100
