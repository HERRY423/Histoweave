"""Unit tests for the 5 external-validation preparers (synthetic inputs, no network).

Each test exercises the pure-logic functions of a preparer — truth resolution,
label filtering, stratified subsampling, pathology point-in-polygon assignment
— with tiny synthetic inputs, mirroring ``test_allen_mouse_brain_preparer.py``.
No dataset is downloaded; the heavy ``build()`` functions are integration-tested
by actually running the preparers in the benchmark step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "benchmark_external_validation"

# Make the package + external-validation dir importable.
for p in (str(ROOT / "src"), str(EXT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (and other decorators that look up
    # sys.modules[cls.__module__]) resolve correctly.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


# ---------------------------------------------------------------------------
# Shared pathology helper (used by lung + ovarian + lymph-node preparers)
# ---------------------------------------------------------------------------

PATHOLOGY = _load_module("pathology_domains", ROOT / "src/histoweave/datasets/pathology_domains.py")


def _square_geojson(label: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": label}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                },
            }
        ],
    }


def test_pathology_assigns_cells_to_polygons():
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "tumor"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "stroma"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[5, 0], [10, 0], [10, 5], [5, 5], [5, 0]]],
                },
            },
        ],
    }
    xy = np.array([[1, 1], [6, 1], [20, 20], [2.5, 2.5]], dtype=float)
    labels = PATHOLOGY.assign_pathology_domains(xy, geo)
    assert list(labels)[:2] == ["tumor", "stroma"]
    assert pd.isna(labels.iloc[2])  # outside every polygon
    assert labels.iloc[3] == "tumor"


def test_pathology_marks_conflicting_overlaps_ambiguous():
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "tumor"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"classification": {"name": "stroma"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[4, 0], [10, 0], [10, 6], [4, 6], [4, 0]]],
                },
            },
        ],
    }
    # Cell at (5,1) is inside both the tumor and stroma polygons -> ambiguous
    # (only when prefer_smaller=False; with prefer_smaller=True the first
    # polygon wins on equal-area ties).
    xy = np.array([[5, 1]], dtype=float)
    labels = PATHOLOGY.assign_pathology_domains(xy, geo, prefer_smaller=False)
    assert labels.iloc[0] == "ambiguous"


def test_pathology_prefers_smaller_polygon_on_overlap():
    """With prefer_smaller=True (default), a nested smaller polygon wins."""
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "tumor"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"classification": {"name": "lymphoid_aggregate"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[3, 3], [5, 3], [5, 5], [3, 5], [3, 3]]],
                },
            },
        ],
    }
    # Cell at (4,4) is inside both the large tumor and the small lymphoid
    # aggregate. With prefer_smaller=True (default), the smaller polygon wins.
    xy = np.array([[4, 4]], dtype=float)
    labels = PATHOLOGY.assign_pathology_domains(xy, geo)
    assert labels.iloc[0] == "lymphoid_aggregate"


def test_pathology_raises_when_no_polygon_intersects():
    geo = _square_geojson("tumor", 0, 0, 1, 1)
    xy = np.array([[100, 100]], dtype=float)
    with pytest.raises(ValueError, match="no labelled pathology polygon"):
        PATHOLOGY.assign_pathology_domains(xy, geo)


def test_pathology_label_property_override():
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"custom": {"region": "necrosis"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    xy = np.array([[0.5, 0.5]], dtype=float)
    labels = PATHOLOGY.assign_pathology_domains(xy, geo, label_property="custom.region")
    assert labels.iloc[0] == "necrosis"


def test_stratified_indices_respects_limit_and_groups():
    labels = pd.Series(["a"] * 10 + ["b"] * 10 + ["c"] * 5)
    idx = PATHOLOGY.stratified_indices(labels, limit=5, seed=0)
    assert len(idx) == 5
    # Every group present in the input should be represented (quota >= 1).
    chosen = labels.iloc[idx].unique().tolist()
    assert set(chosen) == {"a", "b", "c"}


def test_stratified_indices_passthrough_when_small():
    labels = pd.Series(["a", "b", "c"])
    idx = PATHOLOGY.stratified_indices(labels, limit=10, seed=0)
    assert list(idx) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Xenium common builder (lung / ovarian) — synthetic GeoJSON + tiny matrix
# ---------------------------------------------------------------------------

XENIUM_COMMON = _load_module("_xenium_pathology_common", EXT / "_xenium_pathology_common.py")


def test_xenium_spec_defaults():
    from prepare_xenium_lung_cancer import SPEC as lung_spec
    from prepare_xenium_ovarian_cancer import SPEC as ovarian_spec

    assert lung_spec.name == "xenium_lung_cancer"
    assert ovarian_spec.name == "xenium_ovarian_cancer"
    assert lung_spec.default_label_property == "classification.name"
    assert "10x" in lung_spec.source.lower()


def test_xenium_build_with_synthetic_inputs(tmp_path):
    """End-to-end build() on a 4-cell synthetic Xenium bundle.

    Two cells inside the 'tumor' polygon, one inside 'stroma', one outside
    (excluded). Verifies domain_truth, obsm['spatial'], layers['counts'],
    and the receipt JSON.
    """
    import h5py
    import scipy.sparse as sp

    # 4 cells x 3 genes count matrix, written in 10x legacy h5 format so
    # sc.read_10x_h5 can parse it.
    X = sp.csr_matrix(np.array([[5, 0, 1], [3, 2, 0], [0, 4, 2], [1, 1, 1]], dtype=np.float32))
    barcodes = np.array([b"cell1", b"cell2", b"cell3", b"cell4"], dtype="S")
    gene_ids = np.array([b"GENE1", b"GENE2", b"GENE3"], dtype="S")
    gene_names = np.array([b"GENE1", b"GENE2", b"GENE3"], dtype="S")
    matrix_h5 = tmp_path / "cell_feature_matrix.h5"
    with h5py.File(matrix_h5, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("barcodes", data=barcodes)
        grp.create_dataset("data", data=X.data.astype(np.float32))
        grp.create_dataset("indices", data=X.indices.astype(np.int64))
        grp.create_dataset("indptr", data=X.indptr.astype(np.int64))
        grp.create_dataset("shape", data=np.array([3, 4], dtype=np.int64))
        feats = grp.create_group("features")
        feats.create_dataset("id", data=gene_ids)
        feats.create_dataset("name", data=gene_names)
        # scanpy's v3 reader requires feature_type under matrix/features.
        feats.create_dataset("feature_type", data=np.array([b"Gene Expression"] * 3, dtype="S"))

    # cells metadata with centroids matching the GeoJSON frame.
    meta = pd.DataFrame(
        {
            "cell_id": ["cell1", "cell2", "cell3", "cell4"],
            "x_centroid": [0.5, 6.5, 20.0, 0.5],
            "y_centroid": [0.5, 0.5, 20.0, 0.5],
        }
    )
    meta_path = tmp_path / "cells.csv"
    meta.to_csv(meta_path, index=False)

    # GeoJSON: tumor square [0,0]-[5,5], stroma square [5,0]-[10,5].
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "tumor"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"classification": {"name": "stroma"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[5, 0], [10, 0], [10, 5], [5, 5], [5, 0]]],
                },
            },
        ],
    }
    geo_path = tmp_path / "pathology.geojson"
    geo_path.write_text(json.dumps(geo))

    out = tmp_path / "xenium_test.h5ad"
    args = argparse.Namespace(
        matrix=matrix_h5,
        metadata=meta_path,
        pathology_geojson=geo_path,
        output=out,
        id_column="cell_id",
        x_column="x_centroid",
        y_column="y_centroid",
        label_property=None,
        geojson_scale=1.0,
        geojson_offset_x=0.0,
        geojson_offset_y=0.0,
        exclude_label=None,
        min_cells_per_domain=1,
        max_cells=100,
        seed=42,
    )
    spec = XENIUM_COMMON.XeniumDatasetSpec(
        name="xenium_test",
        source="synthetic",
        source_url="",
        license="test",
        schema_version="test.v1",
    )
    receipt = XENIUM_COMMON.build_xenium_pathology_bundle(args, spec)

    assert receipt["name"] == "xenium_test"
    assert receipt["n_obs"] == 3  # cell4 (outside) excluded
    assert set(receipt["domains"]) == {"tumor", "stroma"}
    assert receipt["sha256"] != ""

    import anndata as ad

    built = ad.read_h5ad(out)
    assert "domain_truth" in built.obs.columns
    assert "spatial" in built.obsm
    assert built.obsm["spatial"].shape == (3, 2)
    assert "counts" in built.layers
    assert built.obs["truth_source"].iloc[0] == "10x_pathology_annotation"
    # No NaNs in domain_truth.
    assert not built.obs["domain_truth"].isna().any()


# ---------------------------------------------------------------------------
# Visium HD CRC — annotation CSV loader + label filtering
# ---------------------------------------------------------------------------

CRC = _load_module("prepare_visium_hd_crc", EXT / "prepare_visium_hd_crc.py")


def test_crc_annotation_loader_resolves_barcode_and_label(tmp_path):
    csv = tmp_path / "ann.csv"
    csv.write_text(
        "barcode,label\ns_016um_00186_00418-1,Neoplasm\n"
        "s_016um_00174_00236-1,Non-neoplastic Epithelium\n"
    )
    ann = CRC._load_annotation(csv)
    assert list(ann.columns) == ["barcode", "pathology_label"]
    assert ann["pathology_label"].tolist() == ["Neoplasm", "Non-neoplastic Epithelium"]
    assert ann["barcode"].iloc[0] == "s_016um_00186_00418-1"


def test_crc_invalid_labels_set_covers_common_junk():
    for bad in ["", "nan", "none", "unknown", "unannotated", "NA"]:
        assert bad.lower() in {x.lower() for x in CRC.INVALID_LABELS}


def test_crc_stratified_indices_keeps_all_groups():
    labels = pd.Series(["Neoplasm"] * 20 + ["Stroma"] * 10 + ["Smooth Muscle"] * 5)
    idx = CRC._stratified_indices(labels, limit=6, seed=0)
    assert len(idx) == 6
    chosen = set(labels.iloc[idx].unique().tolist())
    assert chosen == {"Neoplasm", "Stroma", "Smooth Muscle"}


# ---------------------------------------------------------------------------
# Visium mouse brain — expected anatomical regions + label validation
# ---------------------------------------------------------------------------

VMB = _load_module("prepare_visium_mouse_brain", EXT / "prepare_visium_mouse_brain.py")


def test_visium_mouse_brain_expected_regions_present():
    # The 15 Allen-reference anatomical regions the squidpy dataset carries.
    assert len(VMB.ANATOMICAL_REGIONS) == 15
    for region in ("L1", "L6", "Hippocampus", "Fiber_tract", "Striatum"):
        assert region in VMB.ANATOMICAL_REGIONS


def test_visium_mouse_brain_invalid_labels_cover_junk():
    for bad in ["", "nan", "none", "unknown"]:
        assert bad.lower() in {x.lower() for x in VMB.INVALID_LABELS}


# ---------------------------------------------------------------------------
# Allen MERFISH brain section — anatomical truth resolution (strict policy)
# ---------------------------------------------------------------------------

ALLEN = _load_module(
    "prepare_allen_merfish_brain_section", EXT / "prepare_allen_merfish_brain_section.py"
)


def test_allen_prefers_parcellation_division():
    obs = pd.DataFrame(
        {
            "parcellation_division": ["Isocortex", "Hippocampal formation"],
            "parcellation_structure": ["Somatosensory", "CA1"],
            "subclass": ["IT", "Oligo"],
        }
    )
    labels, column = ALLEN._resolve_truth(obs, region_column=None)
    assert labels.tolist() == ["Isocortex", "Hippocampal formation"]
    assert column == "parcellation_division"


def test_allen_rejects_missing_anatomical_column():
    obs = pd.DataFrame({"subclass": ["IT", "Oligo"]})
    with pytest.raises(ValueError, match="not accepted as primary"):
        ALLEN._resolve_truth(obs, region_column=None)


def test_allen_respects_explicit_region_column():
    obs = pd.DataFrame(
        {
            "parcellation_division": ["Isocortex"],
            "parcellation_structure": ["Somatosensory"],
        }
    )
    labels, column = ALLEN._resolve_truth(obs, region_column="parcellation_structure")
    assert column == "parcellation_structure"
    assert labels.tolist() == ["Somatosensory"]


def test_allen_valid_mask_drops_invalid_labels():
    labels = pd.Series(["Isocortex", "", "nan", "unknown", "Hippocampal formation"], dtype="string")
    mask = ALLEN._valid_mask(labels)
    assert mask.tolist() == [True, False, False, False, True]


def test_allen_stratified_indices_respects_limit():
    labels = pd.Series(["Isocortex"] * 20 + ["Hippocampal formation"] * 10)
    idx = ALLEN._stratified_indices(labels, limit=4, seed=0)
    assert len(idx) == 4
    chosen = set(labels.iloc[idx].unique().tolist())
    assert chosen == {"Isocortex", "Hippocampal formation"}
