"""Prepare a Xenium Human Lung Cancer bundle with pathology ground truth.

Source: 10x Genomics "Preview Data: FFPE Human Lung Cancer with Xenium
Multimodal Cell Segmentation" —
https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard

This is the Xenium v1 (XA v2.0 preview) human lung cancer FFPE dataset that
ships a pathologist-annotated GeoJSON aligned to the post-Xenium H&E. The
Xenium output bundle (``Xenium_V1_humanLung_Cancer_FFPE_outs.zip``) contains
``cell_feature_matrix.h5`` and ``cells.csv.gz``; the supplemental
``Xenium_V1_humanLung_Cancer_FFPE_annotation.geojson`` carries the pathology
polygons.

Ground truth comes ONLY from the pathologist annotation polygons supplied
with the dataset (QuPath-exported GeoJSON, aligned to the post-Xenium H&E).
Cells outside polygons or in conflicting overlaps are excluded; predicted
cell-type labels are never used as spatial-domain truth.

Download (command line)::

    curl -O https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip
    curl -O https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_annotation.geojson
    unzip Xenium_V1_humanLung_Cancer_FFPE_outs.zip

Then run::

    python benchmark_external_validation/prepare_xenium_lung_cancer.py \
        --matrix Xenium_V1_humanLung_Cancer_FFPE_outs/cell_feature_matrix.h5 \
        --metadata Xenium_V1_humanLung_Cancer_FFPE_outs/cells.csv.gz \
        --pathology-geojson Xenium_V1_humanLung_Cancer_FFPE_annotation.geojson
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _xenium_pathology_common import (
    XeniumDatasetSpec,
    add_common_xenium_args,
    build_xenium_pathology_bundle,
)

SPEC = XeniumDatasetSpec(
    name="xenium_lung_cancer",
    source="10x Xenium FFPE Human Lung Cancer (XA v2.0 multimodal cell segmentation preview)",
    source_url=(
        "https://www.10xgenomics.com/datasets/"
        "preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard"
    ),
    license="10x Genomics EULA (CC BY 4.0 per dataset page)",
    schema_version="histoweave.xenium.lung_cancer.bundle.v1",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_xenium_args(
        parser,
        default_output=root / "datasets_cache" / "xenium" / "xenium_lung_cancer.h5ad",
    )
    build_xenium_pathology_bundle(parser.parse_args(), SPEC)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
