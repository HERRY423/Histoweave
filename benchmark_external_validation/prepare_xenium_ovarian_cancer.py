"""Prepare a Xenium Prime Human Ovarian Cancer bundle with pathology ground truth.

Source: 10x Genomics "FFPE Human Ovarian Cancer with 5K Human Pan Tissue and
Pathways Panel plus 100 Custom Genes" —
https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer

The dataset ships pathologist annotations (QuPath v0.5.1) with the colour
legend: red = tumor, black = necrosis, blue = smooth muscle, dark green =
fallopian tube, light green = ovary. These histological region labels are the
spatial-domain ground truth; predicted cell-type labels are never used.

The Xenium Prime output bundle uses ``cells.parquet`` (not ``cells.csv.gz``).
Download (command line)::

    curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip
    curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_annotation.geojson
    unzip Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip

Then run::

    python benchmark_external_validation/prepare_xenium_ovarian_cancer.py \
        --matrix Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs/cell_feature_matrix.h5 \
        --metadata Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs/cells.parquet \
        --pathology-geojson Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_annotation.geojson
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
    name="xenium_ovarian_cancer",
    source="10x Xenium Prime FFPE Human Ovarian Cancer (5K Pan Tissue + 100 custom)",
    source_url="https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer",
    license="10x Genomics EULA (CC BY 4.0 per dataset page)",
    schema_version="histoweave.xenium.ovarian_cancer.bundle.v1",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_xenium_args(
        parser,
        default_output=root / "datasets_cache" / "xenium" / "xenium_ovarian_cancer.h5ad",
    )
    build_xenium_pathology_bundle(parser.parse_args(), SPEC)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
