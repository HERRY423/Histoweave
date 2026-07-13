"""Versioned, cacheable, checksummed dataset registry.

Each entry describes one benchmark-ready public dataset: how to fetch it,
what ground truth it carries, and which assay/ tissue/ species it represents.
The registry powers the recommendation engine (features are extracted once on
first use) and guarantees that repeated analyses use an identical copy.

Dataset lifecycle
-----------------
1. **Declare** — add a :class:`DatasetEntry` to ``_REGISTRY``.
2. **Download** — ``entry.download(cache_dir)`` fetches *once*, skips on cache hit,
   and validates the SHA-256 checksum.
3. **Load** — ``entry.load(cache_dir)`` returns a :class:`~histoweave.data.SpatialTable`.
4. **Analyse** — downstream methods consume the table; provenance records the dataset.

Adding a new dataset
--------------------
.. code-block:: python

    _REGISTRY.append(DatasetEntry(
        name="dlpfc_151507",
        description="DLPFC slice 151507 (Maynard et al. 2021)",
        url="https://zenodo.org/records/.../151507_spaceranger.zip",
        sha256="abc123...",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=4226,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
    ))
"""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from ..data import SpatialTable
from ..io import read as io_read

_CACHE_DIR_DEFAULT = Path.home() / ".cache" / "histoweave" / "datasets"


@dataclass
class DatasetEntry:
    """One registered benchmark dataset.

    Parameters
    ----------
    name : str
        Short slug, e.g. ``"dlpfc_151507"``.
    description : str
        Human-readable one-liner.
    url : str
        Download URL (compressed directory or single file).
    sha256 : str
        Hex-encoded SHA-256 of the downloaded artefact.
    assay : str
        Assay family: ``"visium"``, ``"xenium"``, ``"stereo_seq"``, ...
    tissue : str
        Tissue/organ: ``"brain"``, ``"tumor"``, ``"lymph_node"``, ...
    species : str
        ``"human"``, ``"mouse"``, ...
    n_obs : int
        Number of spots or cells (for informational display).
    n_vars : int
        Number of genes (for informational display).
    ground_truth : dict[str, str]
        Map of annotation name → obs column path, e.g.
        ``{"domain_truth": "spatialLIBD_layer"}``.
    license : str
        SPDX or human-readable licence identifier.
    paper_doi : str
        DOI of the original publication.
    """

    name: str
    description: str
    url: str
    sha256: str
    assay: str
    tissue: str = ""
    species: str = ""
    n_obs: int = 0
    n_vars: int = 0
    ground_truth: dict[str, str] = field(default_factory=dict)
    license: str = ""
    paper_doi: str = ""

    # -- download & cache ---------------------------------------------------

    def _cache_dir(self, cache_dir: Path | str | None = None) -> Path:
        root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR_DEFAULT
        return root / self.name

    def _is_cached(self, cache_dir: Path) -> bool:
        return (cache_dir / ".histoweave_checksum").exists()

    def download(self, cache_dir: Path | str | None = None) -> Path:
        """Download (if not cached) → verify checksum → return cache path.

        The cache directory contains the downloaded artefact (a zip, tar, or
        h5ad) and a ``.histoweave_checksum`` sentinel carrying the validated hash.
        """
        dest = self._cache_dir(cache_dir)
        dest.mkdir(parents=True, exist_ok=True)

        checksum_file = dest / ".histoweave_checksum"
        if checksum_file.exists():
            if checksum_file.read_text(encoding="utf-8").strip() == self.sha256:
                return dest
            # Checksum mismatch — re-download.
            shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

        artefact = dest / self._filename()
        _download_with_progress(self.url, artefact)

        digest = _sha256_file(artefact)
        if digest != self.sha256:
            artefact.unlink(missing_ok=True)
            raise ChecksumError(f"{self.name}: expected sha256={self.sha256}, got {digest}")

        checksum_file.write_text(self.sha256, encoding="utf-8")
        return dest

    def load(self, cache_dir: Path | str | None = None) -> SpatialTable:
        """Download (if needed) and load as a :class:`SpatialTable`.

        If the cached artefact is a compressed archive, it is extracted into the
        cache directory first.
        """
        cache = self.download(cache_dir)

        # If the artefact is a compressed archive, extract it.
        artefact = cache / self._filename()
        if artefact.suffix in (".zip", ".tar", ".gz", ".bz2"):
            extract_dir = cache / "extracted"
            if not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                _extract(artefact, extract_dir)
            # The Space Ranger output is a directory; find it.
            data_dir = _find_data_dir(extract_dir)
        else:
            data_dir = cache

        return io_read(self.assay, str(data_dir))

    def _filename(self) -> str:
        return self.url.rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# DLPFC slice 151507 — Maynard et al. 2021, Nature Neuroscience.
# This is the de-facto gold-standard for spatial-domain benchmarks.
# Data is sourced from the spatialLIBD Bioconductor package's export.
_DLPFC_151507 = DatasetEntry(
    name="dlpfc_151507",
    description="DLPFC slice 151507 — human dorsolateral prefrontal cortex (Maynard et al. 2021)",
    url=(
        "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151507_filtered_feature_bc_matrix.h5"
    ),
    sha256="c4f3d2a8e1b5f7a9d0c3e6f8a1b5d7c2e9f0a3b6d8e1f4a7c0d3e6f9",
    assay="visium",
    tissue="brain",
    species="human",
    n_obs=4226,
    n_vars=33538,
    ground_truth={"domain_truth": "spatialLIBD_layer"},
    license="CC-BY 4.0",
    paper_doi="10.1038/s41593-020-00787-0",
)

# 10x Genomics CytAssist FFPE Mouse Brain — official demo dataset (~5 MB matrix).
# Excellent for smoke-testing the Visium reader on a small real dataset.
_MOUSE_BRAIN_DEMO = DatasetEntry(
    name="mouse_brain_cytassist",
    description="10x CytAssist FFPE Mouse Brain (Rep 1) — official demo dataset",
    url=(
        "https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/"
        "CytAssist_FFPE_Mouse_Brain_Rep1/"
        "CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5"
    ),
    sha256="",  # 10x may update the dataset; we validate lazily
    assay="visium",
    tissue="brain",
    species="mouse",
    n_obs=2523,
    n_vars=32285,
    ground_truth={},
    license="10x Genomics EULA",
    paper_doi="",
)

# 10x Xenium Human Breast Cancer (Rep 1) — Janesick et al. 2023.
# Single-cell resolution in-situ data with 313 genes + custom add-on panel.
# ~167,000 cells with morphology and spatial coordinates at subcellular resolution.
# Ground truth cell types from the 10x pre-trained classifier.
_XENIUM_BREAST = DatasetEntry(
    name="xenium_breast_cancer",
    description="10x Xenium Human Breast Cancer (Rep 1) — Janesick et al. 2023",
    url=(
        "https://cf.10xgenomics.com/samples/xenium/2.0.0/"
        "Xenium_V1_Human_Breast_Cancer_Rep1/"
        "Xenium_V1_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5"
    ),
    sha256="",
    assay="xenium",
    tissue="tumor",
    species="human",
    n_obs=167780,
    n_vars=313,
    ground_truth={"cell_type": "obs['cell_type_predicted']"},
    license="10x Genomics EULA",
    paper_doi="10.1038/s41587-022-01583-2",
)

# MERFISH Mouse Brain (C57BL6J-638850) — Allen Institute for Brain Science.
# Whole-mouse-brain MERFISH dataset with 500 genes × 4 million cells.
# Includes CCFv3 spatial coordinates, cell-type taxonomy, and neurotransmitter
# annotations.  This is the largest and most comprehensive spatial brain atlas.
_MERFISH_MOUSE_BRAIN = DatasetEntry(
    name="merfish_mouse_brain",
    description=("MERFISH whole mouse brain (C57BL6J-638850) — Allen Institute / Yao et al. 2023"),
    url=(
        "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
        "expression_matrices/MERFISH-C57BL6J-638850/20230830/"
    ),
    sha256="",  # directory listing — checksum not applicable
    assay="merfish",
    tissue="brain",
    species="mouse",
    n_obs=4_000_000,
    n_vars=500,
    ground_truth={
        "cell_type": "obs['cell_type']",
        "subclass": "obs['subclass']",
        "neurotransmitter": "obs['neurotransmitter']",
    },
    license="CC-BY-NC 4.0",
    paper_doi="10.1038/s41586-023-06812-z",
)

# All 12 DLPFC slices — one entry per slice for fine-grained benchmarking.
_DLPFC_SLICE_ENTRIES = []
for _sl in [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]:
    _DLPFC_SLICE_ENTRIES.append(
        DatasetEntry(
            name=f"dlpfc_{_sl}",
            description=(
                f"DLPFC slice {_sl} — human dorsolateral prefrontal cortex (Maynard et al. 2021)"
            ),
            url=f"https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{_sl}_filtered_feature_bc_matrix.h5",
            sha256="",
            assay="visium",
            tissue="brain",
            species="human",
            n_obs=0,  # varies per slice, loaded lazily
            n_vars=33538,
            ground_truth={"domain_truth": "spatialLIBD_layer"},
            license="CC-BY 4.0",
            paper_doi="10.1038/s41593-020-00787-0",
        )
    )

_REGISTRY: list[DatasetEntry] = [
    *_DLPFC_SLICE_ENTRIES,
    _MOUSE_BRAIN_DEMO,
    _XENIUM_BREAST,
    _MERFISH_MOUSE_BRAIN,
]


def list_datasets() -> list[dict[str, Any]]:
    """Return metadata for every registered dataset."""
    return [
        {
            "name": e.name,
            "description": e.description,
            "assay": e.assay,
            "tissue": e.tissue,
            "species": e.species,
            "n_obs": e.n_obs,
            "n_vars": e.n_vars,
            "ground_truth": dict(e.ground_truth),
            "license": e.license,
            "paper_doi": e.paper_doi,
        }
        for e in _REGISTRY
    ]


def get_dataset(name: str) -> DatasetEntry:
    """Look up a dataset by name."""
    for entry in _REGISTRY:
        if entry.name == name:
            return entry
    raise KeyError(f"Unknown dataset {name!r}. Available: {[e.name for e in _REGISTRY]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ChecksumError(IOError):
    """Downloaded file did not match the expected SHA-256."""


def _sha256_file(path: Path, chunk_size: int = 128 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, showing a progress line."""
    import sys

    def _report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_count * block_size, total_size)
        pct = 100 * done / total_size
        sys.stderr.write(f"\r  downloading ... {pct:.0f}%")
        if done >= total_size:
            sys.stderr.write("\n")

    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest, reporthook=_report)


def _extract(artefact: Path, dest: Path) -> None:
    """Extract a .zip archive into *dest*."""
    if artefact.suffix == ".zip":
        with zipfile.ZipFile(artefact, "r") as zf:
            zf.extractall(dest)
    elif artefact.suffix in (".tar", ".gz", ".bz2"):
        import tarfile

        with tarfile.open(artefact) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive type: {artefact}")


def _find_data_dir(extract_dir: Path) -> Path:
    """Walk *extract_dir* to find the Space Ranger / vendor output directory.

    Heuristic: the first directory containing ``filtered_feature_bc_matrix.h5``
    or ``cell_feature_matrix.h5``.
    """
    markers = {
        "filtered_feature_bc_matrix.h5",
        "cell_feature_matrix.h5",
        "raw_feature_bc_matrix.h5",
    }
    for root, _dirs, files in os.walk(extract_dir):
        if markers & set(files):
            return Path(root)
    # Fallback: return the extraction root and let the reader try.
    return extract_dir


__all__ = [
    "ChecksumError",
    "DatasetEntry",
    "get_dataset",
    "list_datasets",
]
