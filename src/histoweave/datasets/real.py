"""Versioned, cacheable, checksummed dataset registry.

Each entry describes one benchmark-ready public dataset: how to fetch it,
what ground truth it carries, and which assay/tissue/species it represents.
The registry powers the recommendation engine (features are extracted once on
first use) and guarantees that repeated analyses use an identical copy.

Dataset lifecycle
-----------------
1. **Declare** — add a :class:`DatasetEntry` to ``_REGISTRY``.
2. **Download** — ``entry.download(cache_dir)`` fetches *once*, skips on cache hit,
   and validates the SHA-256 checksum.
3. **Load** — ``entry.load(cache_dir)`` returns a :class:`~histoweave.data.SpatialTable`.
4. **Analyse** — downstream methods consume the table; provenance records the dataset.

DLPFC label baking
------------------
DLPFC slices are shipped as **self-contained, checksummed h5ad bundles** built
by ``scripts/build_dlpfc_bundles.py``.  Each bundle carries the raw counts,
``obs['spatialLIBD_layer']`` (canonical manual layer annotation from
Maynard et al. 2021), ``obs['domain_truth']`` (alias for the benchmark
harness), and ``obsm['spatial']`` (pxl_col, pxl_row).  This means
``entry.load()`` performs a single SHA-256-guarded download and returns a
ready-to-benchmark :class:`SpatialTable` — no separate label-fetch step.

Local bundles
-------------
URLs beginning with ``local://`` are resolved against the repository root
(``<repo>/datasets_cache/dlpfc/...``).  This lets the same registry work for
development (local mirror) and production (Zenodo mirror) with a single
one-character URL rewrite.
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

# Root used to resolve local:// URLs.  Points at the repository root by default
# so ``local://datasets_cache/dlpfc/dlpfc_151507.h5ad`` maps to
# ``<repo>/datasets_cache/dlpfc/dlpfc_151507.h5ad``.  Override via the
# HISTOWEAVE_LOCAL_DATA env var for tests.
_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path.cwd()
)


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
        Download URL (``https://``, ``http://``, or ``local://<relpath>``).
    sha256 : str
        Hex-encoded SHA-256 of the downloaded artefact.
    assay : str
        Assay family: ``"visium"``, ``"xenium"``, ``"stereo_seq"``, ``"merfish"``.
    tissue : str
        Tissue/organ: ``"brain"``, ``"tumor"``, ...
    species : str
        ``"human"``, ``"mouse"``, ...
    n_obs : int
        Number of spots or cells.
    n_vars : int
        Number of genes.
    ground_truth : dict[str, str]
        Map of annotation name → ``obs`` column path, e.g.
        ``{"domain_truth": "obs['spatialLIBD_layer']"}``.
    license : str
        SPDX or human-readable licence identifier.
    paper_doi : str
        DOI of the original publication.
    is_h5ad_bundle : bool
        True when the artefact is a pre-built h5ad (counts + labels baked in).
        For such bundles ``load()`` calls ``anndata.read_h5ad()`` directly and
        constructs a :class:`SpatialTable` from ``obs['domain_truth']`` +
        ``obsm['spatial']``, bypassing vendor-specific reader logic.
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
    is_h5ad_bundle: bool = False

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
            recorded = checksum_file.read_text(encoding="utf-8").strip()
            artefact = dest / self._filename()
            expected = self.sha256 or recorded
            if artefact.exists() and expected and _sha256_file(artefact) == expected:
                return dest
            # Missing or corrupted cache: remove it and fetch a verified copy.
            shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

        artefact = dest / self._filename()
        if self.url.startswith("local://"):
            rel = self.url[len("local://") :].lstrip("/")
            configured_root = Path(os.environ.get("HISTOWEAVE_LOCAL_DATA", _REPOSITORY_ROOT))
            candidates = [configured_root / rel]
            rel_path = Path(rel)
            if rel_path.parts and rel_path.parts[0] == "datasets_cache":
                candidates.append(configured_root.joinpath(*rel_path.parts[1:]))
            candidates.append(_REPOSITORY_ROOT / rel)
            src = next((candidate for candidate in candidates if candidate.exists()), None)
            if src is None:
                raise FileNotFoundError(
                    f"{self.name}: local artefact not found; tried {candidates}. "
                    "Run scripts/build_dlpfc_bundles.py (DLPFC) or set "
                    "HISTOWEAVE_LOCAL_DATA to either the repository or datasets_cache root."
                )
            # copyfile: S3 FUSE forbids the metadata-copying of copy2.
            shutil.copyfile(src, artefact)
        else:
            _download_with_progress(self.url, artefact)

        digest = _sha256_file(artefact)
        if self.sha256 and digest != self.sha256:
            artefact.unlink(missing_ok=True)
            raise ChecksumError(f"{self.name}: expected sha256={self.sha256}, got {digest}")
        # If sha256 is empty, record the observed digest so subsequent calls are pinned.
        recorded = self.sha256 or digest
        checksum_file.write_text(recorded, encoding="utf-8")
        return dest

    def load(self, cache_dir: Path | str | None = None) -> SpatialTable:
        """Download (if needed) and load as a :class:`SpatialTable`.

        For h5ad bundles (``is_h5ad_bundle=True``) the artefact is read
        directly with ``anndata.read_h5ad`` and mapped into a SpatialTable
        with ``obs['domain_truth']`` and ``obsm['spatial']`` preserved.

        For vendor-format artefacts (zip / tar / raw h5) the reader is
        dispatched by ``assay``.
        """
        cache = self.download(cache_dir)
        artefact = cache / self._filename()

        if self.is_h5ad_bundle:
            return _load_h5ad_bundle(artefact)

        if artefact.suffix in (".zip", ".tar", ".gz", ".bz2"):
            extract_dir = cache / "extracted"
            if not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                _extract(artefact, extract_dir)
            data_dir = _find_data_dir(extract_dir)
        else:
            data_dir = cache

        return io_read(self.assay, str(data_dir))

    def _filename(self) -> str:
        return self.url.rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# H5AD bundle loader
# ---------------------------------------------------------------------------


def _load_h5ad_bundle(artefact: Path) -> SpatialTable:
    """Read a bundled h5ad file directly into a :class:`SpatialTable`.

    The bundle is expected to carry:

    * ``X`` — count matrix (sparse or dense).
    * ``obs['domain_truth']`` or ``obs['spatialLIBD_layer']`` — ground-truth
      domain labels.  If both are present, ``domain_truth`` wins.
    * ``obsm['spatial']`` — 2-D coordinates.

    All ``uns`` provenance keys are forwarded (e.g. ``dlpfc_source_urls``,
    ``dlpfc_bundle_version``).
    """
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(artefact)

    if "spatial" not in adata.obsm:
        raise ValueError(f"h5ad bundle {artefact} is missing obsm['spatial']")

    # Prefer the harness-standard column name; alias if only the canonical one exists.
    obs = adata.obs.copy()
    if "domain_truth" not in obs.columns and "spatialLIBD_layer" in obs.columns:
        obs["domain_truth"] = obs["spatialLIBD_layer"]

    return SpatialTable(
        X=adata.X,
        obs=obs,
        var=adata.var.copy(),
        obsm={
            "spatial": np.asarray(adata.obsm["spatial"], dtype=float),
        },
        uns=dict(adata.uns),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# --- DLPFC slices (12 bundles) ---------------------------------------------
# Auto-generated by scripts/build_dlpfc_bundles.py.  Every checksum below was
# computed on the actual on-disk artefact; regenerate with the build script to
# refresh.  Layer labels are baked into the bundle (schema v1).

_DLPFC_SLICE_ENTRIES = [
    DatasetEntry(
        name="dlpfc_151507",
        description="DLPFC slice 151507 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151507.h5ad",
        sha256="639839200d01012c7967310627692ea785cc0f8a95e350c71d4f26dbb894ff78",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=4221,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151508",
        description="DLPFC slice 151508 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151508.h5ad",
        sha256="276a57ca312cc0927971f81535661365ad3bd06fc4661c0df16b83dfe81f8bb7",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=4381,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151509",
        description="DLPFC slice 151509 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151509.h5ad",
        sha256="bca3d130b21e680c40f8d09605469a693376d3cff5c6789c89fc780a1303d52e",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=4788,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151510",
        description="DLPFC slice 151510 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151510.h5ad",
        sha256="cf6530ada7dd502b395528de3d2dffe45eb373a528e19cc7bc72d901774cf193",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=4595,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151669",
        description="DLPFC slice 151669 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151669.h5ad",
        sha256="11d9ea11169a3e23fbffc89977f5c082c2365c679139f0550568b4994f0b5ece",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3645,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151670",
        description="DLPFC slice 151670 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151670.h5ad",
        sha256="23440b89a14caf8549523d3f2668689ae3de97deff89edc6c88df54b4b2e632d",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3484,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151671",
        description="DLPFC slice 151671 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151671.h5ad",
        sha256="a2658a292027edaded49bf30c6a1897d7b617a98919ead83b568d193a007efbb",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=4093,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151672",
        description="DLPFC slice 151672 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151672.h5ad",
        sha256="ac069789e0c4cefa39b9294d2485b8638705f566ac832c23f3749ccd29b8bb5d",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3888,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151673",
        description="DLPFC slice 151673 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151673.h5ad",
        sha256="22033dc65de247a4457f398228b7fd24facfad4dddf48a0b9e9f9b248391a626",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3611,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151674",
        description="DLPFC slice 151674 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151674.h5ad",
        sha256="4db3370d0f9c264273c0fd5f65768bceb266823cc324328ae45a9607ff061cc0",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3635,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151675",
        description="DLPFC slice 151675 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151675.h5ad",
        sha256="2d45c5f68ceec0b1f55656a4aaacc03a24266262a7e1191a6d84ff49e6c4bc37",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3566,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
    DatasetEntry(
        name="dlpfc_151676",
        description="DLPFC slice 151676 — spatialLIBD-labelled human prefrontal cortex",
        url="local://datasets_cache/dlpfc/dlpfc_151676.h5ad",
        sha256="6e84e4e049bd8406e19c252f11ad8168212c91ab3cceb0d56c1e4ddf2e83d986",
        assay="visium",
        tissue="brain",
        species="human",
        n_obs=3431,
        n_vars=33538,
        ground_truth={"domain_truth": "obs['spatialLIBD_layer']"},
        license="CC-BY 4.0",
        paper_doi="10.1038/s41593-020-00787-0",
        is_h5ad_bundle=True,
    ),
]

# --- Non-DLPFC entries ------------------------------------------------------

# 10x Genomics CytAssist FFPE Mouse Brain — small demo dataset (~5 MB).
_MOUSE_BRAIN_DEMO = DatasetEntry(
    name="mouse_brain_cytassist",
    description="10x CytAssist FFPE Mouse Brain (Rep 1) — official demo dataset",
    url=(
        "https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/"
        "CytAssist_FFPE_Mouse_Brain_Rep1/"
        "CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5"
    ),
    sha256="",  # 10x may update; validated lazily
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
# Bundled subsample (~50 K cells) with domain_truth from the 10x cell-type
# predictor collapsed to tissue compartments; produced by
# ``benchmark_crossplatform/prepare_xenium.py``.
_XENIUM_BREAST = DatasetEntry(
    name="xenium_breast_cancer",
    description="10x Xenium Human Breast Cancer Rep 1 (~50K-cell subsample) — Janesick et al. 2023",
    url="local://datasets_cache/xenium/xenium_breast_cancer.h5ad",
    sha256="",  # populated by prepare_xenium.py on first build
    assay="xenium",
    tissue="tumor",
    species="human",
    n_obs=50000,
    n_vars=313,
    ground_truth={"domain_truth": "obs['domain_truth']"},
    license="10x Genomics EULA",
    paper_doi="10.1038/s41587-022-01583-2",
    is_h5ad_bundle=True,
)

# MERFISH Mouse Brain (C57BL6J-638850) — 3 anterior sections, ~60 K cells each,
# from the Allen Brain Cell Atlas (Yao et al. 2023).  Subclass labels are
# collapsed to 10 CCF parent regions; produced by
# ``benchmark_crossplatform/prepare_merfish.py``.
_MERFISH_MOUSE_BRAIN = DatasetEntry(
    name="merfish_mouse_brain",
    description="MERFISH mouse brain — three labelled anterior sections (Yao et al. 2023)",
    url="local://datasets_cache/merfish/merfish_mouse_brain.h5ad",
    sha256="",  # populated by prepare_merfish.py on first build
    assay="merfish",
    tissue="brain",
    species="mouse",
    n_obs=180000,
    n_vars=500,
    ground_truth={"domain_truth": "obs['domain_truth']"},
    license="CC-BY-NC 4.0",
    paper_doi="10.1038/s41586-023-06812-z",
    is_h5ad_bundle=True,
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
            "is_h5ad_bundle": e.is_h5ad_bundle,
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
    """Extract a .zip or tar archive into *dest*."""
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
    """Walk *extract_dir* to find the Space Ranger / vendor output directory."""
    markers = {
        "filtered_feature_bc_matrix.h5",
        "cell_feature_matrix.h5",
        "raw_feature_bc_matrix.h5",
    }
    for root, _dirs, files in os.walk(extract_dir):
        if markers & set(files):
            return Path(root)
    return extract_dir


__all__ = [
    "ChecksumError",
    "DatasetEntry",
    "get_dataset",
    "list_datasets",
]
