"""Build ``leaderboard/data.json`` from the benchmark CSV artefacts.

Read order (any missing file is skipped with a warning, so this stays a
one-shot script even on partial local check-outs):

  * ``5x15_spatial_aware/benchmark_long.csv``
      ``dataset, method, family, seed, ari, seconds, n_domains_truth``
      Preferred source. The `family` column feeds the filter chip.

  * ``5x10_dlpfc_benchmark/benchmark_long.csv``  (fallback)
      Same schema minus `family` and `n_domains_truth`; family is inferred
      from a hard-coded sklearn list.

  * ``benchmark_crossplatform/benchmark_long.csv``  (cross-platform, Task 3)
      Same schema as the 5×15 CSV. Records are concatenated verbatim.

Dataset metadata is looked up from the h5ad bundles via a small on-disk
manifest at ``benchmark_crossplatform/dataset_manifest.json`` when
available, otherwise from a small hard-coded fallback for the DLPFC slices.

The output is written to ``leaderboard/data.json`` with a stable schema
that ``main.js`` consumes.

Usage:
    python leaderboard/generate.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data.json"

SKLEARN_METHODS = {
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "dbscan",
    "gaussian_mixture",
    "kmeans",
    "mean_shift",
    "minibatch_kmeans",
    "optics",
    "spectral",
}

SOTA_METHODS = {
    "spagcn",
    "graphst",
    "stagate",
    "bayesspace",
    "banksy",
    "banksy_py",
    "rctd",
}

# One-line descriptions for the "Methods included" panel. Keep short.
METHOD_DESCRIPTIONS = {
    "agglomerative": "Ward-linkage hierarchical clustering on the PCA + spatial embedding.",
    "birch": "BIRCH incremental clustering (memory-frugal baseline).",
    "bisecting_kmeans": "Divisive k-means; useful when clusters vary in size.",
    "dbscan": "Density-based DBSCAN; strong on well-separated compartments.",
    "gaussian_mixture": "GMM with full covariances on the embedding.",
    "kmeans": "Vanilla k-means baseline on PCA + spatial embedding.",
    "mean_shift": "Non-parametric mean-shift; no k required.",
    "minibatch_kmeans": "Mini-batch k-means; the fast large-n baseline.",
    "optics": "Ordering points to identify clustering structure.",
    "spectral": "Normalised-cut spectral clustering on kNN affinity.",
    "banksy_py": "Python BANKSY: own + neighbourhood-mean expression, PCA + KMeans.",
    "banksy": "Bioconductor BANKSY (official R implementation).",
    "spagcn": "SpaGCN graph-convolutional spatial domains (official).",
    "graphst": "GraphST contrastive graph representation + fixed-q clustering.",
    "stagate": "STAGATE graph-attention autoencoder + fixed-q clustering.",
    "bayesspace": "BayesSpace Bayesian spatial clustering (Bioconductor).",
    "harmony_kmeans": "Harmony-integrated PCA (spatial quadrants as pseudo-batches) + KMeans.",
    "moran_spectral": "Moran's I gene ranking → spectral clustering on the spatial graph.",
    "spatialde_kmeans": "SpatialDE-ranked HVGs → PCA + KMeans.",
    "nnsvg_kmeans": "nnSVG-ranked HVGs → PCA + KMeans.",
}


def _method_family(name: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    base = name.split("@", 1)[0]
    if base in SKLEARN_METHODS or name in SKLEARN_METHODS:
        return "sklearn"
    if base in SOTA_METHODS or name in SOTA_METHODS:
        return "sota"
    return "spatial_aware"


def _dataset_contract(dataset_id: str) -> dict:
    """Attach task-contract fields from the HistoWeave dataset registry when known."""
    try:
        from histoweave.datasets import get_dataset, list_datasets
    except Exception:
        return {
            "task": "spatial_domain",
            "ground_truth_kind": "spatial_domain",
            "study": None,
        }
    candidates = [dataset_id, f"dlpfc_{dataset_id}"]
    available = {row["name"] for row in list_datasets()}
    for name in candidates:
        if name in available:
            entry = get_dataset(name)
            return {
                "task": entry.analysis_task,
                "ground_truth_kind": entry.ground_truth_kind,
                "label_key": entry.label_key,
                "study": entry.study,
                "registry_name": entry.name,
                "license": entry.license,
            }
    return {
        "task": "spatial_domain",
        "ground_truth_kind": "spatial_domain",
        "study": None,
    }


# Fallback platform metadata for slices whose manifest is missing.
DATASET_FALLBACK = {
    "151507": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151508": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151509": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151510": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151669": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 8},
    "151670": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 5},
    "151671": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 5},
    "151672": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 5},
    "151673": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151674": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151675": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "151676": {"platform": "Visium", "tissue": "DLPFC", "n_domains": 7},
    "merfish_mouse_hypothalamus": {
        "platform": "MERFISH",
        "tissue": "mouse hypothalamus",
        "n_domains": 7,
    },
    "merfish_mouse_brain": {"platform": "MERFISH", "tissue": "mouse brain", "n_domains": 10},
    "xenium_breast_cancer": {"platform": "Xenium", "tissue": "human breast cancer", "n_domains": 5},
    "visium_hd_crc": {
        "platform": "Visium HD",
        "tissue": "human colorectal cancer (FFPE)",
        "n_domains": 7,
    },
    "xenium_lung_cancer": {
        "platform": "Xenium",
        "tissue": "human lung adenocarcinoma (FFPE)",
        "n_domains": 5,
    },
    "xenium_ovarian_cancer": {
        "platform": "Xenium Prime",
        "tissue": "human ovarian cancer",
        "n_domains": 6,
    },
    "visium_mouse_brain": {
        "platform": "Visium v2",
        "tissue": "mouse brain (coronal, H&E)",
        "n_domains": 15,
    },
    "allen_merfish_brain_section": {
        "platform": "MERFISH",
        "tissue": "mouse whole-brain (single coronal section)",
        "n_domains": 8,
    },
}


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _read_long_csv(path: Path) -> list[dict]:
    """Return list of dicts (or []) from a benchmark_long.csv file."""
    if not path.exists():
        _log(f"[skip] {path} does not exist")
        return []
    import csv

    rows: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):

            def _num(k, source=row):
                v = source.get(k, "")
                if v == "" or v is None:
                    return None
                try:
                    x = float(v)
                    return None if math.isnan(x) else x
                except ValueError:
                    return None

            method_name = (row.get("config") or row.get("method") or "").strip()
            if not method_name:
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": method_name,
                    "seed": int(row.get("seed") or 0),
                    "ari": _num("ari"),
                    "seconds": _num("seconds"),
                    "family": _method_family(method_name, row.get("family") or None),
                }
            )
    return rows


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _n_obs_for(dataset: str) -> int | None:
    """Best-effort spot/cell count from the h5ad bundle when present.

    Uses ``anndata.read_h5ad(..., backed='r')`` when available so we don't
    load the full expression matrix into memory. Falls back to a raw h5py
    scan for the obs group, then to None on any error.
    """
    root = Path(
        os.environ.get(
            "HISTOWEAVE_LOCAL_DATA",
            REPO_ROOT,
        )
    )
    candidates = [
        root / "datasets_cache" / "dlpfc" / f"dlpfc_{dataset}.h5ad",
        root / "datasets_cache" / "merfish" / f"{dataset}.h5ad",
        root / "datasets_cache" / "slideseqv2" / f"{dataset}.h5ad",
        root / "datasets_cache" / "xenium" / f"{dataset}.h5ad",
        root / "benchmark_external_validation" / "bundles" / f"{dataset}.h5ad",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            import anndata as ad  # optional

            a = ad.read_h5ad(p, backed="r")
            n = int(a.n_obs)
            a.file.close()
            return n
        except Exception:
            pass
        try:
            import h5py  # optional

            with h5py.File(p, "r") as h:
                # anndata writes obs['_index'] (string) which has shape (n_obs,)
                if "obs" in h and "_index" in h["obs"]:
                    return int(h["obs/_index"].shape[0])
        except Exception:
            pass
    return None


def _sparsity_for(dataset: str) -> float | None:
    """Read exact sparse-matrix density from a local h5ad without loading X."""
    root = Path(os.environ.get("HISTOWEAVE_LOCAL_DATA", REPO_ROOT))
    candidates = [
        root / "datasets_cache" / "dlpfc" / f"dlpfc_{dataset}.h5ad",
        root / "datasets_cache" / "merfish" / f"{dataset}.h5ad",
        root / "datasets_cache" / "xenium" / f"{dataset}.h5ad",
        root / "benchmark_external_validation" / "bundles" / f"{dataset}.h5ad",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                matrix = handle["X"]
                if not isinstance(matrix, h5py.Group) or "data" not in matrix:
                    continue
                shape = tuple(int(value) for value in matrix.attrs["shape"])
                nnz = int(matrix["data"].shape[0])
                return round(1.0 - nnz / (shape[0] * shape[1]), 6)
        except (ImportError, KeyError, OSError, TypeError, ValueError):
            continue
    return None


def build() -> dict:
    records: list[dict] = []

    # -- collect records ------------------------------------------------------
    for candidate in [
        REPO_ROOT / "5x15_spatial_aware" / "benchmark_long.csv",
        REPO_ROOT / "5x15_spatial_aware" / "sota_benchmark_long.csv",
        REPO_ROOT / "5x10_dlpfc_benchmark" / "benchmark_long.csv",
        REPO_ROOT / "benchmark_crossplatform" / "benchmark_long.csv",
        REPO_ROOT / "benchmark_external_validation" / "benchmark_long.csv",
    ]:
        rows = _read_long_csv(candidate)
        if rows:
            _log(f"[read] {candidate.relative_to(REPO_ROOT)}: {len(rows)} rows")
            records.extend(rows)

    if not records:
        raise RuntimeError(
            "No benchmark_long.csv found. Expected one of "
            "5x15_spatial_aware/, 5x10_dlpfc_benchmark/, benchmark_crossplatform/, "
            "benchmark_external_validation/."
        )

    # -- distinct datasets + methods ------------------------------------------
    ds_ids = sorted({r["dataset"] for r in records})
    method_names = sorted({r["method"] for r in records})

    # -- dataset metadata + task contracts ------------------------------------
    manifest: dict[str, dict] = {}
    for manifest_path in (
        REPO_ROOT / "benchmark_crossplatform" / "dataset_manifest.json",
        REPO_ROOT / "benchmark_external_validation" / "dataset_manifest.json",
    ):
        manifest.update(_read_manifest(manifest_path))
    datasets = []
    for did in ds_ids:
        meta = manifest.get(did) or DATASET_FALLBACK.get(did) or {}
        contract = _dataset_contract(did)
        # Refuse to advertise self-supervised labels as spatial-domain GT.
        if contract.get("ground_truth_kind") in {"self_supervised", "leiden", "louvain"}:
            contract["task"] = "cell_type"
            contract["excluded_from_domain_leaderboard"] = True
        datasets.append(
            {
                "id": did,
                "platform": meta.get("platform") or contract.get("platform") or "unknown",
                "tissue": meta.get("tissue", "unknown"),
                "n_obs": meta.get("n_obs") or _n_obs_for(did),
                "n_domains": meta.get("n_domains"),
                "sparsity": meta.get("sparsity") or _sparsity_for(did),
                "task": contract.get("task"),
                "ground_truth_kind": contract.get("ground_truth_kind"),
                "study": contract.get("study"),
                "registry_name": contract.get("registry_name"),
                "license": contract.get("license"),
            }
        )

    # -- method metadata ------------------------------------------------------
    methods = []
    for m in method_names:
        base = m.split("@", 1)[0]
        family = _method_family(m)
        methods.append(
            {
                "name": m,
                "base_method": base,
                "family": family,
                "summary": METHOD_DESCRIPTIONS.get(m) or METHOD_DESCRIPTIONS.get(base, ""),
            }
        )

    # -- final feed -----------------------------------------------------------
    return {
        "protocol": "histoweave.leaderboard.v2",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "task_default": "spatial_domain",
        "submission_schema": "histoweave.external_submission.v1",
        "datasets": datasets,
        "methods": methods,
        "records": records,
    }


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, indent=2))
    _log(
        f"[write] {OUT}: {len(data['records'])} records, "
        f"{len(data['datasets'])} datasets, {len(data['methods'])} methods"
    )


if __name__ == "__main__":
    main()
