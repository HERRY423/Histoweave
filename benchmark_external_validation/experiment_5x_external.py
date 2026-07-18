"""External-validation 5-dataset × 15-method landscape with cell-bootstrap CIs.

Runs the Task-2 method list (10 sklearn + 5 spatial-aware) on the 5 external
validation datasets prepared by the sibling ``prepare_*.py`` scripts:

  * visium_hd_crc            — Visium HD human colorectal cancer (pathologist)
  * xenium_lung_cancer       — Xenium human lung adenocarcinoma (pathology)
  * xenium_ovarian_cancer    — Xenium Prime human ovarian cancer (pathology)
  * visium_mouse_brain       — Visium v2 mouse brain (Allen anatomical regions)
  * allen_merfish_brain_section — MERFISH mouse brain single section (Allen CCF)

This is the **cross-study generalization** counterpart to the within-study
5x10/5x15 DLPFC benchmarks. The 5 datasets span 4 platforms, 2 organisms, 4
tissues, and 4 independent studies, all with strict region ground truth
(anatomical / pathology / manual — never cell-type predictions). The
companion ``recommender_loocv_external.py`` uses this landscape to test
whether the recommendation engine beats the global-best baseline once the
landscape is diverse.

Design choices (mirrors ``benchmark_crossplatform/experiment_7x15.py``):
  * Datasets above ``N_MAX`` cells are stratified-subsampled per (dataset, seed)
    so every method sees the same slice.
  * Bootstrap CIs are refit-free: once a method predicts labels on the full
    evaluation slice, we draw ``N_BOOT`` × 80% cell subsamples and recompute
    ARI on each.
  * Per-(method, dataset, seed) checkpoints so a crash loses only the last cell.

Outputs (in HISTOWEAVE_EXT_OUT, default this directory):
  * benchmark_long.csv            — per-cell records with bootstrap CI
  * performance_matrix_mean.csv   — 5 × 15 mean ARI
  * performance_matrix_std.csv    — 5 × 15 std ARI
  * bootstrap_ci.csv              — per (method, dataset) mean + 2.5/97.5 pctl
  * benchmark_5x_external.json    — self-describing summary
  * dataset_manifest.json         — metadata for the leaderboard
  * manifest.json                 — file hashes + protocol tag
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

# --- Import path shim so the 5x15 spatial-aware adapters resolve.
_HERE = Path(__file__).resolve().parent
_5X15 = _HERE.parent / "5x15_spatial_aware"
for p in (str(_HERE), str(_5X15)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapters import (  # noqa: E402
    banksy_py_adapter,
    harmony_adapter,
    moran_adapter,
    nnsvg_adapter,
    spatialde_adapter,
)

from histoweave.benchmark.landscape import run_task_landscape  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins import MethodCategory  # noqa: E402

_LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OUT = Path(os.environ.get("HISTOWEAVE_EXT_OUT", _HERE))
CHK = Path(os.environ.get("HISTOWEAVE_EXT_CHECKPOINT", _HERE / "checkpoints"))
OUT.mkdir(parents=True, exist_ok=True)
CHK.mkdir(parents=True, exist_ok=True)

# (dataset_id, platform, tissue, organism)
DATASETS = [
    ("visium_hd_crc", "Visium HD", "human colorectal cancer", "human"),
    ("xenium_lung_cancer", "Xenium", "human lung adenocarcinoma", "human"),
    ("xenium_ovarian_cancer", "Xenium Prime", "human ovarian cancer", "human"),
    ("visium_mouse_brain", "Visium v2", "mouse brain", "mouse"),
    ("allen_merfish_brain_section", "MERFISH", "mouse brain", "mouse"),
]
DATASET_IDS = [d[0] for d in DATASETS]

SKLEARN_METHODS = [
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
]
SPATIAL_METHODS = [
    "banksy_py",
    "spatialde_kmeans",
    "nnsvg_kmeans",
    "harmony_kmeans",
    "moran_spectral",
]
METHODS = SKLEARN_METHODS + SPATIAL_METHODS
SEEDS = [42, 1, 2]
PROTOCOL = "histoweave.external_validation.5x15.v1"

N_MAX = 15_000
N_BOOT = 100
BOOT_FRAC = 0.80

# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


def _bundle_path(dataset: str) -> Path:
    """Return the h5ad path for a dataset id."""
    root = Path(os.environ.get("HISTOWEAVE_LOCAL_DATA", _HERE.parent))
    if dataset == "visium_hd_crc":
        return root / "datasets_cache" / "visium_hd_crc" / "visium_hd_crc.h5ad"
    if dataset == "visium_mouse_brain":
        return root / "datasets_cache" / "visium" / "visium_mouse_brain.h5ad"
    if dataset == "allen_merfish_brain_section":
        return root / "datasets_cache" / "merfish" / "allen_merfish_brain_section.h5ad"
    if dataset.startswith("xenium"):
        return root / "datasets_cache" / "xenium" / f"{dataset}.h5ad"
    raise KeyError(f"unknown dataset: {dataset}")


def _stratified_subsample(n_obs: int, labels: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_obs <= n_max:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.round(counts / n_obs * n_max).astype(int))
    diff = int(quota.sum() - n_max)
    if diff > 0:
        order = np.argsort(-counts)
        for i in order[:diff]:
            if quota[i] > 1:
                quota[i] -= 1
    elif diff < 0:
        order = np.argsort(-counts)
        for i in order[:(-diff)]:
            quota[i] += 1
    keep = []
    for u, q in zip(uniq, quota, strict=True):
        idx = np.where(labels == u)[0]
        pick = rng.choice(idx, size=min(int(q), idx.size), replace=False)
        keep.append(pick)
    return np.sort(np.concatenate(keep))


def load_dataset(dataset: str, seed: int) -> tuple[SpatialTable, int, np.ndarray]:
    """Load a dataset as (SpatialTable, n_domains, subsample_indices)."""
    p = _bundle_path(dataset)
    if not p.exists():
        raise FileNotFoundError(f"bundle not found: {p}")
    a = sc.read_h5ad(p)
    truth = a.obs.get("domain_truth", a.obs.get("spatialLIBD_layer"))
    truth = np.asarray(truth.astype(str).values)

    seed_material = f"{dataset}:{seed}".encode()
    sub_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    n_original = int(a.n_obs)
    idx = _stratified_subsample(a.n_obs, truth, N_MAX, sub_seed)
    a = a[idx].copy()
    truth = truth[idx]

    counts = a.layers.get("counts", a.X)
    X = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical(truth)},
        index=a.obs_names.astype(str),
    )
    var = pd.DataFrame(index=a.var_names.astype(str))
    tab = SpatialTable(
        X=X.astype(np.float32),
        obs=obs,
        var=var,
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=np.float32)},
        uns={"slice_id": dataset, "n_original": n_original},
    )
    tab.layers["counts"] = X
    n_domains = int(pd.Categorical(truth).categories.size)
    return tab, n_domains, idx


# --------------------------------------------------------------------------
# Spatial-aware dispatch (matches 5x15 / 7x15)
# --------------------------------------------------------------------------


def _adapter_labels(method: str, tab: SpatialTable, seed: int, n_domains: int):
    counts = tab.layers.get("counts") if hasattr(tab, "layers") else None
    X = counts if counts is not None else tab.X
    spatial = tab.obsm["spatial"]
    gene_names = tab.var.index.tolist()
    if method == "banksy_py":
        return banksy_py_adapter.run(X, spatial, seed=seed, n_domains=n_domains)
    if method == "harmony_kmeans":
        return harmony_adapter.run(X, spatial, seed=seed, n_domains=n_domains)
    if method == "moran_spectral":
        return moran_adapter.run(X, spatial, seed=seed, n_domains=n_domains)
    if method == "spatialde_kmeans":
        return spatialde_adapter.run(X, spatial, gene_names, seed=seed, n_domains=n_domains)
    if method == "nnsvg_kmeans":
        return nnsvg_adapter.run(X, spatial, gene_names, seed=seed, n_domains=n_domains)
    raise KeyError(f"unknown spatial-aware method: {method}")


def _bootstrap_ari(
    truth: np.ndarray, pred: np.ndarray, n_boot: int, frac: float, seed: int
) -> tuple[float, float, float]:
    if len(truth) != len(pred):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(truth)
    m = max(2, int(round(frac * n)))
    aris = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        aris[i] = adjusted_rand_score(truth[idx], pred[idx])
    return (
        float(np.mean(aris)),
        float(np.percentile(aris, 2.5)),
        float(np.percentile(aris, 97.5)),
    )


def _run_cell(method: str, dataset: str, seed: int, tab: SpatialTable, n_domains: int) -> dict:
    ckpt = CHK / f"{method}__{dataset}__seed{seed}.json"
    if ckpt.exists():
        return json.loads(ckpt.read_text())

    truth = tab.obs["domain_truth"].astype(str).values
    t0 = time.time()
    try:
        if method in SKLEARN_METHODS:
            ls = run_task_landscape(
                {dataset: tab},
                category=MethodCategory.DOMAIN_DETECTION,
                methods=[method],
                extra_params_factory=lambda _d: {"n_domains": n_domains, "random_state": seed},
            )
            ari_point = float(ls.performance[dataset][method])
            # Re-fit to recover labels for the bootstrap (same pattern as 7x15).
            from histoweave.plugins import create_method
            from histoweave.plugins.builtin.normalize import LogNormalize  # noqa: F401

            normed = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(tab.copy())
            cls = create_method(MethodCategory.DOMAIN_DETECTION, method)
            valid = set(cls.params.keys())
            kw = {}
            for k, v in {"n_domains": n_domains, "random_state": seed}.items():
                if k in valid:
                    kw[k] = v
            fitted = create_method(MethodCategory.DOMAIN_DETECTION, method, **kw).run(normed.copy())
            pred = np.asarray(fitted.obs["domain"].astype(str).values)
        else:
            pred = np.asarray(_adapter_labels(method, tab, seed, n_domains))
            pred = pred.astype(str)
            ari_point = float(adjusted_rand_score(truth, pred))
        elapsed = time.time() - t0
        ari_mean, ci_lo, ci_hi = _bootstrap_ari(truth, pred, N_BOOT, BOOT_FRAC, seed=seed)
        payload = {
            "ari": ari_point,
            "ari_bootstrap_mean": ari_mean,
            "ari_ci_lo": ci_lo,
            "ari_ci_hi": ci_hi,
            "seconds": elapsed,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        payload = {
            "ari": None,
            "ari_bootstrap_mean": None,
            "ari_ci_lo": None,
            "ari_ci_hi": None,
            "seconds": elapsed,
            "error": str(exc)[:400],
        }
        _log(f" [FAIL] {method}@{dataset} seed={seed}: {exc}")
    ckpt.write_text(json.dumps(payload))
    return payload


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    import gc

    per_seed_records: list[dict] = []
    dataset_meta: dict[str, dict] = {}

    for dataset, platform, tissue, organism in DATASETS:
        _log(f"\n===== {dataset} =====")
        tab0, k0, _ = load_dataset(dataset, seed=SEEDS[0])
        dataset_meta[dataset] = {
            "platform": platform,
            "tissue": tissue,
            "organism": organism,
            "n_obs": int(tab0.n_obs),
            "n_domains": int(k0),
            "n_original": int(tab0.uns.get("n_original", tab0.n_obs)),
            "subsampled": bool(tab0.uns.get("n_original", tab0.n_obs) > tab0.n_obs),
        }
        del tab0
        gc.collect()

        for seed in SEEDS:
            tab, k, _ = load_dataset(dataset, seed=seed)
            _log(f" [seed {seed}] {tab.n_obs} × {tab.n_vars}, k={k}")
            for method in METHODS:
                pay = _run_cell(method, dataset, seed, tab, k)
                per_seed_records.append(
                    {
                        "dataset": dataset,
                        "platform": platform,
                        "organism": organism,
                        "method": method,
                        "family": "sklearn" if method in SKLEARN_METHODS else "spatial_aware",
                        "seed": seed,
                        "ari": pay.get("ari"),
                        "ari_bootstrap_mean": pay.get("ari_bootstrap_mean"),
                        "ari_ci_lo": pay.get("ari_ci_lo"),
                        "ari_ci_hi": pay.get("ari_ci_hi"),
                        "seconds": pay.get("seconds"),
                        "n_obs_subsample": int(tab.n_obs),
                    }
                )
                ari_txt = f"{pay['ari']:.3f}" if pay.get("ari") is not None else "n/a"
                ci_txt = (
                    f"[{pay['ari_ci_lo']:.3f},{pay['ari_ci_hi']:.3f}]"
                    if pay.get("ari_ci_lo") is not None
                    else ""
                )
                _log(
                    f" {method:>18s} @ {dataset} seed={seed}: "
                    f"ARI={ari_txt} {ci_txt} ({pay['seconds']:.1f}s)",
                )
            del tab
            gc.collect()

    # ---------------------------------------------------------------- outputs
    df = pd.DataFrame(per_seed_records)
    df.to_csv(OUT / "benchmark_long.csv", index=False)

    piv_mean = df.pivot_table(
        index="dataset", columns="method", values="ari", aggfunc="mean"
    ).reindex(index=DATASET_IDS, columns=METHODS)
    piv_std = df.pivot_table(
        index="dataset", columns="method", values="ari", aggfunc="std"
    ).reindex(index=DATASET_IDS, columns=METHODS)
    piv_mean.to_csv(OUT / "performance_matrix_mean.csv")
    piv_std.to_csv(OUT / "performance_matrix_std.csv")

    boot = df.groupby(["dataset", "method"], as_index=False).agg(
        {
            "ari_bootstrap_mean": "mean",
            "ari_ci_lo": "mean",
            "ari_ci_hi": "mean",
            "ari": "mean",
        }
    )
    boot["platform"] = boot["dataset"].map(lambda d: dataset_meta[d]["platform"])
    boot.to_csv(OUT / "bootstrap_ci.csv", index=False)

    with open(OUT / "dataset_manifest.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    summary = {
        "protocol": PROTOCOL,
        "task": "external_validation_domain_detection",
        "metric": "ARI",
        "higher_is_better": True,
        "datasets": DATASET_IDS,
        "methods": METHODS,
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "bootstrap_fraction": BOOT_FRAC,
        "n_max_cells": N_MAX,
        "dataset_meta": dataset_meta,
        "notes": [
            "5 external validation datasets spanning 4 platforms, 2 organisms, "
            "4 tissues, 4 independent studies; strict region ground truth.",
            f"Datasets above {N_MAX} cells are stratified-subsampled per "
            "(dataset, seed) using a hash-derived sub_seed so every method "
            "sees the same slice.",
            f"Bootstrap: {N_BOOT} × {int(BOOT_FRAC * 100)}% cell resamples per cell, refit-free.",
        ],
    }
    with open(OUT / "benchmark_5x_external.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = {"protocol": PROTOCOL, "artifacts": []}
    for p in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json")):
        if p.name == "manifest.json":
            continue
        h = hashlib.sha256()
        h.update(p.read_bytes())
        manifest["artifacts"].append(
            {"path": p.name, "sha256": h.hexdigest(), "bytes": p.stat().st_size}
        )
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log("\n===== DONE =====")
    _log(piv_mean.round(3))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
