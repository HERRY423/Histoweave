"""Cross-platform 7 × 15 benchmark with cell-bootstrap CIs.

Runs the Task-2 method list (10 sklearn + 5 spatial-aware) on 7 datasets:
5 DLPFC Visium slices (from Task 1) + MERFISH mouse hypothalamus
(Moffitt 2018, via squidpy) + SlideseqV2 mouse hippocampus (Stickels 2021,
via squidpy). Method + dataset checkpoints are shared with the 5×15 harness
so partial results carry over.

Design choices
--------------
* **Subsampling for tractability.** MERFISH (73 K) and SlideseqV2 (42 K) are
  too large for the sklearn baselines' full-affinity solvers on a 16 GB
  worker (spectral would allocate ~10 GB for an n×n graph). Every dataset
  above ``N_MAX`` cells is subsampled *stratified by ``domain_truth``* to
  ``N_MAX``; the same seeded subsample is reused for every method so the
  comparison is fair. The subsample seed is derived from the outer ``seed``
  argument so bootstrap CIs are computed on the actual subsampled slice.
* **Bootstrap CIs, refit-free.** Once a method has predicted labels on the
  full evaluation slice, we draw ``N_BOOT`` × 80 % cell subsamples and
  recompute ARI on each — no refit. This is the standard "evaluation
  bootstrap" (D3 in the plan).
* **Checkpoints.** ``<method>__<dataset>__seed<seed>.json`` stores the
  point ARI + bootstrap ``[lo, hi]`` at 2.5 / 97.5 percentiles. Crash loses
  only the last cell.

Outputs (in HISTOWEAVE_7x15_OUT):
  * ``benchmark_long.csv``            per-cell records with bootstrap CI
  * ``performance_matrix_7x15_mean.csv`` / ``_std.csv``
  * ``bootstrap_ci.csv``              per (method, dataset) mean + 2.5/97.5 pctl
  * ``benchmark_7x15.json``           self-describing summary
  * ``dataset_manifest.json``         metadata for the leaderboard
  * ``manifest.json``                 file hashes + protocol tag
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

# --- Import path shim so 5x15 adapters + this file's siblings both resolve.
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

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OUT = Path(os.environ.get("HISTOWEAVE_7x15_OUT", _HERE))
CHK = Path(os.environ.get("HISTOWEAVE_7x15_CHECKPOINT", _HERE / "checkpoints"))
OUT.mkdir(parents=True, exist_ok=True)
CHK.mkdir(parents=True, exist_ok=True)

DLPFC_SLICES = ["151673", "151674", "151507", "151669", "151670"]
CROSSPLAT_DATASETS = [
    ("xenium_breast_cancer", "Xenium", "human breast cancer"),
    ("merfish_mouse_brain", "MERFISH", "mouse brain"),
]
ALL_DATASETS = DLPFC_SLICES + [d[0] for d in CROSSPLAT_DATASETS]

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
PROTOCOL = "histoweave.crossplatform.7x15.v1"

# Cap dataset size for tractability (n×n graph methods on >20k cells OOM a
# 16 GB worker). All methods see the same seeded subsample so the comparison
# stays fair; bootstrap CIs are computed on the subsampled slice.
N_MAX = 15000
N_BOOT = 100
BOOT_FRAC = 0.80


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _bundle_path(dataset: str) -> Path:
    """Return the h5ad path for a dataset id."""
    root = Path(
        os.environ.get(
            "HISTOWEAVE_LOCAL_DATA",
            _HERE.parent,
        )
    )
    if dataset in DLPFC_SLICES:
        return root / "datasets_cache" / "dlpfc" / f"dlpfc_{dataset}.h5ad"
    if dataset.startswith("merfish"):
        return root / "datasets_cache" / "merfish" / f"{dataset}.h5ad"
    if dataset.startswith("xenium"):
        return root / "datasets_cache" / "xenium" / f"{dataset}.h5ad"
    raise KeyError(f"unknown dataset: {dataset}")


def _stratified_subsample(n_obs: int, labels: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    """Return sorted indices of a stratified-by-label subsample of size n_max."""
    if n_obs <= n_max:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    # Proportional per-class quota; guarantee at least 1 per class present in labels.
    uniq, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.round(counts / n_obs * n_max).astype(int))
    # Adjust to hit n_max exactly by trimming/growing the largest bins.
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
    """Load a dataset as (SpatialTable, n_domains, subsample_indices).

    Datasets above ``N_MAX`` cells are stratified-subsampled with a seed
    derived from ``(dataset, seed)`` so the subset is deterministic per
    (dataset, seed) but differs across seeds.
    """
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
    if "counts" in a.layers or hasattr(counts, "todense"):
        tab.layers["counts"] = X
    n_domains = int(pd.Categorical(truth).categories.size)
    return tab, n_domains, idx


# --------------------------------------------------------------------------
# Spatial-aware dispatch (matches 5×15)
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
    """Return (mean, ci_lo, ci_hi) of ARI over n_boot × frac cell resamples."""
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
            # Reuse the landscape harness's single-cell path so param filtering
            # (dbscan lacks n_domains etc.) works identically to 5×15.
            ls = run_task_landscape(
                {dataset: tab},
                category=MethodCategory.DOMAIN_DETECTION,
                methods=[method],
                extra_params_factory=lambda _d: {"n_domains": n_domains, "random_state": seed},
            )
            ari_point = float(ls.performance[dataset][method])
            # The landscape harness records labels in obs['domain']; we need
            # them for the bootstrap. Since run_task_landscape swallows the
            # data.copy, re-fit the labels here so the bootstrap sees the
            # same predictions. Cheaper than refactoring the harness.
            from histoweave.plugins import create_method
            from histoweave.plugins.builtin.normalize import LogCP10K  # noqa: F401

            normed = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(tab.copy())
            # Determine which params this specific method accepts.
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
        _log(f"  [FAIL] {method}@{dataset} seed={seed}: {exc}")
    ckpt.write_text(json.dumps(payload))
    return payload


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    import gc

    per_seed_records: list[dict] = []
    dataset_meta: dict[str, dict] = {}

    for dataset in ALL_DATASETS:
        _log(f"\n===== {dataset} =====")
        # We need the metadata (platform / n_domains / n_obs) which is the
        # same across seeds, so peek at seed 42 once and cache.
        tab0, k0, _ = load_dataset(dataset, seed=SEEDS[0])
        platform = (
            "Visium"
            if dataset in DLPFC_SLICES
            else "MERFISH"
            if dataset.startswith("merfish")
            else "Xenium"
            if dataset.startswith("xenium")
            else "unknown"
        )
        dataset_meta[dataset] = {
            "platform": platform,
            "tissue": (
                "DLPFC"
                if dataset in DLPFC_SLICES
                else "mouse brain"
                if dataset.startswith("merfish")
                else "human breast cancer"
                if dataset.startswith("xenium")
                else "unknown"
            ),
            "n_obs": int(tab0.n_obs),
            "n_domains": int(k0),
            "n_original": int(tab0.uns.get("n_original", tab0.n_obs)),
            "subsampled": bool(tab0.uns.get("n_original", tab0.n_obs) > tab0.n_obs),
        }
        del tab0
        gc.collect()

        for seed in SEEDS:
            tab, k, _ = load_dataset(dataset, seed=seed)
            _log(f"  [seed {seed}] {tab.n_obs} × {tab.n_vars}, k={k}")
            for method in METHODS:
                pay = _run_cell(method, dataset, seed, tab, k)
                per_seed_records.append(
                    {
                        "dataset": dataset,
                        "platform": platform,
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
                    f"    {method:>18s} @ {dataset} seed={seed}: "
                    f"ARI={ari_txt} {ci_txt} ({pay['seconds']:.1f}s)",
                )
            del tab
            gc.collect()

    # ---------------------------------------------------------------- outputs
    df = pd.DataFrame(per_seed_records)
    df.to_csv(OUT / "benchmark_long.csv", index=False)

    # 7 × 15 mean / std matrix on point ARI
    piv_mean = df.pivot_table(
        index="dataset", columns="method", values="ari", aggfunc="mean"
    ).reindex(index=ALL_DATASETS, columns=METHODS)
    piv_std = df.pivot_table(
        index="dataset", columns="method", values="ari", aggfunc="std"
    ).reindex(index=ALL_DATASETS, columns=METHODS)
    piv_mean.to_csv(OUT / "performance_matrix_7x15_mean.csv")
    piv_std.to_csv(OUT / "performance_matrix_7x15_std.csv")

    # Bootstrap CI table averaged over seeds
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

    # Save dataset manifest for the leaderboard
    with open(_HERE / "dataset_manifest.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    summary = {
        "protocol": PROTOCOL,
        "task": "cross_platform_domain_detection",
        "metric": "ARI",
        "higher_is_better": True,
        "datasets": ALL_DATASETS,
        "methods": METHODS,
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "bootstrap_fraction": BOOT_FRAC,
        "n_max_cells": N_MAX,
        "dataset_meta": dataset_meta,
        "notes": [
            f"Datasets above {N_MAX} cells are stratified-subsampled per "
            "(dataset, seed) using a hash-derived sub_seed so every method "
            "sees the same slice.",
            f"Bootstrap: {N_BOOT} × {int(BOOT_FRAC * 100)}% cell resamples per cell, refit-free.",
        ],
    }
    with open(OUT / "benchmark_7x15.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Manifest with hashes
    manifest = {"protocol": PROTOCOL, "artifacts": []}
    for p in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json")):
        if p.name == "manifest.json":
            continue
        h = hashlib.sha256()
        h.update(p.read_bytes())
        manifest["artifacts"].append(
            {
                "path": p.name,
                "sha256": h.hexdigest(),
                "bytes": p.stat().st_size,
            }
        )
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log("\n===== DONE =====")
    _log(piv_mean.round(3))


if __name__ == "__main__":
    main()
