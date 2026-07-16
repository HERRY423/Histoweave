"""Seven-real-dataset by 19-method cross-tissue domain benchmark.

The five Maynard DLPFC sections are supplemented with Human Lymph Node
(Xenium Prime pathology annotations) and Mouse Brain (Allen Brain Atlas
MERFISH anatomical annotations).  Every method sees the same deterministic,
truth-stratified subset for a dataset/seed pair.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BASE = _ROOT / "5x15_spatial_aware"
for path in (str(_ROOT / "src"), str(_BASE), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment_5x15_methods import (  # noqa: E402
    METHODS,
    SKLEARN_METHODS,
    SOTA_METHODS,
)

from histoweave.data import SpatialTable  # noqa: E402

OUT = Path(os.environ.get("HISTOWEAVE_7x19_OUT", _HERE))
CHK = Path(os.environ.get("HISTOWEAVE_7x19_CHECKPOINT", _HERE / "checkpoints"))
OUT.mkdir(parents=True, exist_ok=True)
CHK.mkdir(parents=True, exist_ok=True)

DLPFC_SLICES = ["151673", "151674", "151507", "151669", "151670"]
NON_DLPFC_DATASETS = {
    "xenium_human_lymph_node": {
        "platform": "Xenium Prime",
        "tissue": "human lymph node",
        "truth_source": "10x pathology annotation polygons",
    },
    "merfish_mouse_brain": {
        "platform": "MERFISH",
        "tissue": "mouse brain",
        "truth_source": "Allen CCF anatomical division/region",
    },
}
DATASETS = DLPFC_SLICES + list(NON_DLPFC_DATASETS)
SEEDS = [42, 1, 2]
N_MAX = 15_000
PROTOCOL = "histoweave.cross_tissue.7x19.v1"


def _log(message: object) -> None:
    logging.getLogger(__name__).info("%s", message)


def _adapter_labels(
    method: str,
    tab: SpatialTable,
    seed: int,
    n_domains: int,
):
    """Expose the shared 19-method adapter dispatch to isolated workers."""
    from experiment_5x15_methods import _adapter_labels as dispatch

    return dispatch(method, tab, seed, n_domains)


def _bundle_path(dataset: str) -> Path:
    root = Path(os.environ.get("HISTOWEAVE_LOCAL_DATA", _ROOT))
    if dataset in DLPFC_SLICES:
        return root / "datasets_cache" / "dlpfc" / f"dlpfc_{dataset}.h5ad"
    if dataset.startswith("xenium"):
        return root / "datasets_cache" / "xenium" / f"{dataset}.h5ad"
    if dataset.startswith("merfish"):
        return root / "datasets_cache" / "merfish" / f"{dataset}.h5ad"
    raise KeyError(f"unknown dataset: {dataset}")


def _stratified_subsample(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(labels) <= limit:
        return np.arange(len(labels))
    series = pd.Series(labels)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for indices in series.groupby(series, observed=True).indices.values():
        quota = max(1, round(len(indices) / len(series) * limit))
        selected.append(rng.choice(indices, min(quota, len(indices)), replace=False))
    merged = np.unique(np.concatenate(selected))
    if len(merged) > limit:
        merged = rng.choice(merged, limit, replace=False)
    elif len(merged) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), merged, assume_unique=False)
        merged = np.concatenate([merged, rng.choice(remaining, limit - len(merged), False)])
    return np.sort(merged)


def load_dataset(dataset: str, seed: int = 42) -> tuple[SpatialTable, int]:
    import scanpy as sc

    path = _bundle_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark bundle missing: {path}. Run the matching prepare_*.py script first."
        )
    adata = sc.read_h5ad(path)
    if "domain_truth" not in adata.obs or "spatial" not in adata.obsm:
        raise ValueError(f"{path} must contain obs['domain_truth'] and obsm['spatial']")
    truth = adata.obs["domain_truth"].astype(str).to_numpy()
    if pd.isna(truth).any() or np.isin(truth, ["nan", "unknown", "unmapped", "ambiguous"]).any():
        raise ValueError(f"{dataset} contains invalid domain_truth labels")
    sub_seed = int.from_bytes(hashlib.sha256(f"{dataset}:{seed}".encode()).digest()[:4], "big")
    keep = _stratified_subsample(truth, N_MAX, sub_seed)
    adata = adata[keep].copy()
    truth = truth[keep]
    counts = adata.layers.get("counts", adata.X)
    matrix = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    obs = pd.DataFrame({"domain_truth": pd.Categorical(truth)}, index=adata.obs_names.astype(str))
    for coordinate in ("array_row", "array_col"):
        if coordinate in adata.obs:
            obs[coordinate] = adata.obs[coordinate].to_numpy()
    tab = SpatialTable(
        X=matrix.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=adata.var_names.astype(str)),
        obsm={"spatial": np.asarray(adata.obsm["spatial"], dtype=np.float32)},
        uns={
            "dataset_id": dataset,
            "n_original": int(adata.uns.get("n_original", len(keep))),
            "assay": "visium"
            if dataset in DLPFC_SLICES
            else "xenium"
            if dataset.startswith("xenium")
            else "merfish",
        },
    )
    tab.layers["counts"] = matrix.astype(np.float32)
    return tab, int(pd.Series(truth).nunique())


def unsupported_reason(method: str, dataset: str) -> str | None:
    if method == "bayesspace" and dataset not in DLPFC_SLICES:
        return "BayesSpace requires Visium array_row/array_col coordinates"
    return None


def _checkpoint(method: str, dataset: str, seed: int) -> Path:
    return CHK / f"{method}__{dataset}__seed{seed}.json"


def _run_in_subprocess(method: str, dataset: str, seed: int, n_domains: int) -> dict:
    checkpoint = _checkpoint(method, dataset, seed)
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    reason = unsupported_reason(method, dataset)
    if reason:
        payload = {"ari": None, "seconds": 0.0, "status": "unsupported", "error": reason}
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    python = os.environ.get(f"HISTOWEAVE_{method.upper()}_PYTHON", sys.executable)
    code = (
        "import json, sys, time, numpy as np\n"
        f"sys.path[:0] = [{str(_HERE)!r}, {str(_BASE)!r}, {str(_ROOT / 'src')!r}]\n"
        "from experiment_7x19 import load_dataset, _adapter_labels\n"
        "from histoweave.benchmark.landscape import run_task_landscape\n"
        "from histoweave.plugins import MethodCategory\n"
        "from sklearn.metrics import adjusted_rand_score\n"
        f"method={method!r}; dataset={dataset!r}; seed={seed}; k={n_domains}; "
        f"checkpoint={str(checkpoint)!r}\n"
        "tab, _ = load_dataset(dataset, seed); started=time.time()\n"
        "try:\n"
        "    if method in " + repr(SKLEARN_METHODS) + ":\n"
        "        result=run_task_landscape(\n"
        "            {dataset: tab}, category=MethodCategory.DOMAIN_DETECTION,\n"
        "            methods=[method],\n"
        "            extra_params_factory=lambda _d: {'n_domains': k, 'random_state': seed},\n"
        "        )\n"
        "        ari=float(result.performance[dataset][method])\n"
        "    else:\n"
        "        labels=_adapter_labels(method, tab, seed, k)\n"
        "        truth=tab.obs['domain_truth'].astype(str).values\n"
        "        ari=float(adjusted_rand_score(truth, labels))\n"
        "    payload={'ari': ari if np.isfinite(ari) else None, 'seconds': time.time()-started, "
        "'status': 'success'}\n"
        "except Exception as exc:\n"
        "    payload={'ari': None, 'seconds': time.time()-started, 'status': 'failed', "
        "'error': str(exc)[:400]}\n"
        "open(checkpoint, 'w').write(json.dumps(payload))\n"
    )
    started = time.time()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(_ROOT / "src"), str(_BASE), str(_HERE))),
        "HISTOWEAVE_LOCAL_DATA": os.environ.get("HISTOWEAVE_LOCAL_DATA", str(_ROOT)),
    }
    timeout = (
        int(os.environ.get("HISTOWEAVE_SOTA_TIMEOUT", "7200")) if method in SOTA_METHODS else 900
    )
    try:
        result = subprocess.run(
            [python, "-u", "-c", code], capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode != 0 or not checkpoint.exists():
            error = (result.stderr or "no output").strip().splitlines()[-1]
            checkpoint.write_text(
                json.dumps(
                    {
                        "ari": None,
                        "seconds": time.time() - started,
                        "status": "failed",
                        "error": error[:400],
                    }
                ),
                encoding="utf-8",
            )
    except subprocess.TimeoutExpired:
        checkpoint.write_text(
            json.dumps(
                {
                    "ari": None,
                    "seconds": time.time() - started,
                    "status": "failed",
                    "error": "timeout",
                }
            ),
            encoding="utf-8",
        )
    return json.loads(checkpoint.read_text(encoding="utf-8"))


def main() -> None:
    rows: list[dict[str, object]] = []
    dataset_meta: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        preview, k = load_dataset(dataset, SEEDS[0])
        meta = (
            {"platform": "Visium", "tissue": "human DLPFC", "truth_source": "manual cortical layer"}
            if dataset in DLPFC_SLICES
            else NON_DLPFC_DATASETS[dataset]
        )
        dataset_meta[dataset] = {
            **meta,
            "n_obs_evaluation": int(preview.n_obs),
            "n_domains": k,
            "bundle": str(_bundle_path(dataset)),
        }
        del preview
        for seed in SEEDS:
            for method in METHODS:
                payload = _run_in_subprocess(method, dataset, seed, k)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "family": "sklearn"
                        if method in SKLEARN_METHODS
                        else "sota"
                        if method in SOTA_METHODS
                        else "spatial_aware",
                        "seed": seed,
                        "ari": payload.get("ari"),
                        "seconds": payload.get("seconds"),
                        "status": payload.get("status"),
                        "error": payload.get("error"),
                        "n_domains_truth": k,
                    }
                )
                _log(
                    f"{method}@{dataset} seed={seed}: {payload.get('status')} "
                    f"ARI={payload.get('ari')}"
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "benchmark_long.csv", index=False)
    grouped = frame.groupby(["dataset", "method"])["ari"].agg(["mean", "std", "count"])
    mean = grouped["mean"].unstack("method").reindex(index=DATASETS, columns=METHODS)
    std = grouped["std"].unstack("method").reindex(index=DATASETS, columns=METHODS)
    mean.to_csv(OUT / "performance_matrix_7x19_mean.csv")
    std.to_csv(OUT / "performance_matrix_7x19_std.csv")
    (OUT / "dataset_manifest.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    summary = {
        "protocol": PROTOCOL,
        "task": "cross_tissue_spatial_domain_detection",
        "metric": "ARI",
        "datasets": DATASETS,
        "methods": METHODS,
        "seeds": SEEDS,
        "n_max_cells": N_MAX,
        "unsupported_cells": {"bayesspace": list(NON_DLPFC_DATASETS)},
        "dataset_meta": dataset_meta,
    }
    (OUT / "benchmark_7x19.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts = []
    for path in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json")):
        if path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    (OUT / "manifest.json").write_text(
        json.dumps({"protocol": PROTOCOL, "artifacts": artifacts}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
