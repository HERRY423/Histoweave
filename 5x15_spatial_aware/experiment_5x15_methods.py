"""HistoWeave 5-dataset x 19-method spatial-aware performance landscape on DLPFC.

Extends ``5x10_dlpfc_benchmark/experiment_5x10_dlpfc.py`` from 10 sklearn
methods to 19 methods (10 sklearn + 5 spatial-aware + 4 SOTA):

* Sklearn family (unchanged from the 5×10 harness):
    agglomerative, birch, bisecting_kmeans, dbscan, gaussian_mixture, kmeans,
    mean_shift, minibatch_kmeans, optics, spectral
* Spatial-aware family (new adapters under ``adapters/``):
    banksy_py, spatialde_kmeans, nnsvg_kmeans, harmony_kmeans, moran_spectral
* Published SOTA spatial-domain family (official external backends):
    spagcn, graphst, bayesspace, stagate

Metric: ARI vs `spatialLIBD_layer`. 3 seeds (42/1/2). n_domains per slice
matches the true layer count read from the h5ad bundle.

Outputs are written next to this script (or ``HISTOWEAVE_5x15_OUT`` if set):
  * ``benchmark_long.csv``, ``performance_matrix_mean.csv``, ``_std.csv``
  * ``timings_mean.csv``, ``benchmark.json``, ``manifest.json``
  * ``heatmap_5x19.svg`` (produced by :file:`make_heatmap.py`)
  * ``report_5x19.md`` (produced by :file:`make_report.py`)

Bundle path: reads the 12 DLPFC h5ad bundles from Task 1 either from
``HISTOWEAVE_LOCAL_DATA=/path/to/histoweave_upgrade`` via the registry
(preferred, exercises the ``local://`` code path) or from ``HISTOWEAVE_DLPFC_DATA``
(a legacy escape-hatch that reads the same files directly).
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
from sklearn.metrics import adjusted_rand_score

# --- import path shim so adapters/ + svg_domain_pipeline.py both resolve ---
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from adapters import (  # noqa: E402
    banksy_py_adapter,
    bayesspace_adapter,
    graphst_adapter,
    harmony_adapter,
    moran_adapter,
    nnsvg_adapter,
    spagcn_adapter,
    spatialde_adapter,
    stagate_adapter,
)
from sota_runner import checkpoint_metadata, run_sota_cell  # noqa: E402

from histoweave.data import SpatialTable  # noqa: E402

OUT = Path(os.environ.get("HISTOWEAVE_5x15_OUT", _HERE))
CHK = Path(os.environ.get("HISTOWEAVE_5x15_CHECKPOINT", _HERE / "checkpoints"))
OUT.mkdir(parents=True, exist_ok=True)
CHK.mkdir(parents=True, exist_ok=True)

# Same 5 slices as the 5×10 harness, ordered by difficulty.
SLICES = ["151673", "151674", "151507", "151669", "151670"]

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
SOTA_METHODS = ["spagcn", "graphst", "bayesspace", "stagate"]
METHODS = SKLEARN_METHODS + SPATIAL_METHODS + SOTA_METHODS
SEEDS = [42, 1, 2]
PROTOCOL = "histoweave.landscape.dlpfc_spatial_sota.v2"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _bundle_path(sid: str) -> Path:
    """Return the h5ad path for a slice, preferring the registry path."""
    override = os.environ.get("HISTOWEAVE_DLPFC_DATA")
    if override:
        p = Path(override) / f"{sid}.h5ad"
        if p.exists():
            return p
    root = os.environ.get(
        "HISTOWEAVE_LOCAL_DATA",
        str(Path(__file__).resolve().parents[1]),
    )
    return Path(root) / "datasets_cache" / "dlpfc" / f"dlpfc_{sid}.h5ad"


def load_slice(sid: str) -> tuple[SpatialTable, int]:
    """Load a DLPFC bundle as (SpatialTable, n_domains)."""
    import scanpy as sc

    p = _bundle_path(sid)
    if not p.exists():
        raise FileNotFoundError(
            f"DLPFC bundle missing at {p}. "
            "Set HISTOWEAVE_LOCAL_DATA to your histoweave_upgrade root, "
            "or HISTOWEAVE_DLPFC_DATA to a folder of raw {sid}.h5ad files."
        )
    a = sc.read_h5ad(p)
    counts = a.layers.get("counts", a.X)
    X = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    truth = a.obs.get("domain_truth", a.obs.get("spatialLIBD_layer"))
    truth = pd.Categorical(truth.astype(str).values)
    obs = pd.DataFrame({"domain_truth": truth}, index=a.obs_names.astype(str))
    for coord in ("array_row", "array_col"):
        if coord in a.obs:
            obs[coord] = a.obs[coord].to_numpy()
    var = pd.DataFrame(index=a.var_names.astype(str))
    # float32 keeps 5 slices below 2 GB combined and matches downstream
    # adapters' precision — sklearn methods internally upcast when needed.
    tab = SpatialTable(
        X=X.astype(np.float32),
        obs=obs,
        var=var,
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=np.float32)},
        uns={"slice_id": sid},
    )
    n_domains = int(pd.Series(truth).nunique())
    return tab, n_domains


# --------------------------------------------------------------------------
# Spatial-aware dispatch
# --------------------------------------------------------------------------


def _adapter_labels(
    method: str,
    tab: SpatialTable,
    seed: int,
    n_domains: int,
) -> np.ndarray:
    counts = tab.layers.get("counts") if hasattr(tab, "layers") else None
    X = counts if counts is not None else tab.X
    spatial = tab.obsm["spatial"]
    gene_names = tab.var.index.tolist()
    array_coords = None
    if {"array_row", "array_col"}.issubset(tab.obs.columns):
        array_coords = tab.obs[["array_row", "array_col"]].to_numpy()
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
    if method == "spagcn":
        return spagcn_adapter.run(
            X,
            spatial,
            gene_names,
            seed=seed,
            n_domains=n_domains,
            array_coords=array_coords,
        )
    if method == "graphst":
        return graphst_adapter.run(X, spatial, gene_names, seed=seed, n_domains=n_domains)
    if method == "bayesspace":
        return bayesspace_adapter.run(
            X,
            spatial,
            gene_names,
            seed=seed,
            n_domains=n_domains,
            array_coords=array_coords,
        )
    if method == "stagate":
        return stagate_adapter.run(X, spatial, gene_names, seed=seed, n_domains=n_domains)
    raise KeyError(f"unknown spatial-aware method: {method}")


def _run_spatial_cell(
    method: str, sid: str, seed: int, tab: SpatialTable, n_domains: int
) -> tuple[float, float]:
    ckpt = CHK / f"{method}__{sid}__seed{seed}.json"
    if ckpt.exists():
        cached = json.loads(ckpt.read_text())
        ari = cached.get("ari")
        return (float(ari) if ari is not None else float("nan")), cached["seconds"]
    t0 = time.time()
    try:
        labels = _adapter_labels(method, tab, seed=seed, n_domains=n_domains)
        truth = tab.obs["domain_truth"].astype(str).values
        ari = float(adjusted_rand_score(truth, labels))
    except Exception as exc:  # noqa: BLE001
        # Record NaN + reason; the aggregator surfaces the failure in the report.
        ari = float("nan")
        elapsed = time.time() - t0
        ckpt.write_text(json.dumps({"ari": None, "seconds": elapsed, "error": str(exc)[:400]}))
        _log(f"  [FAIL] {method}@{sid} seed={seed}: {exc}")
        return ari, elapsed
    elapsed = time.time() - t0
    ckpt.write_text(json.dumps({"ari": ari, "seconds": elapsed}))
    return ari, elapsed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def _run_sklearn_cell(
    method: str, sid: str, seed: int, tab: SpatialTable, n_domains: int
) -> tuple[float, float]:
    """Run a single sklearn method on a single (slice, seed) cell.

    Because ``run_task_landscape`` + ``SklearnClusterMethod.run`` allocate
    several copies of the expression matrix per call and the Python GC does
    not always release them promptly, running 10 methods in a single
    process climbs monotonically past 15 GB on Visium DLPFC slices. We
    isolate each method in a fresh subprocess so the OS reclaims memory
    between calls. Checkpoint files (JSON) are the handoff channel; a
    crash loses only the running cell.
    """
    ckpt = CHK / f"{method}__{sid}__seed{seed}.json"
    if ckpt.exists():
        cached = json.loads(ckpt.read_text())
        ari = cached.get("ari")
        return (float(ari) if ari is not None else float("nan")), cached["seconds"]

    import subprocess

    script = (
        "import sys, os, json, time, numpy as np\n"
        f"sys.path.insert(0, {json.dumps(str(_HERE))})\n"
        f"sys.path.insert(0, {json.dumps(str(_HERE.parent / 'src'))})\n"
        "from experiment_5x15 import load_slice\n"
        "from histoweave.plugins import MethodCategory\n"
        "from histoweave.benchmark.landscape import run_task_landscape\n"
        f"sid = {json.dumps(sid)}; method = {json.dumps(method)}; "
        f"seed = {seed}; n_domains = {n_domains}\n"
        f"ckpt_path = {json.dumps(str(ckpt))}\n"
        "tab, _ = load_slice(sid)\n"
        "t0 = time.time()\n"
        "try:\n"
        "    ls = run_task_landscape(\n"
        "        {sid: tab},\n"
        "        category=MethodCategory.DOMAIN_DETECTION,\n"
        "        methods=[method],\n"
        "        extra_params_factory=lambda _d: {\n"
        "            'n_domains': n_domains, 'random_state': seed,\n"
        "        },\n"
        "    )\n"
        "    ari = float(ls.performance[sid][method])\n"
        "    payload = {'ari': ari if np.isfinite(ari) else None,\n"
        "               'seconds': time.time() - t0}\n"
        "except Exception as exc:\n"
        "    payload = {'ari': None, 'seconds': time.time() - t0,\n"
        "               'error': str(exc)[:400]}\n"
        "with open(ckpt_path, 'w') as f:\n"
        "    json.dump(payload, f)\n"
        "sys.stdout.write(json.dumps(payload) + '\\n')\n"
    )
    t0 = time.time()
    try:
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(_HERE.parent / "src"), str(_HERE))),
            "HISTOWEAVE_LOCAL_DATA": os.environ.get("HISTOWEAVE_LOCAL_DATA", str(_HERE.parent)),
        }
        r = subprocess.run(
            [sys.executable, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if r.returncode != 0 or not ckpt.exists():
            elapsed = time.time() - t0
            err = (r.stderr or "").strip().splitlines()[-1:] if r.stderr else []
            payload = {
                "ari": None,
                "seconds": elapsed,
                "error": (err[0] if err else "no output")[:400],
            }
            ckpt.write_text(json.dumps(payload))
            _log(
                f"  [FAIL] {method}@{sid} seed={seed}: rc={r.returncode} stderr={payload['error']}",
            )
            return float("nan"), elapsed
        payload = json.loads(ckpt.read_text())
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        payload = {"ari": None, "seconds": elapsed, "error": "timeout"}
        ckpt.write_text(json.dumps(payload))
        _log(f"  [TIMEOUT] {method}@{sid} seed={seed}")
        return float("nan"), elapsed
    ari = payload.get("ari")
    return (float(ari) if ari is not None else float("nan")), payload["seconds"]


def main() -> None:
    # Iterate slice-by-slice, seed-by-seed with explicit gc between slices so
    # 5×19 fits comfortably in a 16 GB sandbox. ``run_task_landscape`` copies
    # the SpatialTable per method internally; holding all 5 slices resident
    # plus concurrent copies is what OOMed a prior 32 GB run.
    import gc

    n_domains_map: dict[str, int] = {}
    per_seed_perf: dict[int, dict[str, dict[str, float]]] = {
        s: {sid: {} for sid in SLICES} for s in SEEDS
    }
    per_seed_time: dict[int, dict[str, dict[str, float]]] = {
        s: {sid: {} for sid in SLICES} for s in SEEDS
    }

    for sid in SLICES:
        tab, k = load_slice(sid)
        n_domains_map[sid] = k
        _log(f"\n[load] {sid}: {tab.n_obs} spots, {tab.n_vars} genes, k={k}")

        for seed in SEEDS:
            _log(f"  --- seed {seed} ---")
            for method in SKLEARN_METHODS:
                ari, secs = _run_sklearn_cell(method, sid, seed, tab, k)
                per_seed_perf[seed][sid][method] = ari
                per_seed_time[seed][sid][method] = secs
                _log(f"  {method:>18s} @ {sid} seed={seed}: ARI={ari:.3f} ({secs:.1f}s)")
            for method in SPATIAL_METHODS:
                ari, secs = _run_spatial_cell(method, sid, seed, tab, k)
                per_seed_perf[seed][sid][method] = ari
                per_seed_time[seed][sid][method] = secs
                _log(f"  {method:>18s} @ {sid} seed={seed}: ARI={ari:.3f} ({secs:.1f}s)")
            for method in SOTA_METHODS:
                ari, secs = run_sota_cell(
                    method, sid, seed, k, benchmark_dir=_HERE, checkpoint_dir=CHK
                )
                per_seed_perf[seed][sid][method] = ari
                per_seed_time[seed][sid][method] = secs
                _log(f"  {method:>18s} @ {sid} seed={seed}: ARI={ari:.3f} ({secs:.1f}s)")

        # Free this slice before loading the next.
        del tab
        gc.collect()

    # ---- aggregate --------------------------------------------------------
    long_rows: list[dict] = []
    for sid in SLICES:
        for m in METHODS:
            for s in SEEDS:
                status, error = checkpoint_metadata(m, sid, s, checkpoint_dir=CHK)
                long_rows.append(
                    {
                        "dataset": sid,
                        "method": m,
                        "family": (
                            "sklearn"
                            if m in SKLEARN_METHODS
                            else "sota"
                            if m in SOTA_METHODS
                            else "spatial_aware"
                        ),
                        "seed": s,
                        "ari": _f(per_seed_perf[s][sid].get(m, np.nan)),
                        "seconds": per_seed_time[s][sid].get(m),
                        "n_domains_truth": n_domains_map[sid],
                        "status": status,
                        "error": error,
                    }
                )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "benchmark_long.csv", index=False)

    agg = long_df.groupby(["dataset", "method"])["ari"].agg(["mean", "std", "count"]).reset_index()
    mean_mat = agg.pivot(index="dataset", columns="method", values="mean").reindex(
        index=SLICES, columns=METHODS
    )
    std_mat = agg.pivot(index="dataset", columns="method", values="std").reindex(
        index=SLICES, columns=METHODS
    )
    mean_mat.to_csv(OUT / "performance_matrix_mean.csv")
    std_mat.to_csv(OUT / "performance_matrix_std.csv")

    time_agg = (
        long_df.dropna(subset=["seconds"])
        .groupby(["dataset", "method"])["seconds"]
        .mean()
        .reset_index()
    )
    time_agg.to_csv(OUT / "timings_mean.csv", index=False)

    master = {
        "protocol": PROTOCOL,
        "task": "domain_detection",
        "metric": "ARI",
        "higher_is_better": True,
        "datasets": SLICES,
        "methods": METHODS,
        "sklearn_methods": SKLEARN_METHODS,
        "spatial_methods": SPATIAL_METHODS,
        "sota_methods": SOTA_METHODS,
        "seeds": SEEDS,
        "n_domains_truth": n_domains_map,
        "performance_matrix_mean": _js(
            {
                sid: {
                    m: float(mean_mat.loc[sid, m]) if np.isfinite(mean_mat.loc[sid, m]) else None
                    for m in METHODS
                }
                for sid in SLICES
            }
        ),
        "best_method_per_slice": _js(
            {
                sid: max(
                    ((m, v) for m, v in mean_mat.loc[sid].to_dict().items() if np.isfinite(v)),
                    key=lambda kv: kv[1],
                    default=(None, None),
                )[0]
                for sid in SLICES
            }
        ),
        "limitations": [
            "5 slices from one study (Maynard 2021 human DLPFC) => within-study validation only.",
            "SpatialDE/nnSVG are ranking-only; downstream clustering is fixed "
            "to top-N -> PCA -> KMeans.",
            "banksy_py is a from-scratch reimplementation of the BANKSY recipe "
            "(not the Bioconductor Banksy).",
            "GraphST and STAGATE embeddings use the same fixed-q full-covariance "
            "Gaussian-mixture downstream clustering; BayesSpace and SpaGCN retain "
            "their native clustering procedures.",
            "Official SOTA backends have incompatible legacy dependencies and are "
            "therefore run through method-specific isolated Python interpreters.",
        ],
    }
    with open(OUT / "benchmark.json", "w") as f:
        json.dump(master, f, indent=2)

    manifest = {"protocol": PROTOCOL, "artifacts": []}
    for p in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json")):
        if p.name == "manifest.json":
            continue
        manifest["artifacts"].append({"path": p.name, "sha256": _sha(p), "bytes": p.stat().st_size})
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log("\n=== DONE ===")
    _log(f"mean_mat shape: {mean_mat.shape}")
    _log(mean_mat.round(3))


def _f(v):
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _js(v):
    if isinstance(v, dict):
        return {str(k): _js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
