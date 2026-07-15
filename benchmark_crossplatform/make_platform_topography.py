"""Platform topography figure for the cross-platform benchmark.

For each dataset, extract the standard histoweave dataset-feature vector
(mean spot depth, sparsity, sample-cluster balance, etc.), PCA-reduce to
2-D, and scatter with points sized by best-ARI and coloured by winning
method. The figure lands as ``platform_topography.svg`` next to this
script and as a PNG copy for the report.

Usage:
    python benchmark_crossplatform/make_platform_topography.py

Reads:
    performance_matrix_7x15_mean.csv  (this folder)
    dataset_manifest.json             (this folder, produced by experiment_7x15.py)
    dataset h5ad bundles              (per HISTOWEAVE_LOCAL_DATA)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from histoweave.benchmark.features import (  # noqa: E402
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)
from histoweave.data import SpatialTable  # noqa: E402

# Phylo palette
PALETTE = {
    "banksy_py": "#0279EE",
    "harmony_kmeans": "#75A025",
    "moran_spectral": "#FF9400",
    "spatialde_kmeans": "#FD9BED",
    "nnsvg_kmeans": "#E9ED4C",
    # sklearn family — shades of grey
    "kmeans": "#333333",
    "spectral": "#555555",
    "agglomerative": "#777777",
    "gaussian_mixture": "#999999",
    "birch": "#aaaaaa",
    "bisecting_kmeans": "#bbbbbb",
    "minibatch_kmeans": "#cccccc",
    "mean_shift": "#666666",
    "dbscan": "#444444",
    "optics": "#222222",
}


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _bundle_path(dataset: str) -> Path:
    root = Path(os.environ.get("HISTOWEAVE_LOCAL_DATA", _ROOT))
    if dataset.startswith("merfish"):
        return root / "datasets_cache" / "merfish" / f"{dataset}.h5ad"
    if dataset.startswith("xenium"):
        return root / "datasets_cache" / "xenium" / f"{dataset}.h5ad"
    return root / "datasets_cache" / "dlpfc" / f"dlpfc_{dataset}.h5ad"


def _load_min(dataset: str, n_max: int = 8000) -> SpatialTable:
    """Load a minimal SpatialTable for feature extraction; subsample for speed."""
    p = _bundle_path(dataset)
    a = sc.read_h5ad(p)
    if a.n_obs > n_max:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(a.n_obs, size=n_max, replace=False))
        a = a[idx].copy()
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    obs = pd.DataFrame(index=a.obs_names.astype(str))
    var = pd.DataFrame(index=a.var_names.astype(str))
    return SpatialTable(
        X=X.astype(np.float32),
        obs=obs,
        var=var,
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=np.float32)},
        uns={"slice_id": dataset},
    )


def build_topography() -> None:
    mat_path = _HERE / "performance_matrix_7x15_mean.csv"
    if not mat_path.exists():
        raise FileNotFoundError(f"{mat_path} not found — run experiment_7x15.py first.")
    perf = pd.read_csv(mat_path, index_col=0)
    manifest = json.loads((_HERE / "dataset_manifest.json").read_text())
    datasets = list(perf.index)

    # ---- Feature vectors ---------------------------------------------------
    feats: dict[str, np.ndarray] = {}
    for did in datasets:
        tab = _load_min(did)
        f = extract_features(tab, include_domain=False)
        feats[did] = feature_vector(f, order=RECOMMENDATION_FEATURE_ORDER)
        _log(f"[features] {did}: {feats[did]}")

    F = np.vstack([feats[d] for d in datasets])
    # z-score columns, then PCA
    mean = np.nan_to_num(F.mean(axis=0))
    std = np.nan_to_num(F.std(axis=0), nan=1.0)
    std[std == 0] = 1.0
    Fz = (np.nan_to_num(F) - mean) / std
    # 2-D SVD
    U, S, _ = np.linalg.svd(Fz - Fz.mean(axis=0), full_matrices=False)
    scores = U[:, :2] * S[:2]

    # ---- Per-dataset best method + ARI ------------------------------------
    best_method = {}
    best_ari = {}
    for d in datasets:
        row = perf.loc[d].dropna()
        if row.empty:
            best_method[d] = None
            best_ari[d] = 0.0
        else:
            best_method[d] = str(row.idxmax())
            best_ari[d] = float(row.max())

    # ---- Figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for i, d in enumerate(datasets):
        m = best_method[d] or "kmeans"
        colour = PALETTE.get(m, "#0279EE")
        size = 60 + 800 * max(0, best_ari[d])
        ax.scatter(
            scores[i, 0],
            scores[i, 1],
            s=size,
            c=colour,
            alpha=0.85,
            edgecolor="#000",
            linewidth=0.6,
            zorder=3,
        )
        # Small platform tag
        plat = manifest.get(d, {}).get("platform", "")
        ax.annotate(
            f"{d}\n{plat}",
            (scores[i, 0], scores[i, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            zorder=4,
        )

    # Legend: methods that actually win somewhere
    winners = sorted(set(m for m in best_method.values() if m is not None))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE.get(m, "#0279EE"),
            markeredgecolor="#000",
            markersize=8,
            label=m,
        )
        for m in winners
    ]
    if handles:
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=9, title="Winning method")

    ax.set_xlabel("Dataset-feature PC1")
    ax.set_ylabel("Dataset-feature PC2")
    ax.set_title(
        "Cross-platform topography — each dataset sized by its best ARI, coloured by winner"
    )
    ax.axhline(0, color="#ccc", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#ccc", linewidth=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    svg = _HERE / "platform_topography.svg"
    png = _HERE / "platform_topography.png"
    fig.savefig(svg, format="svg")
    fig.savefig(png, format="png", dpi=150)
    plt.close(fig)
    _log(f"[write] {svg}")
    _log(f"[write] {png}")


if __name__ == "__main__":
    build_topography()
