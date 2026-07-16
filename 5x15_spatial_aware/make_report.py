"""Generate ``report_5x19.md`` from the aggregated benchmark artefacts.

Reads ``performance_matrix_mean.csv``, ``performance_matrix_std.csv``,
``timings_mean.csv``, and ``benchmark.json``. Emits a self-contained
markdown report with:

* an executive summary (best method per slice, sklearn vs spatial-aware
  overall mean),
* the mean/std ARI table (formatted, top-3 methods per slice **bolded**),
* per-method runtime medians,
* per-method commentary,
* the ``heatmap_5x19.svg`` figure link.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
SKLEARN = {
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
SPATIAL = {
    "banksy_py",
    "spatialde_kmeans",
    "nnsvg_kmeans",
    "harmony_kmeans",
    "moran_spectral",
    "spagcn",
    "graphst",
    "bayesspace",
    "stagate",
}

METHOD_NOTES = {
    "spagcn": "Official SpaGCN 1.2.7 graph-convolutional domain model using "
    "expression and coordinates (histology disabled because the benchmark bundle "
    "does not contain a registered tissue image); includes Visium hex-grid refinement.",
    "graphst": "Official GraphST graph self-supervised contrastive embedding, followed "
    "by the benchmark's fixed-q full-covariance Gaussian-mixture clustering.",
    "bayesspace": "Official Bioconductor BayesSpace spatialPreprocess + spatialCluster "
    "pipeline with truth-derived q and 10,000 MCMC iterations.",
    "stagate": "Official PyTorch-Geometric STAGATE graph-attention autoencoder with an "
    "adaptive six-neighbour radius, followed by fixed-q Gaussian-mixture clustering.",
    "banksy_py": "Python port of the BANKSY neighbourhood-averaged expression recipe "
    "(k_geom=15, λ=0.8). Deterministic per seed; no external R/Bioconductor "
    "dependency. Runs in seconds because it operates on 2000 HVGs.",
    "harmony_kmeans": "Spatial quadrants (median-split) treated as pseudo-batches, "
    "PCA(30) → harmonypy → KMeans on the corrected embedding. Meant to "
    "probe whether Harmony's batch correction transfers to spatial "
    "compartments.",
    "moran_spectral": "Vectorised Moran's I on a kNN(k=15) row-stochastic graph, top-500 "
    "genes → PCA → SpectralClustering on the symmetrised graph. A "
    "graph-first analogue of the SVG → KMeans recipe.",
    "spatialde_kmeans": "SpatialDE (Svensson 2018) run on a 500-gene dispersion prefilter, "
    "sorted by qval; the top 500 genes drive the standard "
    "log1p+scale → PCA(20) → KMeans pipeline.",
    "nnsvg_kmeans": "Bioconductor nnSVG (Weber 2023) via a local R subprocess reading a "
    "temporary h5ad; same downstream pipeline as SpatialDE. Provides a "
    "more principled SVG ranking (nearest-neighbour Gaussian process).",
    "kmeans": "Standard k-means baseline on a spatial-fused PCA embedding (n_pcs=15, "
    "spatial_weight=0.3, n_init=4). This is the workhorse to beat.",
    "agglomerative": "Ward-linkage hierarchical clustering on the same embedding.",
    "gaussian_mixture": "Full-covariance GMM on the embedding.",
    "birch": "BIRCH — a memory-frugal but competitive baseline on Visium-scale data.",
    "bisecting_kmeans": "Divisive k-means; useful when cluster sizes vary.",
    "minibatch_kmeans": "Mini-batch k-means; the fast large-n baseline.",
    "spectral": "Normalised-cut spectral clustering on kNN affinity.",
    "dbscan": "Density-based; poor on Visium spot geometry — reference case for "
    "failure mode 'everything is one cluster'.",
    "mean_shift": "Bandwidth-picked mean shift; typically over-merges layers.",
    "optics": "OPTICS ordering; a slower but more label-stable DBSCAN cousin.",
}


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def _bold_top3(values: pd.Series) -> list[str]:
    """Return list of str formatted 'x.xxx', top-3 wrapped in bold."""
    finite = values.dropna().sort_values(ascending=False)
    top3 = set(finite.index[:3].tolist())
    return [
        f"**{v:.3f}**"
        if (not pd.isna(v) and idx in top3)
        else ("n/a" if pd.isna(v) else f"{v:.3f}")
        for idx, v in values.items()
    ]


def build(out: Path) -> None:
    mean = pd.read_csv(_HERE / "performance_matrix_mean.csv", index_col=0)
    std = pd.read_csv(_HERE / "performance_matrix_std.csv", index_col=0)
    if (_HERE / "timings_mean.csv").exists():
        tim = pd.read_csv(_HERE / "timings_mean.csv")
    else:
        tim = pd.DataFrame(columns=["dataset", "method", "seconds"])
    bench = json.loads((_HERE / "benchmark.json").read_text())

    # Winners per slice
    winners = []
    for sid in mean.index:
        row = mean.loc[sid].dropna()
        if row.empty:
            winners.append((sid, None, None))
        else:
            winners.append((sid, str(row.idxmax()), float(row.max())))

    # Family mean summary
    def _mask(cols):
        return [c for c in cols if c in mean.columns]

    sk_cols = _mask(SKLEARN)
    sa_cols = _mask(SPATIAL)
    sk_mean = float(np.nanmean(mean[sk_cols].values)) if sk_cols else float("nan")
    sa_mean = float(np.nanmean(mean[sa_cols].values)) if sa_cols else float("nan")

    lines: list[str] = []
    lines.append(f"# {len(mean.index)} × {len(mean.columns)} spatial-aware benchmark — DLPFC")
    lines.append("")
    lines.append(
        f"Protocol: `{bench.get('protocol', '—')}`.  "
        f"5 slices × {len(mean.columns)} methods × "
        f"{len(bench.get('seeds', []))} seeds ({bench.get('seeds', [])})."
    )
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"* Sklearn baselines mean ARI: **{sk_mean:.3f}**")
    lines.append(f"* Spatial-aware methods mean ARI: **{sa_mean:.3f}**")
    delta = sa_mean - sk_mean
    verdict = (
        "spatial-aware methods win overall"
        if delta > 0.02
        else "spatial-aware methods trail slightly"
        if delta < -0.02
        else "spatial-aware methods and sklearn baselines are within noise"
    )
    lines.append(f"* Overall verdict: **{verdict}** (Δ = {delta:+.3f}).")
    lines.append("")
    lines.append("### Best method per slice")
    lines.append("")
    lines.append("| slice | best method | ARI | family |")
    lines.append("| --- | --- | --- | --- |")
    for sid, m, v in winners:
        fam = "spatial" if m in SPATIAL else "sklearn"
        v_s = f"{v:.3f}" if v is not None else "n/a"
        lines.append(f"| `{sid}` | `{m or '—'}` | {v_s} | {fam} |")
    lines.append("")
    lines.append("## ARI table")
    lines.append("")
    lines.append("Numbers are the mean ARI over 3 seeds. Top-3 methods per slice are **bolded**.")
    lines.append("")
    lines.append("| slice | " + " | ".join(mean.columns) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(mean.columns)) + " |")
    for sid in mean.index:
        row = mean.loc[sid]
        cells = _bold_top3(row)
        lines.append(f"| `{sid}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Standard deviations across seeds (same layout):")
    lines.append("")
    lines.append("| slice | " + " | ".join(std.columns) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(std.columns)) + " |")
    for sid in std.index:
        cells = []
        for col in std.columns:
            v = std.loc[sid, col]
            cells.append("n/a" if pd.isna(v) else f"{v:.3f}")
        lines.append(f"| `{sid}` | " + " | ".join(cells) + " |")
    lines.append("")

    # Timing summary — median across slices per method
    if not tim.empty:
        med = tim.groupby("method")["seconds"].median().sort_values()
        lines.append("## Runtime (per-cell median, seconds)")
        lines.append("")
        lines.append("| method | family | median seconds |")
        lines.append("| --- | --- | --- |")
        for m, v in med.items():
            fam = "spatial" if m in SPATIAL else "sklearn"
            lines.append(f"| `{m}` | {fam} | {v:.1f} |")
        lines.append("")

    lines.append("## Per-method commentary")
    lines.append("")
    for m in mean.columns:
        note = METHOD_NOTES.get(m, "—")
        row = mean[m].dropna()
        best_slice = str(row.idxmax()) if not row.empty else "—"
        best_ari = float(row.max()) if not row.empty else float("nan")
        lines.append(f"### `{m}` ({'spatial-aware' if m in SPATIAL else 'sklearn'})")
        lines.append("")
        lines.append(note)
        if not row.empty:
            lines.append(f"Best slice: `{best_slice}` at ARI **{best_ari:.3f}**.")
        lines.append("")

    lines.append("## Heatmap")
    lines.append("")
    lines.append(
        f"![{len(mean.index)}×{len(mean.columns)} ARI heatmap](heatmap_5x19.svg)"
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in bench.get("limitations", []):
        lines.append(f"* {lim}")
    lines.append("")

    out.write_text("\n".join(lines))
    _log(f"[write] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "report_5x19.md"))
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
