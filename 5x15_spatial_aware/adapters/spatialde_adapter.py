"""SpatialDE (Svensson 2018) → top-N → PCA → KMeans pipeline.

Ranks genes by SpatialDE q-value / LLR, then delegates clustering to
:mod:`svg_domain_pipeline`. Requires the ``SpatialDE`` Python package (patched
in-place for scipy>=1.12; the sandbox provisioning script wraps
``scipy.misc.derivative`` with a central-difference shim so the import succeeds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from svg_domain_pipeline import cluster_from_svg_ranking  # noqa: E402


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def _rank_by_spatialde(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    n_candidates: int = 500,
) -> list[str]:
    """Return a SpatialDE-ranked gene list (best first).

    SpatialDE fits ~10 kernel models per gene and scales roughly linearly in
    n_candidates × n_cells. On Visium slices (~3-4k spots) pre-filtering to
    the top-500 most-dispersed genes keeps a single call under ~4 min and
    matches the downstream ``n_top=500`` used by the KMeans stage.
    """
    import SpatialDE  # patched at install time

    X = _to_dense(X_counts).astype(float)
    var = X.var(axis=0)
    # rank by dispersion, keep top n_candidates non-constant genes
    keep_idx = np.argsort(-var)[:n_candidates]
    keep_idx = np.asarray([j for j in keep_idx if var[j] > 0], dtype=int)
    Xk = X[:, keep_idx]
    gnames = [str(gene_names[j]) for j in keep_idx]
    # log1p(cp10k) — SpatialDE expects positive real-valued expression
    sums = Xk.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    Xk = np.log1p(Xk / sums * 1e4)

    counts_df = pd.DataFrame(Xk, columns=gnames)
    coord_df = pd.DataFrame(np.asarray(spatial), columns=["x", "y"])

    results = SpatialDE.run(coord_df, counts_df)
    # SpatialDE returns per-gene LL and q-values; lower qval / higher LLR = better
    results = results.sort_values("qval", ascending=True)
    return results["g"].astype(str).tolist()


def run(
    X_counts,
    spatial: np.ndarray,
    gene_names,
    seed: int,
    n_domains: int,
    n_top: int = 500,
) -> np.ndarray:
    ranked = _rank_by_spatialde(X_counts, spatial, gene_names)
    return cluster_from_svg_ranking(
        X_counts,
        ranked_genes=ranked,
        all_genes=list(gene_names),
        n_domains=n_domains,
        seed=seed,
        n_top=n_top,
    )
