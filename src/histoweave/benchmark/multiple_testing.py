"""Multiple-testing correction and false-discovery-rate control.

Provides the classical procedures used in genomics / method-comparison
reviews without pulling in statsmodels:

* Benjamini–Hochberg (BH) step-up FDR
* Benjamini–Yekutieli (BY) FDR under arbitrary dependence
* Holm–Bonferroni family-wise error rate (FWER) control
* Bonferroni FWER

All functions are pure NumPy and operate on 1-D p-value arrays (NaNs
preserved).  They are shared by:

* method-ranking pairwise tests (:mod:`histoweave.benchmark.stats_review`)
* gene-level SVG statistics (e.g. Moran's I p-values)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

FdrMethod = Literal["bh", "by", "holm", "bonferroni"]


def fdr_adjust(
    p_values: np.ndarray | list[float],
    *,
    method: FdrMethod = "bh",
    alpha: float = 0.05,
) -> np.ndarray:
    """Return adjusted p-values (q-values for BH/BY) for *p_values*.

    Parameters
    ----------
    p_values
        One-dimensional array of raw p-values in ``[0, 1]`` (NaN allowed).
    method
        ``"bh"`` (default), ``"by"``, ``"holm"``, or ``"bonferroni"``.
    alpha
        Used only for documentation of the decision rule; the returned
        vector is the full set of adjusted p-values, independent of alpha.
        Callers reject nulls with ``q <= alpha``.
    """
    del alpha  # decision threshold is caller-side
    p = np.asarray(p_values, dtype=float).ravel()
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    if np.any((p[finite] < 0) | (p[finite] > 1)):
        raise ValueError("p_values must lie in [0, 1]")

    m = int(finite.sum())
    order = np.argsort(p[finite], kind="mergesort")
    ranked = p[finite][order]
    adj = _adjust_sorted(ranked, m=m, method=method)

    # Undo the sort and enforce monotonicity already handled in helpers.
    restored = np.empty(m, dtype=float)
    restored[order] = adj
    out[finite] = np.clip(restored, 0.0, 1.0)
    return out


def reject_nulls(
    p_values: np.ndarray | list[float],
    *,
    method: FdrMethod = "bh",
    alpha: float = 0.05,
) -> np.ndarray:
    """Boolean mask of rejected null hypotheses at level *alpha*."""
    q = fdr_adjust(p_values, method=method, alpha=alpha)
    return np.isfinite(q) & (q <= float(alpha))


def pairwise_fdr_table(
    p_matrix: np.ndarray,
    method_names: list[str],
    *,
    method: FdrMethod = "bh",
    alpha: float = 0.05,
) -> dict[str, object]:
    """Flatten an upper-triangle pairwise p-value matrix, adjust, and reassemble.

    Returns a JSON-serialisable dict with:
    * ``pairs``: list of {method_a, method_b, p_raw, p_adj, significant}
    * ``n_tests``, ``n_significant``, ``method``, ``alpha``
    """
    names = list(method_names)
    n = len(names)
    mat = np.asarray(p_matrix, dtype=float)
    if mat.shape != (n, n):
        raise ValueError(f"p_matrix shape {mat.shape} != ({n}, {n})")

    pairs_idx: list[tuple[int, int]] = []
    raw: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_idx.append((i, j))
            raw.append(float(mat[i, j]))

    adj = fdr_adjust(raw, method=method, alpha=alpha)
    pairs: list[dict[str, object]] = []
    n_sig = 0
    for (i, j), p_raw, p_adj in zip(pairs_idx, raw, adj, strict=True):
        sig = bool(np.isfinite(p_adj) and p_adj <= alpha)
        n_sig += int(sig)
        pairs.append(
            {
                "method_a": names[i],
                "method_b": names[j],
                "p_raw": None if not np.isfinite(p_raw) else float(p_raw),
                "p_adj": None if not np.isfinite(p_adj) else float(p_adj),
                "significant": sig,
            }
        )
    return {
        "pairs": pairs,
        "n_tests": len(pairs),
        "n_significant": n_sig,
        "method": method,
        "alpha": float(alpha),
    }


def _adjust_sorted(ranked: np.ndarray, *, m: int, method: FdrMethod) -> np.ndarray:
    if method == "bonferroni":
        return np.minimum(ranked * m, 1.0)

    if method == "holm":
        # Step-down: p_(i) * (m - i + 1), then enforce non-decreasing from left.
        scales = np.arange(m, 0, -1, dtype=float)
        adj = ranked * scales
        # Monotone from the left (increasing).
        for i in range(1, m):
            adj[i] = max(adj[i], adj[i - 1])
        return np.minimum(adj, 1.0)

    if method == "bh":
        # Step-up: p_(i) * m / i, then enforce non-increasing from the right.
        ranks = np.arange(1, m + 1, dtype=float)
        adj = ranked * m / ranks
        for i in range(m - 2, -1, -1):
            adj[i] = min(adj[i], adj[i + 1])
        return np.minimum(adj, 1.0)

    if method == "by":
        # BY = BH with harmonic factor c(m) = sum_{k=1}^m 1/k.
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
        ranks = np.arange(1, m + 1, dtype=float)
        adj = ranked * m * c_m / ranks
        for i in range(m - 2, -1, -1):
            adj[i] = min(adj[i], adj[i + 1])
        return np.minimum(adj, 1.0)

    raise ValueError(f"unknown multiple-testing method: {method!r}")
