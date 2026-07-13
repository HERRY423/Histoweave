"""Spatially variable gene (SVG) detection via Moran's I.

Moran's I is the classical spatial autocorrelation statistic: it measures whether
a gene's expression pattern is more spatially clustered than expected by chance.
Genes with high positive Moran's I are "spatially variable" — they exhibit
expression hotspots that follow tissue architecture rather than random scatter.

This is the first stage that was entirely unimplemented in the Phase-0 scaffold.
The implementation uses a spatial k-NN graph (via ``scipy.spatial``) and computes
Moran's I per gene, returning a ranked list and writing per-gene statistics into
``var`` so the benchmarking harness and report can consume them.
"""

from __future__ import annotations

import numpy as np

from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register


@register
class MoransISVG(Method):
    """Per-gene spatial autocorrelation via Moran's I on a k-NN graph.

    For each gene, Moran's I is computed as::

        I = (N / W) * (sum_i sum_j w_ij (x_i - x̄)(x_j - x̄)) / (sum_i (x_i - x̄)²)

    where ``w_ij`` = 1 if *j* is among the *k* nearest spatial neighbours of *i*
    (or vice versa for symmetry), and 0 otherwise.  ``W`` is the sum of all weights.
    """

    spec = MethodSpec(
        name="morans_i",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        version="0.1.0",
        summary="Per-gene Moran's I spatial autocorrelation on a k-NN graph.",
        params=(
            ParamSpec("k", "int", 6, "Number of spatial neighbours per spot/cell."),
            ParamSpec("n_top", "int", 50, "Number of top SVG genes to flag in uns."),
            ParamSpec("key_added", "str", "morans_i", "var column for the statistic."),
        ),
        assumptions=("obsm['spatial'] present.", "Normalised expression recommended."),
        wraps="scipy.spatial + manual Moran's I",
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        coords = data.spatial
        if coords is None:
            raise ValueError("obsm['spatial'] is required for Moran's I SVG detection")

        k = int(min(self.params["k"], data.n_obs - 1))
        W, W_sum = _build_spatial_weight_matrix(coords, k)

        X = np.asarray(data.X, dtype=float)
        Xc = X - X.mean(axis=0, keepdims=True)  # centre per gene

        # Moran's I per gene: I_g = (N/W_sum) * (Xc[:,g]^T @ W @ Xc[:,g]) / sum(Xc[:,g]^2)
        # Xc.T @ W has shape (n_vars, n_obs); element-wise * Xc.T then sum over obs.
        numerator = ((Xc.T @ W) * Xc.T).sum(axis=1)  # (n_vars,)
        denominator = (Xc**2).sum(axis=0) + 1e-12
        morans_i = (data.n_obs / W_sum) * numerator / denominator

        data.var[self.params["key_added"]] = morans_i

        # Flag the top n SVG genes in uns for the report.
        top_k = min(self.params["n_top"], data.n_vars)
        top_idx = np.argsort(morans_i)[::-1][:top_k]
        data.uns["svg"] = {
            "method": "morans_i",
            "top_genes": [
                {"gene": str(data.var_names[i]), "morans_i": float(morans_i[i])}
                for i in top_idx
            ],
        }
        return self.finalize(data, step="svg")


def _build_spatial_weight_matrix(
    coords: np.ndarray, k: int
) -> tuple[np.ndarray, float]:
    """Build a symmetric, unweighted k-NN adjacency matrix from spatial coordinates.

    Returns ``(W, W_sum)`` where ``W`` is a sparse CSR matrix.  The graph is made
    symmetric (undirected) by setting ``W_ij = 1`` if *either* i is among j's
    neighbours *or* j is among i's neighbours.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import KDTree

    n = coords.shape[0]
    k = int(min(k, n - 1))
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # k+1 because self is included

    # Build sparse symmetric adjacency: rows = source, cols = target
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()  # skip self (col 0)

    data_arr = np.ones(len(rows), dtype=np.float64)
    W_directed = csr_matrix((data_arr, (rows, cols)), shape=(n, n))

    # Make symmetric: W = max(W, W^T) element-wise, which is W + W^T - W * W^T
    # Simpler: W_sym = (W + W^T).sign()  (clip to 1 after addition)
    W_sym = (W_directed + W_directed.T).sign()  # binary, symmetric
    W_sum = float(W_sym.sum())
    if W_sum == 0:
        W_sum = 1.0
    return W_sym, W_sum
