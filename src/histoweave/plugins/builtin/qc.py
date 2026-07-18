"""Built-in quality-control method (assay-aware, MAD-based filtering)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register


@register
class BasicQC(Method):
    """Compute per-cell QC metrics and filter low-quality observations.

    Mirrors the scverse best practice referenced in the plan: median-absolute-deviation
    (MAD) based outlier filtering on total counts and detected genes, plus a hard cap
    on mitochondrial fraction. Metrics are written to ``obs`` so downstream reporting
    and benchmarking can use them.
    """

    spec = MethodSpec(
        name="basic_qc",
        category=MethodCategory.QC,
        version="0.1.0",
        summary="Per-cell QC metrics + MAD-based outlier filtering.",
        params=(
            ParamSpec("n_mads", "float", 5.0, "MAD multiplier for outlier bounds.", minimum=0),
            ParamSpec("min_counts", "int", 10, "Absolute floor on total counts.", minimum=0),
            ParamSpec("min_genes", "int", 3, "Absolute floor on detected genes.", minimum=0),
            ParamSpec(
                "max_mito_pct",
                "float",
                40.0,
                "Max % counts in mito genes.",
                minimum=0,
                maximum=100,
            ),
            ParamSpec("mito_prefix", "str", "MT-", "Var-name prefix marking mito genes."),
        ),
        assumptions=("Counts-like X (non-negative).",),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        X = data.X
        total_counts = np.asarray(X.sum(axis=1)).ravel()
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()

        if "mito" in data.var:
            mito_mask = data.var["mito"].to_numpy(dtype=bool)
        else:
            # Match the prefix against gene symbols. Real readers index var by a stable
            # feature id (e.g. ENSEMBL) and keep the symbol in a column, so fall back to
            # var_names only when no symbol column is present.
            prefix = self.params["mito_prefix"]
            names: Any = data.var_names
            for col in ("feature_name", "gene_name", "symbol", "symbols"):
                if col in data.var.columns:
                    names = data.var[col].to_numpy()
                    break
            mito_mask = np.array([str(v).startswith(prefix) for v in names])
        mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_mito = np.where(total_counts > 0, mito_counts / total_counts * 100, 0.0)

        data.obs["total_counts"] = total_counts
        data.obs["n_genes_by_counts"] = n_genes
        data.obs["pct_counts_mito"] = pct_mito

        keep = (
            self._mad_ok(np.log1p(total_counts))
            & self._mad_ok(np.log1p(n_genes))
            & (total_counts >= self.params["min_counts"])
            & (n_genes >= self.params["min_genes"])
            & (pct_mito <= self.params["max_mito_pct"])
        )

        data.uns["qc"] = {
            "n_obs_before": int(data.n_obs),
            "n_obs_after": int(keep.sum()),
            "n_removed": int((~keep).sum()),
            "median_counts": float(np.median(total_counts)),
            "median_genes": float(np.median(n_genes)),
            "median_pct_mito": float(np.median(pct_mito)),
        }
        filtered = data.subset_obs(keep)
        return self.finalize(filtered)

    def _mad_ok(self, values: np.ndarray) -> np.ndarray:
        """Boolean mask of values within ``n_mads`` MADs of the median."""
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.ones_like(values, dtype=bool)
        # 1.4826 scales MAD to be a consistent estimator of the std for normal data.
        bound = self.params["n_mads"] * mad * 1.4826
        return np.abs(values - median) <= bound
