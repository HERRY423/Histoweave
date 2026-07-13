"""SpatialDE adapter using :meth:`Method._run_via_anndata`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ...data import SpatialTable
from ..interfaces import (
    BackendRequirement,
    Method,
    MethodCategory,
    MethodImplementation,
    MethodSpec,
    ParamSpec,
)
from ..registry import register

if TYPE_CHECKING:
    from anndata import AnnData


@register
class SpatialDESVG(Method):
    """Identify spatially variable genes with the original SpatialDE model."""

    spec = MethodSpec(
        name="spatialde",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        version="0.1.0",
        summary="SpatialDE Gaussian-process test for spatially variable genes.",
        params=(
            ParamSpec("layer", "str|None", None, "Expression layer; None uses X."),
            ParamSpec("n_top", "int", 50, "Genes retained in the ranked SVG summary.", minimum=1),
            ParamSpec(
                "qval_threshold",
                "float",
                0.05,
                "FDR threshold used for the spatialde_significant flag.",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        assumptions=(
            "obsm['spatial'] contains at least x/y coordinates.",
            "Non-negative counts or normalized expression are supplied.",
        ),
        assays=("*",),
        wraps="SpatialDE (Svensson et al., 2018)",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(
            BackendRequirement("SpatialDE", "==1.1.3", "spatialde"),
            BackendRequirement("NaiveDE", "required by SpatialDE", "spatialde"),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        return self._run_via_anndata(data)

    def run_on_anndata(self, adata: AnnData) -> AnnData:  # type: ignore[valid-type]
        try:
            import NaiveDE
            import SpatialDE
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SpatialDE and NaiveDE are required for the spatialde method. "
                "Install with: pip install 'histoweave-spatial[spatialde]'"
            ) from exc

        result = adata.copy()
        if "spatial" not in result.obsm:
            raise ValueError("obsm['spatial'] is required for SpatialDE")
        coords_array = np.asarray(result.obsm["spatial"])
        if coords_array.ndim != 2 or coords_array.shape[1] < 2:
            raise ValueError("obsm['spatial'] must have at least two coordinate columns")

        layer = self.params["layer"]
        if layer is not None and layer not in result.layers:
            raise KeyError(f"SpatialDE expression layer {layer!r} does not exist")
        matrix = result.X if layer is None else result.layers[layer]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        expression = np.asarray(matrix, dtype=float)
        if not np.isfinite(expression).all():
            raise ValueError("SpatialDE input expression contains non-finite values")
        if (expression < 0).any():
            raise ValueError("SpatialDE input expression must be non-negative")

        obs_names = pd.Index(result.obs_names.astype(str))
        var_names = pd.Index(result.var_names.astype(str))
        counts = pd.DataFrame(expression, index=obs_names, columns=var_names)
        coordinates = pd.DataFrame(coords_array[:, :2], index=obs_names, columns=["x", "y"])
        sample_info = coordinates.copy()
        sample_info["total_counts"] = counts.sum(axis=1).clip(lower=1.0)

        stabilized = NaiveDE.stabilize(counts.T).T
        residual = NaiveDE.regress_out(sample_info, stabilized.T, "np.log(total_counts)").T
        de_results = SpatialDE.run(coordinates, residual)
        ranked = _index_spatialde_results(de_results, var_names)

        column_map = {
            "FSV": "spatialde_fsv",
            "pval": "spatialde_pval",
            "qval": "spatialde_qval",
            "l": "spatialde_length_scale",
        }
        for source, target in column_map.items():
            values = ranked[source] if source in ranked else np.nan
            result.var[target] = pd.Series(values, index=var_names).reindex(var_names).to_numpy()

        qvals = pd.to_numeric(result.var["spatialde_qval"], errors="coerce")
        result.var["spatialde_significant"] = (
            (qvals <= float(self.params["qval_threshold"])).fillna(False).to_numpy()
        )

        ordered = ranked.copy()
        if "qval" in ordered:
            columns = ["qval", "FSV"] if "FSV" in ordered else ["qval"]
            ascending = [True, False] if "FSV" in ordered else [True]
            ordered = ordered.sort_values(columns, ascending=ascending, na_position="last")
        elif "FSV" in ordered:
            ordered = ordered.sort_values("FSV", ascending=False, na_position="last")
        ordered = ordered.head(min(int(self.params["n_top"]), len(ordered)))
        result.uns["svg"] = {
            "method": "spatialde",
            "top_genes": [_svg_record(gene, row) for gene, row in ordered.iterrows()],
        }
        return result


def _index_spatialde_results(de_results: Any, var_names: pd.Index) -> pd.DataFrame:
    """Normalize SpatialDE's version-dependent gene column/index convention."""
    ranked = pd.DataFrame(de_results).copy()
    gene_column = next((name for name in ("g", "gene") if name in ranked), None)
    if gene_column is not None:
        ranked.index = ranked.pop(gene_column).astype(str)
    else:
        ranked.index = ranked.index.astype(str)
    if ranked.index.has_duplicates:
        raise RuntimeError("SpatialDE returned duplicate gene identifiers")
    if not ranked.index.isin(var_names).any() and len(ranked) == len(var_names):
        ranked.index = var_names
    return ranked.reindex(var_names)


def _svg_record(gene: Any, row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {"gene": str(gene)}
    for source, target in (("FSV", "fsv"), ("pval", "pval"), ("qval", "qval"), ("l", "l")):
        if source in row and pd.notna(row[source]):
            record[target] = float(row[source])
    return record
