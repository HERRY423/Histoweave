"""LIANA+ consensus ligand-receptor inference adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ...data import SpatialTable
from ..interfaces import (
    BackendRequirement,
    Method,
    MethodCategory,
    MethodImplementation,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register

if TYPE_CHECKING:
    from anndata import AnnData


@register
class LianaPlusCCC(Method):
    """Run LIANA+'s robust rank aggregation over real LR methods."""

    spec = MethodSpec(
        name="liana_plus",
        category=MethodCategory.CELL_CELL_COMMUNICATION,
        version="0.1.0",
        summary="LIANA+ consensus ligand-receptor inference with optional spatial weighting.",
        params=(
            ParamSpec("groupby", "str", "cell_type", "obs column defining cell identities."),
            ParamSpec("resource_name", "str", "consensus", "LIANA resource name."),
            ParamSpec(
                "expr_prop",
                "float",
                0.1,
                "Minimum expressing-cell proportion.",
                minimum=0.0,
                maximum=1.0,
            ),
            ParamSpec("min_cells", "int", 5, "Minimum cells per identity.", minimum=1),
            ParamSpec("n_perms", "int|None", 1000, "Permutation count; None disables.", minimum=1),
            ParamSpec("seed", "int", 1337, "Permutation seed.", minimum=0),
            ParamSpec("n_jobs", "int", 1, "Parallel workers.", minimum=1),
            ParamSpec("use_raw", "bool", False, "Use AnnData.raw instead of X/layer."),
            ParamSpec("layer", "str|None", None, "Expression layer when use_raw=False."),
            ParamSpec(
                "spatial_weighted",
                "bool",
                True,
                "Pass obsm['spatial'] to LIANA+'s spatial-aware scoring.",
            ),
            ParamSpec("key_added", "str", "liana_res", "uns key for the result table."),
            ParamSpec("verbose", "bool", False, "Enable LIANA progress output."),
        ),
        assumptions=(
            "obs[groupby] contains biologically meaningful cell identities.",
            "Expression is natural-log normalized unless use_raw/layer specifies otherwise.",
            "Gene identifiers match the selected LIANA ligand-receptor resource.",
        ),
        assays=("xenium", "cosmx", "merscope", "visium"),
        maturity=MethodMaturity.BETA,
        wraps="LIANA+ rank_aggregate",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("liana", ">=1.2", "liana"),),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        return self._run_via_anndata(data)

    def run_on_anndata(self, adata: AnnData) -> AnnData:  # type: ignore[valid-type]
        try:
            import liana as li
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LIANA+ is required for this method. "
                "Install with: pip install 'histoweave-spatial[liana]'"
            ) from exc

        result = adata.copy()
        groupby = self.params["groupby"]
        if groupby not in result.obs:
            raise KeyError(f"LIANA+ groupby column {groupby!r} does not exist")
        if result.obs[groupby].isna().any():
            raise ValueError(f"LIANA+ groupby column {groupby!r} contains missing values")
        if result.obs[groupby].nunique() < 2:
            raise ValueError("LIANA+ requires at least two cell identities")
        layer = self.params["layer"]
        if layer is not None and layer not in result.layers:
            raise KeyError(f"LIANA+ expression layer {layer!r} does not exist")
        spatial_key = None
        if self.params["spatial_weighted"]:
            if "spatial" not in result.obsm:
                raise ValueError("LIANA+ spatial weighting requires obsm['spatial']")
            spatial_key = "spatial"

        key_added = self.params["key_added"]
        li.mt.rank_aggregate(
            result,
            groupby=groupby,
            resource_name=self.params["resource_name"],
            expr_prop=float(self.params["expr_prop"]),
            min_cells=int(self.params["min_cells"]),
            n_perms=self.params["n_perms"],
            seed=int(self.params["seed"]),
            n_jobs=int(self.params["n_jobs"]),
            use_raw=bool(self.params["use_raw"]),
            layer=layer,
            spatial_key=spatial_key,
            key_added=key_added,
            inplace=True,
            verbose=bool(self.params["verbose"]),
        )
        if key_added not in result.uns:
            raise RuntimeError(f"LIANA+ did not create uns[{key_added!r}]")
        table = pd.DataFrame(result.uns[key_added])
        required = {"source", "target", "ligand_complex", "receptor_complex"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise RuntimeError(f"LIANA+ result is missing required columns: {missing}")
        result.uns[key_added] = table
        result.uns["ccc"] = {
            "method": "liana_plus",
            "result_key": key_added,
            "n_interactions": int(len(table)),
            "spatial_weighted": bool(self.params["spatial_weighted"]),
        }
        return result
