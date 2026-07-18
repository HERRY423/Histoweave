"""cell2location spatial deconvolution adapter.

The adapter consumes a gene-by-cell-type reference signature matrix from
``adata.uns[reference_key]`` and runs the real ``cell2location.models.Cell2location``
model. It never substitutes marker scoring when the optional backend is absent.
"""

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
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register
from ._validation import validate_count_matrix

if TYPE_CHECKING:
    from anndata import AnnData


@register
class Cell2LocationDeconvolution(Method):
    """Estimate absolute cell-type abundance with cell2location."""

    spec = MethodSpec(
        name="cell2location",
        category=MethodCategory.DECONVOLUTION,
        version="0.1.0",
        summary="Bayesian spatial cell-type abundance estimation with cell2location.",
        params=(
            ParamSpec(
                "reference_key",
                "str",
                "cell2location_reference",
                "uns key containing a genes x cell-types reference DataFrame.",
            ),
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec("batch_key", "str|None", None, "Optional spatial batch column."),
            ParamSpec(
                "min_shared_genes",
                "int",
                2,
                "Minimum genes shared with the reference signature matrix.",
                minimum=2,
            ),
            ParamSpec(
                "n_cells_per_location",
                "float",
                8.0,
                "Expected cells per capture location; tissue-specific.",
                minimum=0.1,
            ),
            ParamSpec(
                "detection_alpha",
                "float",
                20.0,
                "Prior controlling location-specific detection sensitivity.",
                minimum=0.1,
            ),
            ParamSpec("max_epochs", "int", 30000, "Training epochs.", minimum=1),
            ParamSpec("batch_size", "int|None", None, "Training minibatch size.", minimum=1),
            ParamSpec("posterior_batch_size", "int", 2500, "Posterior batch size.", minimum=1),
            ParamSpec(
                "posterior_quantile",
                "str",
                "q05",
                "Posterior summary exported as abundance.",
                choices=("q05", "q50", "q95"),
            ),
            ParamSpec("use_gpu", "bool", False, "Use a CUDA device when available."),
            ParamSpec("abundance_key", "str", "cell_abundance", "obsm abundance key."),
            ParamSpec("proportion_key", "str", "proportions", "obsm normalized key."),
        ),
        assumptions=(
            "Raw non-negative integer-like counts are available in the selected layer.",
            "uns[reference_key] is a gene-by-cell-type expression signature matrix.",
            "n_cells_per_location is estimated from assay geometry or histology.",
        ),
        assays=("visium", "slide_seq", "stereo_seq"),
        maturity=MethodMaturity.BETA,
        wraps="cell2location.models.Cell2location",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("cell2location", ">=0.1.4", "cell2location"),),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        self._reference_from_uns(data.uns, self.params["reference_key"])
        return self._run_via_anndata(data)

    @staticmethod
    def _reference_from_uns(uns: dict[str, Any], reference_key: str) -> pd.DataFrame:
        if reference_key not in uns:
            raise KeyError(
                f"cell2location reference {reference_key!r} is missing from uns; "
                "provide a genes x cell-types signature DataFrame"
            )
        reference = pd.DataFrame(uns[reference_key]).copy()
        if reference.empty or reference.columns.empty:
            raise ValueError("cell2location reference signature matrix is empty")
        try:
            values = reference.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("cell2location reference signatures must be numeric") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("cell2location reference signatures must be finite and non-negative")
        if (values.sum(axis=0) <= 0).any():
            raise ValueError("every cell type in the cell2location reference must be expressed")
        reference.iloc[:, :] = values
        return reference

    def run_on_anndata(self, adata: AnnData) -> AnnData:
        try:
            from cell2location.models import Cell2location
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "cell2location is required for this method. "
                "Install with: pip install 'histoweave-spatial[cell2location]'"
            ) from exc

        result = adata.copy()
        layer = self.params["layer"]
        if layer is not None and layer not in result.layers:
            raise KeyError(f"cell2location count layer {layer!r} does not exist")
        count_matrix = result.X if layer is None else result.layers[layer]
        validate_count_matrix(count_matrix, method="cell2location")
        batch_key = self.params["batch_key"]
        if batch_key is not None and batch_key not in result.obs:
            raise KeyError(f"cell2location batch column {batch_key!r} does not exist")

        reference_key = self.params["reference_key"]
        reference = self._reference_from_uns(dict(result.uns), reference_key)
        reference.index = reference.index.astype(str)
        reference.columns = reference.columns.astype(str)
        if reference.index.has_duplicates or reference.columns.has_duplicates:
            raise ValueError("cell2location reference genes and cell types must be unique")

        var_names = pd.Index(result.var_names.astype(str))
        shared = var_names[var_names.isin(reference.index)]
        minimum_shared = int(self.params["min_shared_genes"])
        if len(shared) < minimum_shared:
            raise ValueError(
                "cell2location has too few shared genes between data and reference: "
                f"{len(shared)} < min_shared_genes={minimum_shared}"
            )
        model_adata = result[:, shared].copy()
        reference = reference.loc[shared]

        Cell2location.setup_anndata(
            adata=model_adata,
            layer=layer,
            batch_key=batch_key,
        )
        model = Cell2location(
            model_adata,
            cell_state_df=reference,
            N_cells_per_location=float(self.params["n_cells_per_location"]),
            detection_alpha=float(self.params["detection_alpha"]),
        )
        model.train(
            max_epochs=int(self.params["max_epochs"]),
            batch_size=self.params["batch_size"],
            train_size=1,
            accelerator="gpu" if bool(self.params["use_gpu"]) else "cpu",
        )

        quantile = self.params["posterior_quantile"]
        exported = model.export_posterior(
            model_adata,
            use_quantiles=True,
            add_to_obsm=[quantile],
            sample_kwargs={
                "batch_size": int(self.params["posterior_batch_size"]),
                "use_gpu": bool(self.params["use_gpu"]),
            },
        )
        exported = model_adata if exported is None else exported
        posterior_key = f"{quantile}_cell_abundance_w_sf"
        if posterior_key not in exported.obsm:
            raise RuntimeError(f"cell2location posterior is missing obsm[{posterior_key!r}]")
        abundance_value = exported.obsm[posterior_key]
        cell_types = [str(value) for value in reference.columns]
        if isinstance(abundance_value, pd.DataFrame):
            abundance_frame = abundance_value.copy()
            abundance_frame.index = abundance_frame.index.astype(str)
            expected_obs = pd.Index(result.obs_names.astype(str))
            if (
                abundance_frame.index.has_duplicates
                or not expected_obs.isin(abundance_frame.index).all()
            ):
                raise RuntimeError(
                    "cell2location abundance rows do not match the input observations"
                )
            abundance_frame = abundance_frame.reindex(expected_obs)
            posterior_cell_types = [str(value) for value in abundance_frame.columns]
            if set(posterior_cell_types) != set(cell_types):
                raise RuntimeError(
                    "cell2location abundance columns do not match reference cell types"
                )
            abundance = abundance_frame.loc[:, cell_types].to_numpy()
        else:
            abundance = np.asarray(abundance_value)
        abundance = np.asarray(abundance, dtype=float)
        if abundance.shape != (result.n_obs, reference.shape[1]):
            raise RuntimeError(
                "cell2location abundance shape does not match observations/cell types: "
                f"{abundance.shape} vs {(result.n_obs, reference.shape[1])}"
            )
        if not np.isfinite(abundance).all() or (abundance < 0).any():
            raise RuntimeError("cell2location returned invalid abundance values")

        denominator = abundance.sum(axis=1, keepdims=True)
        proportions = np.divide(
            abundance,
            denominator,
            out=np.zeros_like(abundance),
            where=denominator > 0,
        )
        result.obsm[self.params["abundance_key"]] = abundance
        result.obsm[self.params["proportion_key"]] = proportions
        result.uns["deconvolution"] = {
            "method": "cell2location",
            "cell_types": cell_types,
            "abundance_key": self.params["abundance_key"],
            "proportion_key": self.params["proportion_key"],
            "posterior_summary": quantile,
            "shared_genes": int(len(shared)),
            "zero_abundance_locations": int(np.count_nonzero(denominator.ravel() == 0)),
        }
        return result
