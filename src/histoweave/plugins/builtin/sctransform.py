"""SCTransform normalization through the shared R container bridge."""

from __future__ import annotations

import numpy as np

from ...data import SpatialTable
from ..interfaces import (
    BackendRequirement,
    MethodCategory,
    MethodImplementation,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register
from ._r_base import RContainerMethod
from ._validation import validate_count_matrix


@register
class SCTransformNormalization(RContainerMethod):
    """Variance-stabilize UMI counts with ``sctransform::vst``."""

    spec = MethodSpec(
        name="sctransform",
        category=MethodCategory.NORMALIZATION,
        version="0.1.0",
        summary="SCTransform regularized negative-binomial variance stabilization.",
        params=(
            ParamSpec("layer", "str|None", None, "Count layer; None uses X."),
            ParamSpec(
                "vst_flavor",
                "str",
                "v2",
                "SCTransform regularization flavor.",
                choices=("v1", "v2"),
            ),
            ParamSpec(
                "residual_type",
                "str",
                "pearson",
                "Residual type returned in X.",
                choices=("pearson", "deviance"),
            ),
            ParamSpec("min_cells", "int", 5, "Minimum expressing cells per gene.", minimum=1),
            ParamSpec(
                "n_cells",
                "int|None",
                None,
                "Optional number of observations sampled for model fitting.",
                minimum=2,
            ),
        ),
        assumptions=(
            "Selected matrix contains raw non-negative UMI counts.",
            "Residual output is dense and requires memory proportional to observations x genes.",
            "The histoweave-r image contains sctransform and its compiled dependencies.",
        ),
        assays=("visium", "xenium", "cosmx", "merscope", "stereo_seq"),
        maturity=MethodMaturity.BETA,
        wraps="sctransform::vst",
        language="container",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("sctransform", ">=0.4.0", runtime="r"),),
    )
    r_script = "/usr/local/bin/histoweave-sctransform.R"

    def _validate_input(self, data: SpatialTable) -> None:
        layer = self.params["layer"]
        if layer is not None and layer not in data.layers:
            raise KeyError(f"SCTransform count layer {layer!r} does not exist")
        matrix = data.X if layer is None else data.layers[layer]
        validate_count_matrix(matrix, method="SCTransform")
        if data.n_obs < 2 or data.n_vars < 2:
            raise ValueError("SCTransform requires at least two observations and two genes")
        if int(self.params["min_cells"]) > data.n_obs:
            raise ValueError(
                f"SCTransform min_cells={self.params['min_cells']} exceeds n_obs={data.n_obs}"
            )
        n_cells = self.params["n_cells"]
        if n_cells is not None and int(n_cells) > data.n_obs:
            raise ValueError(f"SCTransform n_cells={n_cells} exceeds n_obs={data.n_obs}")

    def _build_r_args(self, data: SpatialTable) -> list[str]:
        layer = "" if self.params["layer"] is None else self.params["layer"]
        return [
            f"layer={layer}",
            f"vst_flavor={self.params['vst_flavor']}",
            f"residual_type={self.params['residual_type']}",
            f"min_cells={self.params['min_cells']}",
            f"n_cells={'' if self.params['n_cells'] is None else self.params['n_cells']}",
        ]

    def _validate_r_output(self, data: SpatialTable) -> None:
        if not data.uns.get("sctransform_normalized", False):
            raise RuntimeError("SCTransform output is missing completion metadata")
        if "counts" not in data.layers:
            raise RuntimeError("SCTransform output did not preserve raw counts")
        if "sctransform_modeled" not in data.var:
            raise RuntimeError("SCTransform output is missing per-gene modeled flags")
        if not data.var["sctransform_modeled"].astype(bool).any():
            raise RuntimeError("SCTransform did not model any genes")
        values = (
            data.X.data
            if hasattr(data.X, "data") and not isinstance(data.X, np.ndarray)
            else data.X
        )
        if not np.isfinite(np.asarray(values)).all():
            raise RuntimeError("SCTransform returned non-finite residuals")
