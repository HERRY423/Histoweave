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
        backends=(BackendRequirement("sctransform", "R package", "sctransform", runtime="r"),),
    )
    r_script = "/usr/local/bin/histoweave-sctransform.R"

    def _validate_input(self, data: SpatialTable) -> None:
        layer = self.params["layer"]
        if layer is not None and layer not in data.layers:
            raise KeyError(f"SCTransform count layer {layer!r} does not exist")
        matrix = data.X if layer is None else data.layers[layer]
        values = (
            matrix.data
            if hasattr(matrix, "data") and not isinstance(matrix, np.ndarray)
            else matrix
        )
        values = np.asarray(values)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("SCTransform input counts must be finite and non-negative")

    def _build_r_args(self, data: SpatialTable) -> list[str]:
        layer = "" if self.params["layer"] is None else self.params["layer"]
        return [
            f"layer={layer}",
            f"vst_flavor={self.params['vst_flavor']}",
            f"residual_type={self.params['residual_type']}",
            f"min_cells={self.params['min_cells']}",
        ]

    def _validate_r_output(self, data: SpatialTable) -> None:
        if not data.uns.get("sctransform_normalized", False):
            raise RuntimeError("SCTransform output is missing completion metadata")
        values = (
            data.X.data
            if hasattr(data.X, "data") and not isinstance(data.X, np.ndarray)
            else data.X
        )
        if not np.isfinite(np.asarray(values)).all():
            raise RuntimeError("SCTransform returned non-finite residuals")
