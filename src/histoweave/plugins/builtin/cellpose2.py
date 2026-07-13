"""Cellpose 2 image-segmentation adapter."""

from __future__ import annotations

import numpy as np

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


@register
class Cellpose2Segmentation(Method):
    """Segment a registered tissue image with the real Cellpose 2 model."""

    spec = MethodSpec(
        name="cellpose2",
        category=MethodCategory.SEGMENTATION,
        version="0.1.0",
        summary="Cell/nucleus instance segmentation with Cellpose 2 pretrained models.",
        params=(
            ParamSpec("image_key", "str", "image", "Input key in SpatialTable.images."),
            ParamSpec("mask_key", "str", "cellpose_masks", "Output label-image key."),
            ParamSpec(
                "model_type",
                "str",
                "cyto2",
                "Cellpose 2 pretrained model.",
                choices=("cyto", "cyto2", "nuclei"),
            ),
            ParamSpec("gpu", "bool", False, "Use CUDA/MPS when supported by Cellpose."),
            ParamSpec("diameter", "float|None", None, "Expected object diameter.", minimum=1.0),
            ParamSpec("channels", "list", [0, 0], "[cytoplasm, nucleus] channel selection."),
            ParamSpec("channel_axis", "int|None", None, "Image channel axis."),
            ParamSpec("z_axis", "int|None", None, "Image z axis."),
            ParamSpec("do_3d", "bool", False, "Run Cellpose 3D dynamics."),
            ParamSpec("normalize", "bool", True, "Apply Cellpose intensity normalization."),
            ParamSpec("invert", "bool", False, "Invert image intensities."),
            ParamSpec("flow_threshold", "float", 0.4, "Flow-error threshold.", minimum=0.0),
            ParamSpec("cellprob_threshold", "float", 0.0, "Cell probability threshold."),
            ParamSpec("min_size", "int", 15, "Minimum object size in pixels.", minimum=1),
        ),
        assumptions=(
            "SpatialTable.images[image_key] is a numeric 2D/3D image array.",
            "Channel and z-axis parameters match the image layout.",
            "Model weights are available locally or may be downloaded at runtime.",
        ),
        assays=("xenium", "cosmx", "merscope", "visium"),
        maturity=MethodMaturity.BETA,
        wraps="cellpose 2.x CellposeModel",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("cellpose", ">=2,<3", "cellpose2"),),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        try:
            from cellpose import models
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Cellpose 2 is required for segmentation. "
                "Install with: pip install 'histoweave-spatial[cellpose2]'"
            ) from exc

        image_key = self.params["image_key"]
        if image_key not in data.images:
            raise KeyError(f"Cellpose input image {image_key!r} does not exist")
        image = np.asarray(data.images[image_key])
        if image.ndim not in (2, 3, 4):
            raise ValueError(
                f"Cellpose image must be 2D/3D with optional channels, got {image.shape}"
            )
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("Cellpose input image must contain finite numeric values")
        channels = list(self.params["channels"])
        if len(channels) != 2 or not all(isinstance(value, int) for value in channels):
            raise ValueError("Cellpose channels must be a two-integer list")

        result = data.copy()
        model = models.CellposeModel(
            gpu=bool(self.params["gpu"]),
            model_type=self.params["model_type"],
        )
        evaluated = model.eval(
            image,
            diameter=self.params["diameter"],
            channels=channels,
            channel_axis=self.params["channel_axis"],
            z_axis=self.params["z_axis"],
            do_3D=bool(self.params["do_3d"]),
            normalize=bool(self.params["normalize"]),
            invert=bool(self.params["invert"]),
            flow_threshold=float(self.params["flow_threshold"]),
            cellprob_threshold=float(self.params["cellprob_threshold"]),
            min_size=int(self.params["min_size"]),
        )
        if not isinstance(evaluated, tuple) or not evaluated:
            raise RuntimeError("Cellpose returned an unexpected result")
        masks = np.asarray(evaluated[0])
        expected_shape = tuple(
            size for axis, size in enumerate(image.shape) if axis != self.params["channel_axis"]
        )
        if masks.shape != expected_shape:
            raise RuntimeError(
                f"Cellpose mask shape {masks.shape} does not match image plane {expected_shape}"
            )
        if not np.issubdtype(masks.dtype, np.integer) or (masks < 0).any():
            raise RuntimeError("Cellpose masks must be non-negative integer labels")

        mask_key = self.params["mask_key"]
        result.images[mask_key] = masks
        result.uns["segmentation"] = {
            "method": "cellpose2",
            "image_key": image_key,
            "mask_key": mask_key,
            "model_type": self.params["model_type"],
            "n_instances": int(masks.max(initial=0)),
        }
        return self.finalize(result, step="segmentation")
