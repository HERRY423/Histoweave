"""scANVI semi-supervised cell-type annotation adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...data import SpatialTable
from ..interfaces import (
    Method,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register

if TYPE_CHECKING:
    from anndata import AnnData


@register
class SCANVIAnnotation(Method):
    """Train SCVI then scANVI on partially labelled observations."""

    spec = MethodSpec(
        name="scanvi",
        category=MethodCategory.ANNOTATION,
        version="0.1.0",
        summary="Semi-supervised cell annotation using scVI-tools scANVI.",
        params=(
            ParamSpec("labels_key", "str", "cell_type_seed", "Partial labels in obs."),
            ParamSpec("unlabeled_category", "str", "Unknown", "Unlabelled category value."),
            ParamSpec("batch_key", "str|None", None, "Optional batch column."),
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec("n_latent", "int", 30, "Latent dimensions.", minimum=2),
            ParamSpec("n_layers", "int", 2, "Encoder/decoder layers.", minimum=1),
            ParamSpec("scvi_epochs", "int", 200, "Unsupervised pretraining epochs.", minimum=1),
            ParamSpec("scanvi_epochs", "int", 100, "Semi-supervised epochs.", minimum=1),
            ParamSpec("accelerator", "str", "auto", "Lightning accelerator."),
            ParamSpec("devices", "str|int", "auto", "Lightning devices selection."),
            ParamSpec("seed", "int", 0, "scvi-tools random seed.", minimum=0),
            ParamSpec("key_added", "str", "cell_type", "Predicted label column."),
            ParamSpec("confidence_key", "str", "scanvi_confidence", "Confidence column."),
            ParamSpec("probability_key", "str", "scanvi_probabilities", "obsm probability key."),
            ParamSpec("latent_key", "str", "X_scanvi", "obsm latent embedding key."),
        ),
        assumptions=(
            "Selected layer contains raw non-negative counts.",
            "obs[labels_key] contains labelled cells and the unlabeled category.",
            "Labelled and unlabelled observations share the same feature space.",
        ),
        assays=("xenium", "cosmx", "merscope", "visium"),
        maturity=MethodMaturity.BETA,
        wraps="scvi.model.SCANVI",
        language="python",
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        return self._run_via_anndata(data)

    def run_on_anndata(self, adata: AnnData) -> AnnData:  # type: ignore[valid-type]
        try:
            import scvi
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scvi-tools is required for scANVI. "
                "Install with: pip install 'histoweave-spatial[scanvi]'"
            ) from exc

        result = adata.copy()
        labels_key = self.params["labels_key"]
        if labels_key not in result.obs:
            raise KeyError(f"scANVI labels column {labels_key!r} does not exist")
        labels = result.obs[labels_key].astype(str)
        unlabeled = self.params["unlabeled_category"]
        if unlabeled not in set(labels):
            raise ValueError(f"scANVI labels must include unlabeled category {unlabeled!r}")
        if not (labels != unlabeled).any():
            raise ValueError("scANVI requires at least one labelled observation")
        batch_key = self.params["batch_key"]
        if batch_key is not None and batch_key not in result.obs:
            raise KeyError(f"scANVI batch column {batch_key!r} does not exist")
        layer = self.params["layer"]
        if layer is not None and layer not in result.layers:
            raise KeyError(f"scANVI count layer {layer!r} does not exist")

        scvi.settings.seed = int(self.params["seed"])
        scvi.model.SCVI.setup_anndata(
            result,
            layer=layer,
            batch_key=batch_key,
            labels_key=labels_key,
        )
        vae = scvi.model.SCVI(
            result,
            n_latent=int(self.params["n_latent"]),
            n_layers=int(self.params["n_layers"]),
        )
        trainer = {
            "accelerator": self.params["accelerator"],
            "devices": self.params["devices"],
        }
        vae.train(max_epochs=int(self.params["scvi_epochs"]), **trainer)
        model = scvi.model.SCANVI.from_scvi_model(
            vae,
            unlabeled_category=unlabeled,
        )
        model.train(max_epochs=int(self.params["scanvi_epochs"]), **trainer)

        predictions = model.predict(adata=result, soft=False)
        probabilities_value = model.predict(adata=result, soft=True)
        probabilities = (
            probabilities_value.to_numpy()
            if hasattr(probabilities_value, "to_numpy")
            else np.asarray(probabilities_value)
        )
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != result.n_obs:
            raise RuntimeError("scANVI returned an invalid probability matrix")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("scANVI returned non-finite probabilities")

        predicted = np.asarray(predictions).reshape(-1).astype(str)
        if predicted.shape[0] != result.n_obs:
            raise RuntimeError("scANVI returned the wrong number of labels")
        if hasattr(probabilities_value, "columns"):
            classes = [str(value) for value in probabilities_value.columns]
        else:
            classes = sorted(set(labels) - {unlabeled})
            if len(classes) != probabilities.shape[1]:
                classes = [f"class_{index}" for index in range(probabilities.shape[1])]

        latent = np.asarray(model.get_latent_representation(adata=result))
        if latent.ndim != 2 or latent.shape[0] != result.n_obs:
            raise RuntimeError("scANVI returned an invalid latent representation")
        result.obs[self.params["key_added"]] = pd.Categorical(predicted)
        result.obs[self.params["confidence_key"]] = probabilities.max(axis=1)
        result.obsm[self.params["probability_key"]] = probabilities
        result.obsm[self.params["latent_key"]] = latent
        result.uns["annotation"] = {
            "method": "scanvi",
            "classes": classes,
            "probability_key": self.params["probability_key"],
            "latent_key": self.params["latent_key"],
        }
        return result
