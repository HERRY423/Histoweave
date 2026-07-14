"""scANVI semi-supervised cell-type annotation adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("scvi-tools", ">=1.3", "scanvi"),),
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
        if result.obs[labels_key].isna().any():
            raise ValueError("scANVI labels must not contain missing values")
        labels = result.obs[labels_key].astype(str)
        unlabeled = self.params["unlabeled_category"]
        if unlabeled not in set(labels):
            raise ValueError(f"scANVI labels must include unlabeled category {unlabeled!r}")
        labelled_classes = sorted(set(labels) - {unlabeled})
        if len(labelled_classes) < 2:
            raise ValueError("scANVI requires at least two labelled cell-type classes")
        result.obs[labels_key] = pd.Categorical(labels)
        batch_key = self.params["batch_key"]
        if batch_key is not None and batch_key not in result.obs:
            raise KeyError(f"scANVI batch column {batch_key!r} does not exist")
        layer = self.params["layer"]
        if layer is not None and layer not in result.layers:
            raise KeyError(f"scANVI count layer {layer!r} does not exist")
        count_matrix = result.X if layer is None else result.layers[layer]
        validate_count_matrix(count_matrix, method="scANVI")

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
        expected_obs = pd.Index(result.obs_names.astype(str))
        if isinstance(probabilities_value, pd.DataFrame):
            probability_frame = probabilities_value.copy()
            probability_frame.index = probability_frame.index.astype(str)
            if probability_frame.index.has_duplicates or probability_frame.columns.has_duplicates:
                raise RuntimeError("scANVI returned duplicate probability labels")
            if not expected_obs.isin(probability_frame.index).all():
                raise RuntimeError("scANVI probability rows do not match input observations")
            probability_frame = probability_frame.reindex(expected_obs)
            probabilities = probability_frame.to_numpy()
        else:
            probabilities = np.asarray(probabilities_value)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != result.n_obs:
            raise RuntimeError("scANVI returned an invalid probability matrix")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("scANVI returned non-finite probabilities")
        if (probabilities < -1e-6).any() or (probabilities > 1 + 1e-6).any():
            raise RuntimeError("scANVI probabilities must lie between zero and one")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4):
            raise RuntimeError("scANVI probability rows must sum to one")

        if isinstance(predictions, pd.Series):
            predicted_series = predictions.copy()
            predicted_series.index = predicted_series.index.astype(str)
            if (
                predicted_series.index.has_duplicates
                or not expected_obs.isin(predicted_series.index).all()
            ):
                raise RuntimeError("scANVI prediction rows do not match input observations")
            predicted = predicted_series.reindex(expected_obs).to_numpy().astype(str)
        else:
            predicted = np.asarray(predictions).reshape(-1).astype(str)
        if predicted.shape[0] != result.n_obs:
            raise RuntimeError("scANVI returned the wrong number of labels")
        if hasattr(probabilities_value, "columns"):
            classes = [str(value) for value in probabilities_value.columns]
        else:
            classes = labelled_classes
            if len(classes) != probabilities.shape[1]:
                classes = [f"class_{index}" for index in range(probabilities.shape[1])]

        latent = np.asarray(model.get_latent_representation(adata=result), dtype=float)
        if latent.ndim != 2 or latent.shape[0] != result.n_obs:
            raise RuntimeError("scANVI returned an invalid latent representation")
        if not np.isfinite(latent).all():
            raise RuntimeError("scANVI returned a non-finite latent representation")
        result.obs[self.params["key_added"]] = pd.Categorical(predicted)
        result.obs[self.params["confidence_key"]] = probabilities.max(axis=1)
        result.obsm[self.params["probability_key"]] = probabilities
        result.obsm[self.params["latent_key"]] = latent
        result.uns["annotation"] = {
            "method": "scanvi",
            "classes": classes,
            "probability_key": self.params["probability_key"],
            "latent_key": self.params["latent_key"],
            "labels_key": labels_key,
            "unlabeled_category": unlabeled,
        }
        return result
