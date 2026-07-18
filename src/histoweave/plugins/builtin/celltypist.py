"""CellTypist pretrained-model annotation adapter."""

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

if TYPE_CHECKING:
    from anndata import AnnData


@register
class CellTypistAnnotation(Method):
    """Annotate observations with a named or local CellTypist model."""

    spec = MethodSpec(
        name="celltypist",
        category=MethodCategory.ANNOTATION,
        version="0.1.0",
        summary="Pretrained CellTypist logistic-regression cell annotation.",
        params=(
            ParamSpec(
                "model",
                "str",
                "Immune_All_Low.pkl",
                "Built-in model name or path to a custom CellTypist model.",
            ),
            ParamSpec("majority_voting", "bool", True, "Refine labels by over-clusters."),
            ParamSpec(
                "mode",
                "str",
                "best match",
                "CellTypist classification mode.",
                choices=("best match", "prob match"),
            ),
            ParamSpec(
                "p_thres",
                "float",
                0.5,
                "Probability threshold in prob-match mode.",
                minimum=0.0,
                maximum=1.0,
            ),
            ParamSpec(
                "min_prop",
                "float",
                0.0,
                "Minimum dominant-cell proportion for majority voting.",
                minimum=0.0,
                maximum=1.0,
            ),
            ParamSpec("key_added", "str", "cell_type", "Predicted label column."),
            ParamSpec("confidence_key", "str", "celltypist_confidence", "Confidence column."),
            ParamSpec("probability_key", "str", "celltypist_probabilities", "obsm score key."),
        ),
        assumptions=(
            "X is log1p-normalized to 10,000 counts per observation.",
            "Gene symbols match the organism and namespace used by the selected model.",
            "Model downloads require network access unless a local model path is used.",
        ),
        assays=("xenium", "cosmx", "merscope", "visium"),
        maturity=MethodMaturity.BETA,
        wraps="celltypist.annotate",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("celltypist", ">=1.7", "celltypist"),),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        return self._run_via_anndata(data)

    def run_on_anndata(self, adata: AnnData) -> AnnData:
        try:
            import celltypist
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CellTypist is required for this method. "
                "Install with: pip install 'histoweave-spatial[celltypist]'"
            ) from exc

        result = adata.copy()
        annotations = celltypist.annotate(
            result,
            model=self.params["model"],
            majority_voting=bool(self.params["majority_voting"]),
            mode=self.params["mode"],
            p_thres=float(self.params["p_thres"]),
            min_prop=float(self.params["min_prop"]),
        )
        labels = pd.DataFrame(annotations.predicted_labels).copy()
        if labels.index.is_unique:
            labels = labels.reindex(result.obs_names)
        preferred = (
            "majority_voting"
            if self.params["majority_voting"] and "majority_voting" in labels
            else "predicted_labels"
        )
        if preferred not in labels:
            raise RuntimeError("CellTypist result is missing predicted labels")
        predicted = labels[preferred].astype(str)
        if predicted.isna().any() or len(predicted) != result.n_obs:
            raise RuntimeError("CellTypist returned incomplete predicted labels")

        probability_value = annotations.probability_matrix
        probability = (
            probability_value.to_numpy()
            if hasattr(probability_value, "to_numpy")
            else np.asarray(probability_value)
        )
        probability = np.asarray(probability, dtype=float)
        if probability.ndim != 2 or probability.shape[0] != result.n_obs:
            raise RuntimeError("CellTypist returned an invalid probability matrix")
        if hasattr(probability_value, "columns"):
            classes = [str(value) for value in probability_value.columns]
        else:
            classes = [f"class_{index}" for index in range(probability.shape[1])]

        if "conf_score" in labels:
            confidence = pd.to_numeric(labels["conf_score"], errors="coerce").to_numpy()
        else:
            confidence = probability.max(axis=1)
        result.obs[self.params["key_added"]] = pd.Categorical(predicted.to_numpy())
        result.obs[self.params["confidence_key"]] = confidence
        result.obsm[self.params["probability_key"]] = probability
        result.uns["annotation"] = {
            "method": "celltypist",
            "model": self.params["model"],
            "classes": classes,
            "probability_key": self.params["probability_key"],
            "majority_voting": bool(self.params["majority_voting"]),
        }
        return result
