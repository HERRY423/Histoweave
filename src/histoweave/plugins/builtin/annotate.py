"""Built-in marker-based cell/domain annotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..._math import zscore
from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register
from ._markers import resolve_markers


@register
class MarkerScoreAnnotation(Method):
    """Assign each observation the label whose marker set it scores highest on.

    A simple, transparent stand-in for reference-mapping methods (scANVI / scArches).
    Marker sets come from the ``marker_genes`` parameter or ``uns['marker_genes']``.
    """

    spec = MethodSpec(
        name="marker_score",
        category=MethodCategory.ANNOTATION,
        version="0.1.0",
        summary="Mean z-scored marker expression -> argmax label.",
        params=(
            ParamSpec("marker_genes", "dict|None", None, "label -> [genes]; else uns."),
            ParamSpec("key_added", "str", "cell_type", "obs column for the result."),
            ParamSpec("score_key", "str", "annotation_score", "obs column for the top score."),
        ),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        markers = self.params["marker_genes"] or data.uns.get("marker_genes")
        if not markers:
            raise ValueError("No marker_genes given and uns['marker_genes'] is absent")

        resolved = resolve_markers(data, markers)
        Z = zscore(data.X)  # z-score per gene so sets are comparable
        labels = resolved.labels
        score_cols = [Z[:, idx].mean(axis=1) for idx in resolved.indices]
        score_matrix = np.vstack(score_cols).T  # cells x labels

        best = score_matrix.argmax(axis=1)
        data.obs[self.params["key_added"]] = pd.Categorical([labels[i] for i in best])
        data.obs[self.params["score_key"]] = score_matrix[np.arange(data.n_obs), best]
        data.uns["annotation_marker_resolution"] = resolved.diagnostics()
        return self.finalize(data, step="annotation")
