"""Built-in deconvolution method (marker-gene mean ratio).

A deliberately simple reference implementation — the equivalent of a "mean-marker"
baseline — that every real deconvolution method (RCTD, cell2location, SPOTlight,
RCTD, stereoscope) must outperform. It estimates per-spot cell-type proportions
as the normalised mean z-scored expression of each type's marker genes.
"""

from __future__ import annotations

import numpy as np

from ..._math import zscore
from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register
from ._markers import resolve_markers


@register
class MarkerDeconvolution(Method):
    """Baseline deconvolution: marker-gene mean → softmax proportions.

    For each cell type and each spot, compute the mean z-scored expression across
    that type's marker genes, then normalise across types (softmax-style) to get
    per-spot proportions. Writes the result to ``obsm['proportions']``.
    """

    spec = MethodSpec(
        name="marker_deconv",
        category=MethodCategory.DECONVOLUTION,
        version="0.1.0",
        summary="Marker-gene mean-ratio baseline deconvolution.",
        params=(
            ParamSpec(
                "marker_genes", "dict|None", None, "label -> [genes]; else uns['marker_genes']."
            ),
            ParamSpec(
                "proportion_key", "str", "proportions", "obsm key for the estimated proportions."
            ),
        ),
        assumptions=("Normalized X (log1p-transformed).",),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        markers = self.params["marker_genes"] or data.uns.get("marker_genes")
        if not markers:
            raise ValueError("No marker_genes parameter and uns['marker_genes'] is absent")

        resolved = resolve_markers(data, markers)
        cell_types = resolved.labels
        n_ct = len(cell_types)
        Z = zscore(data.X)

        # Per-spot × per-cell-type score matrix.
        scores: np.ndarray = np.zeros((data.n_obs, n_ct), dtype=float)
        for j, idx in enumerate(resolved.indices):
            scores[:, j] = Z[:, idx].mean(axis=1)

        # Softmax normalisation to proportions.
        scores -= scores.max(axis=1, keepdims=True)  # numerical stability
        exp_scores = np.exp(scores)
        proportions = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        data.obsm[self.params["proportion_key"]] = proportions
        data.uns["deconvolution"] = {
            "cell_types": cell_types,
            "marker_resolution": resolved.diagnostics(),
        }
        return self.finalize(data, step="deconvolution")
