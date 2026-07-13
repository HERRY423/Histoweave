"""Built-in normalization method (library-size + log1p)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register


@register
class LogNormalize(Method):
    """Total-count normalize to ``target_sum`` then natural-log1p transform.

    The plan lists analytic Pearson residuals and scVI-family integration as methods to
    wrap; this simple, well-understood default is the reference implementation. Before
    overwriting ``X`` with the log-normalized matrix, the raw counts are stashed in
    ``layers['counts']`` so later steps (e.g. SVG detection that prefers counts) can
    recover them. Layers are shape-aligned to ``X``, so the preserved counts follow the
    observations through any later subsetting instead of drifting out of alignment.
    """

    spec = MethodSpec(
        name="log1p_cp10k",
        category=MethodCategory.NORMALIZATION,
        version="0.1.0",
        summary="Library-size normalization (CP10k) + log1p.",
        params=(
            ParamSpec(
                "target_sum", "float", 1e4, "Counts per cell after scaling.", minimum=1e-12
            ),
        ),
        assumptions=("Counts-like X (non-negative).",),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        # `counts` is whatever X currently holds: the raw count matrix on a first run,
        # but the *already-normalized* values if this method is applied twice. The name
        # reflects the intended/first-run case; the setdefault below is what keeps that
        # distinction from mattering.
        counts = data.X.astype(float)
        # Stash the pre-normalization matrix under layers['counts'] — but only if it is
        # not already there. setdefault makes the *first* run authoritative: a re-run
        # will not clobber the true raw counts with the normalized `counts` it sees.
        data.layers.setdefault("counts", counts.copy())
        try:
            from scipy.sparse import issparse
        except ModuleNotFoundError:  # pragma: no cover - sparse X requires SciPy
            sparse_counts = False
        else:
            sparse_counts = issparse(counts)

        if sparse_counts:
            matrix = cast(Any, counts)
            totals = np.asarray(matrix.sum(axis=1)).ravel()
            totals[totals == 0] = 1.0
            normed = matrix.multiply((self.params["target_sum"] / totals)[:, None]).tocsr()
            normed.data = np.log1p(normed.data)
            data.X = normed
        else:
            totals = counts.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0
            data.X = np.log1p(counts / totals * self.params["target_sum"])
        data.uns["normalization"] = {
            "method": self.spec.name,
            "target_sum": self.params["target_sum"],
            "counts_layer": "counts",
        }
        return self.finalize(data)
