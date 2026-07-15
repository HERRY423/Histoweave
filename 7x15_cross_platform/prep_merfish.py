"""Prepare the MERFISH mouse hypothalamic preoptic dataset (Moffitt et al. 2018).

Source: squidpy.datasets.merfish() -> scverse example data (73,655 cells x 161 genes).
Proxy domain label: ``Cell_class`` (major neuronal / non-neuronal cell classes).
"""

from __future__ import annotations

import squidpy as sq
from _prep_common import finalize

if __name__ == "__main__":
    a = sq.datasets.merfish()
    finalize(
        a,
        dataset_id="merfish",
        platform="MERFISH",
        label_col="Cell_class",
        drop_labels=("Ambiguous",),
    )
