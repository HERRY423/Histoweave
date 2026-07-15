"""Prepare the Slide-seqV2 mouse hippocampus dataset (Stickels et al. 2021).

Source: squidpy.datasets.slideseqv2() -> scverse example data (41,786 beads x 4,000 genes).
Proxy domain label: ``cluster`` (RCTD/transcriptomic cell-type cluster). X is already
log-normalized, so _prep_common recovers pseudo-counts via expm1 for the counts layer.
"""

from __future__ import annotations

import squidpy as sq
from _prep_common import finalize

if __name__ == "__main__":
    a = sq.datasets.slideseqv2()
    finalize(
        a,
        dataset_id="slideseqv2",
        platform="Slide-seqV2",
        label_col="cluster",
    )
