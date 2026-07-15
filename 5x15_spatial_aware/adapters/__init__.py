"""Task-2 spatial-aware adapters.

Each adapter exposes a `run(...) -> np.ndarray[int]` label function callable
from :mod:`experiment_5x15`.
"""

from . import (  # noqa: F401
    banksy_py_adapter,
    harmony_adapter,
    moran_adapter,
    nnsvg_adapter,
    spatialde_adapter,
)
