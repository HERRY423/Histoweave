"""Ingestion layer: vendor/community formats -> canonical SpatialTable.

Real readers delegate to ``spatialdata-io`` (installed via the ``spatial`` extra) and
then convert to the canonical object. Until that stack is wired up, the concrete
readers raise a clear, actionable error rather than pretending to work — the plan is
explicit that ingestion for Visium/HD, Xenium and a sequencing assay is Phase-1 scope.
"""

from __future__ import annotations

import abc

from ..data import SpatialTable


class Reader(abc.ABC):
    """Base class for assay ingestion adapters."""

    #: Assay family identifier, e.g. "visium", "xenium".
    assay: str
    #: Human-readable vendor/platform label.
    platform: str

    @abc.abstractmethod
    def read(self, path: str) -> SpatialTable:
        """Read ``path`` (a vendor output directory/file) into a SpatialTable."""

    def _require_spatialdata_io(self):
        try:
            import spatialdata_io  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                f"Reading {self.platform} data requires spatialdata-io. "
                "Install the extra with: pip install 'histoweave-spatial[spatial]'"
            ) from exc
        return spatialdata_io
