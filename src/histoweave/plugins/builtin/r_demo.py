"""Containerized R-method reference plugin.

This is the **spike** that proves the Python ↔ R bridge works: the plugin
writes the input to ``.h5ad`` (the interchange format), shells out to an R
script, and reads the transformed ``.h5ad`` back.  The shared machinery lives
in :class:`RContainerMethod` so that adding a new R/Bioconductor method only
requires writing ``_build_r_args`` — ~10 lines of differential logic instead
of ~120.

In production the R script lives inside a container image and is called by the
Nextflow process; for local testing ``Rscript`` on the host PATH is sufficient.
"""

from __future__ import annotations

from ...data import SpatialTable
from ..interfaces import MethodCategory, MethodSpec, ParamSpec
from ..registry import register
from ._r_base import RContainerMethod

_R_SCRIPT_NAME = "histoweave-sc-transform.R"


@register
class RLogNormalize(RContainerMethod):
    """R-side log-normalisation — a containerization reference implementation.

    Requires either:

    * ``anndata`` + ``Rscript`` on the host PATH (dev / CI), or
    * a container image with the R script installed at the expected path
      (production, via Nextflow).
    """

    spec = MethodSpec(
        name="r_lognorm",
        category=MethodCategory.NORMALIZATION,
        version="0.1.0",
        summary="R-side library-size log1p normalisation (container spike).",
        params=(
            ParamSpec("target_sum", "float", 1e4, "Counts per cell after scaling."),
        ),
        assumptions=("anndata installed (extra: spatial); R + anndata R package.",),
        wraps="R::base",
        language="container",
    )

    r_script = "/usr/local/bin/" + _R_SCRIPT_NAME

    # -- differential logic (10 lines vs the old 116) ------------------------

    def _build_r_args(self, data: SpatialTable) -> list[str]:
        """Forward user parameters as R script CLI arguments."""
        return [f"target_sum={self.params['target_sum']}"]

    # -- script discovery (override for dev-mode source-tree fallback) -------

    def _find_r_script(self):
        """Container path first; source-tree second (for dev / CI)."""
        from pathlib import Path

        container_path = Path(self.r_script)
        if container_path.exists():
            return container_path
        source_path = (
            Path(__file__).resolve().parents[4]
            / "workflows"
            / "containers"
            / "histoweave-r"
            / _R_SCRIPT_NAME
        )
        if source_path.exists():
            return source_path
        raise FileNotFoundError(
            f"Cannot find {_R_SCRIPT_NAME} — "
            f"expected at {container_path} or {source_path}"
        )
