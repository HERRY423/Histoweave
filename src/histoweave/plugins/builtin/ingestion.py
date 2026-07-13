"""Registered ingestion methods wrapping the native assay readers.

Ingestion methods differ from analysis methods in one fundamental way: they
*create* a :class:`~histoweave.data.SpatialTable` from a file-system source
rather than *transforming* an existing one.  The ``data`` argument to
:meth:`run` is therefore ignored; all configuration lives in the method
parameters.

This module fills the ``ingestion`` category (previously empty) and allows
``histoweave list-methods --category ingestion`` to report available readers.
"""

from __future__ import annotations

from ...data import SpatialTable
from ...io import read as io_read
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register

# Method-name → io assay key (the name prefix isn't always the assay slug).
_ASSAY_MAP: dict[str, str] = {
    "visium_reader": "visium",
    "xenium_reader": "xenium",
    "stereoseq_reader": "stereo_seq",
    "merfish_reader": "merfish",
    "cosmx_reader": "cosmx",
    "merscope_reader": "merscope",
    "slideseq_reader": "slideseq",
}


class _IngestionMethod(Method):
    """Base for methods that create a table from a file-system source.

    Subclasses only need to set ``spec``.
    """

    def run(self, data: SpatialTable) -> SpatialTable:
        """Ingest from the path declared in ``self.params`` (ignores *data*)."""
        path = str(self.params["path"])
        engine = str(self.params["engine"])
        assay = _ASSAY_MAP[self.spec.name]
        table = io_read(assay, path, engine=engine)
        return self.finalize(table, step="ingestion")


# ---------------------------------------------------------------------------
# Visium
# ---------------------------------------------------------------------------
@register
class VisiumIngestion(_IngestionMethod):
    """Ingest 10x Visium / Visium HD Space Ranger output."""

    spec = MethodSpec(
        name="visium_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest 10x Visium Space Ranger output directory.",
        params=(
            ParamSpec("path", "str", "", "Space Ranger output directory."),
            ParamSpec(
                "engine",
                "str",
                "native",
                "Reader engine: 'native' (lightweight) or 'spatialdata' (full).",
                choices=("native", "spatialdata"),
            ),
        ),
        assays=("visium",),
    )


# ---------------------------------------------------------------------------
# Xenium
# ---------------------------------------------------------------------------
@register
class XeniumIngestion(_IngestionMethod):
    """Ingest 10x Xenium output (in-situ / onboard analysis)."""

    spec = MethodSpec(
        name="xenium_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest 10x Xenium output directory.",
        params=(
            ParamSpec("path", "str", "", "Xenium output directory."),
            ParamSpec(
                "engine",
                "str",
                "native",
                "Reader engine: 'native' (lightweight) or 'spatialdata' (full).",
                choices=("native", "spatialdata"),
            ),
            ParamSpec(
                "gene_expression_only",
                "bool",
                True,
                "Keep only 'Gene Expression' features (native engine).",
            ),
        ),
        assays=("xenium",),
    )


# ---------------------------------------------------------------------------
# Stereo-seq
# ---------------------------------------------------------------------------
@register
class StereoSeqIngestion(_IngestionMethod):
    """Ingest Stereo-seq GEF output (via spatialdata-io bridge)."""

    spec = MethodSpec(
        name="stereoseq_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest Stereo-seq output (requires spatialdata-io).",
        params=(
            ParamSpec("path", "str", "", "Stereo-seq output directory."),
            ParamSpec(
                "engine",
                "str",
                "spatialdata",
                "Reader engine (spatialdata-io required for .gef format).",
                choices=("spatialdata",),
            ),
        ),
        assays=("stereo_seq",),
        assumptions=("spatialdata-io installed (extra: spatial).",),
    )


# ---------------------------------------------------------------------------
# MERFISH (Vizgen MERSCOPE platform via spatialdata-io)
# ---------------------------------------------------------------------------
@register
class MerfishIngestion(_IngestionMethod):
    """Ingest Vizgen MERFISH output (cell_by_gene + cell_metadata CSV)."""

    spec = MethodSpec(
        name="merfish_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest Vizgen MERFISH output directory.",
        params=(
            ParamSpec("path", "str", "", "MERFISH output directory."),
            ParamSpec(
                "engine",
                "str",
                "spatialdata",
                "Reader engine (spatialdata-io).",
                choices=("spatialdata",),
            ),
        ),
        assays=("merfish",),
        assumptions=("spatialdata-io installed (extra: spatial).",),
    )


# ---------------------------------------------------------------------------
# CosMx (NanoString CosMx SMI via spatialdata-io)
# ---------------------------------------------------------------------------
@register
class CosMxIngestion(_IngestionMethod):
    """Ingest NanoString CosMx SMI output."""

    spec = MethodSpec(
        name="cosmx_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest NanoString CosMx SMI output.",
        params=(
            ParamSpec("path", "str", "", "CosMx output directory."),
            ParamSpec(
                "engine",
                "str",
                "spatialdata",
                "Reader engine (spatialdata-io).",
                choices=("spatialdata",),
            ),
        ),
        assays=("cosmx",),
        assumptions=("spatialdata-io installed (extra: spatial).",),
    )


# ---------------------------------------------------------------------------
# MERSCOPE (Vizgen MERSCOPE via spatialdata-io)
# ---------------------------------------------------------------------------
@register
class MerscopeIngestion(_IngestionMethod):
    """Ingest Vizgen MERSCOPE output."""

    spec = MethodSpec(
        name="merscope_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest Vizgen MERSCOPE output.",
        params=(
            ParamSpec("path", "str", "", "MERSCOPE output directory."),
            ParamSpec(
                "engine",
                "str",
                "spatialdata",
                "Reader engine (spatialdata-io).",
                choices=("spatialdata",),
            ),
        ),
        assays=("merscope",),
        assumptions=("spatialdata-io installed (extra: spatial).",),
    )


# ---------------------------------------------------------------------------
# Slide-seq (Rodriques et al. 2019 via spatialdata-io)
# ---------------------------------------------------------------------------
@register
class SlideSeqIngestion(_IngestionMethod):
    """Ingest Slide-seq V2 puck data."""

    spec = MethodSpec(
        name="slideseq_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest Slide-seq V2 puck data.",
        params=(
            ParamSpec("path", "str", "", "Slide-seq output directory."),
            ParamSpec(
                "engine",
                "str",
                "spatialdata",
                "Reader engine (spatialdata-io).",
                choices=("spatialdata",),
            ),
        ),
        assays=("slideseq",),
        assumptions=("spatialdata-io installed (extra: spatial).",),
    )
