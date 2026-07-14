"""nnSVG spatially-variable-gene detection through the shared R container bridge.

nnSVG (Weber et al., 2023) ranks spatially variable genes by fitting a
nearest-neighbour Gaussian process (via BRISC) to each gene's expression as a
function of spatial location, and scoring genes by a likelihood-ratio test
against a non-spatial null. Unlike the built-in Moran's I / Geary's C statistics,
nnSVG models the length-scale of spatial variation per gene and returns a
calibrated, multiple-testing-corrected ranking.

The heavy R/Bioconductor computation runs inside the ``histoweave-r`` container
image; this plugin only supplies the parameter marshalling and output contract,
reusing :class:`RContainerMethod` for the h5ad round-trip and provenance.
"""

from __future__ import annotations

from ...data import SpatialTable
from ..interfaces import MethodCategory, MethodSpec, ParamSpec
from ..registry import register
from ._r_base import RContainerMethod


@register
class NNSVG(RContainerMethod):
    """Bioconductor nnSVG spatially variable gene detection.

    Writes three per-gene columns into ``var`` on completion:

    * ``nnsvg_rank`` — integer rank (1 = most spatially variable),
    * ``nnsvg_LR_stat`` — likelihood-ratio statistic,
    * ``nnsvg_padj`` — Benjamini–Hochberg adjusted p-value.

    The top ``n_top`` genes by rank are recorded in ``uns['nnsvg_top_genes']`` so
    the benchmark harness and report can consume them the same way they consume
    the native SVG methods.
    """

    spec = MethodSpec(
        name="nnsvg",
        category=MethodCategory.SPATIALLY_VARIABLE_GENES,
        version="0.1.0",
        summary="nnSVG nearest-neighbour Gaussian-process SVG detection (Bioconductor).",
        params=(
            ParamSpec(
                "n_top", "int", 50, "Number of top-ranked SVGs to flag in uns.", minimum=1
            ),
            ParamSpec(
                "n_neighbors", "int", 10,
                "Nearest neighbours for the BRISC GP approximation.", minimum=2,
            ),
            ParamSpec(
                "order", "str", "AMMD",
                "BRISC ordering scheme for the nearest-neighbour GP.",
                choices=("AMMD", "Sum_coords"),
            ),
            ParamSpec(
                "n_threads", "int", 1, "Threads for the per-gene GP fits.", minimum=1
            ),
            ParamSpec(
                "assay_name", "str", "logcounts",
                "SpatialExperiment assay nnSVG reads (log-normalised recommended).",
            ),
            ParamSpec("seed", "int", 0, "Random seed for reproducible fits.", minimum=0),
        ),
        assumptions=(
            "obsm['spatial'] contains two-dimensional coordinates.",
            "X (or the named assay) holds log-normalised expression; nnSVG assumes "
            "a continuous, roughly Gaussian response.",
            "The histoweave-r image contains Bioconductor nnSVG.",
        ),
        assays=("visium", "xenium", "cosmx", "merscope", "slideseq", "stereoseq"),
        wraps="Bioconductor::nnSVG",
        language="container",
    )
    r_script = "/usr/local/bin/histoweave-nnsvg.R"

    def _validate_input(self, data: SpatialTable) -> None:
        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for nnSVG SVG detection")
        if data.spatial.shape[1] != 2:
            raise ValueError("nnSVG requires exactly two-dimensional spatial coordinates")

    def _build_r_args(self, data: SpatialTable) -> list[str]:
        return [
            f"n_top={self.params['n_top']}",
            f"n_neighbors={self.params['n_neighbors']}",
            f"order={self.params['order']}",
            f"n_threads={self.params['n_threads']}",
            f"assay_name={self.params['assay_name']}",
            f"seed={self.params['seed']}",
        ]

    def _validate_r_output(self, data: SpatialTable) -> None:
        missing = [c for c in ("nnsvg_rank", "nnsvg_LR_stat", "nnsvg_padj") if c not in data.var]
        if missing:
            raise RuntimeError(f"nnSVG output is missing var columns: {missing}")
