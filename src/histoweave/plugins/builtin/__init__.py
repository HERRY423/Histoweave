"""Built-in reference and wrapped analysis methods."""

from __future__ import annotations

_REGISTERED = False


def register_all() -> None:
    """Import the built-in method modules so their ``@register`` decorators run."""
    global _REGISTERED
    if _REGISTERED:
        return
    from . import (  # noqa: F401
        annotate,
        banksy,
        banksy_py,
        cell2location,
        cellpose2,
        celltypist,
        deconv,
        deep_learning,
        domains,
        extended_native,
        ingestion,
        integration,
        liana_plus,
        nnsvg,
        normalize,
        qc,
        r_demo,
        research_context,
        research_preprocessing,
        research_spatial,
        scanvi,
        sctransform,
        sklearn_clustering,
        sota_domains,
        spatial_graph,
        spatial_svg,
        spatialde,
    )
    from .release_manifest import apply_builtin_release_manifest

    apply_builtin_release_manifest()
    _REGISTERED = True


__all__ = ["register_all"]
