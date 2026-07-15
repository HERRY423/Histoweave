"""Build a compact LLM-facing catalog from the live plugin registry."""

from __future__ import annotations

from typing import Any

from ..plugins import list_methods


def build_catalog(*, assay: str | None = None) -> list[dict[str, Any]]:
    """Return every active method and its executable parameter contract."""
    catalog = []
    for method in list_methods(assay=assay):
        if method.get("deprecated"):
            continue
        catalog.append(
            {
                "category": method["category"],
                "name": method["name"],
                "version": method["version"],
                "summary": method["summary"],
                "assays": method["assays"],
                "maturity": method["maturity"],
                "assumptions": method["assumptions"],
                "params": method["params"],
            }
        )
    return catalog
