"""Shared marker-gene namespace resolution for reference methods."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ...data import SpatialTable

_SYMBOL_COLUMNS = ("feature_name", "gene_symbol", "symbol")


@dataclass(frozen=True)
class ResolvedMarkers:
    """Validated marker indices plus diagnostics suitable for ``uns``."""

    labels: list[str]
    indices: list[list[int]]
    matched: dict[str, list[str]]
    unmatched: dict[str, list[str]]
    namespaces: list[str]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "unmatched": self.unmatched,
            "namespaces": self.namespaces,
        }


def resolve_markers(
    data: SpatialTable,
    markers: Mapping[str, Iterable[str]],
) -> ResolvedMarkers:
    """Resolve stable feature IDs and common gene-symbol columns without silent loss.

    Native 10x ingestion intentionally indexes ``var`` by stable feature ID while
    retaining symbols in ``var['feature_name']``.  Reference annotation methods must
    therefore search both namespaces.  Every label is required to match at least one
    feature; otherwise argmax/softmax would emit plausible but scientifically invalid
    labels or proportions.
    """

    if not isinstance(markers, Mapping) or not markers:
        raise TypeError("marker_genes must be a non-empty mapping of label -> genes")

    aliases: dict[str, int] = {}
    folded: dict[str, set[int]] = {}
    namespaces = ["var.index"]

    def add_alias(value: Any, position: int) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none"}:
            return
        aliases.setdefault(text, position)
        folded.setdefault(text.casefold(), set()).add(position)

    for position, name in enumerate(data.var_names):
        add_alias(name, position)

    for column in _SYMBOL_COLUMNS:
        if column not in data.var:
            continue
        namespaces.append(f"var[{column!r}]")
        for position, name in enumerate(data.var[column].tolist()):
            add_alias(name, position)

    labels: list[str] = []
    all_indices: list[list[int]] = []
    matched: dict[str, list[str]] = {}
    unmatched: dict[str, list[str]] = {}

    for raw_label, raw_genes in markers.items():
        label = str(raw_label)
        if isinstance(raw_genes, str) or not isinstance(raw_genes, Iterable):
            raise TypeError(f"marker_genes[{label!r}] must be an iterable of gene names")

        positions: list[int] = []
        seen: set[int] = set()
        matched_names: list[str] = []
        unmatched_names: list[str] = []
        for raw_gene in raw_genes:
            gene = str(raw_gene).strip()
            resolved_position: int | None = aliases.get(gene)
            if resolved_position is None:
                candidates = folded.get(gene.casefold(), set())
                resolved_position = next(iter(candidates)) if len(candidates) == 1 else None
            if resolved_position is None:
                unmatched_names.append(gene)
                continue
            matched_names.append(gene)
            if resolved_position not in seen:
                seen.add(resolved_position)
                positions.append(resolved_position)

        if not positions:
            available = ", ".join(namespaces)
            raise ValueError(
                f"No markers for label {label!r} matched the dataset ({available}). "
                "Check feature IDs/gene symbols and organism-specific capitalization."
            )

        labels.append(label)
        all_indices.append(positions)
        matched[label] = matched_names
        unmatched[label] = unmatched_names

    return ResolvedMarkers(labels, all_indices, matched, unmatched, namespaces)
