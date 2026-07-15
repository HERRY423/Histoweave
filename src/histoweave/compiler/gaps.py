"""Auditable capability-gap logging."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .schema import CompiledPlan

_HEADER = "| Timestamp (UTC) | Question | Missing concept | Degraded to |\n|---|---|---|---|\n"


def append_gaps(plan: CompiledPlan, path: str | Path) -> Path | None:
    if not plan.gaps:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(
            "# Spatial Pipeline Compiler capability gaps\n\n" + _HEADER, encoding="utf-8"
        )
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    question = plan.question.replace("|", "\\|").replace("\n", " ")
    with output.open("a", encoding="utf-8") as handle:
        for gap in plan.gaps:
            concept = gap.concept.replace("|", "\\|")
            degraded = gap.degraded_to.replace("|", "\\|")
            handle.write(f"| {timestamp} | {question} | {concept} | {degraded} |\n")
    return output
