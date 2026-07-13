"""Example external HistoWeave plugin package."""

from __future__ import annotations


def register_all() -> None:
    """Entry-point hook: import method modules so their @register decorators run."""
    from . import method  # noqa: F401


__all__ = ["register_all"]
