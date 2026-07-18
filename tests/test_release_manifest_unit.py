"""Unit tests for the audited release maturity manifest."""

from histoweave.plugins import list_methods
from histoweave.plugins.builtin.release_manifest import (
    VALIDATED_METHODS,
    VALIDATION_EVIDENCE,
)


def test_validated_methods_have_evidence() -> None:
    assert len(VALIDATED_METHODS) >= 3
    assert VALIDATED_METHODS <= set(VALIDATION_EVIDENCE)


def test_list_methods_reflects_validated() -> None:
    rows = {row["name"]: row for row in list_methods()}
    for name in VALIDATED_METHODS:
        assert rows[name]["maturity"] == "validated"
