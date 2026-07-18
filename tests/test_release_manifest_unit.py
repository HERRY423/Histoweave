"""Unit tests for the audited release maturity manifest."""

from histoweave.plugins import list_methods
from histoweave.plugins.builtin.release_manifest import (
    CONTRACT_VALIDATED_METHODS,
    MULTI_DATASET_EVIDENCE_METHODS,
    SCIENTIFIC_VALIDATED_METHODS,
    VALIDATED_METHODS,
    VALIDATION_EVIDENCE,
)


def test_validated_methods_have_evidence() -> None:
    assert len(VALIDATED_METHODS) == 10
    assert VALIDATED_METHODS == SCIENTIFIC_VALIDATED_METHODS
    assert VALIDATED_METHODS <= set(VALIDATION_EVIDENCE)
    assert len(CONTRACT_VALIDATED_METHODS) == 3
    assert len(MULTI_DATASET_EVIDENCE_METHODS) == 13


def test_list_methods_reflects_scientific_and_contract() -> None:
    rows = {row["name"]: row for row in list_methods()}
    for name in SCIENTIFIC_VALIDATED_METHODS:
        assert rows[name]["maturity"] == "validated"
    for name in CONTRACT_VALIDATED_METHODS:
        assert rows[name]["maturity"] == "contract_validated"
