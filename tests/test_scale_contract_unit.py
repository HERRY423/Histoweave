"""Unit tests for large-table scale contracts."""

from histoweave.datasets.scale_contract import (
    SCALE_CONTRACTS,
    registry_scale_table,
    scale_contract_for_assay,
)


def test_known_contracts_exist() -> None:
    for name in ("visium_standard", "xenium_50k", "merfish_100k", "merfish_atlas"):
        assert name in SCALE_CONTRACTS


def test_xenium_large_n_selects_full_slide() -> None:
    contract = scale_contract_for_assay("xenium", 150_000)
    assert contract.name == "xenium_full_slide"
    plan = contract.plan_for(200_000)
    assert plan["sparse_required"] is True
    assert plan["subsample_to"] == contract.recommended_subsample


def test_registry_scale_table_non_empty() -> None:
    rows = registry_scale_table()
    assert len(rows) >= 10
    assert all("scale_contract" in row for row in rows)
