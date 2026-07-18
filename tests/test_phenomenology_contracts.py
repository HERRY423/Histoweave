from __future__ import annotations

from histoweave.benchmark.phenomenology_contracts import (
    PHENOMENOLOGY_METHODS,
    EvaluationRole,
    ResourceClass,
    build_evaluation_contracts,
    capability_matrix_rows,
    freeze_release_manifest,
)
from histoweave.datasets import SpatialPhenomenon
from histoweave.plugins import MethodCategory, MethodReference, get_method
from histoweave.plugins.builtin.release_manifest import RESEARCH_METHODS


def test_frozen_manifest_matches_audited_release_sets_only() -> None:
    manifest = freeze_release_manifest()
    names = {method.name for method in manifest.methods}
    assert names == PHENOMENOLOGY_METHODS
    assert names.isdisjoint(RESEARCH_METHODS)
    assert len(names) == 54
    assert len(manifest.manifest_hash) == 64
    assert manifest.to_dict()["method_count"] == len(names)


def test_manifest_is_deterministic_and_category_sorted() -> None:
    first = freeze_release_manifest()
    second = freeze_release_manifest()
    assert first.manifest_hash == second.manifest_hash
    assert [method.name for method in first.methods] == sorted(PHENOMENOLOGY_METHODS)


def test_each_release_method_has_exactly_one_explicit_contract() -> None:
    manifest = freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)
    assert len(contracts) == len(manifest.methods)
    for method in manifest.methods:
        reference = MethodReference(method.category, method.name, method.version)
        contract = contracts[reference]
        assert contract.metrics
        assert any(metric.primary for metric in contract.metrics)
        assert contract.applicable_phenomena
        assert contract.required_inputs


def test_category_roles_preserve_scientific_interpretation() -> None:
    manifest = freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)

    def role(category: MethodCategory, name: str) -> EvaluationRole:
        spec = get_method(category, name).spec
        return contracts[MethodReference(category, name, spec.version)].role

    assert role(MethodCategory.INGESTION, "visium_reader") is EvaluationRole.INGESTION_FIDELITY
    assert (
        role(MethodCategory.NORMALIZATION, "log1p_cp10k")
        is EvaluationRole.PREPROCESSING_PRESERVATION
    )
    assert role(MethodCategory.INTEGRATION, "harmony") is EvaluationRole.REPRESENTATION_INTEGRATION
    assert role(MethodCategory.DOMAIN_DETECTION, "banksy_py") is EvaluationRole.DIRECT_INFERENCE


def test_not_applicable_is_declared_not_converted_to_failure() -> None:
    manifest = freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)
    deconv_spec = get_method(MethodCategory.DECONVOLUTION, "marker_deconv").spec
    deconv = contracts[
        MethodReference(MethodCategory.DECONVOLUTION, "marker_deconv", deconv_spec.version)
    ]
    assert deconv.is_applicable(SpatialPhenomenon.MIXTURE)
    assert not deconv.is_applicable(SpatialPhenomenon.GRADIENT)

    ccc_spec = get_method(MethodCategory.CELL_CELL_COMMUNICATION, "liana_plus").spec
    ccc = contracts[
        MethodReference(MethodCategory.CELL_CELL_COMMUNICATION, "liana_plus", ccc_spec.version)
    ]
    assert set(ccc.applicable_phenomena) == {
        SpatialPhenomenon.HOTSPOT,
        SpatialPhenomenon.MIXTURE,
    }


def test_heavy_budget_class_is_declarative() -> None:
    manifest = freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)
    cell2location = get_method(MethodCategory.DECONVOLUTION, "cell2location").spec
    reference = MethodReference(
        MethodCategory.DECONVOLUTION,
        "cell2location",
        cell2location.version,
    )
    assert contracts[reference].resource_class is ResourceClass.HEAVY


def test_tuning_spaces_reference_real_parameters_and_have_at_most_four_candidates() -> None:
    manifest = freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)
    for reference, contract in contracts.items():
        params = {
            param.name for param in get_method(reference.category, reference.name).spec.params
        }
        for parameter, values in contract.tuning_space:
            assert parameter in params
            assert 1 < len(values) <= 4


def test_capability_matrix_has_six_rows_per_method_without_rank() -> None:
    manifest = freeze_release_manifest()
    rows = capability_matrix_rows(manifest)
    assert len(rows) == len(manifest.methods) * len(SpatialPhenomenon)
    assert all("rank" not in row for row in rows)
    assert {row["role"] for row in rows} == {role.value for role in EvaluationRole}
