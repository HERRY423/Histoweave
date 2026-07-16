from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histoweave import datasets
from histoweave.compiler import (
    CompiledPlan,
    CompiledStep,
    CompilerSchemaError,
    CompilerValidationError,
    build_catalog,
    load_plan,
    run_compiled,
    save_plan,
    validate_plan,
)
from histoweave.compiler import (
    compile as compile_question,
)
from histoweave.compiler.executor import _encode_nextflow_params, _nextflow_params
from histoweave.compiler.prompts import build_messages
from histoweave.compiler.serialization import catalog_digest, seal_plan
from histoweave.compiler.templates import MOCK_TEMPLATES, template_for_question


def test_catalog_comes_from_live_registry() -> None:
    rows = build_catalog()
    pairs = {(row["category"], row["name"]) for row in rows}
    assert ("qc", "basic_qc") in pairs
    assert ("domain_detection", "banksy") in pairs
    assert all("params" in row and not row.get("deprecated") for row in rows)


def test_mock_compiler_emits_valid_gap_aware_plan(tmp_path) -> None:
    data = datasets.make_synthetic(seed=3)
    data.uns["assay"] = "xenium"
    gap_file = tmp_path / "COMPILER_GAPS.md"

    plan = compile_question(
        "What are the immune-escape mechanisms at the invasive margin?",
        data=data,
        provider="mock",
        gaps_path=gap_file,
    )

    assert plan.dry_run is True
    assert [step.method for step in plan.steps][:2] == ["basic_qc", "log1p_cp10k"]
    assert {step.method for step in plan.steps} >= {"banksy", "spatial_graph", "liana_plus"}
    assert plan.gaps and "invasive-margin" in plan.gaps[0].concept
    assert "immune-escape" in gap_file.read_text(encoding="utf-8")
    json.dumps(plan.to_dict())


def test_mock_compiler_seals_stable_versioned_plan() -> None:
    first = compile_question("Find spatial domains", provider="mock")
    second = compile_question("Find spatial domains", provider="mock")

    assert first.schema_version == 1
    assert first.attempt_count == 1
    assert first.plan_id == second.plan_id
    assert first.plan_id.startswith("hwc1_")
    assert len(first.plan_id.removeprefix("hwc1_")) == 24
    assert all(character in "0123456789abcdef" for character in first.plan_id[5:])
    assert first.catalog_digest == catalog_digest(build_catalog())
    assert first.catalog_digest.startswith("sha256:")
    assert len(first.catalog_digest.removeprefix("sha256:")) == 64


def test_plan_save_load_roundtrip_with_live_catalog_check(tmp_path) -> None:
    plan = compile_question("Find spatial domains", provider="mock")
    path = save_plan(plan, tmp_path / "plan.json")

    restored = load_plan(path, require_catalog_match=True)

    assert restored.to_dict() == plan.to_dict()


def test_strict_load_uses_actual_catalog_scope_not_model_assay_guess(tmp_path) -> None:
    plan = compile_question("Analyze a Xenium tumor section", provider="mock")
    assert plan.assay_assumed == "xenium"
    path = save_plan(plan, tmp_path / "xenium-assumption-plan.json")

    restored = load_plan(path, require_catalog_match=True)

    assert restored.plan_id == plan.plan_id


def test_compiled_steps_pin_versions_defaults_and_pipeline_metadata() -> None:
    plan = compile_question("Find spatial domains", provider="mock")
    catalog = {
        (row["category"], row["name"]): row
        for row in build_catalog()
    }

    for step in plan.steps:
        registered = catalog[(step.category, step.method)]
        assert step.method_version == registered["version"]
        assert step.params == {
            param["name"]: param["default"] for param in registered["params"]
        }
    assert [step.method_version for step in plan.pipeline_steps] == [
        step.method_version for step in plan.steps
    ]


def test_numpy_integer_context_is_normalized_for_compilation() -> None:
    data = datasets.make_synthetic(seed=12)
    data.uns["n_domains"] = np.int64(5)

    plan = compile_question("Find spatial domains", data=data, provider="mock")

    assert plan.steps


def test_non_finite_context_is_rejected_before_provider_call(monkeypatch) -> None:
    import histoweave.compiler as compiler_module

    data = datasets.make_synthetic(seed=13)
    data.uns["n_domains"] = float("nan")
    provider_called = False

    def unexpected_provider_call(**kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider must not receive a non-finite compiler context")

    monkeypatch.setattr(compiler_module, "request_plan", unexpected_provider_call)
    with pytest.raises(ValueError, match="finite"):
        compile_question("Find spatial domains", data=data, provider="mock")
    assert provider_called is False


def test_load_plan_rejects_tampered_payload(tmp_path) -> None:
    plan = compile_question("Find spatial domains", provider="mock")
    path = save_plan(plan, tmp_path / "plan.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rationale"] = "tampered after compilation"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CompilerSchemaError, match="integrity"):
        load_plan(path)


def test_load_plan_can_reject_catalog_drift(tmp_path, monkeypatch) -> None:
    import histoweave.compiler.serialization as serialization

    plan = compile_question("Find spatial domains", provider="mock")
    path = save_plan(plan, tmp_path / "plan.json")
    live_catalog = build_catalog()
    monkeypatch.setattr(
        serialization,
        "build_catalog",
        lambda *, assay=None: [
            *live_catalog,
            {
                "category": "domain_detection",
                "name": "future_method",
                "version": "99.0",
            },
        ],
    )

    with pytest.raises(CompilerValidationError, match="catalog digest"):
        load_plan(path, require_catalog_match=True)

    assert load_plan(path, require_catalog_match=False).plan_id == plan.plan_id


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_compiler_schema_rejects_non_finite_json_numbers(value) -> None:
    with pytest.raises(CompilerSchemaError, match="finite"):
        CompiledPlan.from_dict(
            {
                "rationale": "invalid numeric payload",
                "steps": [
                    {
                        "category": "qc",
                        "method": "basic_qc",
                        "params": {"min_counts": value},
                    }
                ],
            },
            question="test",
        )


def test_validator_rejects_unknown_method_and_backwards_order() -> None:
    unknown = CompiledPlan.from_dict(
        {
            "rationale": "test",
            "steps": [{"category": "qc", "method": "invented", "params": {}, "purpose": "x"}],
        },
        question="test",
    )
    with pytest.raises(CompilerValidationError, match="No method"):
        validate_plan(unknown)

    backwards = CompiledPlan.from_dict(
        {
            "rationale": "test",
            "steps": [
                {"category": "normalization", "method": "log1p_cp10k", "params": {}},
                {"category": "qc", "method": "basic_qc", "params": {}},
            ],
        },
        question="test",
    )
    with pytest.raises(CompilerValidationError, match="later-stage"):
        validate_plan(backwards)


def test_validator_accepts_integration_between_normalization_and_domains() -> None:
    plan = CompiledPlan.from_dict(
        {
            "rationale": "correct batches before spatial analysis",
            "steps": [
                {"category": "normalization", "method": "log1p_cp10k", "params": {}},
                {"category": "integration", "method": "combat", "params": {}},
                {"category": "domain_detection", "method": "kmeans", "params": {}},
            ],
        },
        question="test",
    )
    assert validate_plan(plan) is plan


@pytest.mark.parametrize("template", MOCK_TEMPLATES, ids=lambda row: row["name"])
def test_all_seven_mock_templates_compile_to_registered_plans(template, tmp_path) -> None:
    plan = compile_question(
        template["question"],
        provider="mock",
        gaps_path=tmp_path / "gaps.md",
    )
    assert validate_plan(plan) is plan
    assert plan.steps[-1].method == template["plan"]["steps"][-1]["method"]


def test_prompt_contains_seven_registry_backed_few_shot_examples() -> None:
    messages = build_messages("Find domains", build_catalog())
    assistants = [message for message in messages if message["role"] == "assistant"]
    assert len(MOCK_TEMPLATES) == 7
    assert len(assistants) == 7
    assert all(json.loads(message["content"])["steps"] for message in assistants)


def test_mock_template_domains_are_canonical() -> None:
    assert tuple(template["name"] for template in MOCK_TEMPLATES) == (
        "tumor",
        "brain",
        "developmental",
        "immune",
        "drug",
        "cross_section",
        "generic",
    )


@pytest.mark.parametrize(
    ("question", "template_name"),
    [
        ("Profile carcinoma architecture", "tumor"),
        ("Resolve cortical layers", "brain"),
        ("Segment nuclei in an embryo", "developmental"),
        ("Rank macrophage ligand-receptor communication", "immune"),
        ("Map treatment response after therapy", "drug"),
        ("Integrate serial cross-sections", "cross_section"),
        ("Explore spatial structure", "generic"),
    ],
)
def test_mock_question_routes_to_domain_template(question, template_name) -> None:
    expected = next(row for row in MOCK_TEMPLATES if row["name"] == template_name)
    assert template_for_question(question) == expected["plan"]


def test_cli_ask_plan_only_json(tmp_path, capsys) -> None:
    from histoweave.cli import main
    from histoweave.io import write_bundle

    bundle = tmp_path / "sample.ttab"
    plan_path = tmp_path / "compiled-plan.json"
    write_bundle(datasets.make_synthetic(seed=4), bundle)
    rc = main(
        [
            "ask",
            "Which genes are spatially variable?",
            "--in",
            str(bundle),
            "--model",
            "mock",
            "--plan-only",
            "--json",
            "--gaps-file",
            str(tmp_path / "gaps.md"),
            "--plan-out",
            str(plan_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    saved = load_plan(plan_path, require_catalog_match=True)
    assert rc == 0
    assert payload["steps"][-1]["method"] == "spatialde"
    assert payload["dry_run"] is True
    assert payload["plan_id"] == saved.plan_id
    assert saved.to_dict() == payload


def test_cli_ask_yes_passes_non_mutating_execution_confirmation(
    tmp_path, capsys, monkeypatch
) -> None:
    import histoweave.compiler as compiler_module
    from histoweave.cli import main
    from histoweave.io import write_bundle

    bundle = tmp_path / "sample.ttab"
    write_bundle(datasets.make_synthetic(seed=17), bundle)
    captured = {}

    def fake_run_compiled(plan, *, data, out, confirmed):
        captured.update(
            plan=plan,
            data=data,
            out=out,
            confirmed=confirmed,
        )
        return data

    monkeypatch.setattr(compiler_module, "run_compiled", fake_run_compiled)
    rc = main(
        [
            "ask",
            "Find spatial domains",
            "--in",
            str(bundle),
            "--model",
            "mock",
            "--yes",
            "--out",
            str(tmp_path / "compiled.html"),
            "--gaps-file",
            str(tmp_path / "gaps.md"),
        ]
    )
    capsys.readouterr()

    assert rc == 0
    assert captured["confirmed"] is True
    assert captured["plan"].dry_run is True
    assert captured["plan"].plan_id.startswith("hwc1_")


def test_run_compiled_rejects_unsealed_plan_before_writing(tmp_path) -> None:
    plan = CompiledPlan.from_dict(
        {
            "rationale": "manual unsealed plan",
            "steps": [
                {
                    "category": "qc",
                    "method": "basic_qc",
                    "params": {},
                }
            ],
        },
        question="test",
    )
    output = tmp_path / "unsealed.html"

    with pytest.raises(CompilerSchemaError, match="plan_id"):
        run_compiled(
            plan,
            data=datasets.make_synthetic(seed=16),
            out=output,
            confirmed=True,
        )

    assert not output.exists()


def test_run_compiled_requires_explicit_confirmation_before_writing(tmp_path) -> None:
    plan = compile_question("Find spatial domains", provider="mock")
    output = tmp_path / "compiled.html"

    with pytest.raises(RuntimeError, match="confirmed=True"):
        run_compiled(plan, data=datasets.make_synthetic(seed=9), out=output)

    assert not output.exists()


def test_run_compiled_rejects_tampering_before_any_write(tmp_path) -> None:
    plan = compile_question(
        "Find spatial domains and annotate their cell types.",
        provider="mock",
        executor="nextflow",
    )
    plan.rationale += " tampered"
    output = tmp_path / "tampered.html"

    with pytest.raises(CompilerSchemaError, match="integrity"):
        run_compiled(
            plan,
            data=datasets.make_synthetic(seed=14),
            out=output,
            confirmed=True,
        )

    assert not output.exists()
    assert not output.with_suffix(".params.json").exists()
    assert not output.with_suffix(".input.ttab").exists()


@pytest.mark.parametrize("attribute", ["dry_run", "attempt_count"])
def test_run_compiled_identity_covers_execution_metadata(attribute, tmp_path) -> None:
    plan = compile_question(
        "Find spatial domains and annotate their cell types.",
        provider="mock",
        executor="nextflow",
    )
    tampered = False if attribute == "dry_run" else plan.attempt_count + 1
    setattr(plan, attribute, tampered)
    output = tmp_path / f"tampered-{attribute}.html"

    with pytest.raises(CompilerSchemaError, match="integrity"):
        run_compiled(
            plan,
            data=datasets.make_synthetic(seed=15),
            out=output,
            confirmed=True,
        )

    assert not output.with_suffix(".params.json").exists()
    assert not output.with_suffix(".input.ttab").exists()


def test_nextflow_params_match_flat_workflow_schema(tmp_path) -> None:
    plan = compile_question(
        "Find spatial domains and annotate their cell types.",
        provider="mock",
        executor="nextflow",
    )
    payload = _nextflow_params(
        plan,
        bundle_path=tmp_path / "sample.ttab",
        outdir=tmp_path / "results",
    )
    assert payload["steps"] == "qc,normalize,domain_detection,annotation,report"
    assert payload["qc_method"] == "basic_qc"
    assert payload["normalize_method"] == "log1p_cp10k"
    assert payload["domain_method"] == "banksy"
    assert payload["annotation_method"] == "marker_score"
    assert payload["n_domains"] == 5
    assert "question" not in payload
    assert not isinstance(payload["steps"], list)


def test_nextflow_params_forward_every_pinned_step_version(tmp_path) -> None:
    plan = compile_question(
        "Find spatial domains and annotate their cell types.",
        provider="mock",
        executor="nextflow",
    )
    payload = _nextflow_params(
        plan,
        bundle_path=tmp_path / "sample.ttab",
        outdir=tmp_path / "results",
    )
    version_keys = {
        "qc": "qc_version",
        "normalization": "normalize_version",
        "domain_detection": "domain_version",
        "annotation": "annotation_version",
        "deconvolution": "deconvolution_version",
    }

    for step in plan.steps:
        assert payload[version_keys[step.category]] == step.method_version


def test_nextflow_params_reject_unsupported_compiler_stage(tmp_path) -> None:
    plan = compile_question(
        "Which genes are spatially variable?",
        provider="mock",
        executor="nextflow",
    )
    with pytest.raises(ValueError, match="in-process"):
        _nextflow_params(plan, bundle_path=tmp_path / "sample.ttab", outdir=tmp_path)


def test_nextflow_params_preserve_nested_json_values() -> None:
    assert _encode_nextflow_params({"markers": {"T cell": ["CD3D", "CD3E"]}}) == [
        'markers={"T cell":["CD3D","CD3E"]}'
    ]


def test_nextflow_unsupported_stage_fails_before_writing(tmp_path) -> None:
    plan = compile_question(
        "Which genes are spatially variable?",
        provider="mock",
        executor="nextflow",
    )
    output = tmp_path / "unsupported.html"

    with pytest.raises(ValueError, match="in-process"):
        run_compiled(
            plan,
            data=datasets.make_synthetic(seed=11),
            out=output,
            confirmed=True,
        )

    assert not output.with_suffix(".params.json").exists()
    assert not output.with_suffix(".input.ttab").exists()


def test_nextflow_handoff_bundle_keeps_compiler_metadata(tmp_path) -> None:
    from histoweave.io import read_bundle

    plan = compile_question(
        "Find spatial domains and annotate their cell types.",
        provider="mock",
        executor="nextflow",
    )
    handoff = run_compiled(
        plan,
        data=datasets.make_synthetic(seed=10),
        out=tmp_path / "compiled.html",
        confirmed=True,
    )
    payload = json.loads(Path(handoff["params_path"]).read_text(encoding="utf-8"))
    bundled = read_bundle(payload["bundle"])
    assert bundled.uns["run_manifest"]["compiler"]["question"] == plan.question
    assert payload["steps"].endswith(",report")


def test_compiled_report_displays_compiler_metadata(tmp_path) -> None:
    plan = compile_question("Find spatial domains", provider="mock")
    kmeans_version = next(
        row["version"]
        for row in build_catalog()
        if row["category"] == "domain_detection" and row["name"] == "kmeans"
    )
    plan.steps[-1] = CompiledStep(
        category="domain_detection",
        method="kmeans",
        params={"n_domains": 3},
        purpose="Detect spatial domains.",
        method_version=kmeans_version,
    )
    seal_plan(
        plan,
        catalog_fingerprint=plan.catalog_digest,
        catalog_assay=plan.catalog_assay,
        attempt_count=plan.attempt_count,
    )
    output = tmp_path / "compiled.html"
    run_compiled(
        plan,
        data=datasets.make_synthetic(seed=8),
        out=output,
        confirmed=True,
    )
    html = output.read_text(encoding="utf-8")
    assert "Compiler plan" in html
    assert "Find spatial domains" in html
    assert "mock" in html
    assert "in-process" in html
