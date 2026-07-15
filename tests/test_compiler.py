from __future__ import annotations

import json
from pathlib import Path

import pytest

from histoweave import datasets
from histoweave.compiler import (
    CompiledPlan,
    CompiledStep,
    CompilerValidationError,
    build_catalog,
    validate_plan,
)
from histoweave.compiler import (
    compile as compile_question,
)
from histoweave.compiler.executor import _encode_nextflow_params, _nextflow_params
from histoweave.compiler.prompts import build_messages
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
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["steps"][-1]["method"] == "spatialde"
    assert payload["dry_run"] is True


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


def test_nextflow_handoff_bundle_keeps_compiler_metadata(tmp_path) -> None:
    from histoweave.compiler import run_compiled
    from histoweave.io import read_bundle

    plan = compile_question(
        "Find spatial domains and annotate their cell types.",
        provider="mock",
        executor="nextflow",
    )
    plan.dry_run = False
    handoff = run_compiled(
        plan,
        data=datasets.make_synthetic(seed=10),
        out=tmp_path / "compiled.html",
    )
    payload = json.loads(Path(handoff["params_path"]).read_text(encoding="utf-8"))
    bundled = read_bundle(payload["bundle"])
    assert bundled.uns["run_manifest"]["compiler"]["question"] == plan.question
    assert payload["steps"].endswith(",report")


def test_compiled_report_displays_compiler_metadata(tmp_path) -> None:
    from histoweave.compiler import run_compiled

    plan = compile_question("Find spatial domains", provider="mock")
    plan.steps[-1] = CompiledStep(
        category="domain_detection",
        method="kmeans",
        params={"n_domains": 3},
        purpose="Detect spatial domains.",
    )
    plan.dry_run = False
    output = tmp_path / "compiled.html"
    run_compiled(plan, data=datasets.make_synthetic(seed=8), out=output)
    html = output.read_text(encoding="utf-8")
    assert "Compiler plan" in html
    assert "Find spatial domains" in html
    assert "mock" in html
    assert "in-process" in html
