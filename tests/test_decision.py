"""Scientific contracts for the evidence-governed decision layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from histoweave import DecisionPolicy, decide_from_bundle
from histoweave.benchmark import (
    DecisionAction,
    DecisionEngine,
    ISUSResult,
    LandscapeResult,
    MethodScore,
    Recommendation,
    build_decision_card,
    extract_features,
    feature_vector,
)
from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER
from histoweave.cli import main
from histoweave.datasets import make_synthetic
from histoweave.io import write_bundle


def _recommendation(
    *,
    beats: bool | None,
    ground_truth_kind: str = "spatial_domain",
) -> Recommendation:
    ranked = [
        MethodScore(
            method="local@sw0.8",
            score=0.82,
            confidence=0.80,
            wins=2,
            neighbour_scores={"ref_a": 0.84, "ref_b": 0.80},
            uncertainty=0.02,
            support=2,
            coverage=1.0,
            base_method="local",
            spatial_context_policy="sw0.8",
        ),
        MethodScore(
            method="global",
            score=0.79,
            confidence=0.75,
            wins=0,
            neighbour_scores={"ref_a": 0.78, "ref_b": 0.80},
            uncertainty=0.01,
            support=2,
            coverage=1.0,
            base_method="global",
        ),
    ]
    return Recommendation(
        task="spatial_domain",
        dataset_name="query",
        ranked_methods=ranked,
        neighbours=[
            {
                "name": "ref_a",
                "similarity": 0.9,
                "task": "spatial_domain",
                "ground_truth_kind": ground_truth_kind,
            },
            {
                "name": "ref_b",
                "similarity": 0.8,
                "task": "spatial_domain",
                "ground_truth_kind": ground_truth_kind,
            },
        ],
        global_best_method="global",
        global_best_score=0.79,
        beats_global_best_baseline=beats,
        selection_regret_vs_global_best=-0.03 if beats else 0.03,
    )


def _heldout(beats: bool = True) -> dict[str, object]:
    return {
        "protocol": "external_holdout",
        "n_queries": 12,
        "beats_global_best": beats,
    }


def test_negative_baseline_returns_global_default_and_json_is_safe():
    card = build_decision_card(_recommendation(beats=False))
    assert card.action is DecisionAction.GLOBAL_DEFAULT
    assert card.primary_set == ["global"]
    assert "local@sw0.8" in card.comparison_set
    assert card.can_personalise is False
    payload = card.to_dict()
    assert payload["protocol"] == "histoweave.evidence_decision.v1"
    json.dumps(payload, allow_nan=False)


def test_reference_neighbour_advantage_is_not_heldout_validation():
    card = build_decision_card(_recommendation(beats=True))
    assert card.action is DecisionAction.EVIDENCE_REQUIRED
    assert card.primary_set == []
    assert card.can_personalise is False
    check = next(item for item in card.checks if item.name == "heldout_validation")
    assert check.status.value == "not_evaluated"


def test_grouped_holdout_and_exact_pareto_front_enable_set_valued_action():
    pareto = {
        "dataset": "query",
        "frontier": ["local@sw0.8", "global"],
        "ranks": {"local@sw0.8": 0, "global": 0},
        "table": {
            "local@sw0.8": {"accuracy": 0.82, "speed": 4.0},
            "global": {"accuracy": 0.79, "speed": 2.0},
        },
    }
    card = build_decision_card(
        _recommendation(beats=True),
        pareto=pareto,
        validation=_heldout(),
    )
    assert card.action is DecisionAction.PERSONALISED_SET
    assert card.primary_set == ["local@sw0.8", "global"]
    assert card.can_personalise is True


def test_different_spatial_policy_is_not_treated_as_same_pareto_point():
    pareto = {
        "dataset": "query",
        "frontier": ["local@sw0.0"],
        "ranks": {"local@sw0.0": 0},
        "table": {"local@sw0.0": {"accuracy": 0.81, "speed": 2.0}},
    }
    card = build_decision_card(
        _recommendation(beats=True),
        pareto=pareto,
        validation=_heldout(),
    )
    assert card.action is DecisionAction.EVIDENCE_REQUIRED
    assert card.primary_set == []


def test_isus_is_posthoc_and_does_not_change_preexecution_action():
    recommendation = _recommendation(beats=False)
    without = build_decision_card(recommendation)
    descriptor = ISUSResult(
        dataset="query",
        isus=0.05,
        i_d_e=0.8,
        i_d_se=0.84,
        i_d_s_given_e=0.04,
        band="expression-sufficient",
        n_obs=100,
        n_domains=3,
        n_pcs=10,
        k=3,
        flags=["Interpretation thresholds are provisional."],
    )
    with_isus = build_decision_card(recommendation, isus=descriptor)
    assert with_isus.action is without.action is DecisionAction.GLOBAL_DEFAULT
    assert with_isus.primary_set == without.primary_set
    assert with_isus.evidence_roles["spatial_utility"] == "posthoc_label_conditioned_descriptor"


def test_cluster_proxy_for_spatial_domain_fails_closed():
    card = build_decision_card(
        _recommendation(beats=True, ground_truth_kind="cluster_proxy"),
        validation=_heldout(),
    )
    assert card.action is DecisionAction.ABSTAIN
    assert card.primary_set == []


def test_bundled_external_negative_result_cannot_unlock_personalisation():
    validation_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark_external_validation"
        / "decision_validation.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    card = build_decision_card(_recommendation(beats=True), validation=validation)
    assert validation["beats_global_best"] is False
    assert card.action is DecisionAction.GLOBAL_DEFAULT
    assert card.can_personalise is False


def _knowledge_base(tmp_path):
    datasets = {
        "spatial_a": make_synthetic(n_cells=60, n_genes=16, seed=1),
        "spatial_b": make_synthetic(n_cells=70, n_genes=16, seed=2),
        "proxy": make_synthetic(n_cells=80, n_genes=16, seed=3),
    }
    features = {
        name: feature_vector(
            extract_features(data, include_domain=False),
            order=RECOMMENDATION_FEATURE_ORDER,
        )
        for name, data in datasets.items()
    }
    landscape = LandscapeResult(
        performance={
            "spatial_a": {"kmeans": 0.8, "spectral": 0.7},
            "spatial_b": {"kmeans": 0.7, "spectral": 0.8},
            "proxy": {"kmeans": 0.1, "spectral": 0.95},
        },
        features=features,
        embedding={},
        best_method={"spatial_a": "kmeans", "spatial_b": "spectral", "proxy": "spectral"},
        niches={"kmeans": ["spatial_a"], "spectral": ["spatial_b", "proxy"]},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=2,
        dataset_count=3,
        task="spatial_domain",
        metric="ARI",
        dataset_meta={
            "spatial_a": {"task": "spatial_domain", "ground_truth_kind": "spatial_domain"},
            "spatial_b": {"task": "spatial_domain", "ground_truth_kind": "spatial_domain"},
            "proxy": {"task": "cell_type", "ground_truth_kind": "cluster_proxy"},
        },
    )
    path = tmp_path / "knowledge.json"
    from histoweave.benchmark import MethodRecommender

    MethodRecommender(landscape).save_knowledge_base(path)
    return path, datasets


def test_decision_engine_hard_filters_cross_task_evidence(tmp_path):
    path, datasets = _knowledge_base(tmp_path)
    card = DecisionEngine(path, k_neighbours=3).decide(
        datasets["spatial_a"],
        dataset_name="query",
        task="spatial_domain",
    )
    neighbour_names = {item["name"] for item in card.recommendation["neighbours"]}
    assert "proxy" not in neighbour_names


def test_decide_cli_writes_identical_json(tmp_path, capsys):
    path, datasets = _knowledge_base(tmp_path)
    bundle = write_bundle(datasets["spatial_a"], tmp_path / "query.ttab")
    output = tmp_path / "decision.json"
    rc = main(
        [
            "decide",
            "--in",
            str(bundle),
            "--knowledge-base",
            str(path),
            "--task",
            "spatial_domain",
            "--dataset-name",
            "query",
            "--json",
            "--out",
            str(output),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] in {"global_default", "evidence_required"}
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert np.isfinite(payload["recommendation"]["ranked_methods"][0]["score"])


def test_python_bundle_api_and_cli_share_policy_and_execution_path(tmp_path, capsys):
    path, datasets = _knowledge_base(tmp_path)
    bundle = write_bundle(datasets["spatial_a"], tmp_path / "query.ttab")
    python_output = tmp_path / "python_decision.json"
    cli_output = tmp_path / "cli_decision.json"
    policy = DecisionPolicy(
        shortlist_size=1,
        min_support=1,
        min_rank_support_score=0.1,
        severe_failure_threshold=0.8,
        require_baseline_advantage=False,
        require_heldout_validation=False,
    )

    python_card = decide_from_bundle(
        bundle,
        knowledge_base=path,
        task="spatial_domain",
        dataset_name="query",
        policy=policy,
        out=python_output,
    )
    rc = main(
        [
            "decide",
            "--in",
            str(bundle),
            "--knowledge-base",
            str(path),
            "--task",
            "spatial_domain",
            "--dataset-name",
            "query",
            "--shortlist-size",
            "1",
            "--min-support",
            "1",
            "--min-rank-support-score",
            "0.1",
            "--severe-failure-threshold",
            "0.8",
            "--allow-no-baseline-advantage",
            "--allow-no-heldout-validation",
            "--json",
            "--out",
            str(cli_output),
        ]
    )

    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload == python_card.to_dict()
    assert json.loads(python_output.read_text(encoding="utf-8")) == cli_payload
    assert json.loads(cli_output.read_text(encoding="utf-8")) == cli_payload
    assert cli_payload["policy"] == policy.to_dict()
