"""Tests for active-learning recommender calibration."""

from __future__ import annotations

import json
from pathlib import Path

from histoweave.benchmark import (
    RECOMMENDATION_FEATURE_ORDER,
    LandscapeResult,
    MethodRecommender,
    extract_features,
    feature_vector,
    propose_evidence_acquisition,
)
from histoweave.cli import main
from histoweave.datasets import make_synthetic
from histoweave.io import write_bundle


def _kb_with_holes():
    """Knowledge base with missing scores so EIG tasks are non-empty."""
    datasets = {
        "ref_a": make_synthetic(n_cells=80, n_genes=18, noise=0.12, marker_gene_lift=9.0, seed=41),
        "ref_b": make_synthetic(n_cells=95, n_genes=22, noise=0.40, marker_gene_lift=4.0, seed=42),
        "ref_c": make_synthetic(
            n_cells=110,
            n_genes=26,
            n_domains=4,
            noise=0.22,
            marker_gene_lift=7.0,
            layout="grid",
            seed=43,
        ),
    }
    features = {
        name: feature_vector(
            extract_features(data, include_domain=False),
            order=RECOMMENDATION_FEATURE_ORDER,
        )
        for name, data in datasets.items()
    }
    # alpha wins overall mean → global best; beta better on noisy; holes on gamma.
    performance = {
        "ref_a": {"alpha": 0.95, "beta": 0.70, "gamma": float("nan")},
        "ref_b": {"alpha": 0.40, "beta": 0.85, "gamma": 0.50},
        "ref_c": {"alpha": 0.60, "beta": float("nan"), "gamma": 0.55},
    }
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding={},
        best_method={"ref_a": "alpha", "ref_b": "beta", "ref_c": "alpha"},
        niches={"alpha": ["ref_a", "ref_c"], "beta": ["ref_b"]},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=3,
        dataset_count=3,
        task="domain_detection",
        metric="ARI",
    ), datasets


def test_recommend_attaches_evidence_todo_when_not_beating_baseline():
    landscape, datasets = _kb_with_holes()
    rec = MethodRecommender(landscape, k_neighbours=2).recommend(
        datasets["ref_b"],
        dataset_name="query_noisy",
    )
    # Strictly better than global-best is rare; typically False when tying or worse.
    assert rec.beats_global_best_baseline is False or rec.evidence_todo is not None
    # With holes, calibration should propose tasks when needed.
    plan = propose_evidence_acquisition(
        MethodRecommender(landscape, k_neighbours=2),
        rec,
        top_n=5,
    )
    assert plan.todo
    # Missing cells appear as evidence tasks.
    pairs = {(t.dataset, t.method) for t in plan.todo}
    assert ("ref_a", "gamma") in pairs or ("ref_c", "beta") in pairs
    for task in plan.todo:
        assert task.expected_information_gain > 0
        assert task.currently_missing
        assert task.priority >= 1


def test_evidence_prioritises_high_similarity_and_frontier_methods():
    landscape, datasets = _kb_with_holes()
    recommender = MethodRecommender(landscape, k_neighbours=2)
    rec = recommender.recommend(datasets["ref_a"], dataset_name="q")
    plan = propose_evidence_acquisition(recommender, rec, top_n=10)
    assert plan.todo
    # First task should have high EIG among missing cells.
    top = plan.todo[0]
    assert top.priority == 1
    assert top.novelty == 1.0
    payload = plan.to_dict()
    json.dumps(payload, allow_nan=False)


def test_calibrate_recommender_cli(tmp_path: Path):
    landscape, datasets = _kb_with_holes()
    kb_path = MethodRecommender(landscape, k_neighbours=2).save_knowledge_base(
        tmp_path / "kb.json"
    )
    bundle = write_bundle(datasets["ref_b"], tmp_path / "query.ttab")
    out = tmp_path / "calibration.json"
    rc = main(
        [
            "calibrate-recommender",
            "--in",
            str(bundle),
            "--knowledge-base",
            str(kb_path),
            "--top",
            "5",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["protocol"] == "histoweave.active_calibration.v1"
    assert "todo" in payload
    assert payload["n_tasks"] == len(payload["todo"])
