import json

import numpy as np
import pytest

from histoweave.benchmark import (
    RECOMMENDATION_FEATURE_ORDER,
    LandscapeResult,
    MethodRecommender,
    extract_features,
    feature_vector,
)
from histoweave.benchmark.features import _expression_features, _hopkins_statistic
from histoweave.cli import main
from histoweave.datasets import make_synthetic
from histoweave.io import write_bundle


def _knowledge_base() -> tuple[LandscapeResult, dict[str, object]]:
    datasets = {
        "small_clean": make_synthetic(
            n_cells=70, n_genes=18, noise=0.10, marker_gene_lift=9.0, seed=11
        ),
        "medium_noisy": make_synthetic(
            n_cells=95, n_genes=22, noise=0.45, marker_gene_lift=4.0, seed=12
        ),
        "large_regular": make_synthetic(
            n_cells=120,
            n_genes=26,
            n_domains=4,
            noise=0.20,
            marker_gene_lift=7.0,
            layout="grid",
            seed=13,
        ),
    }
    features = {
        name: feature_vector(
            extract_features(data, include_domain=False),
            order=RECOMMENDATION_FEATURE_ORDER,
        )
        for name, data in datasets.items()
    }
    performance = {
        "small_clean": {"alpha": 0.95, "beta": float("nan")},
        "medium_noisy": {"alpha": 0.45, "beta": 0.82},
        "large_regular": {"alpha": 0.55, "beta": 0.75},
    }
    result = LandscapeResult(
        performance=performance,
        features=features,
        embedding={},
        best_method={
            "small_clean": "alpha",
            "medium_noisy": "beta",
            "large_regular": "beta",
        },
        niches={"alpha": ["small_clean"], "beta": ["medium_noisy", "large_regular"]},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=2,
        dataset_count=3,
        task="domain_detection",
        metric="ARI",
    )
    return result, datasets


def test_expression_entropy_is_scale_invariant():
    expression = np.array([[1.0, 3.0, 0.0], [2.0, 1.0, 4.0]])
    first = _expression_features(expression)["expression_entropy"]
    second = _expression_features(expression * 100.0)["expression_entropy"]
    assert first == pytest.approx(second)
    assert 0.0 <= first <= np.log2(expression.shape[1])


def test_hopkins_excludes_sampled_point_itself():
    rng = np.random.default_rng(2)
    uniform = rng.uniform(0, 1, size=(200, 2))
    clustered = np.vstack(
        [
            rng.normal((0.2, 0.2), 0.025, size=(100, 2)),
            rng.normal((0.8, 0.8), 0.025, size=(100, 2)),
        ]
    )
    uniform_score = _hopkins_statistic(uniform)
    clustered_score = _hopkins_statistic(clustered)
    assert 0.30 < uniform_score < 0.70
    assert clustered_score > uniform_score


def test_recommender_uses_target_free_features_and_reference_scaler():
    knowledge_base, datasets = _knowledge_base()
    query = datasets["small_clean"].copy()
    recommender = MethodRecommender(knowledge_base, k_neighbours=1)
    before = recommender.recommend(query, dataset_name="query")

    query.obs["domain"] = "leaked-label"
    after = recommender.recommend(query, dataset_name="query")

    assert before.neighbours[0]["name"] == "small_clean"
    assert before.neighbours[0]["similarity"] == pytest.approx(1.0)
    assert before.best() is not None
    assert before.best().method == "alpha"
    assert before.feature_order == RECOMMENDATION_FEATURE_ORDER
    assert before.feature_vector == pytest.approx(after.feature_vector)


def test_missing_method_score_uses_method_specific_weight_and_roundtrips(tmp_path):
    knowledge_base, datasets = _knowledge_base()
    recommender = MethodRecommender(knowledge_base, k_neighbours=3)
    recommendation = recommender.recommend(datasets["small_clean"])
    beta = next(method for method in recommendation.ranked_methods if method.method == "beta")
    assert 0.75 <= beta.score <= 0.82
    assert beta.support == 2
    assert 0.0 < beta.coverage < 1.0

    path = recommender.save_knowledge_base(tmp_path / "knowledge.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert raw["performance"]["small_clean"]["beta"] is None
    loaded = MethodRecommender(path, k_neighbours=3).recommend(datasets["small_clean"])
    assert loaded.to_dict()["ranked_methods"] == recommendation.to_dict()["ranked_methods"]


def test_recommend_cli_json_and_output_file(tmp_path, capsys):
    knowledge_base, datasets = _knowledge_base()
    kb_path = MethodRecommender(knowledge_base).save_knowledge_base(tmp_path / "kb.json")
    bundle = write_bundle(datasets["small_clean"], tmp_path / "query.ttab")
    output = tmp_path / "recommendation.json"

    rc = main(
        [
            "recommend",
            "--in",
            str(bundle),
            "--knowledge-base",
            str(kb_path),
            "--dataset-name",
            "held_out",
            "--json",
            "--out",
            str(output),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["dataset_name"] == "held_out"
    assert payload["feature_order"] == RECOMMENDATION_FEATURE_ORDER
    assert payload["neighbours"][0]["name"] == "small_clean"
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_recommend_cli_rejects_invalid_k(tmp_path, capsys):
    knowledge_base, datasets = _knowledge_base()
    kb_path = MethodRecommender(knowledge_base).save_knowledge_base(tmp_path / "kb.json")
    bundle = write_bundle(datasets["small_clean"], tmp_path / "query.ttab")
    rc = main(
        [
            "recommend",
            "--in",
            str(bundle),
            "--knowledge-base",
            str(kb_path),
            "--k-neighbours",
            "0",
        ]
    )
    assert rc == 2
    assert "k_neighbours" in capsys.readouterr().err


def test_empty_knowledge_base_fails_closed(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "feature_order": list(RECOMMENDATION_FEATURE_ORDER),
                "features": {},
                "performance": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least two datasets"):
        MethodRecommender(path)
