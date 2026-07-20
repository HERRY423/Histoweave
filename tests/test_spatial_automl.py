"""Spatial AutoML compiler tests."""

from __future__ import annotations

import json
from pathlib import Path

from histoweave.automl import compute_pareto_front, run_spatial_automl
from histoweave.automl.compiler import MethodRunResult
from histoweave.benchmark import (
    RECOMMENDATION_FEATURE_ORDER,
    LandscapeResult,
    extract_features,
    feature_vector,
)
from histoweave.cli import main
from histoweave.datasets import make_synthetic
from histoweave.io import write_bundle


def _tiny_knowledge_base(path: Path) -> Path:
    datasets = {
        "ref_a": make_synthetic(
            n_cells=80, n_genes=20, noise=0.15, marker_gene_lift=8.0, seed=21
        ),
        "ref_b": make_synthetic(
            n_cells=100, n_genes=24, noise=0.40, marker_gene_lift=4.0, seed=22
        ),
        "ref_c": make_synthetic(
            n_cells=110,
            n_genes=28,
            n_domains=4,
            noise=0.22,
            marker_gene_lift=7.0,
            layout="grid",
            seed=23,
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
        "ref_a": {"kmeans": 0.92, "spectral": 0.88, "gaussian_mixture": 0.90},
        "ref_b": {"kmeans": 0.55, "spectral": 0.78, "gaussian_mixture": 0.72},
        "ref_c": {"kmeans": 0.70, "spectral": 0.85, "gaussian_mixture": 0.80},
    }
    landscape = LandscapeResult(
        performance=performance,
        features=features,
        embedding={},
        best_method={"ref_a": "kmeans", "ref_b": "spectral", "ref_c": "spectral"},
        niches={"kmeans": ["ref_a"], "spectral": ["ref_b", "ref_c"]},
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=3,
        dataset_count=3,
        task="domain_detection",
        metric="ARI",
    )
    kb = path / "landscape.json"
    # MethodRecommender accepts LandscapeResult or path; write via save_knowledge_base.
    from histoweave.benchmark.recommend import MethodRecommender

    MethodRecommender(landscape, k_neighbours=2).save_knowledge_base(kb)
    return kb


def test_compute_pareto_front_marks_non_dominated():
    runs = [
        MethodRunResult(
            method="fast",
            success=True,
            seconds=0.1,
            quality_score=0.5,
            spatial_coherence=0.5,
            silhouette=0.2,
            consensus_agreement=0.4,
            recommendation_score=0.5,
        ),
        MethodRunResult(
            method="quality",
            success=True,
            seconds=2.0,
            quality_score=0.9,
            spatial_coherence=0.9,
            silhouette=0.6,
            consensus_agreement=0.8,
            recommendation_score=0.9,
        ),
        MethodRunResult(
            method="fail",
            success=False,
            seconds=0.01,
            error="boom",
        ),
    ]
    points = compute_pareto_front(runs)
    by_name = {p.method: p for p in points}
    assert by_name["quality"].is_pareto
    assert by_name["fast"].is_pareto
    assert by_name["quality"].pareto_rank == 1
    assert by_name["fast"].pareto_rank == 1
    assert by_name["fast"].objectives["speed"] == 0.1
    assert by_name["fail"].pareto_rank == 99


def test_run_spatial_automl_end_to_end(tmp_path: Path):
    kb = _tiny_knowledge_base(tmp_path)
    data = make_synthetic(n_cells=90, n_genes=22, n_domains=3, seed=30)
    result = run_spatial_automl(
        data,
        "Find spatial domains for my Visium liver cancer data.",
        knowledge_base=kb,
        dataset_name="visium_hcc",
        top_k=2,
        methods=["kmeans", "spectral"],
        use_compiler=True,
        compiler_model="mock",
        seed=0,
        out_dir=tmp_path / "automl",
        write_report=True,
        platform="visium",
    )
    assert result.ranked_methods
    assert set(result.ranked_methods) <= {"kmeans", "spectral"}
    assert any(r.success for r in result.method_runs)
    payload = result.to_dict()
    json.dumps(payload, allow_nan=False, default=str)
    report = tmp_path / "automl" / "automl_report.html"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "AutoML" in text or "Pareto" in text
    assert result.neighbours
    assert result.decision_card is not None
    assert result.decision_card["action"] in {"global_default", "evidence_required"}
    assert (tmp_path / "automl" / "decision_card.json").is_file()


def test_automl_cli(tmp_path: Path):
    kb = _tiny_knowledge_base(tmp_path)
    data = make_synthetic(n_cells=75, n_genes=20, seed=11)
    bundle = tmp_path / "data.ttab"
    write_bundle(data, bundle)
    out = tmp_path / "cli_automl"
    rc = main(
        [
            "automl",
            "Find spatial domains for my Visium data.",
            "--in",
            str(bundle),
            "--knowledge-base",
            str(kb),
            "--out-dir",
            str(out),
            "--methods",
            "kmeans,spectral",
            "--top",
            "2",
            "--no-compiler",
            "--json",
        ]
    )
    assert rc == 0
    assert (out / "automl_result.json").is_file()
    assert (out / "automl_report.html").is_file()
