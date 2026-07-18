"""Digital-twin generator and validation tests."""

from __future__ import annotations

import json
from pathlib import Path

from histoweave.benchmark import run_digital_twin_validation
from histoweave.cli import main
from histoweave.datasets import (
    TWIN_MATCH_FEATURES,
    make_digital_twin,
    make_synthetic,
)
from histoweave.io import write_bundle


def test_make_digital_twin_plants_truth_and_matches_shape():
    real = make_synthetic(n_cells=120, n_genes=40, n_domains=4, seed=7)
    # Drop truth to mimic an unlabelled user upload.
    real.obs = real.obs.drop(columns=["domain_truth"])
    pack = make_digital_twin(real, seed=0, n_trials=6, max_cells=120, max_genes=40)
    twin = pack.twin
    assert "domain_truth" in twin.obs.columns
    assert twin.n_obs <= 120
    assert twin.n_vars <= 40
    assert twin.spatial is not None
    assert pack.match.match_cosine >= 0.0
    assert set(pack.match.feature_order) == set(TWIN_MATCH_FEATURES)
    assert len(TWIN_MATCH_FEATURES) == 13
    payload = pack.to_dict()
    assert payload["n_domains"] >= 2
    # JSON-safe (no NaN)
    json.dumps(payload, allow_nan=False)


def test_digital_twin_validation_ranks_methods(tmp_path: Path):
    real = make_synthetic(n_cells=90, n_genes=30, n_domains=3, seed=3)
    real.obs = real.obs.drop(columns=["domain_truth"])
    result = run_digital_twin_validation(
        real,
        methods=["kmeans", "spectral"],
        dataset_name="toy",
        seed=1,
        n_trials=4,
        max_cells=90,
        max_genes=30,
        out_dir=tmp_path / "twin_out",
        write_report=True,
    )
    assert result.predicted_ranking
    assert result.best_method() in {"kmeans", "spectral"}
    assert (tmp_path / "twin_out" / "digital_twin_validation.json").is_file()
    assert (tmp_path / "twin_out" / "digital_twin_report.html").is_file()
    html = (tmp_path / "twin_out" / "digital_twin_report.html").read_text(encoding="utf-8")
    assert "Digital-twin" in html or "digital-twin" in html.lower()
    # At least one finite score expected on easy synthetic-like twin.
    scores = [r.get("score") for r in result.leaderboard if r.get("score") is not None]
    assert scores
    assert max(float(s) for s in scores) > 0.0


def test_digital_twin_cli(tmp_path: Path):
    data = make_synthetic(n_cells=70, n_genes=24, seed=5)
    data.obs = data.obs.drop(columns=["domain_truth"])
    bundle = tmp_path / "data.ttab"
    write_bundle(data, bundle)
    out = tmp_path / "cli_twin"
    rc = main(
        [
            "digital-twin",
            "--in",
            str(bundle),
            "--out-dir",
            str(out),
            "--methods",
            "kmeans",
            "--n-trials",
            "3",
            "--max-cells",
            "70",
            "--max-genes",
            "24",
            "--json",
        ]
    )
    assert rc == 0
    assert (out / "leaderboard.json").is_file()


def test_twin_preserves_coordinate_count_when_capped():
    real = make_synthetic(n_cells=200, n_genes=50, seed=9)
    pack = make_digital_twin(real, seed=2, max_cells=80, max_genes=25, n_trials=3)
    assert pack.twin.n_obs == 80
    assert pack.twin.n_vars == 25
    assert pack.twin.spatial is not None
    assert len(pack.twin.spatial) == 80
