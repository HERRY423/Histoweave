import csv
import hashlib
import json

from histoweave.benchmark import (
    FIGURE3_DATASETS,
    FIGURE3_METHODS,
    run_figure3_experiment,
)
from histoweave.datasets.synthetic import make_benchmark_suite
from histoweave.plugins import list_methods


def test_figure3_protocol_has_ten_runnable_methods_and_three_datasets():
    registered = {
        item["name"]
        for item in list_methods("domain_detection")
        if item["language"] == "python"
    }
    assert len(FIGURE3_METHODS) == 10
    assert len(set(FIGURE3_METHODS)) == 10
    assert set(FIGURE3_METHODS) <= registered
    assert FIGURE3_DATASETS == ("clean_easy", "noisy_hard", "sparse_scattered")


def test_benchmark_suite_seed_is_deterministic():
    first = make_benchmark_suite(seed=17)
    second = make_benchmark_suite(seed=17)
    for name in FIGURE3_DATASETS:
        assert (first.datasets[name].X == second.datasets[name].X).all()
        assert (
            first.datasets[name].obsm["spatial"]
            == second.datasets[name].obsm["spatial"]
        ).all()


def test_figure3_experiment_emits_validated_artifacts(tmp_path):
    result = run_figure3_experiment(tmp_path, seed=9)

    matrix = list(
        csv.DictReader(result.performance_matrix_path.open(encoding="utf-8"))
    )
    assert len(matrix) == 3
    assert list(matrix[0]) == ["dataset", *FIGURE3_METHODS]

    long_rows = list(
        csv.DictReader(result.benchmark_long_path.open(encoding="utf-8"))
    )
    assert len(long_rows) == 30
    assert all(row["status"] == "success" for row in long_rows)

    recommendation = json.loads(result.recommendation_path.read_text(encoding="utf-8"))
    assert recommendation["summary"]["n_queries"] == 3
    assert 0.0 <= recommendation["summary"]["top1_accuracy"] <= 1.0
    assert 0.0 <= recommendation["summary"]["top3_accuracy"] <= 1.0

    recommendation_csv = list(
        csv.DictReader(result.recommendation_csv_path.open(encoding="utf-8"))
    )
    assert len(recommendation_csv) == 3

    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "share_with_caveats"
    assert all(validation["checks"].values())

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 8
    for artifact in manifest["artifacts"]:
        path = tmp_path / artifact["path"]
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
