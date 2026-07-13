from histoweave.benchmark import deconvolution_task, domain_detection_task, run_benchmark
from histoweave.plugins import list_methods


def test_benchmark_produces_ranked_leaderboard():
    result = run_benchmark(domain_detection_task())
    assert result.task == "domain_detection"
    assert result.leaderboard
    # Ranked best-first, ranks are 1..n.
    ranks = [row["rank"] for row in result.leaderboard]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [row["score"] for row in result.leaderboard]
    assert scores == sorted(scores, reverse=True)


def test_benchmark_best_method_is_reasonable():
    # This is the single, integration-level guard on recovery *accuracy*: it exercises
    # the full normalize -> PCA -> neighbourhood -> clustering -> ARI path end to end.
    # The per-method unit tests intentionally leave accuracy to this test to avoid overlap.
    result = run_benchmark(domain_detection_task())
    best = result.best()
    assert best is not None
    # On the default synthetic sample the best method lands at ARI ~0.99 regardless
    # of which method claims the top spot, so a real regression (a broken normalizer,
    # PCA, or clustering) drops well below this floor.
    assert best["score"] > 0.90
    # At least one method should score well on clean synthetic data.
    high_scorers = [r for r in result.leaderboard if r["score"] > 0.85]
    assert len(high_scorers) >= 2, "fewer than 2 methods score >0.85 on clean data"


def test_benchmark_writes_scores_back_to_registry():
    """After a benchmark run, list_methods surfaces the scores for every method."""
    result = run_benchmark(domain_detection_task())
    methods = {m["name"]: m["benchmark"] for m in list_methods("domain_detection")}
    # Every method that appeared in the leaderboard should have a benchmark entry.
    for row in result.leaderboard:
        method_name = row["method"]
        assert method_name in methods, f"{method_name} missing from registry"
        bench = methods[method_name]
        assert "domain_detection" in bench, f"{method_name} missing benchmark entry"
        entry = bench["domain_detection"]
        assert isinstance(entry["rank"], int)
        assert isinstance(entry["score"], int | float)
        assert entry["score"] >= 0.0, f"{method_name} score is NaN (method crashed)"


def test_deconvolution_benchmark_runs_and_ranks():
    result = run_benchmark(deconvolution_task())
    assert result.task == "deconvolution"
    assert len(result.leaderboard) >= 1
    best = result.best()
    assert best is not None
    assert best["method"] == "marker_deconv"
    # 1 − RMSD: reasonable recovery should score > 0.7 on clean synthetic data.
    assert best["score"] > 0.70
