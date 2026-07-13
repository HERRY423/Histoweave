"""Extended coverage for benchmark harness, landscape, and SVG task paths."""

import numpy as np
import pytest

from histoweave.benchmark import (
    BenchmarkResult,
    LandscapeResult,
    domain_detection_task,
    get_task,
    landscape_svg,
    run_benchmark,
    run_landscape,
    run_multi_landscape,
    run_task_landscape,
    svg_task,
)
from histoweave.datasets.synthetic import (
    make_benchmark_suite,
    make_mixture_synthetic,
    make_synthetic,
)
from histoweave.plugins import MethodCategory

# ── harness ────────────────────────────────────────────────────────────────


def test_harness_svg_task_runs_and_ranks():
    """SVG task finds marker genes via precision@k on synthetic data."""
    data = make_synthetic(seed=1, n_genes=30, marker_genes_per_domain=3)
    task = svg_task(dataset=data, k=5)
    assert task.name == "svg"
    assert task.category == MethodCategory.SPATIALLY_VARIABLE_GENES
    result = run_benchmark(task)
    assert len(result.leaderboard) >= 1
    # At least one method should recover some marker genes on clean data.
    scores = [r["score"] for r in result.leaderboard if r.get("score", -1) > 0]
    assert len(scores) >= 1, "no SVG method recovered any marker genes"


def test_harness_get_task_unknown_key_raises():
    with pytest.raises(KeyError, match="Unknown task 'nonexistent'"):
        get_task("nonexistent")


def test_harness_get_task_returns_each_factory():
    for name in ("domain_detection", "deconvolution", "svg"):
        task = get_task(name)
        assert task.name == name


def test_harness_benchmark_result_best_returns_none_when_empty():
    br = BenchmarkResult(task="test", metric="score", leaderboard=[])
    assert br.best() is None


def test_harness_method_params_passed_to_create_method():
    """method_params dict reaches the method constructor."""
    result = run_benchmark(
        domain_detection_task(),
        methods=["kmeans"],
        method_params={"kmeans": {"n_domains": 3}},
    )
    assert result.leaderboard[0]["method"] == "kmeans"
    assert result.leaderboard[0]["score"] > 0.8


def test_harness_methods_subset_runs_only_requested():
    result = run_benchmark(
        domain_detection_task(),
        methods=["kmeans", "spectral"],
    )
    method_names = {r["method"] for r in result.leaderboard}
    assert method_names == {"kmeans", "spectral"}
    assert len(result.leaderboard) == 2


def test_harness_failing_method_scores_inf():
    """A method that crashes should get -inf score, not crash the harness."""
    # spectral with random_state=None throws on some data; give it invalid params
    result = run_benchmark(
        domain_detection_task(),
        methods=["mean_shift"],  # auto bandwidth may fail on tiny data
    )
    for row in result.leaderboard:
        if row.get("error"):
            assert row["score"] == float("-inf")
        else:
            assert isinstance(row["score"], float)


# ── landscape ──────────────────────────────────────────────────────────────


def test_landscape_result_performance_matrix():
    """performance_matrix() returns (n_datasets, n_methods) array."""
    ds = {
        "a": make_synthetic(seed=0, n_cells=100),
        "b": make_synthetic(seed=1, n_cells=100),
    }
    lr = run_landscape(ds, methods=["kmeans", "spectral"], n_domains_override=3)
    mat = lr.performance_matrix()
    assert mat.shape == (2, 2)
    assert not np.any(np.isnan(mat)), "all methods should succeed on clean data"


def test_landscape_result_method_order():
    ds = {"a": make_synthetic(seed=0, n_cells=100)}
    lr = run_landscape(ds, methods=["kmeans", "spectral"], n_domains_override=3)
    order = lr.method_order()
    assert "kmeans" in order
    assert "spectral" in order


def test_landscape_result_dataset_order():
    ds = {"b": make_synthetic(seed=0, n_cells=100), "a": make_synthetic(seed=0, n_cells=100)}
    lr = run_landscape(ds, methods=["kmeans"], n_domains_override=3)
    assert lr.dataset_order() == ["a", "b"]


def test_landscape_result_summary():
    ds = {"a": make_synthetic(seed=0, n_cells=100)}
    lr = run_landscape(ds, methods=["kmeans"], n_domains_override=3)
    summary = lr.summary()
    assert "Landscape:" in summary
    assert "kmeans" in summary


def test_landscape_svg_renders_with_data():
    ds = {
        "clean": make_synthetic(seed=0, n_cells=100),
        "noisy": make_synthetic(seed=1, n_cells=100, noise=0.6),
    }
    lr = run_landscape(ds, methods=["kmeans", "spectral"], n_domains_override=3)
    svg = landscape_svg(lr)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "<circle" in svg


def test_landscape_svg_renders_empty_fallback():
    lr = LandscapeResult(
        performance={}, features={}, embedding={},
        best_method={}, niches={}, timings={},
    )
    svg = landscape_svg(lr)
    assert "<svg" in svg
    assert "(no data)" in svg


def test_landscape_deconvolution_scoring():
    """_score_result for deconvolution uses proportions_rmsd."""
    data = make_mixture_synthetic(seed=42, n_spots=50, n_genes=20)
    from histoweave.plugins import create_method

    result = create_method("deconvolution", "marker_deconv").run(data.copy())
    from histoweave.benchmark.landscape import _score_result

    score = _score_result(result, data, MethodCategory.DECONVOLUTION)
    assert 0.0 <= score <= 1.0
    assert score > 0.5, "marker_deconv should recover clean mixture proportions"


def test_landscape_svg_scoring_generic_fallback():
    """_score_result falls back to uns['score'] for unknown categories."""
    from histoweave.benchmark.landscape import _score_result

    data = make_synthetic(seed=0, n_cells=50)
    data.uns["score"] = 0.75
    score = _score_result(data, data, "unknown_category")
    assert score == 0.75


def test_landscape_svg_scoring_missing_score():
    """_score_result returns NaN when no score is available."""
    from histoweave.benchmark.landscape import _score_result

    data = make_synthetic(seed=0, n_cells=50)
    if "score" in data.uns:
        del data.uns["score"]
    score = _score_result(data, data, "unknown_category")
    assert np.isnan(score)


def test_landscape_svg_scoring_svg_precision_zero_on_mismatch():
    """SVG scoring returns 0.0 when no marker genes found."""
    from histoweave.benchmark.landscape import _score_result

    data = make_synthetic(seed=0, n_cells=50, n_genes=20)
    # Remove marker genes so no markers match
    data.uns["marker_genes"] = {}
    data.uns["svg"] = {"top_genes": [{"gene": "gene_000"}, {"gene": "gene_001"}]}
    score = _score_result(data, data, MethodCategory.SPATIALLY_VARIABLE_GENES)
    assert score == 0.0


def test_run_multi_landscape_produces_all_tasks():
    ds = {"synth": make_synthetic(seed=0, n_cells=100)}
    mctx = make_mixture_synthetic(seed=0, n_spots=50, n_genes=20)
    datasets = {"synth_domain": ds["synth"], "synth_mix": mctx}
    multi = run_multi_landscape(datasets)
    names = multi.task_names()
    assert "domain_detection" in names
    for name in names:
        lr = multi[name]
        assert isinstance(lr, LandscapeResult)
        assert lr.dataset_count >= 1


def test_landscape_embedding_single_dataset():
    """Embedding works with a single dataset (SVD branch)."""
    ds = {"only": make_synthetic(seed=0, n_cells=100)}
    lr = run_landscape(ds, methods=["kmeans"], n_domains_override=3)
    emb = lr.embedding
    assert "only" in emb
    assert len(emb["only"]) == 2


def test_landscape_niches_include_top_methods():
    ds = {
        "a": make_synthetic(seed=0, n_cells=100),
        "b": make_synthetic(seed=1, n_cells=100),
    }
    lr = run_landscape(ds, methods=["kmeans", "spectral"], n_domains_override=3)
    # Every method's niche should have at least the datasets where it's best
    total_niche = sum(len(v) for v in lr.niches.values())
    assert total_niche >= len(ds), "every dataset should appear in at least one niche"


def test_run_task_landscape_passes_extra_params():
    """extra_params_factory injects per-method params (n_domains)."""
    ds = {"a": make_synthetic(seed=0, n_cells=600, n_domains=3)}
    lr = run_task_landscape(
        ds,
        category=MethodCategory.DOMAIN_DETECTION,
        methods=["kmeans"],
        extra_params_factory=lambda data: {"n_domains": int(data.obs["domain_truth"].nunique())},
    )
    # n_domains passed from factory produces correct clustering on clean data
    assert lr.performance["a"]["kmeans"] > 0.8


# ── benchmark suite ────────────────────────────────────────────────────────


def test_benchmark_suite_all_presets():
    suite = make_benchmark_suite(seed=42)
    # 6 blob/grid + 2 tumor + 2 developmental = 10 presets
    assert len(suite.datasets) == 10
    expected_base = {
        "clean_easy", "noisy_hard", "many_small_domains",
        "few_large_domains", "dense_regular", "sparse_scattered",
    }
    expected_tissue = {"tumor_mimic", "devel_gradient", "tumor_noisy", "devel_branching"}
    assert set(suite.datasets) >= expected_base
    assert expected_tissue <= set(suite.datasets)
    for _name, ds in suite.datasets.items():
        assert ds.X.shape[0] > 0
        assert "domain_truth" in ds.obs
        assert "spatial" in ds.obsm


def test_benchmark_suite_presets_have_expected_domain_counts():
    suite = make_benchmark_suite(seed=42)
    domain_counts = {
        "clean_easy": 3, "noisy_hard": 3, "many_small_domains": 6,
        "few_large_domains": 2, "dense_regular": 4, "sparse_scattered": 3,
    }
    for name, expected_n in domain_counts.items():
        actual = suite.datasets[name].obs["domain_truth"].nunique()
        assert actual == expected_n, f"{name}: expected {expected_n} domains, got {actual}"


def test_benchmark_suite_len_and_iter():
    suite = make_benchmark_suite(seed=42)
    assert len(suite) == 10
    names = [n for n, _ in suite]
    assert len(names) == 10


# ── features ────────────────────────────────────────────────────────────────


def test_feature_extraction_on_grid_layout():
    """Feature extraction works on grid layout (different code path)."""
    from histoweave.benchmark.features import (
        RECOMMENDATION_FEATURE_ORDER,
        extract_features,
        feature_vector,
    )

    data = make_synthetic(seed=99, n_cells=200, layout="grid")
    feats = extract_features(data, include_domain=False)
    # Target-free feature vector uses RECOMMENDATION_FEATURE_ORDER (16 dims).
    vec = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
    assert vec.shape[0] == len(RECOMMENDATION_FEATURE_ORDER)
    assert not np.any(np.isnan(vec)), "all features should be finite on grid data"


def test_feature_dataframe_output():
    """feature_dataframe returns a proper DataFrame from a dict of datasets."""
    from histoweave.benchmark.features import feature_dataframe

    data = make_synthetic(seed=42, n_cells=100, n_genes=30)
    df = feature_dataframe({"synth": data})
    assert df.shape[0] == 1
    assert "n_obs" in df.columns


def test_features_on_tiny_dataset():
    """Features are stable on minimal data (n=20 cells)."""
    from histoweave.benchmark.features import (
        RECOMMENDATION_FEATURE_ORDER,
        extract_features,
        feature_vector,
    )

    data = make_synthetic(seed=7, n_cells=20, n_genes=10, n_domains=2)
    feats = extract_features(data, include_domain=False)
    vec = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
    assert vec.shape[0] == len(RECOMMENDATION_FEATURE_ORDER)
    # Some features may be NaN on tiny data (effective_rank); that's acceptable.
