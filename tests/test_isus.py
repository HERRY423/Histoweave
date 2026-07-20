from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from histoweave.benchmark.isus import (
    assess_isus_predictor,
    attach_gain_prediction,
    compute_isus,
    compute_isus_from_table,
    extract_spatial_ari_gains_from_long,
    fit_isus_gain_calibration,
    isus_band,
    isus_band_from_permutation,
    mi_discrete_continuous,
    predict_expected_spatial_ari_gain,
)


def test_isus_distinguishes_expression_only_and_spatial_signal():
    rng = np.random.default_rng(0)
    n_obs, n_domains = 600, 3
    labels = rng.integers(0, n_domains, size=n_obs)
    centers = rng.normal(size=(n_domains, 8)) * 3.0
    expression = centers[labels] + rng.normal(size=(n_obs, 8))
    expression_only = compute_isus(
        expression,
        rng.uniform(0, 100, size=(n_obs, 2)),
        labels,
        n_pcs=6,
    )

    coordinates = rng.uniform(0, 100, size=(n_obs, 2))
    spatial_labels = np.clip((coordinates[:, 0] // 34).astype(int), 0, 2)
    weak_centers = rng.normal(size=(n_domains, 8)) * 0.35
    weak_expression = weak_centers[spatial_labels] + rng.normal(size=(n_obs, 8))
    spatial = compute_isus(
        weak_expression,
        coordinates,
        spatial_labels,
        n_pcs=6,
    )

    assert expression_only.isus is not None and expression_only.isus < 0.1
    assert expression_only.band == "expression-sufficient"
    assert expression_only.band_source == "heuristic_absolute"
    assert spatial.isus is not None and spatial.isus > 0.3
    assert spatial.band == "spatial-critical"
    payload = spatial.to_dict()
    assert payload["not_a_pre_predictor"] is True
    assert payload["role"] == "posthoc_label_conditioned_descriptor"
    assert "thresholds" in payload
    json.dumps(payload, allow_nan=False)


def test_public_mi_estimator_and_bands():
    labels = np.repeat([0, 1], 20)
    separated = np.r_[np.zeros((20, 1)), np.ones((20, 1)) * 10]
    assert mi_discrete_continuous(separated, labels) > 0
    assert isus_band(None) == "undetermined"
    assert isus_band(0.05) == "expression-sufficient"
    assert isus_band(0.2) == "modest-spatial-signal"
    assert isus_band(0.4) == "spatial-critical"
    assert isus_band_from_permutation(p_value=0.4, z_score=0.5) == "not_above_null"
    assert isus_band_from_permutation(p_value=0.01, z_score=1.5) == "modest-spatial-signal"
    assert isus_band_from_permutation(p_value=0.01, z_score=4.0) == "spatial-critical"


def test_compute_from_anndata_like_sparse_table():
    rng = np.random.default_rng(4)
    n_obs = 90
    coordinates = rng.uniform(size=(n_obs, 2))
    labels = (coordinates[:, 0] > 0.5).astype(int)
    table = SimpleNamespace(
        X=csr_matrix(rng.poisson(2, size=(n_obs, 12))),
        obs=pd.DataFrame({"domain_truth": labels}),
        obsm={"spatial": coordinates},
        uns={"dataset_name": "sparse-toy"},
    )
    result = compute_isus_from_table(table, n_pcs=5)
    assert result.dataset == "sparse-toy"
    assert result.n_obs == n_obs
    assert result.n_pcs == 5


def test_coordinate_shuffle_null_reports_p_and_z():
    rng = np.random.default_rng(1)
    n_obs, n_domains = 240, 3
    coordinates = rng.uniform(0, 100, size=(n_obs, 2))
    labels = np.clip((coordinates[:, 0] // 34).astype(int), 0, n_domains - 1)
    weak_centers = rng.normal(size=(n_domains, 6)) * 0.3
    expression = weak_centers[labels] + rng.normal(size=(n_obs, 6))

    result = compute_isus(
        expression,
        coordinates,
        labels,
        n_pcs=5,
        n_null=29,
        seed=2,
        alpha=0.05,
    )
    assert result.n_null == 29
    assert result.null_control == "coordinate_shuffle"
    assert result.p_value_i_d_s_given_e is not None
    assert result.p_value_i_d_s_given_e < 0.05
    assert result.z_score_i_d_s_given_e is not None
    assert result.z_score_i_d_s_given_e > 0
    assert result.significant is True
    assert result.band_source == "permutation_z"
    assert result.band in {"modest-spatial-signal", "spatial-critical"}
    assert result.threshold_significant_isus is not None
    payload = result.to_dict()
    assert payload["z_score_i_d_s_given_e"] is not None
    assert payload["band_heuristic"] is not None
    json.dumps(payload, allow_nan=False)

    # Purely expression-driven labels with random coords should not beat the null.
    expr_labels = rng.integers(0, n_domains, size=n_obs)
    centers = rng.normal(size=(n_domains, 6)) * 3.0
    expr = centers[expr_labels] + rng.normal(size=(n_obs, 6))
    nullish = compute_isus(
        expr,
        rng.uniform(0, 100, size=(n_obs, 2)),
        expr_labels,
        n_pcs=5,
        n_null=29,
        seed=3,
        alpha=0.05,
    )
    assert nullish.p_value_i_d_s_given_e is not None
    assert nullish.p_value_i_d_s_given_e > 0.05
    assert nullish.significant is False
    assert nullish.band == "not_above_null"
    assert nullish.band_source == "permutation_z"


def test_extract_spatial_ari_gains_from_bundled_benchmark_long():
    path = Path(__file__).resolve().parents[1] / "5x15_spatial_aware" / "benchmark_long.csv"
    if not path.is_file():
        return
    gains = extract_spatial_ari_gains_from_long(path)
    assert len(gains) >= 2
    assert all(math_isfinite(v) for v in gains.values())


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


def test_fit_gain_calibration_and_predict():
    rng = np.random.default_rng(0)
    n = 12
    isus = np.linspace(0.05, 0.5, n)
    gain = 0.05 + 0.4 * isus + rng.normal(0, 0.005, size=n)
    records = [
        {
            "dataset": f"s{i}",
            "isus": float(isus[i]),
            "spatial_ari_gain": float(gain[i]),
        }
        for i in range(n)
    ]
    calib = fit_isus_gain_calibration(records, min_slices=8)
    assert calib.slope is not None and calib.slope > 0
    assert calib.reliability == "moderate"
    assert calib.r_squared is not None and calib.r_squared > 0.5
    pred = predict_expected_spatial_ari_gain(0.3, calib)
    assert pred.expected_spatial_ari_gain is not None
    assert pred.expected_spatial_ari_gain_low is not None
    assert pred.expected_spatial_ari_gain_high is not None
    assert pred.expected_spatial_ari_gain_low < pred.expected_spatial_ari_gain
    json.dumps(calib.to_dict(), allow_nan=False)
    json.dumps(pred.to_dict(), allow_nan=False)

    # Attach to an ISUSResult with a finite ISUS point estimate
    labels = np.repeat([0, 1], 40)
    centers = rng.normal(size=(2, 6)) * 2.5
    expression = centers[labels] + rng.normal(size=(80, 6))
    toy = compute_isus(
        expression,
        rng.uniform(size=(80, 2)),
        labels,
        n_pcs=4,
        n_null=0,
    )
    assert toy.isus is not None
    attached = attach_gain_prediction(toy, calib)
    assert attached.expected_spatial_ari_gain is not None
    assert attached.gain_prediction_reliability == "moderate"
    assert "downstream_gain" in attached.to_dict()


def test_gain_calibration_marks_unsupported_on_negative_correlation():
    records = [
        {"dataset": "a", "isus": 0.24, "spatial_ari_gain": 0.125},
        {"dataset": "b", "isus": 0.42, "spatial_ari_gain": 0.096},
        {"dataset": "c", "isus": 0.30, "spatial_ari_gain": 0.198},
        {"dataset": "d", "isus": 0.08, "spatial_ari_gain": 0.130},
        {"dataset": "e", "isus": 0.10, "spatial_ari_gain": 0.115},
    ]
    calib = fit_isus_gain_calibration(records)
    assert calib.reliability in {"unsupported", "low"}
    assert calib.slope is not None  # still fitted for honesty / exploration
    pred = calib.predict(0.2)
    assert pred.expected_spatial_ari_gain is not None
    assert pred.reliability in {"unsupported", "low"}


def test_assess_isus_predictor_reports_failure_and_gaps():
    records = [
        {"isus": 0.24, "spatial_ari_gain": 0.125},
        {"isus": 0.42, "spatial_ari_gain": 0.096},
        {"isus": 0.30, "spatial_ari_gain": 0.198},
        {"isus": 0.08, "spatial_ari_gain": 0.130},
        {"isus": 0.10, "spatial_ari_gain": 0.115},
    ]
    assessment = assess_isus_predictor(records)
    assert assessment.n_slices == 5
    assert assessment.spearman_rho is not None
    assert assessment.spearman_rho < 0
    assert assessment.predictor_status == "underpowered"
    assert "sample_size_below_minimum" in assessment.failure_reasons
    assert "correlation_nonpositive" in assessment.failure_reasons
    assert "correlation_not_significant" in assessment.failure_reasons
    payload = assessment.to_dict()
    assert "spearman_pvalue" in payload
    json.dumps(payload, allow_nan=False)


def test_assess_isus_predictor_supported_when_positive_and_large():
    rng = np.random.default_rng(0)
    n = 12
    isus = np.linspace(0.05, 0.5, n) + rng.normal(0, 0.01, size=n)
    gain = 0.05 + 0.4 * isus + rng.normal(0, 0.005, size=n)
    records = [
        {"isus": float(i), "spatial_ari_gain": float(g)} for i, g in zip(isus, gain, strict=True)
    ]
    assessment = assess_isus_predictor(records, min_slices=8)
    assert assessment.predictor_status == "supported"
    assert assessment.spearman_rho is not None and assessment.spearman_rho > 0
    assert assessment.spearman_pvalue is not None and assessment.spearman_pvalue < 0.05


def test_assess_isus_predictor_failed_when_powered_but_nonpositive():
    n = 10
    records = [
        {"isus": float(i), "spatial_ari_gain": float(1.0 - 0.05 * i)} for i in range(n)
    ]
    assessment = assess_isus_predictor(records, min_slices=8)
    assert assessment.predictor_status == "failed"
    assert "correlation_nonpositive" in assessment.failure_reasons
