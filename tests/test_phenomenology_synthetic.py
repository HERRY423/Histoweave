from __future__ import annotations

import numpy as np
import pytest

from histoweave.datasets import (
    ConditionSpec,
    ObservationCondition,
    PhenomenonSpec,
    ScenarioManifest,
    SpatialPhenomenon,
    default_scenario_manifest,
    make_phenomenology_scenario,
    make_phenomenology_suite,
)


def _manifest(
    phenomenon: SpatialPhenomenon,
    condition: ObservationCondition = ObservationCondition.CLEAN,
    *,
    replicate: int = 0,
) -> ScenarioManifest:
    return default_scenario_manifest(
        phenomenon,
        condition,
        replicate=replicate,
        seed=1234,
        n_obs=80,
        n_genes=64,
        image_size=48,
    )


def test_manifest_hash_is_stable_and_condition_sensitive() -> None:
    first = _manifest(SpatialPhenomenon.GRADIENT)
    same = _manifest(SpatialPhenomenon.GRADIENT)
    changed = _manifest(
        SpatialPhenomenon.GRADIENT,
        ObservationCondition.LOW_DEPTH_DROPOUT,
    )
    assert first.manifest_hash == same.manifest_hash
    assert first.manifest_hash != changed.manifest_hash
    assert len(first.manifest_hash) == 64


def test_manifest_validation() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        PhenomenonSpec(SpatialPhenomenon.GRADIENT, effect_size=-1)
    with pytest.raises(ValueError, match="target_zero_fraction"):
        ConditionSpec(ObservationCondition.CLEAN, target_zero_fraction=1.1)
    with pytest.raises(ValueError, match="n_obs"):
        ScenarioManifest(
            PhenomenonSpec(SpatialPhenomenon.GRADIENT),
            ConditionSpec(ObservationCondition.CLEAN),
            replicate=0,
            seed=1,
            n_obs=10,
        )


def test_scenario_is_byte_reproducible_for_key_arrays() -> None:
    manifest = _manifest(SpatialPhenomenon.HOTSPOT)
    first = make_phenomenology_scenario(manifest)
    second = make_phenomenology_scenario(manifest)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.obsm["spatial"], second.obsm["spatial"])
    np.testing.assert_array_equal(
        first.images["synthetic_tissue"], second.images["synthetic_tissue"]
    )
    np.testing.assert_array_equal(
        first.images["segmentation_truth"], second.images["segmentation_truth"]
    )
    assert first.uns["scenario_manifest_hash"] == manifest.manifest_hash


@pytest.mark.parametrize("phenomenon", list(SpatialPhenomenon))
def test_all_phenomena_expose_standard_truth_schema(phenomenon: SpatialPhenomenon) -> None:
    table = make_phenomenology_scenario(_manifest(phenomenon))
    expected_obs = {
        "domain_truth",
        "continuous_truth",
        "hotspot_truth",
        "boundary_distance_truth",
        "branch_truth",
        "pseudotime_truth",
        "cell_type_truth",
        "qc_truth",
        "batch",
    }
    assert expected_obs <= set(table.obs.columns)
    assert table.shape == (80, 64)
    assert table.obsm["proportions_truth"].shape == (80, 4)
    np.testing.assert_allclose(table.obsm["proportions_truth"].sum(axis=1), 1.0)
    assert table.images["synthetic_tissue"].shape == (48, 48, 3)
    assert table.images["segmentation_truth"].shape == (48, 48)
    assert np.asarray(table.X).min() >= 0
    assert set(table.uns["marker_genes"]) == {f"cell_type_{idx}" for idx in range(4)}
    assert np.asarray(table.uns["truth_graph_edges"]).shape[1] == 2


def test_non_sampling_conditions_share_biological_truth_and_coordinates() -> None:
    clean = make_phenomenology_scenario(_manifest(SpatialPhenomenon.BOUNDARY))
    noisy = make_phenomenology_scenario(
        _manifest(SpatialPhenomenon.BOUNDARY, ObservationCondition.LOW_SIGNAL_NOISE)
    )
    np.testing.assert_array_equal(clean.obsm["spatial"], noisy.obsm["spatial"])
    np.testing.assert_array_equal(clean.obs["domain_truth"], noisy.obs["domain_truth"])
    np.testing.assert_array_equal(
        clean.obs["boundary_distance_truth"], noisy.obs["boundary_distance_truth"]
    )
    assert not np.array_equal(clean.X, noisy.X)


def test_low_depth_condition_reaches_registered_dropout_and_depth_loss() -> None:
    clean = make_phenomenology_scenario(_manifest(SpatialPhenomenon.COMPARTMENT))
    low = make_phenomenology_scenario(
        _manifest(SpatialPhenomenon.COMPARTMENT, ObservationCondition.LOW_DEPTH_DROPOUT)
    )
    clean_counts = np.asarray(clean.X)
    low_counts = np.asarray(low.X)
    assert np.mean(low_counts == 0) >= 0.80
    assert np.median(low_counts.sum(axis=1)) < np.median(clean_counts.sum(axis=1))


def test_irregular_sampling_changes_density_and_retains_observation_count() -> None:
    clean = make_phenomenology_scenario(_manifest(SpatialPhenomenon.GRADIENT))
    irregular = make_phenomenology_scenario(
        _manifest(SpatialPhenomenon.GRADIENT, ObservationCondition.IRREGULAR_SAMPLING)
    )
    clean_x = clean.obsm["spatial"][:, 0]
    irregular_x = irregular.obsm["spatial"][:, 0]
    assert irregular.n_obs == clean.n_obs
    assert not np.array_equal(irregular_x, clean_x)
    clean_ratio = np.mean(clean_x < 55) / np.mean(clean_x >= 55)
    irregular_ratio = np.mean(irregular_x < 55) / np.mean(irregular_x >= 55)
    assert irregular_ratio > clean_ratio


def test_batch_condition_is_partial_not_complete_confounding() -> None:
    table = make_phenomenology_scenario(
        _manifest(
            SpatialPhenomenon.COMPARTMENT,
            ObservationCondition.BATCH_PLATFORM_CONFOUNDING,
        )
    )
    batches = table.obs["batch"].to_numpy()
    domains = table.obs["domain_truth"].to_numpy()
    assert set(batches) == {"batch_0", "batch_1"}
    for domain in np.unique(domains):
        assert len(set(batches[domains == domain])) == 2
    assert table.var["batch_shift_truth"].sum() == round(table.n_vars * 0.20)


def test_one_seed_suite_has_exactly_six_by_five_scenarios() -> None:
    suite = make_phenomenology_suite(seeds=(11,), n_obs=40, n_genes=64, image_size=32)
    assert len(suite) == len(SpatialPhenomenon) * len(ObservationCondition) == 30
    assert len({table.uns["scenario_manifest_hash"] for table in suite.values()}) == 30
