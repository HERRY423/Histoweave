"""Donor-stratified bootstrap for L3 discovery panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from histoweave.benchmark.donor_bootstrap import (
    DLPFC_SECTION_TO_DONOR,
    donor_for_slice,
    donor_stratified_bootstrap_l3,
)


def _toy_frame() -> pd.DataFrame:
    """Three donors, two direction_ok L3 components each."""
    rows = []
    # Br5292
    rows.append(
        dict(
            label="a",
            slice_id="dlpfc_151508",
            expected_class="L3_program",
            direction_ok=True,
            l3_delta_rest=0.40,
            myelin_delta_rest=-0.30,
            n=100,
        )
    )
    rows.append(
        dict(
            label="b",
            slice_id="dlpfc_151507",
            expected_class="L3_program",
            direction_ok=True,
            l3_delta_rest=0.20,
            myelin_delta_rest=-0.20,
            n=40,
        )
    )
    # Br5595
    rows.append(
        dict(
            label="c",
            slice_id="dlpfc_151669",
            expected_class="L3_program",
            direction_ok=True,
            l3_delta_rest=0.25,
            myelin_delta_rest=-0.25,
            n=120,
        )
    )
    rows.append(
        dict(
            label="d",
            slice_id="dlpfc_151670",
            expected_class="L3_program",
            direction_ok=True,
            l3_delta_rest=0.15,
            myelin_delta_rest=-0.10,
            n=30,
        )
    )
    # Br8100
    rows.append(
        dict(
            label="e",
            slice_id="dlpfc_151673",
            expected_class="L3_program",
            direction_ok=True,
            l3_delta_rest=0.45,
            myelin_delta_rest=-0.50,
            n=50,
        )
    )
    rows.append(
        dict(
            label="f",
            slice_id="dlpfc_151674",
            expected_class="L3_program",
            direction_ok=True,
            l3_delta_rest=0.30,
            myelin_delta_rest=-0.40,
            n=45,
        )
    )
    # noise: not direction_ok
    rows.append(
        dict(
            label="g",
            slice_id="dlpfc_151676",
            expected_class="L3_program",
            direction_ok=False,
            l3_delta_rest=0.01,
            myelin_delta_rest=0.02,
            n=20,
        )
    )
    # L6 — excluded
    rows.append(
        dict(
            label="h",
            slice_id="dlpfc_151508",
            expected_class="L6_myelin",
            direction_ok=True,
            l3_delta_rest=-0.1,
            myelin_delta_rest=0.5,
            n=150,
        )
    )
    return pd.DataFrame(rows)


def test_donor_mapping():
    assert donor_for_slice("dlpfc_151508") == "Br5292"
    assert donor_for_slice("151673") == "Br8100"
    assert "151669" in DLPFC_SECTION_TO_DONOR


def test_donor_stratified_bootstrap_excludes_zero_in_toy():
    result = donor_stratified_bootstrap_l3(_toy_frame(), n_boot=500, seed=0)
    assert result.n_components == 6
    assert result.n_donors == 3
    assert set(result.donors) == {"Br5292", "Br5595", "Br8100"}
    assert result.point["l3_delta_rest"] > 0
    assert result.point["myelin_delta_rest"] < 0
    # Strong toy signal → CI should exclude 0 for L3
    assert result.ci["l3_delta_rest"]["ci_low"] > 0
    assert result.ci["myelin_delta_rest"]["ci_high"] < 0
    assert result.point["direction_rate"] == pytest.approx(1.0)
    payload = result.to_dict()
    assert payload["protocol"] == "histoweave.donor_bootstrap.v1"
    assert "unstratified" in payload


def test_requires_direction_ok_filter():
    result = donor_stratified_bootstrap_l3(
        _toy_frame(), n_boot=100, seed=1, require_direction_ok=False
    )
    # Includes the non-direction_ok row in Br8100 section 151676 → still 3 donors
    assert result.n_components == 7


def test_cohort_csv_if_present():
    path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "discovery_uncertainty_niches"
        / "results"
        / "cohort"
        / "cohort_component_panel.csv"
    )
    if not path.is_file():
        pytest.skip("cohort panel not generated")
    frame = pd.read_csv(path)
    result = donor_stratified_bootstrap_l3(frame, n_boot=1000, seed=0)
    assert result.n_components >= 10
    assert result.n_donors >= 2
    assert result.point["l3_delta_rest"] == result.point["l3_delta_rest"]  # not NaN
