"""Federation: cross-lab consensus math, tiered-trust hardening, and outliers.

Covers acceptance cases 6 and 7:

6.  consensus math (median-of-lab-means, MAD spread, reproducibility fraction,
    cross-lab CI) and outlier flagging
7.  tiered-trust hardening: unverified -> verified as an independent node
    reproduces a cell within tolerance; irreconcilable disagreement -> disputed
"""

from __future__ import annotations

from federation_helpers import make_bundle

from histoweave.federation import EvidenceStore, build_consensus

# --------------------------------------------------------------------------- #
# 6. Consensus math
# --------------------------------------------------------------------------- #


def _store_with(tmp_path, *bundles) -> EvidenceStore:
    store = EvidenceStore(str(tmp_path / "store.jsonl"))
    for b in bundles:
        store.append_bundle(b, verification_status="unverified")
    return store


def test_single_lab_cell_is_unverified(tmp_path) -> None:
    b, _ = make_bundle("lab-alpha", ari=0.42)
    view = build_consensus(_store_with(tmp_path, b), tolerance=0.05)
    assert len(view.cells) == 1
    cell = view.cells[0]
    assert cell.n_labs == 1
    assert cell.verification_status == "unverified"
    # With one lab the consensus is that lab's score and the CI collapses to a point.
    assert cell.consensus_score == 0.42
    assert cell.cross_lab_ci[0] == cell.cross_lab_ci[1] == 0.42


def test_consensus_score_is_median_of_lab_means(tmp_path) -> None:
    # Three labs, distinct means -> consensus is the median (robust to the tails).
    a, _ = make_bundle("lab-alpha", ari=0.40)
    b, _ = make_bundle("lab-beta", ari=0.42)
    c, _ = make_bundle("lab-gamma", ari=0.44)
    view = build_consensus(_store_with(tmp_path, a, b, c), tolerance=0.05)
    cell = view.cells[0]
    assert cell.n_labs == 3
    assert abs(cell.consensus_score - 0.42) < 1e-9


def test_outlier_is_flagged_but_does_not_move_median(tmp_path) -> None:
    # Two labs agree tightly; a third is a gross outlier. The robust median
    # resists it, and the outlier node is flagged.
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.43)
    c, _ = make_bundle("lab-gamma", ari=0.10)
    view = build_consensus(_store_with(tmp_path, a, b, c), tolerance=0.05)
    cell = view.cells[0]
    assert cell.n_labs == 3
    # Median of {0.42, 0.43, 0.10} = 0.42 -> outlier did not capture consensus.
    assert abs(cell.consensus_score - 0.42) < 1e-9
    assert "lab-gamma" in cell.outlier_node_ids
    assert "lab-alpha" not in cell.outlier_node_ids


def test_reproducibility_fraction(tmp_path) -> None:
    # 2 of 3 labs within tolerance of the consensus -> reproducibility = 2/3.
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.43)
    c, _ = make_bundle("lab-gamma", ari=0.10)
    view = build_consensus(_store_with(tmp_path, a, b, c), tolerance=0.05)
    cell = view.cells[0]
    assert abs(cell.reproducibility - (2.0 / 3.0)) < 1e-9


# --------------------------------------------------------------------------- #
# 7. Tiered-trust hardening
# --------------------------------------------------------------------------- #


def test_two_agreeing_labs_harden_to_verified(tmp_path) -> None:
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.44)  # |Delta| = 0.02 <= 0.05
    view = build_consensus(_store_with(tmp_path, a, b), tolerance=0.05)
    cell = view.cells[0]
    assert cell.n_labs == 2
    assert cell.verification_status == "verified"


def test_two_disagreeing_labs_become_disputed(tmp_path) -> None:
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.10)  # |Delta| = 0.32 > 0.05, no tie-breaker
    view = build_consensus(_store_with(tmp_path, a, b), tolerance=0.05)
    cell = view.cells[0]
    assert cell.n_labs == 2
    assert cell.verification_status == "disputed"
    assert cell.reproducibility < 1.0


def test_third_reproduction_resolves_a_dispute(tmp_path) -> None:
    # Alpha vs Beta disagree; Gamma agrees with Alpha -> a 2/3 majority within
    # tolerance hardens the cell to verified despite Beta remaining an outlier.
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.10)
    c, _ = make_bundle("lab-gamma", ari=0.43)
    view = build_consensus(_store_with(tmp_path, a, b, c), tolerance=0.05)
    cell = view.cells[0]
    assert cell.verification_status == "verified"
    assert "lab-beta" in cell.outlier_node_ids


def test_tolerance_is_configurable(tmp_path) -> None:
    # Two labs at 0.42 and 0.58: each sits 0.08 from the median (0.50). That is
    # outside a 0.05 tolerance (disputed) but inside a 0.10 tolerance (verified).
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.58)
    tight = build_consensus(_store_with(tmp_path, a, b), tolerance=0.05).cells[0]
    loose = build_consensus(_store_with(tmp_path, a, b), tolerance=0.10).cells[0]
    assert tight.verification_status == "disputed"
    assert loose.verification_status == "verified"


def test_track_records_accrue_per_node(tmp_path) -> None:
    a, _ = make_bundle("lab-alpha", ari=0.42)
    b, _ = make_bundle("lab-beta", ari=0.43)
    view = build_consensus(_store_with(tmp_path, a, b), tolerance=0.05)
    tr = {t.node_id: t for t in view.track_records}
    assert set(tr) == {"lab-alpha", "lab-beta"}
    assert tr["lab-alpha"].contributed_cells == 1
    assert tr["lab-beta"].contributed_cells == 1
    # Both were reproduced (they agreed with each other).
    assert tr["lab-alpha"].was_reproduced >= 1
