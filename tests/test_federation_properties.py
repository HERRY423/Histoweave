"""Property-based tests for federation invariants.

Matches the repo's Hypothesis house-style (``pytestmark = pytest.mark.property``).
Two invariants that must hold for *any* well-formed input:

* canonical hashing is deterministic and independent of dict key order;
* the cross-lab consensus score is robust — it always lies within the range of
  the contributing labs' scores and never leaves the ``[min, max]`` envelope,
  no matter how many labs or how extreme one outlier is.
"""

from __future__ import annotations

import pytest
from federation_helpers import make_bundle
from hypothesis import given, settings
from hypothesis import strategies as st

from histoweave.federation import EvidenceStore, build_consensus, canonical_json, content_hash

pytestmark = pytest.mark.property


_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=12),
)


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=8),
        values=_json_scalars,
        max_size=8,
    )
)
@settings(max_examples=50)
def test_canonical_json_stable_under_key_shuffle(d: dict) -> None:
    # Rebuilding the same mapping in a different insertion order must not change
    # the canonical serialization (and therefore the content hash).
    reversed_items = list(d.items())[::-1]
    d2 = dict(reversed_items)
    assert canonical_json(d) == canonical_json(d2)
    assert content_hash(d) == content_hash(d2)


@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=40, deadline=None)
def test_consensus_score_within_lab_envelope(tmp_path_factory, scores: list[float]) -> None:
    # One lab per score, all on the same dataset/method. The consensus (a median
    # of lab means) must never fall outside the observed [min, max] range.
    tmp = tmp_path_factory.mktemp("fed_prop")
    store = EvidenceStore(str(tmp / "store.jsonl"))
    # Round to the store's 6-dp granularity so the envelope check compares like
    # with like (the consensus layer also rounds to 6 dp).
    rounded = [round(s, 6) for s in scores]
    for i, s in enumerate(rounded):
        bundle, _ = make_bundle(f"lab-{i:02d}", ari=s)
        store.append_bundle(bundle)
    view = build_consensus(store, tolerance=0.05)
    cell = view.cells[0]
    assert cell.n_labs == len(scores)
    lo, hi = min(rounded), max(rounded)
    # Slack covers 6-dp rounding on both input and consensus.
    assert lo - 1e-6 <= cell.consensus_score <= hi + 1e-6
    # Reproducibility is always a valid fraction.
    assert 0.0 <= cell.reproducibility <= 1.0
