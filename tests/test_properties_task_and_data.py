"""Property tests for task contracts and SpatialTable construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import sparse

from histoweave.benchmark.task_contract import (
    AnalysisTask,
    GroundTruthKind,
    TaskContract,
    split_method_policy,
)
from histoweave.data import SpatialTable

pytestmark = pytest.mark.property


@given(st.sampled_from(["leiden", "louvain", "proxy_leiden", "self_cluster_labels"]))
@settings(max_examples=20)
def test_spatial_domain_rejects_self_supervised_keys(label_key: str) -> None:
    with pytest.raises(ValueError):
        TaskContract(
            task=AnalysisTask.SPATIAL_DOMAIN,
            ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
            label_key=label_key,
        ).validate()


@given(st.sampled_from([GroundTruthKind.SELF_SUPERVISED, GroundTruthKind.CLUSTER_PROXY]))
@settings(max_examples=10)
def test_spatial_domain_rejects_bad_kinds(kind: GroundTruthKind) -> None:
    with pytest.raises(ValueError):
        TaskContract(
            task=AnalysisTask.SPATIAL_DOMAIN,
            ground_truth_kind=kind,
            label_key="domain_truth",
        ).validate()


@given(
    st.text(min_size=1, max_size=12, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.one_of(
        st.none(), st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz012.")
    ),
)
@settings(max_examples=30)
def test_split_method_policy_roundtrip(method: str, policy: str | None) -> None:
    assume("@" not in method)
    key = method if policy is None else f"{method}@{policy}"
    base, parsed = split_method_policy(key)
    assert base == method
    assert parsed == policy


@given(
    arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 30), st.integers(2, 15)),
        elements=st.floats(0, 20, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=25, deadline=None)
def test_spatial_table_accepts_sparse_and_roundtrips_shape(matrix: np.ndarray) -> None:
    n_obs, n_vars = matrix.shape
    table = SpatialTable(
        X=sparse.csr_matrix(matrix),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
        obsm={"spatial": np.column_stack([np.arange(n_obs), np.zeros(n_obs)])},
    )
    assert table.n_obs == n_obs
    assert table.n_vars == n_vars
    copied = table.copy()
    assert copied.n_obs == n_obs
    assert sparse.issparse(copied.X) or isinstance(copied.X, np.ndarray)
    # Subsample keeps alignment.
    mask = np.zeros(n_obs, dtype=bool)
    mask[::2] = True
    if mask.sum() == 0:
        mask[0] = True
    sub = table.subset_obs(mask)
    assert sub.n_obs == int(mask.sum())
    assert sub.n_vars == n_vars
