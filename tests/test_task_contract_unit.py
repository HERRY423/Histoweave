"""Unit tests for analysis task contracts."""

import numpy as np
import pandas as pd
import pytest

from histoweave.benchmark.task_contract import (
    AnalysisTask,
    GroundTruthKind,
    assert_labels_usable,
    classify_platform,
    default_spatial_context_policy,
)
from histoweave.data import SpatialTable


def test_cell_type_accepts_cluster_proxy() -> None:
    from histoweave.benchmark.task_contract import TaskContract

    TaskContract(
        task=AnalysisTask.CELL_TYPE,
        ground_truth_kind=GroundTruthKind.CLUSTER_PROXY,
        label_key="cluster",
    ).validate()


def test_assert_labels_usable_reads_obs() -> None:
    from histoweave.benchmark.task_contract import TaskContract

    data = SpatialTable(
        X=np.ones((4, 2)),
        obs=pd.DataFrame({"domain_truth": ["a", "a", "b", "b"]}),
        var=pd.DataFrame(index=["g0", "g1"]),
        obsm={"spatial": np.arange(8, dtype=float).reshape(4, 2)},
    )
    contract = TaskContract(
        task=AnalysisTask.SPATIAL_DOMAIN,
        ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
        label_key="domain_truth",
    )
    labels = assert_labels_usable(data, contract)
    assert labels.tolist() == ["a", "a", "b", "b"]


def test_assert_labels_usable_missing_column() -> None:
    from histoweave.benchmark.task_contract import TaskContract

    data = SpatialTable(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    contract = TaskContract(
        task=AnalysisTask.SPATIAL_DOMAIN,
        ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
        label_key="domain_truth",
    )
    with pytest.raises(KeyError):
        assert_labels_usable(data, contract)


def test_classify_platform_aliases() -> None:
    assert classify_platform("10x-Visium") == "visium"
    assert classify_platform("Xenium") == "xenium"
    assert classify_platform(None) is None


def test_default_spatial_context_policy() -> None:
    assert default_spatial_context_policy(AnalysisTask.SPATIAL_DOMAIN) == "high"
    assert default_spatial_context_policy(AnalysisTask.CELL_TYPE) == "off"
