"""Regression tests for scientifically dangerous silent-failure paths."""

import numpy as np
import pandas as pd
import pytest

from histoweave.data import SpatialTable
from histoweave.plugins import create_method


def _feature_id_table() -> SpatialTable:
    return SpatialTable(
        X=np.array([[10.0, 0.0], [0.0, 10.0], [8.0, 1.0], [1.0, 8.0]]),
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(
            {"feature_name": ["EPCAM", "PTPRC"]},
            index=pd.Index(["ENSG000001", "ENSG000002"], name="feature_id"),
        ),
        obsm={"spatial": np.arange(8, dtype=float).reshape(4, 2)},
    )


def test_marker_annotation_resolves_symbols_kept_outside_var_index():
    data = _feature_id_table()
    result = create_method(
        "annotation",
        "marker_score",
        marker_genes={"epithelial": ["EPCAM"], "immune": ["PTPRC"]},
    ).run(data)

    assert list(result.obs["cell_type"]) == [
        "epithelial",
        "immune",
        "epithelial",
        "immune",
    ]
    assert result.obs["annotation_score"].gt(0).all()
    assert result.uns["annotation_marker_resolution"]["matched"]["immune"] == ["PTPRC"]


def test_marker_methods_fail_closed_when_a_label_has_no_matching_features():
    data = _feature_id_table()
    markers = {"epithelial": ["EPCAM"], "invented": ["NOT_A_GENE"]}

    with pytest.raises(ValueError, match="No markers for label 'invented'"):
        create_method("annotation", "marker_score", marker_genes=markers).run(data)
    with pytest.raises(ValueError, match="No markers for label 'invented'"):
        create_method("deconvolution", "marker_deconv", marker_genes=markers).run(data)
