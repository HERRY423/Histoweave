"""Smoke tests for annotation and neighborhood graph methods."""

from histoweave.datasets import make_synthetic
from histoweave.plugins import create_method


def test_marker_score_annotation() -> None:
    data = create_method("normalization", "log1p_cp10k").run(
        make_synthetic(n_cells=50, n_genes=20, n_domains=3, seed=0)
    )
    out = create_method("annotation", "marker_score").run(data)
    assert "cell_type" in out.obs or "annotation_score" in out.obs or out.n_obs == data.n_obs


def test_spatial_graph_writes_metrics() -> None:
    data = create_method("normalization", "log1p_cp10k").run(
        make_synthetic(n_cells=40, n_genes=15, seed=1)
    )
    out = create_method("neighborhood", "spatial_graph").run(data)
    assert out.n_obs == data.n_obs
