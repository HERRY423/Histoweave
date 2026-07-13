"""End-to-end tests for the native Visium/Xenium readers against on-disk fixtures.

The fixtures are written in the real Space Ranger / Xenium directory layout, so these
tests exercise the actual parsing path (10x HDF5 matrix + spatial/cells tables) without
needing a multi-GB vendor download.
"""

import numpy as np
import pytest

from histoweave.datasets import write_visium_fixture, write_xenium_fixture
from histoweave.io import get_reader, read


def test_visium_reader_roundtrips_fixture(tmp_path):
    root = write_visium_fixture(tmp_path / "visium", n_spots=50, n_genes=18, seed=0)
    table = read("visium", str(root))

    assert table.shape == (50, 18)
    assert table.uns["assay"] == "visium"
    # Spatial coordinates land in obsm and match the spot count.
    assert table.spatial.shape == (50, 2)
    # Space Ranger metadata is carried across.
    assert "in_tissue" in table.obs
    assert table.uns["spatial"]["scalefactors"]["spot_diameter_fullres"] == 89.0
    # var is indexed by stable feature id with the symbol kept alongside.
    assert table.var.index.name == "feature_id"
    assert "feature_name" in table.var
    # Counts survive as non-negative integers.
    assert np.all(table.X >= 0)
    assert np.allclose(table.X, np.rint(table.X))
    # Ingestion is recorded in provenance.
    assert table.provenance[-1]["method"] == "visium_reader"


def test_xenium_reader_filters_control_probes(tmp_path):
    root = write_xenium_fixture(
        tmp_path / "xenium", n_cells=40, n_genes=15, seed=0, with_controls=True
    )
    table = read("xenium", str(root))

    # The negative-control probe is dropped: 15 real genes remain.
    assert table.n_vars == 15
    assert set(table.var["feature_type"]) == {"Gene Expression"}
    assert table.uns["assay"] == "xenium"
    assert table.uns["xenium"]["panel_name"] == "synthetic_panel"
    assert "transcript_counts" in table.obs
    assert table.spatial.shape == (40, 2)


def test_xenium_reader_can_keep_controls(tmp_path):
    root = write_xenium_fixture(
        tmp_path / "xenium", n_cells=30, n_genes=12, seed=1, with_controls=True
    )
    table = get_reader("xenium").read(str(root), gene_expression_only=False)
    # With filtering off, the control feature is retained (12 genes + 1 control).
    assert table.n_vars == 13
    assert "Negative Control Probe" in set(table.var["feature_type"])


def test_reader_rejects_unknown_engine(tmp_path):
    root = write_visium_fixture(tmp_path / "visium", n_spots=10, n_genes=6, seed=0)
    with pytest.raises(ValueError, match="engine"):
        read("visium", str(root), engine="bogus")


def test_ingested_fixture_flows_through_pipeline(tmp_path):
    # The whole point of ingestion: a read fixture drives the standard pipeline.
    from histoweave import run_pipeline
    from histoweave.workflow import PipelineStep

    root = write_visium_fixture(tmp_path / "visium", n_spots=120, n_genes=24, n_domains=3, seed=0)
    table = read("visium", str(root))
    steps = [
        PipelineStep("qc", "basic_qc"),
        PipelineStep("normalization", "log1p_cp10k"),
        PipelineStep("domain_detection", "kmeans", {"n_domains": 3}),
    ]
    result = run_pipeline(table, steps)
    assert "domain" in result.obs
    assert result.n_obs <= 120  # QC may drop the injected low-quality spots
    assert "counts" in result.layers  # normalization preserved raw counts
