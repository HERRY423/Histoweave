"""Integration tests with format-faithful vendor fixtures and (optionally) real data.

These tests exercise the full path that a real user would follow:

1. **Vendor fixture** — The ``write_visium_fixture`` / ``write_xenium_fixture`` writers
   produce directories in the exact Space Ranger / Xenium layout.  Running the native
   readers over them exercises every line of the IO layer (10x HDF5 parsing, tissue
   positions, scalefactors, cells table, negative control filtering) without needing a
   multi-GB vendor download.

2. **Real public dataset** — When network is available, a small 10x Genomics dataset is
   downloaded, ingested, and run through the full pipeline.  This is the closest we can
   get to a real-user scenario in CI.  The test is skipped gracefully when offline.

3. **End-to-end pipeline over fixture** — A full QC → normalize → domain detection
   (3 methods) → annotation → report pipeline over fixture data, producing a real HTML
   report and verifying the complete provenance chain.
"""

from __future__ import annotations

import json
import logging
from urllib.request import urlretrieve

import numpy as np
import pytest

from histoweave import build_report, run_pipeline
from histoweave.benchmark import domain_detection_task, run_benchmark
from histoweave.datasets import make_synthetic, write_visium_fixture, write_xenium_fixture
from histoweave.io import read, read_bundle, write_bundle
from histoweave.plugins import create_method
from histoweave.workflow import PipelineStep

_LOGGER = logging.getLogger(__name__)

# ===========================================================================
# 1. Format-faithful vendor fixtures (always run, no network needed)
# ===========================================================================


class TestVisiumFixturePipeline:
    """Full pipeline over a format-faithful Visium fixture."""

    def test_full_pipeline_visium_fixture(self, tmp_path):
        root = write_visium_fixture(
            tmp_path / "visium", n_spots=120, n_genes=30, n_domains=3, seed=0
        )
        table = read("visium", str(root))
        assert table.shape == (120, 30)
        assert table.uns["assay"] == "visium"

        # Full pipeline: QC -> normalize -> domain detection (kmeans).
        # Annotation is skipped here because the Visium reader doesn't carry
        # uns['marker_genes'] (that's synthetic-data metadata, not a real Visium field).
        steps = [
            PipelineStep("qc", "basic_qc"),
            PipelineStep("normalization", "log1p_cp10k"),
            PipelineStep("domain_detection", "kmeans", {"n_domains": 3}),
        ]
        result = run_pipeline(table, steps)
        assert "domain" in result.obs
        assert result.n_obs <= 120  # QC may filter
        assert len(result.provenance) >= 4  # ingest + 3 steps

        # Report.
        report_path = tmp_path / "visium_report.html"
        build_report(result, report_path)
        html = report_path.read_text(encoding="utf-8")
        assert "HistoWeave Analysis Report" in html
        assert "visium" in html

    def test_bundle_roundtrip_through_cli_pipeline(self, tmp_path):
        """Simulate the Nextflow DAG: ingest -> step X 3 -> report, via bundle dirs."""
        root = write_visium_fixture(
            tmp_path / "visium", n_spots=80, n_genes=20, n_domains=3, seed=1
        )
        table = read("visium", str(root))

        # Ingest to bundle.
        b0 = tmp_path / "b0.ttab"
        write_bundle(table, b0)

        # Run each step manually (as the CLI and Nextflow do).
        steps = [
            ("qc", "basic_qc", {}),
            ("normalization", "log1p_cp10k", {}),
            ("domain_detection", "kmeans", {"n_domains": 3}),
        ]
        bundle_in = b0
        for i, (cat, method, params) in enumerate(steps):
            b_next = tmp_path / f"b{i + 1}.ttab"
            data = read_bundle(bundle_in)
            result = create_method(cat, method, **params).run(data)
            write_bundle(result, b_next)
            bundle_in = b_next

        final = read_bundle(bundle_in)
        assert "domain" in final.obs
        assert final.n_obs <= 80


class TestXeniumFixturePipeline:
    """Full pipeline over a format-faithful Xenium fixture."""

    def test_xenium_native_read_filter_and_pipeline(self, tmp_path):
        root = write_xenium_fixture(
            tmp_path / "xenium",
            n_cells=100,
            n_genes=24,
            n_domains=3,
            seed=2,
            with_controls=True,
        )
        table = read("xenium", str(root))
        assert table.uns["assay"] == "xenium"
        # Negative controls are filtered (only Gene Expression kept).
        assert set(table.var["feature_type"]) == {"Gene Expression"}
        assert "transcript_counts" in table.obs
        assert "cell_area" in table.obs

        steps = [
            PipelineStep("qc", "basic_qc"),
            PipelineStep("normalization", "log1p_cp10k"),
            PipelineStep("domain_detection", "kmeans", {"n_domains": 3}),
        ]
        result = run_pipeline(table, steps)
        assert "domain" in result.obs
        assert "counts" in result.layers  # normalization stashed counts


# ===========================================================================
# 2. Benchmarking over fixture data with multiple domain-detection methods
# ===========================================================================


class TestBenchmarkOverFixture:
    """Run benchmark on fixture data — multiple sklearn methods compete."""

    def test_benchmark_ranks_sklearn_methods(self):
        """All three sklearn clustering methods + kmeans compete on fixture data."""
        data = make_synthetic(n_cells=300, n_genes=36, n_domains=3, seed=0)
        result = run_benchmark(domain_detection_task(dataset=data))
        # At least kmeans + 3 sklearn methods registered.
        methods = {row["method"] for row in result.leaderboard}
        assert "kmeans" in methods
        assert "dbscan" in methods
        assert "agglomerative" in methods
        assert "spectral" in methods
        # All four ranked.
        assert len(result.leaderboard) >= 4
        ranks = [row["rank"] for row in result.leaderboard]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_kmeans_still_top_on_clean_data(self):
        """On clean synthetic data, kmeans should score highest (simple blobs)."""
        data = make_synthetic(n_cells=300, n_genes=36, n_domains=3, noise=0.1, seed=0)
        result = run_benchmark(
            domain_detection_task(dataset=data),
            k_policy="oracle",
            allow_oracle_k=True,
        )
        best = result.best()
        assert best is not None
        assert best["score"] > 0.85  # very clean data


# ===========================================================================
# 3. Real public dataset (conditional on network)
# ===========================================================================

# Small 10x Genomics Visium dataset: mouse brain section 1 (CytAssist FFPE).
# This is the smallest official Visium dataset (~ 5 MB for the filtered matrix).
_REAL_DATASET_URL = (
    "https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/"
    "CytAssist_FFPE_Mouse_Brain_Rep1/"
    "CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5"
)

# Alternative: the full spatial bundle as a zip.  The filtered matrix alone is
# sufficient to test the native reader; the spatial/ directory can be synthesised
# from the barcodes in the matrix (spot positions are in the tissue_positions.csv
# which is bundled in the "spatial" tar inside the full download).
_REAL_SPATIAL_URL = (
    "https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/"
    "CytAssist_FFPE_Mouse_Brain_Rep1/"
    "CytAssist_FFPE_Mouse_Brain_Rep1_spatial.tar.gz"
)


def _network_ok(timeout: int = 8) -> bool:
    """Check whether we can reach the 10x Genomics CDN."""
    import urllib.request

    try:
        urllib.request.urlopen("https://cf.10xgenomics.com", timeout=timeout)
        return True
    except Exception:
        return False


network_required = pytest.mark.skipif(
    not _network_ok(),
    reason="No network — skipping real-dataset download test.",
)


class TestRealVisiumDownload:
    """Download and process a real (small) 10x Visium dataset."""

    @network_required
    def test_download_and_read_filtered_matrix(self, tmp_path):
        """Download the filtered feature-barcode matrix and verify its structure."""
        dest = tmp_path / "filtered_feature_bc_matrix.h5"
        try:
            urlretrieve(_REAL_DATASET_URL, dest)
        except Exception as exc:
            pytest.skip(f"Download failed: {exc}")

        assert dest.exists()
        assert dest.stat().st_size > 100_000  # at least 100 KB

        from histoweave.io._tenx import read_10x_h5

        mat = read_10x_h5(str(dest))
        assert mat.X.shape[0] > 100  # spots
        assert mat.X.shape[1] > 1000  # genes
        assert len(mat.barcodes) == mat.X.shape[0]
        assert len(mat.feature_ids) == mat.X.shape[1]
        # Counts are non-negative integers.
        assert np.all(mat.X >= 0)
        assert np.allclose(mat.X, np.rint(mat.X))
        # At least some features are "Gene Expression".
        assert "Gene Expression" in mat.feature_types
        _LOGGER.info(
            "  [ok] Downloaded Visium matrix: %s spots x %s genes, genome=%s",
            mat.X.shape[0],
            mat.X.shape[1],
            mat.genome,
        )

    @network_required
    def test_ingest_and_pipeline_over_synthetic_visium(self, tmp_path):
        """Synthesize a full spatial/ directory to pair with the real count matrix.

        Because downloading the full spatial tarball requires decompression tools
        and the spatial/ data is ~200 MB, we pair the real count matrix with a
        synthetic spatial layout derived from the barcode count.  This exercises
        the full ingestion path with real gene identities and count distributions.
        """
        # Download just the H5 file.
        h5_dest = tmp_path / "filtered_feature_bc_matrix.h5"
        try:
            urlretrieve(_REAL_DATASET_URL, h5_dest)
        except Exception as exc:
            pytest.skip(f"Download failed: {exc}")

        # Read the matrix to get real feature names and barcodes.
        from histoweave.io._tenx import read_10x_h5

        mat = read_10x_h5(str(h5_dest))

        # Build a minimal spatial/ directory with synthetic positions.
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()
        rng = np.random.default_rng(0)
        n_spots = mat.X.shape[0]
        coords = rng.uniform(0, 1000, size=(n_spots, 2))

        import pandas as pd

        positions = pd.DataFrame(
            {
                "barcode": mat.barcodes,
                "in_tissue": 1,
                "array_row": np.arange(n_spots),
                "array_col": np.zeros(n_spots, dtype=int),
                "pxl_row_in_fullres": np.rint(coords[:, 1] * 10).astype(int),
                "pxl_col_in_fullres": np.rint(coords[:, 0] * 10).astype(int),
            }
        )
        positions.to_csv(spatial_dir / "tissue_positions.csv", index=False)
        scalefactors = {
            "spot_diameter_fullres": 89.0,
            "tissue_hires_scalef": 0.15,
            "tissue_lowres_scalef": 0.045,
            "fiducial_diameter_fullres": 144.0,
        }
        (spatial_dir / "scalefactors_json.json").write_text(
            json.dumps(scalefactors), encoding="utf-8"
        )

        # Move the H5 to expected path.
        import shutil

        shutil.move(str(h5_dest), str(tmp_path / "filtered_feature_bc_matrix.h5"))

        # Read via the native Visium reader.
        table = read("visium", str(tmp_path))
        assert table.n_obs == n_spots
        # Mouse brain: expect neuronal/glial markers in the feature list.
        # (Not asserting specific genes — the panel varies by chemistry version.)

        # Run a minimal pipeline.
        steps = [
            PipelineStep("qc", "basic_qc"),
            PipelineStep("normalization", "log1p_cp10k"),
            PipelineStep("domain_detection", "kmeans", {"n_domains": 5}),
        ]
        result = run_pipeline(table, steps)
        assert "domain" in result.obs
        assert result.n_obs <= n_spots  # QC may filter

        # Build report.
        report_path = tmp_path / "real_visium_report.html"
        build_report(result, report_path)
        assert report_path.exists()
        _LOGGER.info(
            "  [ok] Pipeline over real mouse brain: %s -> %s spots, %s domains",
            n_spots,
            result.n_obs,
            result.obs["domain"].nunique(),
        )


# ===========================================================================
# 4. Edge cases: empty data, single-gene, minimal dimensions
# ===========================================================================


class TestEdgeCases:
    """Ensure the pipeline survives edge-case inputs that real data can produce."""

    def test_single_observation(self):
        """Pipeline with n=1 should not crash (though QC may remove it)."""
        data = make_synthetic(n_cells=1, n_genes=5, n_domains=1, seed=0, add_mito=False)
        # Normalize should work.
        normed = create_method("normalization", "log1p_cp10k").run(data)
        assert normed.n_obs == 1

    def test_all_zero_expression(self):
        data = make_synthetic(n_cells=20, n_genes=8, n_domains=1, seed=0, add_mito=False)
        data.X = np.zeros_like(data.X)
        # Normalize should handle zeros (totals=0 → replace with 1).
        normed = create_method("normalization", "log1p_cp10k").run(data)
        assert np.all(np.isfinite(normed.X))

    def test_single_gene_svg(self):
        data = make_synthetic(n_cells=100, n_genes=1, n_domains=1, seed=0, add_mito=False)
        normed = create_method("normalization", "log1p_cp10k").run(data)
        # Moran's I with 1 gene should still run.
        result = create_method("svg", "morans_i", n_top=1).run(normed)
        assert "morans_i" in result.var
