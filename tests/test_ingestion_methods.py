"""Tests for registered ingestion methods (visium, xenium, stereoseq)."""

import pytest

from histoweave.datasets import write_visium_fixture, write_xenium_fixture
from histoweave.plugins import create_method, list_methods


class TestIngestionMethodsRegistered:
    """All seven ingestion methods (3 native + 4 spatialdata-only) are listed."""

    _EXPECTED_INGESTION = {
        "visium_reader", "xenium_reader", "stereoseq_reader",
        "merfish_reader", "cosmx_reader", "merscope_reader", "slideseq_reader",
    }

    def test_ingestion_methods_are_listed(self):
        methods = {m["name"] for m in list_methods(category="ingestion")}
        assert methods == self._EXPECTED_INGESTION

    def test_category_is_not_empty(self):
        methods = list_methods(category="ingestion")
        assert len(methods) == len(self._EXPECTED_INGESTION)

    def test_each_ingestion_method_can_be_created(self):
        for name in ("visium_reader", "xenium_reader", "stereoseq_reader"):
            method = create_method("ingestion", name, path="/tmp/dummy")
            assert method.spec.category.value == "ingestion"
            assert method.spec.name == name


class TestVisiumIngestion:
    """Run a real ingest over the Visium vendor fixture."""

    def test_visium_ingestion_from_fixture(self, tmp_path):
        fixture = write_visium_fixture(
            tmp_path / "visium_bundle",
            n_spots=50, n_genes=20, seed=0,
        )
        result = create_method(
            "ingestion", "visium_reader", path=str(fixture), engine="native",
        ).run(None)  # data argument is ignored by ingestion methods

        assert result.n_obs == 50
        assert result.n_vars == 20
        assert "spatial" in result.obsm
        assert result.obs.columns.tolist() == ["in_tissue", "array_row", "array_col"]
        assert len(result.provenance) >= 1
        assert result.provenance[-1]["step"] == "ingestion"


class TestXeniumIngestion:
    """Run a real ingest over the Xenium vendor fixture."""

    def test_xenium_ingestion_from_fixture(self, tmp_path):
        fixture = write_xenium_fixture(
            tmp_path / "xenium_bundle",
            n_cells=30, n_genes=15, seed=1,
        )
        result = create_method(
            "ingestion", "xenium_reader", path=str(fixture), engine="native",
        ).run(None)

        assert result.n_obs == 30
        assert result.n_vars == 15
        assert "spatial" in result.obsm
        assert "transcript_counts" in result.obs.columns
        assert len(result.provenance) >= 1


class TestIngestionParamsValidation:
    """Parameter validation is enforced."""

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError, match="engine"):
            create_method("ingestion", "visium_reader", path="/tmp", engine="bogus")

    def test_stereoseq_engine_choice(self):
        with pytest.raises(ValueError, match="spatialdata"):
            create_method("ingestion", "stereoseq_reader", path="/tmp", engine="native")
