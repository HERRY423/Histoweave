"""Tests for the versioned dataset registry (download, cache, checksum, metadata)."""

import pytest

from histoweave.datasets import DatasetEntry, get_dataset, list_datasets


class TestRegistryMetadata:
    """Registry queries work without touching the network."""

    def test_list_datasets_returns_all_entries(self):
        datasets = list_datasets()
        assert len(datasets) >= 2
        names = {d["name"] for d in datasets}
        assert "dlpfc_151507" in names
        assert "mouse_brain_cytassist" in names

    def test_get_dataset_returns_entry(self):
        entry = get_dataset("mouse_brain_cytassist")
        assert entry.name == "mouse_brain_cytassist"
        assert entry.assay == "visium"
        assert entry.species == "mouse"
        assert entry.n_obs == 2523

    def test_dlpfc_metadata(self):
        entry = get_dataset("dlpfc_151507")
        assert entry.assay == "visium"
        assert entry.tissue == "brain"
        assert entry.species == "human"
        assert "domain_truth" in entry.ground_truth
        assert entry.paper_doi == "10.1038/s41593-020-00787-0"

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_dataset("nonexistent")


class TestDatasetEntryConstruction:
    """DatasetEntry can be created programmatically for custom datasets."""

    def test_minimal_entry(self):
        e = DatasetEntry(
            name="test",
            description="A test dataset.",
            url="https://example.com/data.zip",
            sha256="abc123",
            assay="visium",
        )
        assert e.name == "test"
        assert e.tissue == ""
        assert e.species == ""
        assert e.ground_truth == {}

    def test_full_entry(self):
        e = DatasetEntry(
            name="full_test",
            description="Full metadata entry.",
            url="https://example.com/data.h5ad",
            sha256="def456",
            assay="xenium",
            tissue="tumor",
            species="human",
            n_obs=10000,
            n_vars=500,
            ground_truth={"domain_truth": "obs['layer']"},
            license="CC-BY 4.0",
            paper_doi="10.1000/example",
        )
        assert e.tissue == "tumor"
        assert e.ground_truth == {"domain_truth": "obs['layer']"}


class TestDatasetCachePath:
    """Cache directory layout follows the convention."""

    def test_cache_dir_default(self):
        e = DatasetEntry(
            name="cache_test", description="", url="https://x.com/d.zip",
            sha256="abc", assay="visium",
        )
        assert e._cache_dir().name == "cache_test"
        assert ".cache" in str(e._cache_dir())

    def test_cache_dir_explicit(self, tmp_path):
        e = DatasetEntry(
            name="explicit_test", description="", url="https://x.com/d.zip",
            sha256="abc", assay="visium",
        )
        assert e._cache_dir(tmp_path) == tmp_path / "explicit_test"


# ---------------------------------------------------------------------------
# Xenium + MERFISH metadata
# ---------------------------------------------------------------------------

class TestXeniumBreast:
    """10x Xenium Human Breast Cancer — metadata contract."""

    def test_metadata(self):
        entry = get_dataset("xenium_breast_cancer")
        assert entry.assay == "xenium"
        assert entry.tissue == "tumor"
        assert entry.species == "human"
        assert entry.n_obs == 167780
        assert entry.n_vars == 313
        assert "cell_type" in entry.ground_truth
        assert entry.paper_doi == "10.1038/s41587-022-01583-2"

    def test_url_is_reachable_from_ci(self):
        """The 10x CDN URL should be accessible (HEAD check, may skip offline)."""
        import urllib.request
        try:
            req = urllib.request.Request(
                get_dataset("xenium_breast_cancer").url,
                method='HEAD', headers={'User-Agent': 'Mozilla/5.0'}
            )
            urllib.request.urlopen(req, timeout=15)
        except Exception:
            pytest.skip("10x CDN unreachable from this environment")


class TestMERFISHMouseBrain:
    """Allen Institute MERFISH whole mouse brain — metadata contract."""

    def test_metadata(self):
        entry = get_dataset("merfish_mouse_brain")
        assert entry.assay == "merfish"
        assert entry.tissue == "brain"
        assert entry.species == "mouse"
        assert entry.n_obs == 4_000_000
        assert entry.n_vars == 500
        assert len(entry.ground_truth) >= 3
        assert "cell_type" in entry.ground_truth
        assert "subclass" in entry.ground_truth
        assert "neurotransmitter" in entry.ground_truth

    def test_license_is_cc_by_nc(self):
        entry = get_dataset("merfish_mouse_brain")
        assert "CC-BY-NC" in entry.license.upper()


class TestAllDatasetsPresent:
    """Full registry coverage check."""

    def test_all_expected_datasets_registered(self):
        datasets = list_datasets()
        # 12 DLPFC slices + mouse_brain + xenium + merfish = 15
        assert len(datasets) >= 15, f"expected >=15 datasets, got {len(datasets)}"
        names = {d["name"] for d in datasets}
        for expected in (
            "dlpfc_151507", "dlpfc_151676",
            "mouse_brain_cytassist",
            "xenium_breast_cancer",
            "merfish_mouse_brain",
        ):
            assert expected in names, f"{expected} missing from registry"

    def test_all_assays_represented(self):
        datasets = list_datasets()
        assays = {d["assay"] for d in datasets}
        assert assays == {"visium", "xenium", "merfish"}
        assert len(assays) == 3

    def test_both_species_represented(self):
        datasets = list_datasets()
        species = {d["species"] for d in datasets}
        assert species == {"human", "mouse"}

    def test_all_have_doi_or_license(self):
        for d in list_datasets():
            assert d["paper_doi"] or d["license"], f"{d['name']} missing both DOI and license"
