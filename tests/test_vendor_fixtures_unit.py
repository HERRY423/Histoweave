"""Unit tests for vendor fixture writers."""

from pathlib import Path

from histoweave.datasets.vendor import write_visium_fixture, write_xenium_fixture
from histoweave.io import read


def test_write_and_read_visium_fixture(tmp_path: Path) -> None:
    root = write_visium_fixture(tmp_path / "visium")
    data = read("visium", str(root))
    assert data.n_obs > 0
    assert data.spatial is not None


def test_write_and_read_xenium_fixture(tmp_path: Path) -> None:
    root = write_xenium_fixture(tmp_path / "xenium")
    data = read("xenium", str(root), engine="native")
    assert data.n_obs > 0
    assert data.spatial is not None
