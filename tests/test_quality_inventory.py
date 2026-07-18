"""Meta-tests: keep typing, property, and coverage inventory honest."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src_py_files() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "src" / "histoweave").rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.endswith(".orig")
    )


def _test_py_files() -> list[Path]:
    return sorted((ROOT / "tests").glob("test_*.py"))


def test_mypy_config_is_not_weak() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = data["tool"]["mypy"]
    assert mypy.get("follow_imports") in {"normal", "silent"}
    assert mypy.get("ignore_missing_imports") is False
    assert mypy.get("check_untyped_defs") is True
    assert mypy.get("no_implicit_optional") is True
    assert mypy.get("warn_unused_configs") is True
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'follow_imports = "skip"' not in raw
    assert "[[tool.mypy.overrides]]" in raw
    # Global silence must be false; per-module overrides may still set true.
    global_pos = raw.index("ignore_missing_imports = false")
    override_pos = raw.index("[[tool.mypy.overrides]]")
    assert global_pos < override_pos


def test_test_to_source_ratio_meets_floor() -> None:
    """Nature-Methods-grade tooling typically keeps test/src ≥ ~0.8.

    Enforce a rising floor so coverage inventory cannot silently collapse.
    """
    n_src = len(_src_py_files())
    n_tests = len(_test_py_files())
    ratio = n_tests / max(n_src, 1)
    assert n_src >= 50
    assert n_tests >= 70, f"expected ≥70 test modules, found {n_tests}"
    assert ratio >= 0.80, f"test/src ratio {ratio:.2f} < 0.80 (tests={n_tests}, src={n_src})"


def test_hypothesis_and_perf_markers_are_registered() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"].get("markers", [])
    joined = "\n".join(markers)
    assert "property:" in joined
    assert "perf:" in joined
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(item.startswith("hypothesis") for item in dev)
    assert any(item.startswith("pandas-stubs") for item in dev)


def test_perf_baselines_exist() -> None:
    path = ROOT / "tests" / "perf_baselines.json"
    assert path.is_file()
    payload = path.read_text(encoding="utf-8")
    assert "knn_indices_n2000_k15" in payload
    assert "slack_factor" in payload
