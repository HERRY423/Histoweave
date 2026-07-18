"""Shared test fixtures and hooks.

The session-level startup hook cleans any stale basetemp left by a previous
run — especially important on Windows, where permission-locked directories
under the system %TEMP% can block ``os.scandir`` and cause every
``tmp_path``-using test to error out. Moving the basetemp into the project
tree (``pyproject.toml`` → ``--basetemp=.pytest_tmp``) and sweeping it here
sidesteps the issue entirely.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from histoweave.plugins.registry import _REGISTRY, list_methods


@pytest.fixture(autouse=True)
def isolate_method_registry() -> None:
    """Prevent tests that register temporary plugins from leaking across cases."""
    list_methods()  # Ensure built-ins and entry-point plugins are loaded before snapshotting.
    snapshot = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def pytest_sessionstart(session) -> None:
    """Remove the project-local basetemp before collection so stale dirs never
    interfere with ``tmp_path`` fixture setup."""
    basetemp = Path.cwd() / ".pytest_tmp"
    if basetemp.is_dir():
        shutil.rmtree(basetemp, ignore_errors=True)
