"""Repository-wide contract for unique logging helper definitions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXCLUDED_PARTS = frozenset({".git", ".venv", "__pycache__"})


def test_python_sources_define_log_helper_at_most_once() -> None:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [
            node.lineno
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_log"
        ]
        if len(definitions) > 1:
            violations.append(f"{path.relative_to(ROOT)}:{definitions}")

    newline = chr(10)
    assert not violations, "duplicate top-level _log helpers:" + newline + newline.join(
        violations
    )
