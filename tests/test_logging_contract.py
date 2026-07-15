"""Repository-wide contracts for standard-library logging usage."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXCLUDED_PARTS = frozenset({".git", ".venv", "__pycache__"})
LOG_METHODS = frozenset({"critical", "debug", "error", "exception", "info", "warning"})
PERCENT_FIELD = re.compile(r"%(?!%)(?:[-+0 #]*\d*(?:\.\d+)?[diouxXeEfFgGcrsa])")


def test_python_sources_do_not_call_builtin_print() -> None:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    builtin_name = "print" + "()"
    assert not violations, f"builtin {builtin_name} calls must use logging instead:\n" + (
        "\n".join(violations)
    )


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

    assert not violations, "duplicate top-level _log helpers:\n" + "\n".join(violations)


def test_runtime_examples_and_healthcheck_do_not_document_print_calls() -> None:
    paths = [
        *sorted((ROOT / "docs").rglob("*.md")),
        ROOT / "workflows/containers/histoweave-python/Dockerfile",
    ]
    violations = [
        str(path.relative_to(ROOT))
        for path in paths
        if ("print" + "(") in path.read_text(encoding="utf-8")
    ]
    assert not violations, "runtime examples must use logging instead: " + ", ".join(violations)


def test_logger_calls_use_literal_balanced_templates() -> None:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LOG_METHODS
            ):
                continue
            location = f"{path.relative_to(ROOT)}:{node.lineno}"
            if not (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                violations.append(f"{location}: message must be a string literal")
                continue
            message = node.args[0].value.replace("%%", "")
            placeholders = len(PERCENT_FIELD.findall(message))
            arguments = len(node.args) - 1
            if placeholders != arguments:
                violations.append(
                    f"{location}: {placeholders} placeholders for {arguments} arguments"
                )

    assert not violations, "invalid logging calls:\n" + "\n".join(violations)
