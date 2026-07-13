"""Method / plugin layer: typed interfaces + a queryable registry.

Importing this package registers the built-in reference methods so the platform is
usable out of the box.
"""

from __future__ import annotations

# Register the built-in reference methods on import.
from . import builtin as _builtin  # noqa: E402
from .coverage import method_coverage_report
from .interfaces import (
    METHOD_MATURITY_POLICIES,
    MaturityPolicy,
    Method,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from .registry import (
    clear_registry,
    create_method,
    get_method,
    list_methods,
    list_plugin_failures,
    register,
)

_builtin.register_all()

__all__ = [
    "Method",
    "MethodCategory",
    "MethodMaturity",
    "MaturityPolicy",
    "METHOD_MATURITY_POLICIES",
    "MethodSpec",
    "ParamSpec",
    "register",
    "get_method",
    "create_method",
    "list_methods",
    "method_coverage_report",
    "list_plugin_failures",
    "clear_registry",
]
