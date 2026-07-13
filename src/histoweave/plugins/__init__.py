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
    BackendRequirement,
    MaturityPolicy,
    Method,
    MethodCategory,
    MethodDeprecation,
    MethodImplementation,
    MethodMaturity,
    MethodReference,
    MethodSpec,
    ParamSpec,
)
from .registry import (
    MethodDeprecationWarning,
    clear_registry,
    create_method,
    get_method,
    list_methods,
    list_plugin_failures,
    migrate_method_params,
    register,
)

_builtin.register_all()

__all__ = [
    "Method",
    "BackendRequirement",
    "MethodDeprecation",
    "MethodDeprecationWarning",
    "MethodImplementation",
    "MethodCategory",
    "MethodMaturity",
    "MaturityPolicy",
    "METHOD_MATURITY_POLICIES",
    "MethodSpec",
    "MethodReference",
    "ParamSpec",
    "register",
    "get_method",
    "create_method",
    "list_methods",
    "migrate_method_params",
    "method_coverage_report",
    "list_plugin_failures",
    "clear_registry",
]
