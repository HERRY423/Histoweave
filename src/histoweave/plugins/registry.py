"""The plugin registry — the machine-readable catalogue that powers method selection.

Methods register themselves (built-ins at import time; third-party packages via the
``histoweave.plugins`` entry-point group). The registry is queryable by category and
assay, and is the surface the benchmarking harness annotates with results.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata

from .interfaces import (
    METHOD_MATURITY_POLICIES,
    Method,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
)

# (category, name) -> Method subclass
_REGISTRY: dict[tuple[MethodCategory, str], type[Method]] = {}
_ENTRYPOINTS_LOADED = False
_PLUGIN_FAILURES: list[dict[str, str]] = []


def register(cls: type[Method]) -> type[Method]:
    """Class decorator that registers a :class:`Method` implementation.

    >>> @register
    ... class MyQC(Method):
    ...     spec = MethodSpec(name="my_qc", category=MethodCategory.QC, version="0.1")
    ...     def run(self, data): ...
    """
    spec = getattr(cls, "spec", None)
    if spec is None:
        raise TypeError(f"{cls.__name__} must define a `spec` (MethodSpec) to register")
    if not isinstance(spec, MethodSpec):
        raise TypeError(f"{cls.__name__}.spec must be a MethodSpec, got {type(spec).__name__}")
    if not spec.name or not spec.version:
        raise ValueError(f"{cls.__name__}.spec requires non-empty name and version")
    param_names = [param.name for param in spec.params]
    if len(param_names) != len(set(param_names)):
        raise ValueError(f"{spec.name}: parameter names must be unique")
    key = (spec.category, spec.name)
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(
            f"A different method is already registered for {spec.category.value}:{spec.name}"
        )
    _REGISTRY[key] = cls
    return cls


def _coerce_category(category: str | MethodCategory) -> MethodCategory:
    return category if isinstance(category, MethodCategory) else MethodCategory(category)


def _coerce_maturity(maturity: str | MethodMaturity) -> MethodMaturity:
    return maturity if isinstance(maturity, MethodMaturity) else MethodMaturity(maturity)


def get_method(category: str | MethodCategory, name: str) -> type[Method]:
    """Look up a registered method class by category and name."""
    _load_entry_points()
    key = (_coerce_category(category), name)
    if key not in _REGISTRY:
        available = [n for (c, n) in _REGISTRY if c == key[0]]
        raise KeyError(f"No method '{name}' for category '{key[0].value}'. Available: {available}")
    return _REGISTRY[key]


def create_method(category: str | MethodCategory, name: str, **params) -> Method:
    """Instantiate a registered method with parameters."""
    return get_method(category, name)(**params)


def list_methods(
    category: str | MethodCategory | None = None,
    assay: str | None = None,
    minimum_maturity: str | MethodMaturity | None = None,
) -> list[dict]:
    """Return metadata for registered methods, optionally filtered.

    This is the data behind ``histoweave list-methods`` and the eventual leaderboards:
    the platform can answer "what methods exist for domain detection, and which lead
    for a Xenium dataset of this size?"
    """
    _load_entry_points()
    cat = _coerce_category(category) if category is not None else None
    minimum_rank = (
        METHOD_MATURITY_POLICIES[_coerce_maturity(minimum_maturity)].rank
        if minimum_maturity is not None
        else None
    )
    out = []
    for (c, name), cls in sorted(_REGISTRY.items(), key=lambda kv: (kv[0][0].value, kv[0][1])):
        if cat is not None and c != cat:
            continue
        spec = cls.spec
        if assay is not None and "*" not in spec.assays and assay not in spec.assays:
            continue
        policy = METHOD_MATURITY_POLICIES[spec.maturity]
        if minimum_rank is not None and policy.rank < minimum_rank:
            continue

        out.append(
            {
                "name": name,
                "category": c.value,
                "version": spec.version,
                "summary": spec.summary,
                "assays": list(spec.assays),
                "wraps": spec.wraps,
                "language": spec.language,
                "modalities": list(spec.modalities),
                "model_family": spec.model_family,
                "is_multimodal": len(spec.modalities) > 1,
                "maturity": spec.maturity.value,
                "maturity_rank": policy.rank,
                "quality_requirements": list(policy.requirements),
                "assumptions": list(spec.assumptions),
                "params": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "default": param.default,
                        "help": param.help,
                        "minimum": param.minimum,
                        "maximum": param.maximum,
                        "choices": list(param.choices) if param.choices is not None else None,
                    }
                    for param in spec.params
                ],
                "benchmark": dict(spec.benchmark),
            }
        )
    return out


def _load_entry_points() -> None:
    """Discover external plugins advertised on the ``histoweave.plugins`` group.

    Each entry point is a callable that registers its methods when invoked. This is
    how the frontier (fast-moving method repos) plugs into the stable core without the
    core depending on it.
    """
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True  # set first so failures don't cause repeated retries
    try:
        eps = metadata.entry_points(group="histoweave.plugins")
    except Exception as exc:  # pragma: no cover - importlib.metadata edge cases
        _PLUGIN_FAILURES.append(
            {
                "entry_point": "<discovery>",
                "value": "histoweave.plugins",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        return
    for ep in eps:
        try:
            hook: Callable[[], None] = ep.load()
            hook()
        except Exception as exc:  # pragma: no cover - broken plugin must not break the core
            _PLUGIN_FAILURES.append(
                {
                    "entry_point": ep.name,
                    "value": ep.value,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )


def list_plugin_failures() -> list[dict[str, str]]:
    """Return diagnostic records for external plugins that could not be loaded."""

    _load_entry_points()
    return [dict(failure) for failure in _PLUGIN_FAILURES]


def clear_registry() -> None:
    """Test helper: reset registry state."""
    global _ENTRYPOINTS_LOADED
    _REGISTRY.clear()
    _PLUGIN_FAILURES.clear()
    _ENTRYPOINTS_LOADED = False
