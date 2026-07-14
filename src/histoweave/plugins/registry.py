"""The plugin registry — the machine-readable catalogue that powers method selection.

Methods register themselves (built-ins at import time; third-party packages via the
``histoweave.plugins`` entry-point group). The registry is queryable by category and
assay, and is the surface the benchmarking harness annotates with results.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from importlib import metadata

from packaging.version import InvalidVersion, Version

from .interfaces import (
    METHOD_MATURITY_POLICIES,
    Method,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
)

# (category, name, wrapper version) -> Method subclass
_REGISTRY: dict[tuple[MethodCategory, str, str], type[Method]] = {}
_ENTRYPOINTS_LOADED = False
_PLUGIN_FAILURES: list[dict[str, str]] = []


class MethodDeprecationWarning(FutureWarning):
    """Visible warning raised when a deprecated method release is selected."""


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
    try:
        Version(spec.version)
    except InvalidVersion as exc:
        raise ValueError(f"{spec.name}: invalid method version {spec.version!r}") from exc
    param_names = [param.name for param in spec.params]
    if len(param_names) != len(set(param_names)):
        raise ValueError(f"{spec.name}: parameter names must be unique")
    key = (spec.category, spec.name, spec.version)
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(
            "A different method is already registered for "
            f"{spec.category.value}:{spec.name}@{spec.version}"
        )
    _REGISTRY[key] = cls
    return cls


def _coerce_category(category: str | MethodCategory) -> MethodCategory:
    return category if isinstance(category, MethodCategory) else MethodCategory(category)


def _coerce_maturity(maturity: str | MethodMaturity) -> MethodMaturity:
    return maturity if isinstance(maturity, MethodMaturity) else MethodMaturity(maturity)


def get_method(
    category: str | MethodCategory,
    name: str,
    version: str | None = None,
    *,
    warn_deprecated: bool = True,
) -> type[Method]:
    """Look up an exact release, or the latest non-deprecated release by default."""
    _load_entry_points()
    cat = _coerce_category(category)
    candidates = _method_versions(cat, name)
    if not candidates:
        available = sorted({n for c, n, _ in _REGISTRY if c == cat})
        raise KeyError(f"No method '{name}' for category '{cat.value}'. Available: {available}")
    if version is None:
        active = [cls for cls in candidates if cls.spec.deprecation is None]
        cls = max(active or candidates, key=lambda item: Version(item.spec.version))
    else:
        try:
            Version(version)
        except InvalidVersion as exc:
            raise ValueError(f"invalid method version {version!r}") from exc
        key = (cat, name, version)
        if key not in _REGISTRY:
            available = [cls.spec.version for cls in candidates]
            raise KeyError(
                f"No release {cat.value}:{name}@{version}. Available versions: {available}"
            )
        cls = _REGISTRY[key]
    if warn_deprecated and cls.spec.deprecation is not None:
        warnings.warn(
            _deprecation_message(cls.spec),
            MethodDeprecationWarning,
            stacklevel=2,
        )
    return cls


def create_method(
    category: str | MethodCategory,
    name: str,
    *,
    version: str | None = None,
    **params,
) -> Method:
    """Instantiate a registered method, optionally pinning its wrapper version."""
    return get_method(category, name, version)(**params)


def _method_versions(category: MethodCategory, name: str) -> list[type[Method]]:
    return sorted(
        [
            cls
            for (cat, method_name, _), cls in _REGISTRY.items()
            if cat == category and method_name == name
        ],
        key=lambda cls: Version(cls.spec.version),
    )


def _deprecation_message(spec: MethodSpec) -> str:
    lifecycle = spec.deprecation
    assert lifecycle is not None
    target = lifecycle.replacement
    message = (
        f"{spec.category.value}:{spec.name}@{spec.version} is deprecated since "
        f"HistoWeave {lifecycle.since}; use "
        f"{_coerce_category(target.category).value}:{target.name}@{target.version}."
    )
    if lifecycle.remove_in:
        message += f" It is scheduled for removal in HistoWeave {lifecycle.remove_in}."
    if lifecycle.reason:
        message += f" {lifecycle.reason}"
    message += " Use migrate_method_params(...) for the declared parameter migration."
    return message


def migrate_method_params(
    category: str | MethodCategory,
    name: str,
    from_version: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Follow declared replacements and safely migrate parameters to an active release.

    Renames are applied one lifecycle hop at a time. Parameters declared as removed
    cause an error instead of being silently dropped. The returned mapping is suitable
    for audit logs and can be passed to ``create_method``.
    """
    current = get_method(category, name, from_version, warn_deprecated=False)
    migrated = dict(params or {})
    path: list[dict[str, object]] = []
    visited: set[tuple[MethodCategory, str, str]] = set()

    while current.spec.deprecation is not None:
        spec = current.spec
        key = (spec.category, spec.name, spec.version)
        if key in visited:
            raise RuntimeError(
                f"method deprecation cycle detected at "
                f"{spec.category.value}:{spec.name}@{spec.version}"
            )
        visited.add(key)
        lifecycle = spec.deprecation
        assert lifecycle is not None
        for old_name, new_name in lifecycle.parameter_renames:
            if old_name not in migrated:
                continue
            if new_name in migrated:
                raise ValueError(
                    f"cannot migrate {old_name!r} to {new_name!r}: both parameters are set"
                )
            migrated[new_name] = migrated.pop(old_name)
        removed = sorted(set(lifecycle.removed_parameters) & set(migrated))
        if removed:
            detail = f" {lifecycle.notes}" if lifecycle.notes else ""
            raise ValueError(
                f"cannot migrate removed parameters {removed} from "
                f"{spec.category.value}:{spec.name}@{spec.version}.{detail}"
            )
        target = lifecycle.replacement
        path.append(
            {
                "from": {
                    "category": spec.category.value,
                    "name": spec.name,
                    "version": spec.version,
                },
                "to": {
                    "category": _coerce_category(target.category).value,
                    "name": target.name,
                    "version": target.version,
                },
                "notes": lifecycle.notes,
            }
        )
        current = get_method(target.category, target.name, target.version, warn_deprecated=False)

    current(**migrated)  # validate the migrated schema without executing the method
    return {
        "category": current.spec.category.value,
        "name": current.spec.name,
        "version": current.spec.version,
        "params": migrated,
        "path": path,
    }


def list_methods(
    category: str | MethodCategory | None = None,
    assay: str | None = None,
    minimum_maturity: str | MethodMaturity | None = None,
    all_versions: bool = False,
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
    if all_versions:
        selected = [(c, name, cls) for (c, name, _), cls in _REGISTRY.items()]
    else:
        selected = []
        method_keys = sorted({(c, name) for c, name, _ in _REGISTRY})
        for c, name in method_keys:
            candidates = _method_versions(c, name)
            active = [cls for cls in candidates if cls.spec.deprecation is None]
            selected.append(
                (c, name, max(active or candidates, key=lambda item: Version(item.spec.version)))
            )
    for c, name, cls in sorted(
        selected,
        key=lambda item: (item[0].value, item[1], Version(item[2].spec.version)),
    ):
        if cat is not None and c != cat:
            continue
        spec = cls.spec
        if assay is not None and "*" not in spec.assays and assay not in spec.assays:
            continue
        policy = METHOD_MATURITY_POLICIES[spec.maturity]
        if minimum_rank is not None and policy.rank < minimum_rank:
            continue

        deprecation = spec.deprecation

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
                "implementation": spec.implementation.value,
                "backends": [
                    {
                        "name": backend.name,
                        "requirement": backend.requirement,
                        "install_extra": backend.install_extra,
                        "runtime": backend.runtime,
                    }
                    for backend in spec.backends
                ],
                "deprecated": deprecation is not None,
                "deprecation": (
                    {
                        "since": deprecation.since,
                        "remove_in": deprecation.remove_in,
                        "reason": deprecation.reason,
                        "replacement": {
                            "category": _coerce_category(deprecation.replacement.category).value,
                            "name": deprecation.replacement.name,
                            "version": deprecation.replacement.version,
                        },
                        "parameter_renames": [
                            list(pair) for pair in deprecation.parameter_renames
                        ],
                        "removed_parameters": list(deprecation.removed_parameters),
                        "notes": deprecation.notes,
                    }
                    if deprecation is not None
                    else None
                ),
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
