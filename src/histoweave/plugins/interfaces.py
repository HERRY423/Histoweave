"""Typed interfaces for analysis methods (the plugin contract).

Each analysis stage (QC, normalization, domain detection, ...) is defined by a *typed
interface*: declared category, inputs/outputs (always :class:`SpatialTable`), a
parameter schema, and stated assumptions. Concrete methods — whether native Python or
a containerized R/Bioconductor step — implement the interface, so the R/Python divide
becomes an implementation detail rather than a user problem.

A machine-readable :class:`MethodSpec` records the metadata (category, version,
parameters, assumptions, benchmark standing) that powers method selection.
"""

from __future__ import annotations

import abc
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

from ..data import Provenance, SpatialTable

if TYPE_CHECKING:
    from anndata import AnnData


class MethodMaturity(str, Enum):
    """Evidence-backed quality level for an analysis-method wrapper."""

    EXPERIMENTAL = "experimental"
    BETA = "beta"
    PRODUCTION = "production"
    VALIDATED = "validated"


@dataclass(frozen=True)
class MaturityPolicy:
    """Required evidence for a method to claim a maturity level."""

    rank: int
    requirements: tuple[str, ...]


METHOD_MATURITY_POLICIES: dict[MethodMaturity, MaturityPolicy] = {
    MethodMaturity.EXPERIMENTAL: MaturityPolicy(
        rank=10,
        requirements=(
            "Stable MethodSpec and declared input/output contract.",
            "Explicit optional-dependency and failure diagnostics.",
            "Deterministic unit or API-contract tests.",
        ),
    ),
    MethodMaturity.BETA: MaturityPolicy(
        rank=20,
        requirements=(
            "Wraps the real upstream implementation; no algorithmic substitute.",
            "Structural validation, provenance, and versioned parameters.",
            "Mock-contract tests plus at least one backend integration path.",
        ),
    ),
    MethodMaturity.PRODUCTION: MaturityPolicy(
        rank=30,
        requirements=(
            "Pinned reproducible runtime or container and operational diagnostics.",
            "Real-data integration tests covering supported assays and failure modes.",
            "Resource bounds, persistence contract, and release ownership.",
        ),
    ),
    MethodMaturity.VALIDATED: MaturityPolicy(
        rank=40,
        requirements=(
            "Meets all production requirements.",
            "Reference-concordance and multi-dataset benchmark thresholds pass.",
            "Scientific outputs have independent review and documented limitations.",
        ),
    ),
}


class MethodCategory(str, Enum):
    """The analysis stages that make up the platform's functional scope (plan §6)."""

    INGESTION = "ingestion"
    QC = "qc"
    NORMALIZATION = "normalization"
    SEGMENTATION = "segmentation"
    ANNOTATION = "annotation"
    DOMAIN_DETECTION = "domain_detection"
    DECONVOLUTION = "deconvolution"
    SPATIALLY_VARIABLE_GENES = "svg"
    NEIGHBORHOOD = "neighborhood"
    CELL_CELL_COMMUNICATION = "ccc"
    INTEGRATION = "integration"


@dataclass(frozen=True)
class ParamSpec:
    """Runtime-validated declaration of a tunable parameter.

    ``type`` uses the deliberately small plugin-schema vocabulary already present in
    HistoWeave (for example ``"int|None"`` or ``"dict|None"``).  Bounds and choices are
    optional so third-party plugins can adopt validation without a schema migration.
    """

    name: str
    type: str
    default: Any = None
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None

    def validate(self, value: Any, *, method: str) -> None:
        """Raise a diagnostic ``TypeError``/``ValueError`` for an invalid value."""

        allowed = tuple(part.strip() for part in self.type.split("|") if part.strip())
        if not allowed or not any(_matches_type(value, part) for part in allowed):
            raise TypeError(
                f"{method}: parameter {self.name!r} expects {self.type}, got {type(value).__name__}"
            )
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"{method}: parameter {self.name!r} must be one of {list(self.choices)!r}, "
                f"got {value!r}"
            )
        if value is not None and self.minimum is not None:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or float(value) < self.minimum
            ):
                raise ValueError(
                    f"{method}: parameter {self.name!r} must be >= {self.minimum}, got {value!r}"
                )
        if value is not None and self.maximum is not None:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or float(value) > self.maximum
            ):
                raise ValueError(
                    f"{method}: parameter {self.name!r} must be <= {self.maximum}, got {value!r}"
                )


@dataclass(frozen=True)
class MethodSpec:
    """Machine-readable metadata for a registered method (feeds the registry)."""

    name: str
    category: MethodCategory
    version: str
    summary: str = ""
    params: tuple[ParamSpec, ...] = ()
    assumptions: tuple[str, ...] = ()
    # Which assay families the method is appropriate for ("*" = assay-agnostic).
    assays: tuple[str, ...] = ("*",)
    maturity: MethodMaturity = MethodMaturity.EXPERIMENTAL
    # Populated by the benchmarking harness; drives in-workflow recommendations.
    benchmark: dict[str, Any] = field(default_factory=dict)
    wraps: str | None = None  # e.g. "squidpy", "Bioconductor::BANKSY"
    language: str = "python"  # "python" | "r" | "container"
    modalities: tuple[str, ...] = ("expression",)
    model_family: str = "statistical"

    def __post_init__(self) -> None:
        """Coerce string values so external plugin manifests remain ergonomic."""
        if not isinstance(self.maturity, MethodMaturity):
            object.__setattr__(self, "maturity", MethodMaturity(self.maturity))
        modalities = tuple(dict.fromkeys(str(item).lower() for item in self.modalities))
        allowed_modalities = {"expression", "image", "spatial", "labels", "shapes"}
        unknown = set(modalities) - allowed_modalities
        if not modalities or unknown:
            raise ValueError(
                f"{self.name}: modalities must be non-empty and use "
                f"{sorted(allowed_modalities)}; unknown={sorted(unknown)}"
            )
        object.__setattr__(self, "modalities", modalities)
        allowed_families = {"statistical", "machine_learning", "deep_learning"}
        if self.model_family not in allowed_families:
            raise ValueError(
                f"{self.name}: model_family must be one of {sorted(allowed_families)}"
            )


class Method(abc.ABC):
    """Base class every analysis plugin implements.

    Subclasses declare a :attr:`spec` and implement :meth:`run`. They should treat
    inputs as immutable (copy before mutating) and must record provenance via
    :meth:`finalize` so results stay reproducible.
    """

    #: Concrete plugins must override this with their :class:`MethodSpec`.
    spec: MethodSpec

    def __init__(self, **params: Any) -> None:
        self.params = self._resolve_params(params)

    def _resolve_params(self, params: dict[str, Any]) -> dict[str, Any]:
        specs = {p.name: p for p in self.spec.params}
        resolved = {name: spec.default for name, spec in specs.items()}
        unknown = set(params) - set(resolved)
        if unknown:
            raise TypeError(
                f"{self.spec.name}: unknown parameter(s) {sorted(unknown)}; "
                f"valid: {sorted(resolved)}"
            )
        resolved.update(params)
        for name, value in resolved.items():
            specs[name].validate(value, method=self.spec.name)
        return resolved

    @abc.abstractmethod
    def run(self, data: SpatialTable) -> SpatialTable:
        """Transform ``data`` and return a new :class:`SpatialTable`.

        Subclasses that prefer to work directly with :class:`anndata.AnnData`
        can implement :meth:`run_on_anndata` and delegate to
        :meth:`_run_via_anndata` from their ``run``, which handles the
        ``SpatialTable ↔ AnnData`` bridge along with spatial-layer preservation
        and provenance:

        .. code-block:: python

            def run(self, data):
                return self._run_via_anndata(data)

            def run_on_anndata(self, adata):
                import scanpy as sc
                sc.pp.normalize_total(adata)
                return adata
        """

    # ------------------------------------------------------------------
    # AnnData bridge (optional hook)
    # ------------------------------------------------------------------

    def run_on_anndata(self, adata: AnnData) -> AnnData:  # type: ignore[valid-type]  # TYPE_CHECKING
        """Optional hook: transform an :class:`anndata.AnnData` in place-or-copy.

        Implementing this method allows the plugin author to write against the
        scanpy / scvi-tools / squidpy ecosystem directly — no manual bridging
        code.  The caller should still override :meth:`run` and delegate to
        :meth:`_run_via_anndata` so that spatial layers and provenance are
        preserved automatically.

        Raises :class:`NotImplementedError` by default — override in the
        subclass.
        """
        raise NotImplementedError(
            f"{self.spec.name}: run_on_anndata is not implemented; "
            "override it or implement run() directly."
        )

    def _run_via_anndata(self, data: SpatialTable, *, step: str | None = None) -> SpatialTable:
        """Bridge: ``SpatialTable → AnnData → run_on_anndata → SpatialTable``.

        Uses :meth:`SpatialTable.to_anndata` and
        :meth:`SpatialTable.from_anndata` to round-trip the molecular layer,
        then re-attaches the spatial layers (``images``, ``shapes``,
        ``obsm['spatial']``) that the AnnData bridge drops.  Provenance is
        appended automatically via :meth:`finalize`.

        Parameters
        ----------
        data : SpatialTable
            Input.  The callee is responsible for copying before mutation
            (typical pattern: copy inside :meth:`run_on_anndata`).
        step : str or None
            Provenance step label (defaults to ``spec.category.value``).
        """
        adata = data.to_anndata()
        result_adata = self.run_on_anndata(adata)
        result = SpatialTable.from_anndata(result_adata)
        # Restore spatial layers dropped by the AnnData bridge.
        result.images = data.images
        result.shapes = data.shapes
        result.obsm.setdefault("spatial", data.spatial)
        return self.finalize(result, step=step or self.spec.category.value)

    def finalize(self, data: SpatialTable, step: str | None = None) -> SpatialTable:
        """Stamp the object with a provenance entry for this method invocation."""
        from .. import __version__

        data.record(
            Provenance(
                step=step or self.spec.category.value,
                method=self.spec.name,
                method_version=self.spec.version,
                params=dict(self.params),
                histoweave_version=__version__,
                container_digest=os.getenv("HISTOWEAVE_CONTAINER_DIGEST"),
                code_revision=os.getenv("HISTOWEAVE_GIT_COMMIT"),
                executor=os.getenv("HISTOWEAVE_EXECUTOR", "in-process"),
            )
        )
        return data

    def __repr__(self) -> str:
        return f"<Method {self.spec.category.value}:{self.spec.name} v{self.spec.version}>"


def _matches_type(value: Any, declared: str) -> bool:
    """Implement the stable, language-neutral subset of the parameter schema."""

    if declared in {"None", "null"}:
        return value is None
    if declared == "int":
        return isinstance(value, Integral) and not isinstance(value, bool)
    if declared in {"float", "number"}:
        return isinstance(value, Real) and not isinstance(value, bool)
    if declared == "bool":
        return isinstance(value, bool)
    if declared == "str":
        return isinstance(value, str)
    if declared in {"dict", "mapping"}:
        return isinstance(value, Mapping)
    if declared in {"list", "sequence"}:
        return isinstance(value, Sequence) and not isinstance(value, str | bytes)
    if declared == "any":
        return True
    raise TypeError(f"unsupported ParamSpec type declaration {declared!r}")
