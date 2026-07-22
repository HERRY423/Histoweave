"""Evaluation contracts and frozen method manifests for phenomenology benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ..datasets.phenomenology import SpatialPhenomenon
from ..plugins import MethodCategory, MethodReference, get_method
from ..plugins.builtin.release_manifest import (
    BETA_METHODS,
    CONTRACT_VALIDATED_METHODS,
    PRODUCTION_METHODS,
    SCIENTIFIC_VALIDATED_METHODS,
)

METHOD_MANIFEST_SCHEMA_VERSION = "1.0.0"
VIRTUAL_ST_METHODS = {
    "virtual_st_morphology",
    "virtual_st_scellst",
    "virtual_st_storm",
}
PHENOMENOLOGY_METHODS = (
    PRODUCTION_METHODS
    | BETA_METHODS
    | set(SCIENTIFIC_VALIDATED_METHODS)
    | set(CONTRACT_VALIDATED_METHODS)
    | {"marker_deconv"}
) - VIRTUAL_ST_METHODS


class EvaluationRole(StrEnum):
    """Scientifically distinct interpretations of a method output."""

    DIRECT_INFERENCE = "direct_inference"
    PREPROCESSING_PRESERVATION = "preprocessing_preservation"
    REPRESENTATION_INTEGRATION = "representation_integration"
    INGESTION_FIDELITY = "ingestion_fidelity"


class ResourceClass(StrEnum):
    """Preregistered execution budget class."""

    STANDARD = "standard"
    HEAVY = "heavy"


@dataclass(frozen=True)
class MetricSpec:
    """A metric declared before method execution."""

    name: str
    direction: str = "maximize"
    primary: bool = False

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("metric direction must be 'maximize' or 'minimize'")


@dataclass(frozen=True)
class MethodEvaluationContract:
    """Role, applicability and scoring contract for one exact method release."""

    reference: MethodReference
    role: EvaluationRole
    required_inputs: tuple[str, ...]
    applicable_phenomena: tuple[SpatialPhenomenon, ...]
    metrics: tuple[MetricSpec, ...]
    resource_class: ResourceClass = ResourceClass.STANDARD
    tuning_space: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def is_applicable(self, phenomenon: SpatialPhenomenon | str) -> bool:
        return SpatialPhenomenon(phenomenon) in self.applicable_phenomena

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": {
                "category": MethodCategory(self.reference.category).value,
                "name": self.reference.name,
                "version": self.reference.version,
            },
            "role": self.role.value,
            "required_inputs": list(self.required_inputs),
            "applicable_phenomena": [item.value for item in self.applicable_phenomena],
            "metrics": [asdict(metric) for metric in self.metrics],
            "resource_class": self.resource_class.value,
            "tuning_space": [
                {"parameter": name, "values": list(values)} for name, values in self.tuning_space
            ],
        }


@dataclass(frozen=True)
class FrozenMethod:
    """Auditable snapshot of one built-in release method."""

    category: str
    name: str
    version: str
    maturity: str
    implementation: str
    wraps: str | None
    language: str
    model_family: str
    modalities: tuple[str, ...]
    defaults: tuple[tuple[str, Any], ...]
    backends: tuple[tuple[str, str, str, str | None], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modalities"] = list(self.modalities)
        payload["defaults"] = {name: value for name, value in self.defaults}
        payload["backends"] = [
            {
                "name": name,
                "requirement": requirement,
                "runtime": runtime,
                "install_extra": install_extra,
            }
            for name, requirement, runtime, install_extra in self.backends
        ]
        return payload


@dataclass(frozen=True)
class FrozenMethodManifest:
    """Content-addressed audited phenomenology-method registry snapshot."""

    methods: tuple[FrozenMethod, ...]
    schema_version: str = METHOD_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_count": len(self.methods),
            "methods": [method.to_dict() for method in self.methods],
        }

    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def freeze_release_manifest() -> FrozenMethodManifest:
    """Freeze audited release methods plus the dependency-light marker baseline.

    The audited release sets are the authority. Every declared name must resolve once,
    and no research or third-party entry point can enter the snapshot by accident.
    """

    release_names = PHENOMENOLOGY_METHODS
    frozen: list[FrozenMethod] = []
    resolved_names: set[str] = set()
    for name in sorted(release_names):
        candidates = []
        for category in MethodCategory:
            try:
                cls = get_method(category, name)
            except KeyError:
                continue
            candidates.append(cls)
        if len(candidates) != 1:
            raise RuntimeError(
                f"release manifest method {name!r} resolved {len(candidates)} times; expected one"
            )
        spec = candidates[0].spec
        if spec.name in resolved_names:
            raise RuntimeError(f"duplicate method name in release manifest: {spec.name}")
        resolved_names.add(spec.name)
        frozen.append(
            FrozenMethod(
                category=spec.category.value,
                name=spec.name,
                version=spec.version,
                maturity=spec.maturity.value,
                implementation=spec.implementation.value,
                wraps=spec.wraps,
                language=spec.language,
                model_family=spec.model_family,
                modalities=tuple(spec.modalities),
                defaults=tuple((param.name, param.default) for param in spec.params),
                backends=tuple(
                    (
                        backend.name,
                        backend.requirement,
                        backend.runtime,
                        backend.install_extra,
                    )
                    for backend in spec.backends
                ),
            )
        )
    if resolved_names != release_names:
        raise RuntimeError(
            "release manifest drift: "
            f"unresolved={sorted(release_names - resolved_names)}, "
            f"unexpected={sorted(resolved_names - release_names)}"
        )
    return FrozenMethodManifest(tuple(frozen))


def build_evaluation_contracts(
    manifest: FrozenMethodManifest | None = None,
) -> dict[MethodReference, MethodEvaluationContract]:
    """Build one explicit evaluation contract for every frozen release method."""

    manifest = manifest or freeze_release_manifest()
    contracts: dict[MethodReference, MethodEvaluationContract] = {}
    for method in manifest.methods:
        category = MethodCategory(method.category)
        reference = MethodReference(category, method.name, method.version)
        if reference in contracts:
            raise RuntimeError(f"duplicate evaluation contract for {reference}")
        role, required, phenomena, metrics = _category_contract(category)
        resource_class = (
            ResourceClass.HEAVY
            if method.model_family == "deep_learning" or method.language in {"r", "container"}
            else ResourceClass.STANDARD
        )
        contracts[reference] = MethodEvaluationContract(
            reference=reference,
            role=role,
            required_inputs=required,
            applicable_phenomena=phenomena,
            metrics=metrics,
            resource_class=resource_class,
            tuning_space=_tuning_space(method.name),
        )
    if len(contracts) != len(manifest.methods):
        raise RuntimeError("each frozen method must have exactly one evaluation contract")
    return contracts


def capability_matrix_rows(
    manifest: FrozenMethodManifest | None = None,
) -> list[dict[str, Any]]:
    """Return a stable, machine-readable method × phenomenon applicability matrix."""

    manifest = manifest or freeze_release_manifest()
    contracts = build_evaluation_contracts(manifest)
    rows: list[dict[str, Any]] = []
    for method in manifest.methods:
        reference = MethodReference(method.category, method.name, method.version)
        contract = contracts[reference]
        for phenomenon in SpatialPhenomenon:
            rows.append(
                {
                    "category": method.category,
                    "method": method.name,
                    "version": method.version,
                    "maturity": method.maturity,
                    "role": contract.role.value,
                    "resource_class": contract.resource_class.value,
                    "phenomenon": phenomenon.value,
                    "applicable": contract.is_applicable(phenomenon),
                    "primary_metrics": ",".join(
                        metric.name for metric in contract.metrics if metric.primary
                    ),
                    "required_inputs": ",".join(contract.required_inputs),
                }
            )
    return rows


def _category_contract(
    category: MethodCategory,
) -> tuple[
    EvaluationRole,
    tuple[str, ...],
    tuple[SpatialPhenomenon, ...],
    tuple[MetricSpec, ...],
]:
    all_phenomena = tuple(SpatialPhenomenon)
    direct: dict[
        MethodCategory,
        tuple[tuple[str, ...], tuple[SpatialPhenomenon, ...], tuple[MetricSpec, ...]],
    ] = {
        MethodCategory.SEGMENTATION: (
            ("image",),
            all_phenomena,
            (
                MetricSpec("instance_ap50", primary=True),
                MetricSpec("mean_matched_iou"),
                MetricSpec("count_error", direction="minimize"),
            ),
        ),
        MethodCategory.ANNOTATION: (
            ("expression", "marker_reference"),
            all_phenomena,
            (
                MetricSpec("macro_f1", primary=True),
                MetricSpec("balanced_accuracy"),
                MetricSpec("rare_type_recall"),
            ),
        ),
        MethodCategory.DOMAIN_DETECTION: (
            ("expression", "spatial"),
            all_phenomena,
            (
                MetricSpec("phenomenon_recovery", primary=True),
                MetricSpec("adjusted_rand_index"),
                MetricSpec("boundary_f1"),
            ),
        ),
        MethodCategory.DECONVOLUTION: (
            ("expression", "reference_profiles"),
            (SpatialPhenomenon.MIXTURE,),
            (
                MetricSpec("proportion_rmse", direction="minimize", primary=True),
                MetricSpec("jensen_shannon_similarity"),
                MetricSpec("cell_type_correlation"),
            ),
        ),
        MethodCategory.SPATIALLY_VARIABLE_GENES: (
            ("expression", "spatial"),
            all_phenomena,
            (
                MetricSpec("gene_pr_auc", primary=True),
                MetricSpec("precision_at_k"),
                MetricSpec("null_fdr_calibration"),
            ),
        ),
        MethodCategory.NEIGHBORHOOD: (
            ("spatial",),
            all_phenomena,
            (
                MetricSpec("edge_f1", primary=True),
                MetricSpec("edge_precision"),
                MetricSpec("edge_recall"),
            ),
        ),
        MethodCategory.CELL_CELL_COMMUNICATION: (
            ("expression", "spatial", "lr_reference"),
            (SpatialPhenomenon.HOTSPOT, SpatialPhenomenon.MIXTURE),
            (
                MetricSpec("lr_pr_auc", primary=True),
                MetricSpec("lr_precision_at_k"),
                MetricSpec("null_fdr_calibration"),
            ),
        ),
    }
    if category is MethodCategory.INGESTION:
        return (
            EvaluationRole.INGESTION_FIDELITY,
            ("vendor_fixture",),
            all_phenomena,
            (
                MetricSpec("roundtrip_fidelity", primary=True),
                MetricSpec("coordinate_fidelity"),
                MetricSpec("metadata_fidelity"),
            ),
        )
    if category in {MethodCategory.QC, MethodCategory.NORMALIZATION}:
        preprocessing_metrics: tuple[MetricSpec, ...] = (
            (
                MetricSpec("qc_auprc", primary=True),
                MetricSpec("normal_retention"),
                MetricSpec("phenomenon_signal_retention"),
            )
            if category is MethodCategory.QC
            else (
                MetricSpec("phenomenon_signal_retention", primary=True),
                MetricSpec("marker_rank_preservation"),
                MetricSpec("library_nuisance_removal"),
            )
        )
        return (
            EvaluationRole.PREPROCESSING_PRESERVATION,
            ("expression", "spatial"),
            all_phenomena,
            preprocessing_metrics,
        )
    if category is MethodCategory.INTEGRATION:
        return (
            EvaluationRole.REPRESENTATION_INTEGRATION,
            ("expression", "spatial", "batch"),
            all_phenomena,
            (
                MetricSpec("biological_neighborhood_conservation", primary=True),
                MetricSpec("batch_mixing"),
                MetricSpec("phenomenon_recoverability"),
                MetricSpec("oversmoothing_penalty", direction="minimize"),
            ),
        )
    if category in direct:
        required, phenomena, metrics = direct[category]
        return EvaluationRole.DIRECT_INFERENCE, required, phenomena, metrics
    raise AssertionError(f"unhandled method category: {category}")


def _tuning_space(method_name: str) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    # Four or fewer candidates, fixed independently of evaluation truth.
    spaces: dict[str, tuple[tuple[str, tuple[Any, ...]], ...]] = {
        "banksy": (("lambda_param", (0.2, 0.5, 0.8)),),
        "banksy_py": (("lambda_param", (0.2, 0.5, 0.8)),),
        "spatial_graph": (("k", (6, 8, 12, 16)),),
        "liana_plus": (("expr_prop", (0.05, 0.1, 0.2)),),
    }
    return spaces.get(method_name, ())
