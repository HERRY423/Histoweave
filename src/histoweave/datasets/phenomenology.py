"""Paired synthetic data for phenomenon-centred spatial benchmarks.

The generators in this module are intentionally independent of any evaluated method.
They expose biological latent fields and technical observation conditions separately,
which makes degradation relative to a paired clean replicate interpretable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..data import Provenance, SpatialTable

PHENOMENOLOGY_SCHEMA_VERSION = "1.0.0"


class SpatialPhenomenon(StrEnum):
    """Orthogonal spatial biology primitives used by the benchmark."""

    COMPARTMENT = "compartment"
    GRADIENT = "gradient"
    HOTSPOT = "hotspot"
    BOUNDARY = "boundary"
    MIXTURE = "mixture"
    BRANCHING = "branching"


class ObservationCondition(StrEnum):
    """One-factor observation processes applied to a biological replicate."""

    CLEAN = "clean"
    LOW_DEPTH_DROPOUT = "low_depth_dropout"
    LOW_SIGNAL_NOISE = "low_signal_noise"
    IRREGULAR_SAMPLING = "irregular_sampling"
    BATCH_PLATFORM_CONFOUNDING = "batch_platform_confounding"


@dataclass(frozen=True)
class PhenomenonSpec:
    """Versioned biological latent-field specification."""

    name: SpatialPhenomenon | str
    effect_size: float = 1.0
    n_regions: int = 3
    hotspot_count: int = 3
    boundary_width: float = 0.08
    branch_angle_degrees: float = 55.0
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", SpatialPhenomenon(self.name))
        if self.effect_size < 0:
            raise ValueError("effect_size must be non-negative")
        if self.n_regions < 2:
            raise ValueError("n_regions must be at least 2")
        if not 1 <= self.hotspot_count <= 3:
            raise ValueError("hotspot_count must lie between 1 and 3")
        if not 0 < self.boundary_width < 0.5:
            raise ValueError("boundary_width must be between 0 and 0.5")
        if not 10 <= self.branch_angle_degrees <= 85:
            raise ValueError("branch_angle_degrees must be between 10 and 85")


@dataclass(frozen=True)
class ConditionSpec:
    """Versioned technical observation specification."""

    name: ObservationCondition | str
    depth_fraction: float = 0.10
    target_zero_fraction: float = 0.85
    signal_fraction: float = 0.35
    dispersion_multiplier: float = 2.0
    batch_log_fold: float = 0.7
    batch_gene_fraction: float = 0.20
    qc_anomaly_fraction: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", ObservationCondition(self.name))
        for attr in (
            "depth_fraction",
            "target_zero_fraction",
            "signal_fraction",
            "batch_gene_fraction",
            "qc_anomaly_fraction",
        ):
            value = float(getattr(self, attr))
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must lie in [0, 1]")
        if self.dispersion_multiplier <= 0:
            raise ValueError("dispersion_multiplier must be positive")
        if self.batch_log_fold < 0:
            raise ValueError("batch_log_fold must be non-negative")


@dataclass(frozen=True)
class ScenarioManifest:
    """Complete, hashable definition of one paired synthetic scenario."""

    phenomenon: PhenomenonSpec
    condition: ConditionSpec
    replicate: int
    seed: int
    n_obs: int = 600
    n_genes: int = 256
    n_cell_types: int = 4
    image_size: int = 256
    schema_version: str = PHENOMENOLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")
        if self.n_obs < 20:
            raise ValueError("n_obs must be at least 20")
        if self.n_genes < 32:
            raise ValueError("n_genes must be at least 32")
        if not 2 <= self.n_cell_types <= 4:
            raise ValueError("n_cell_types must lie between 2 and 4")
        if self.n_genes < 8 * self.n_cell_types:
            raise ValueError("n_genes must provide at least eight markers per cell type")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phenomenon"]["name"] = cast(SpatialPhenomenon, self.phenomenon.name).value
        payload["condition"]["name"] = cast(ObservationCondition, self.condition.name).value
        return payload

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _LatentFields:
    coords: np.ndarray
    domain: np.ndarray
    continuous: np.ndarray
    hotspot: np.ndarray
    boundary_distance: np.ndarray
    branch: np.ndarray
    pseudotime: np.ndarray
    proportions: np.ndarray


def default_scenario_manifest(
    phenomenon: SpatialPhenomenon | str,
    condition: ObservationCondition | str = ObservationCondition.CLEAN,
    *,
    replicate: int = 0,
    seed: int = 1729,
    n_obs: int = 600,
    n_genes: int = 256,
    n_cell_types: int = 4,
    image_size: int = 256,
) -> ScenarioManifest:
    """Construct a scenario using the preregistered default specifications."""

    return ScenarioManifest(
        phenomenon=PhenomenonSpec(phenomenon),
        condition=ConditionSpec(condition),
        replicate=replicate,
        seed=seed,
        n_obs=n_obs,
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        image_size=image_size,
    )


def make_phenomenology_scenario(manifest: ScenarioManifest) -> SpatialTable:
    """Generate one deterministic phenomenon × condition × replicate dataset.

    The random streams for biological fields, expression, technical perturbation and
    imaging are derived independently. Consequently, changing only ``condition`` keeps
    the biological replicate and gene programs paired without accidental changes from
    random-number consumption order.
    """

    phenomenon = SpatialPhenomenon(manifest.phenomenon.name)
    condition = ObservationCondition(manifest.condition.name)
    biological_seed = _derived_seed(manifest.seed, manifest.replicate, "biology")
    expression_seed = _derived_seed(manifest.seed, manifest.replicate, "expression")
    technical_seed = _derived_seed(
        manifest.seed, manifest.replicate, f"condition:{condition.value}"
    )
    image_seed = _derived_seed(manifest.seed, manifest.replicate, "image")

    bio_rng = np.random.default_rng(biological_seed)
    expr_rng = np.random.default_rng(expression_seed)
    tech_rng = np.random.default_rng(technical_seed)
    image_rng = np.random.default_rng(image_seed)

    fields = _make_latent_fields(manifest, bio_rng, tech_rng)
    profiles, marker_genes, gene_names = _make_reference_profiles(manifest, expr_rng)
    spatial_program, spatial_genes = _make_spatial_program(manifest, fields, expr_rng)

    signal_fraction = (
        manifest.condition.signal_fraction
        if manifest.condition.name is ObservationCondition.LOW_SIGNAL_NOISE
        else 1.0
    )
    mu = fields.proportions @ profiles
    mu *= np.exp(signal_fraction * manifest.phenomenon.effect_size * spatial_program)

    library_factor = expr_rng.lognormal(mean=0.0, sigma=0.35, size=manifest.n_obs)
    mu *= library_factor[:, None]
    batch = _make_batches(manifest, fields, tech_rng)
    if manifest.condition.name is ObservationCondition.BATCH_PLATFORM_CONFOUNDING:
        n_shift = max(1, round(manifest.n_genes * manifest.condition.batch_gene_fraction))
        shift_genes = tech_rng.choice(manifest.n_genes, size=n_shift, replace=False)
        batch_one = batch == "batch_1"
        mu[np.ix_(batch_one, shift_genes)] *= np.exp(manifest.condition.batch_log_fold)
    else:
        shift_genes = np.array([], dtype=int)

    if manifest.condition.name is ObservationCondition.LOW_DEPTH_DROPOUT:
        mu *= manifest.condition.depth_fraction

    theta = expr_rng.uniform(4.0, 14.0, size=manifest.n_genes)
    if manifest.condition.name is ObservationCondition.LOW_SIGNAL_NOISE:
        theta /= manifest.condition.dispersion_multiplier
    probability = theta[None, :] / (theta[None, :] + mu)
    counts: np.ndarray = np.asarray(
        tech_rng.negative_binomial(theta[None, :], probability), dtype=np.int32
    )

    if manifest.condition.name is ObservationCondition.LOW_DEPTH_DROPOUT:
        current_zero = float(np.mean(counts == 0))
        if current_zero < manifest.condition.target_zero_fraction:
            extra_dropout = (manifest.condition.target_zero_fraction - current_zero) / (
                1.0 - current_zero
            )
            counts[tech_rng.random(counts.shape) < extra_dropout] = 0

    qc_truth: np.ndarray = np.zeros(manifest.n_obs, dtype=bool)
    n_bad = max(1, round(manifest.n_obs * manifest.condition.qc_anomaly_fraction))
    bad = tech_rng.choice(manifest.n_obs, size=n_bad, replace=False)
    qc_truth[bad] = True
    mito_idx = np.arange(manifest.n_genes - 8, manifest.n_genes)
    counts[bad] = np.rint(counts[bad] * 0.15).astype(np.int32)
    counts[np.ix_(bad, mito_idx)] += tech_rng.poisson(8.0, size=(n_bad, len(mito_idx)))

    image, segmentation = _make_image(manifest, fields, image_rng)
    if manifest.condition.name is ObservationCondition.LOW_SIGNAL_NOISE:
        image = _degrade_image(image, tech_rng)
    elif manifest.condition.name is ObservationCondition.BATCH_PLATFORM_CONFOUNDING:
        image = image.copy()
        image[:, image.shape[1] // 2 :, 0] = np.clip(
            image[:, image.shape[1] // 2 :, 0] * 1.25, 0, 255
        )

    truth_applicability = _truth_applicability(phenomenon)
    obs = pd.DataFrame(
        {
            "domain_truth": [f"domain_{x}" for x in fields.domain],
            "continuous_truth": fields.continuous,
            "hotspot_truth": fields.hotspot.astype(bool),
            "boundary_distance_truth": fields.boundary_distance,
            "branch_truth": ["not_applicable" if x < 0 else f"branch_{x}" for x in fields.branch],
            "pseudotime_truth": fields.pseudotime,
            "cell_type_truth": [f"cell_type_{x}" for x in fields.proportions.argmax(axis=1)],
            "qc_truth": qc_truth,
            "batch": batch,
        },
        index=[f"obs_{idx:05d}" for idx in range(manifest.n_obs)],
    )
    var = pd.DataFrame(
        {
            "mito": np.isin(np.arange(manifest.n_genes), mito_idx),
            "spatial_truth": np.isin(np.arange(manifest.n_genes), spatial_genes),
            "batch_shift_truth": np.isin(np.arange(manifest.n_genes), shift_genes),
        },
        index=gene_names,
    )
    edges = _knn_edges(fields.coords, k=min(8, manifest.n_obs - 1))
    lr_truth = _make_lr_truth(manifest, fields, gene_names)

    table = SpatialTable(
        X=counts.astype(float),
        obs=obs,
        var=var,
        obsm={
            "spatial": fields.coords * 100.0,
            "proportions_truth": fields.proportions,
        },
        layers={"counts": counts.astype(float)},
        images={
            "synthetic_tissue": image,
            "segmentation_truth": segmentation,
        },
        uns={
            "assay": "phenomenology_synthetic",
            "phenomenon": phenomenon.value,
            "condition": condition.value,
            "scenario_manifest": manifest.to_dict(),
            "scenario_manifest_hash": manifest.manifest_hash,
            "truth_schema_version": PHENOMENOLOGY_SCHEMA_VERSION,
            "truth_applicability": truth_applicability,
            "marker_genes": marker_genes,
            "spatial_truth_genes": [gene_names[idx] for idx in spatial_genes],
            "reference_profiles": profiles,
            "reference_cell_types": [f"cell_type_{idx}" for idx in range(manifest.n_cell_types)],
            "truth_graph_edges": edges,
            "lr_truth": lr_truth,
            "n_domains": int(len(np.unique(fields.domain))),
            "paired_replicate_id": f"seed-{manifest.seed}-replicate-{manifest.replicate}",
        },
    )
    table.record(
        Provenance(
            step="ingestion",
            method="make_phenomenology_scenario",
            method_version=PHENOMENOLOGY_SCHEMA_VERSION,
            params={
                "manifest_hash": manifest.manifest_hash,
                "phenomenon": phenomenon.value,
                "condition": condition.value,
                "replicate": manifest.replicate,
                "seed": manifest.seed,
            },
        )
    )
    return table


def make_phenomenology_suite(
    *,
    seeds: tuple[int, ...] = (1729, 2718, 3141, 5772, 8111),
    n_obs: int = 600,
    n_genes: int = 256,
    image_size: int = 256,
) -> dict[str, SpatialTable]:
    """Generate the preregistered 6 × 5 paired scenario suite.

    ``seeds`` defines biological replicates. A one-seed call therefore creates 30
    datasets; the preregistered five-seed call creates 150 datasets.
    """

    suite: dict[str, SpatialTable] = {}
    for replicate, seed in enumerate(seeds):
        for phenomenon in SpatialPhenomenon:
            for condition in ObservationCondition:
                manifest = default_scenario_manifest(
                    phenomenon,
                    condition,
                    replicate=replicate,
                    seed=seed,
                    n_obs=n_obs,
                    n_genes=n_genes,
                    image_size=image_size,
                )
                key = f"{phenomenon.value}__{condition.value}__r{replicate}"
                suite[key] = make_phenomenology_scenario(manifest)
    return suite


def _derived_seed(seed: int, replicate: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{replicate}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _make_latent_fields(
    manifest: ScenarioManifest,
    bio_rng: np.random.Generator,
    tech_rng: np.random.Generator,
) -> _LatentFields:
    n = manifest.n_obs
    coords = bio_rng.uniform(0.02, 0.98, size=(n, 2))
    if manifest.condition.name is ObservationCondition.IRREGULAR_SAMPLING:
        # A deterministic observation transform creates local density imbalance and a
        # tissue hole while retaining n and the replicate-level field parameters.
        coords[:, 0] = np.where(
            coords[:, 0] < 0.65,
            coords[:, 0] * 0.55 / 0.65,
            0.55 + (coords[:, 0] - 0.65) * 0.45 / 0.35,
        )
        hole = np.linalg.norm(coords - np.array([0.70, 0.50]), axis=1) < 0.13
        while np.any(hole):
            coords[hole] = tech_rng.uniform(0.02, 0.98, size=(int(hole.sum()), 2))
            hole = np.linalg.norm(coords - np.array([0.70, 0.50]), axis=1) < 0.13

    x, y = coords[:, 0], coords[:, 1]
    domain: np.ndarray = np.zeros(n, dtype=int)
    continuous: np.ndarray = np.full(n, np.nan)
    hotspot: np.ndarray = np.zeros(n, dtype=bool)
    boundary_distance: np.ndarray = np.full(n, np.nan)
    branch: np.ndarray = np.full(n, -1, dtype=int)
    pseudotime: np.ndarray = np.full(n, np.nan)

    name = manifest.phenomenon.name
    if name is SpatialPhenomenon.COMPARTMENT:
        warped = x + 0.10 * np.sin(2 * np.pi * y)
        domain = np.clip(
            (warped * manifest.phenomenon.n_regions).astype(int),
            0,
            manifest.phenomenon.n_regions - 1,
        )
    elif name is SpatialPhenomenon.GRADIENT:
        continuous = np.clip(0.75 * x + 0.25 * y**2, 0, 1)
        domain = np.clip(
            (continuous * manifest.phenomenon.n_regions).astype(int),
            0,
            manifest.phenomenon.n_regions - 1,
        )
    elif name is SpatialPhenomenon.HOTSPOT:
        centers = np.array([[0.25, 0.30], [0.72, 0.35], [0.53, 0.76]])
        centers = centers[: manifest.phenomenon.hotspot_count]
        rbf = np.exp(
            -np.min(np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
            / (2 * 0.10**2)
        )
        continuous = rbf
        hotspot = rbf >= np.quantile(rbf, 0.75)
        domain = hotspot.astype(int)
    elif name is SpatialPhenomenon.BOUNDARY:
        curve = 0.5 + 0.10 * np.sin(2 * np.pi * y)
        signed = x - curve
        domain = (signed >= 0).astype(int)
        boundary_distance = np.abs(signed)
        continuous = np.exp(-boundary_distance / manifest.phenomenon.boundary_width)
        hotspot = boundary_distance <= manifest.phenomenon.boundary_width
    elif name is SpatialPhenomenon.MIXTURE:
        continuous = np.clip(x, 0, 1)
        hotspot = np.linalg.norm(coords - np.array([0.7, 0.7]), axis=1) < 0.16
        domain = np.clip(
            (x * manifest.phenomenon.n_regions).astype(int), 0, manifest.phenomenon.n_regions - 1
        )
    else:
        angle = np.deg2rad(manifest.phenomenon.branch_angle_degrees)
        origin = np.array([0.18, 0.5])
        directions = np.array(
            [[1.0, 0.0], [np.cos(angle), np.sin(angle)], [np.cos(angle), -np.sin(angle)]]
        )
        rel = coords[:, None, :] - origin[None, None, :]
        projection = np.maximum(np.sum(rel * directions[None, :, :], axis=2), 0)
        closest = origin[None, None, :] + projection[:, :, None] * directions[None, :, :]
        distance = np.linalg.norm(coords[:, None, :] - closest, axis=2)
        branch = distance.argmin(axis=1)
        pseudotime = np.clip(projection[np.arange(n), branch] / 0.85, 0, 1)
        continuous = pseudotime
        domain = branch.copy()
        boundary_distance = distance[np.arange(n), branch]

    logits = np.column_stack(
        [
            1.8 * (0.5 - x),
            1.8 * (x - 0.5),
            1.5 * (0.5 - y),
            1.5 * (y - 0.5),
        ][: manifest.n_cell_types]
    )
    if name is SpatialPhenomenon.MIXTURE:
        logits[:, 2 % manifest.n_cell_types] += 2.2 * hotspot
    logits += bio_rng.normal(0, 0.25, size=logits.shape)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    proportions = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return _LatentFields(
        coords=coords,
        domain=domain,
        continuous=continuous,
        hotspot=hotspot,
        boundary_distance=boundary_distance,
        branch=branch,
        pseudotime=pseudotime,
        proportions=proportions,
    )


def _make_reference_profiles(
    manifest: ScenarioManifest, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, list[str]], list[str]]:
    gene_names = [f"gene_{idx:04d}" for idx in range(manifest.n_genes - 8)] + [
        f"MT-{idx:02d}" for idx in range(8)
    ]
    profiles = rng.lognormal(mean=-0.2, sigma=0.35, size=(manifest.n_cell_types, manifest.n_genes))
    marker_genes: dict[str, list[str]] = {}
    for cell_type in range(manifest.n_cell_types):
        marker_idx = np.arange(cell_type * 8, (cell_type + 1) * 8)
        profiles[cell_type, marker_idx] *= 8.0
        marker_genes[f"cell_type_{cell_type}"] = [gene_names[idx] for idx in marker_idx]
    profiles[:, -8:] *= 0.2
    return profiles, marker_genes, gene_names


def _make_spatial_program(
    manifest: ScenarioManifest,
    fields: _LatentFields,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_spatial = min(48, manifest.n_genes - 8 * manifest.n_cell_types - 8)
    start = 8 * manifest.n_cell_types
    spatial_genes = np.arange(start, start + n_spatial)
    program: np.ndarray = np.zeros((manifest.n_obs, manifest.n_genes), dtype=float)
    name = manifest.phenomenon.name
    if name is SpatialPhenomenon.COMPARTMENT:
        signal = fields.domain / max(1, fields.domain.max())
    elif name is SpatialPhenomenon.HOTSPOT:
        signal = np.nan_to_num(fields.continuous)
    elif name is SpatialPhenomenon.BOUNDARY:
        signal = np.nan_to_num(fields.continuous)
    elif name is SpatialPhenomenon.BRANCHING:
        signal = np.nan_to_num(fields.pseudotime) + 0.25 * fields.branch
    else:
        signal = np.nan_to_num(fields.continuous)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_spatial)
    centered = signal - np.mean(signal)
    program[:, spatial_genes] = centered[:, None] * signs[None, :]
    return program, spatial_genes


def _make_batches(
    manifest: ScenarioManifest,
    fields: _LatentFields,
    rng: np.random.Generator,
) -> np.ndarray:
    if manifest.condition.name is not ObservationCondition.BATCH_PLATFORM_CONFOUNDING:
        return np.full(manifest.n_obs, "batch_0", dtype=object)
    high_state = fields.domain >= np.median(fields.domain)
    probability = np.where(high_state, 0.70, 0.30)
    return np.where(rng.random(manifest.n_obs) < probability, "batch_1", "batch_0")


def _make_image(
    manifest: ScenarioManifest,
    fields: _LatentFields,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    size = manifest.image_size
    image: np.ndarray = np.full((size, size, 3), 236.0, dtype=float)
    segmentation: np.ndarray = np.zeros((size, size), dtype=np.int32)
    palette = np.array([[46, 126, 166], [239, 138, 98], [103, 169, 108], [156, 115, 178]])
    centers = np.clip(np.rint(fields.coords * (size - 1)).astype(int), 0, size - 1)
    radius = max(1, size // 85)
    for idx, ((cx, cy), cell_type) in enumerate(
        zip(centers, fields.proportions.argmax(axis=1), strict=True), start=1
    ):
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        patch = image[y0:y1, x0:x1]
        colour = palette[cell_type % len(palette)] + rng.normal(0, 6, size=3)
        patch[disk] = np.clip(colour, 0, 255)
        label_patch = segmentation[y0:y1, x0:x1]
        label_patch[disk] = idx
    return np.rint(image).astype(np.uint8), segmentation


def _degrade_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(image.astype(float), sigma=(1.2, 1.2, 0))
    grey = blurred.mean(axis=2, keepdims=True)
    low_contrast = grey + 0.35 * (blurred - grey)
    noisy = low_contrast + rng.normal(0, 16, size=image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _knn_edges(coords: np.ndarray, *, k: int) -> np.ndarray:
    tree = cKDTree(coords)
    _, neighbors = tree.query(coords, k=k + 1)
    sources: np.ndarray = np.repeat(np.arange(len(coords)), k)
    targets = neighbors[:, 1:].reshape(-1)
    edges = np.column_stack([sources, targets])
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def _make_lr_truth(
    manifest: ScenarioManifest, fields: _LatentFields, gene_names: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in range(manifest.n_cell_types):
        target = (source + 1) % manifest.n_cell_types
        records.append(
            {
                "source": f"cell_type_{source}",
                "target": f"cell_type_{target}",
                "ligand": gene_names[source * 8],
                "receptor": gene_names[target * 8 + 1],
                "spatially_active": bool(
                    manifest.phenomenon.name
                    in {SpatialPhenomenon.HOTSPOT, SpatialPhenomenon.MIXTURE}
                ),
            }
        )
    return records


def _truth_applicability(phenomenon: SpatialPhenomenon) -> dict[str, bool]:
    return {
        "domain_truth": phenomenon
        in {
            SpatialPhenomenon.COMPARTMENT,
            SpatialPhenomenon.BOUNDARY,
            SpatialPhenomenon.BRANCHING,
        },
        "continuous_truth": phenomenon
        in {
            SpatialPhenomenon.GRADIENT,
            SpatialPhenomenon.HOTSPOT,
            SpatialPhenomenon.BOUNDARY,
            SpatialPhenomenon.MIXTURE,
            SpatialPhenomenon.BRANCHING,
        },
        "hotspot_truth": phenomenon is SpatialPhenomenon.HOTSPOT,
        "boundary_distance_truth": phenomenon is SpatialPhenomenon.BOUNDARY,
        "branch_truth": phenomenon is SpatialPhenomenon.BRANCHING,
        "pseudotime_truth": phenomenon is SpatialPhenomenon.BRANCHING,
        "proportions_truth": phenomenon is SpatialPhenomenon.MIXTURE,
        "segmentation_truth": True,
        "qc_truth": True,
        "truth_graph_edges": True,
        "lr_truth": phenomenon in {SpatialPhenomenon.HOTSPOT, SpatialPhenomenon.MIXTURE},
    }
