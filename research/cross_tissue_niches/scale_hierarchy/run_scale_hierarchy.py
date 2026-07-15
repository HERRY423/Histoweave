"""Screen a conserved spatial-scale hierarchy across three brain datasets.

This is a topology-only candidate screen.  It deliberately does not run SCT or
scVI on the cached MERFISH/Slide-seqV2 matrices because those matrices are not
raw counts.  It asks whether two *pre-specified* spatial processes keep the same
relative scale ordering across DLPFC Visium, hypothalamic MERFISH, and
hippocampal Slide-seqV2:

1. broad neuronal homotypy (E/I marker-field autocorrelation in DLPFC; broad
   neuronal label homotypy in the two cell/bead-resolution datasets), and
2. astrocyte--vascular cross-association (oligodendrocytes are excluded).

Radius is expressed in each sample's native nearest-neighbour (NN) units.  The
null translates each complete spatial field on a 2-D torus and maps it back to
the original coordinates, preserving broad spatial autocorrelation better than
an iid label permutation.  Scale-selected p-values use a max-T statistic.

Outputs are written beside this script under ``results/`` by default.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


SEED = 20260715
DEFAULT_K = (1, 2, 4, 8, 16, 32, 64, 128)
PROCESSES = ("neuronal_homotypy", "astrocyte_vascular")
DOMAINS = ("DLPFC", "MERFISH", "Slide-seqV2")

# Pre-registered fields.  Oligodendrocyte genes are intentionally absent so the
# known white/grey-matter oligodendrocyte axis cannot drive this screen.
MARKERS: dict[str, tuple[str, ...]] = {
    "excitatory": (
        "SLC17A7",
        "CAMK2A",
        "SATB2",
        "CUX1",
        "CUX2",
        "RORB",
        "THEMIS",
        "TLE4",
        "BCL11B",
    ),
    "inhibitory": (
        "GAD1",
        "GAD2",
        "SLC6A1",
        "SLC32A1",
        "DLX1",
        "DLX2",
    ),
    "astrocyte": (
        "AQP4",
        "SLC1A2",
        "SLC1A3",
        "KCNJ10",
        "GLUL",
        "ATP1A2",
        "GJA1",
        "ALDOC",
        "GFAP",
        "ALDH1L1",
    ),
    "vascular": (
        "CLDN5",
        "PECAM1",
        "VWF",
        "EMCN",
        "KDR",
        "RAMP2",
        "SLC2A1",
        "MFSD2A",
        "ABCB1",
        "RGS5",
        "PDGFRB",
    ),
}

DLPFC_DONOR: dict[str, str] = {
    **{str(x): "donor_1" for x in range(151507, 151511)},
    **{str(x): "donor_2" for x in range(151669, 151673)},
    **{str(x): "donor_3" for x in range(151673, 151677)},
}


@dataclass
class SampleResult:
    domain: str
    sample: str
    replication_unit: str
    k_values: np.ndarray
    radius_coord: np.ndarray
    radius_native: np.ndarray
    observed: dict[str, np.ndarray]
    null: dict[str, np.ndarray]
    n_obs: int
    metadata: dict[str, object]


def stable_seed(name: str) -> int:
    return int((SEED + zlib.crc32(name.encode("utf-8"))) % (2**32 - 1))


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-10:
        return np.zeros_like(values, dtype=float)
    return (values - center) / scale


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 8:
        return math.nan
    xx = x[keep] - float(np.mean(x[keep]))
    yy = y[keep] - float(np.mean(y[keep]))
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return float(np.sum(xx * yy) / denom) if denom > 1e-12 else math.nan


def dense_columns(matrix: object, indices: list[int]) -> np.ndarray:
    subset = matrix[:, indices]
    if sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float64)


def module_score(
    counts: object,
    totals: np.ndarray,
    var_names: Iterable[object],
    genes: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    lookup = {str(g).upper(): i for i, g in enumerate(var_names)}
    present = [gene for gene in genes if gene in lookup]
    if len(present) < 2:
        raise RuntimeError(f"fewer than two predeclared genes measured: {present}")
    values = dense_columns(counts, [lookup[g] for g in present])
    values = np.log1p(values * (1e4 / np.maximum(totals, 1.0)[:, None]))
    # Standardise genes within section before averaging so one abundant gene
    # cannot dominate the predeclared field.
    values -= np.mean(values, axis=0, keepdims=True)
    scales = np.std(values, axis=0, keepdims=True)
    scales[scales < 1e-8] = 1.0
    return zscore(np.mean(values / scales, axis=1)), present


def audit_integer_counts(matrix: object, tolerance: float = 1e-6) -> bool:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).all():
        return False
    return bool(
        np.min(values) >= -tolerance and np.max(np.abs(values - np.rint(values))) <= tolerance
    )


def neighbour_graph(
    coords: np.ndarray,
    k_values: np.ndarray,
) -> tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray, int]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"expected n x 2 spatial coordinates, got {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("spatial coordinates contain non-finite values")
    kmax = int(np.max(k_values))
    if len(coords) <= kmax:
        raise ValueError(f"sample has {len(coords)} observations but kmax={kmax}")
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=kmax + 1, workers=-1)
    # Coordinates in all three inputs are unique; the first hit is therefore
    # the observation itself.  Record duplicates as an explicit audit field.
    duplicate_count = int(len(coords) - len(np.unique(coords, axis=0)))
    if duplicate_count:
        raise ValueError(f"found {duplicate_count} duplicate spatial coordinates")
    indices = np.asarray(indices[:, 1:], dtype=np.int32)
    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    radius_coord = np.asarray([np.median(distances[:, int(k) - 1]) for k in k_values], dtype=float)
    d1 = float(radius_coord[0])
    if d1 <= 0:
        raise ValueError("median first-neighbour distance is not positive")
    return tree, indices, radius_coord, radius_coord / d1, duplicate_count


def toroidal_shift_maps(
    coords: np.ndarray,
    tree: cKDTree,
    n_shifts: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    low = np.min(coords, axis=0)
    span = np.ptp(coords, axis=0)
    if np.any(span <= 0):
        raise ValueError(f"zero coordinate span: {span.tolist()}")
    base = coords - low
    for _ in range(n_shifts):
        # Large translations break local alignment; toroidal wrapping retains
        # the broad shape of each target field.  Nearest-point remapping handles
        # irregularly sampled tissue masks without inventing observations.
        shift = rng.uniform(0.20, 0.80, size=2) * span
        query = np.mod(base + shift, span) + low
        yield np.asarray(tree.query(query, k=1, workers=-1)[1], dtype=np.int32)


def local_means(target: np.ndarray, neighbours: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    values = np.asarray(target, dtype=float)[neighbours]
    cumulative = np.cumsum(values, axis=1, dtype=np.float64)
    return np.column_stack([cumulative[:, int(k) - 1] / float(k) for k in k_values])


def field_statistics(
    dominance_anchor: np.ndarray,
    dominance_target: np.ndarray,
    astro_anchor: np.ndarray,
    astro_target: np.ndarray,
    vascular_anchor: np.ndarray,
    vascular_target: np.ndarray,
    neighbours: np.ndarray,
    k_values: np.ndarray,
) -> dict[str, np.ndarray]:
    dominance_lag = local_means(dominance_target, neighbours, k_values)
    astro_lag = local_means(astro_target, neighbours, k_values)
    vascular_lag = local_means(vascular_target, neighbours, k_values)
    astro_high = astro_anchor >= np.quantile(astro_anchor, 0.75)
    vascular_high = vascular_anchor >= np.quantile(vascular_anchor, 0.75)
    neuronal = np.asarray(
        [safe_corr(dominance_anchor, dominance_lag[:, j]) for j in range(len(k_values))]
    )
    cross = np.asarray(
        [
            0.5
            * (
                float(np.mean(vascular_lag[astro_high, j]))
                + float(np.mean(astro_lag[vascular_high, j]))
            )
            for j in range(len(k_values))
        ]
    )
    return {"neuronal_homotypy": neuronal, "astrocyte_vascular": cross}


def label_statistics(
    neuron_anchor: np.ndarray,
    neuron_target: np.ndarray,
    astro_anchor: np.ndarray,
    astro_target: np.ndarray,
    vascular_anchor: np.ndarray,
    vascular_target: np.ndarray,
    neighbours: np.ndarray,
    k_values: np.ndarray,
) -> dict[str, np.ndarray]:
    neuron_rows = np.flatnonzero(neuron_anchor >= 0)
    anchor_codes = neuron_anchor[neuron_rows]
    neighbour_codes = neuron_target[neighbours[neuron_rows]]
    astro_rows = np.flatnonzero(astro_anchor)
    vascular_rows = np.flatnonzero(vascular_anchor)
    neighbour_astro = astro_target[neighbours[vascular_rows]]
    neighbour_vascular = vascular_target[neighbours[astro_rows]]

    neuronal: list[float] = []
    cross: list[float] = []
    for k in k_values:
        kk = int(k)
        codes = neighbour_codes[:, :kk]
        valid = codes >= 0
        denominator = int(np.sum(valid))
        same = (codes == anchor_codes[:, None]) & valid
        neuronal.append(float(np.sum(same) / denominator) if denominator else math.nan)
        cross.append(
            0.5
            * (float(np.mean(neighbour_vascular[:, :kk])) + float(np.mean(neighbour_astro[:, :kk])))
        )
    return {
        "neuronal_homotypy": np.asarray(neuronal),
        "astrocyte_vascular": np.asarray(cross),
    }


def run_field_sample(
    *,
    domain: str,
    sample: str,
    replication_unit: str,
    coords: np.ndarray,
    dominance: np.ndarray,
    astro: np.ndarray,
    vascular: np.ndarray,
    k_values: np.ndarray,
    n_shifts: int,
    metadata: dict[str, object],
) -> SampleResult:
    tree, neighbours, radius_coord, radius_native, _ = neighbour_graph(coords, k_values)
    observed = field_statistics(
        dominance, dominance, astro, astro, vascular, vascular, neighbours, k_values
    )
    null = {process: np.empty((n_shifts, len(k_values)), dtype=float) for process in PROCESSES}
    rng = np.random.default_rng(stable_seed(f"{domain}:{sample}"))
    for shift_index, mapping in enumerate(toroidal_shift_maps(coords, tree, n_shifts, rng)):
        shifted = field_statistics(
            dominance,
            dominance[mapping],
            astro,
            astro[mapping],
            vascular,
            vascular[mapping],
            neighbours,
            k_values,
        )
        for process in PROCESSES:
            null[process][shift_index] = shifted[process]
    return SampleResult(
        domain=domain,
        sample=sample,
        replication_unit=replication_unit,
        k_values=k_values,
        radius_coord=radius_coord,
        radius_native=radius_native,
        observed=observed,
        null=null,
        n_obs=len(coords),
        metadata=metadata,
    )


def run_label_sample(
    *,
    domain: str,
    sample: str,
    replication_unit: str,
    coords: np.ndarray,
    neuron_codes: np.ndarray,
    astro_mask: np.ndarray,
    vascular_mask: np.ndarray,
    k_values: np.ndarray,
    n_shifts: int,
    metadata: dict[str, object],
) -> SampleResult:
    tree, neighbours, radius_coord, radius_native, _ = neighbour_graph(coords, k_values)
    observed = label_statistics(
        neuron_codes,
        neuron_codes,
        astro_mask,
        astro_mask,
        vascular_mask,
        vascular_mask,
        neighbours,
        k_values,
    )
    null = {process: np.empty((n_shifts, len(k_values)), dtype=float) for process in PROCESSES}
    rng = np.random.default_rng(stable_seed(f"{domain}:{sample}"))
    for shift_index, mapping in enumerate(toroidal_shift_maps(coords, tree, n_shifts, rng)):
        shifted = label_statistics(
            neuron_codes,
            neuron_codes[mapping],
            astro_mask,
            astro_mask[mapping],
            vascular_mask,
            vascular_mask[mapping],
            neighbours,
            k_values,
        )
        for process in PROCESSES:
            null[process][shift_index] = shifted[process]
    return SampleResult(
        domain=domain,
        sample=sample,
        replication_unit=replication_unit,
        k_values=k_values,
        radius_coord=radius_coord,
        radius_native=radius_native,
        observed=observed,
        null=null,
        n_obs=len(coords),
        metadata=metadata,
    )


def analyse_dlpfc(
    repo_root: Path,
    k_values: np.ndarray,
    n_shifts: int,
) -> tuple[list[SampleResult], list[dict[str, object]], list[dict[str, object]]]:
    results: list[SampleResult] = []
    marker_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    paths = sorted((repo_root / "datasets_cache" / "dlpfc").glob("dlpfc_*.h5ad"))
    if len(paths) != 12:
        raise RuntimeError(f"expected 12 DLPFC sections, found {len(paths)}")

    for path in paths:
        sample = path.stem.rsplit("_", 1)[-1]
        a = ad.read_h5ad(path)
        counts = a.layers["counts"] if "counts" in a.layers else a.X
        raw_legal = audit_integer_counts(counts)
        if not raw_legal:
            raise RuntimeError(f"DLPFC {sample}: raw count audit failed")
        totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        scores: dict[str, np.ndarray] = {}
        for module, genes in MARKERS.items():
            scores[module], present = module_score(counts, totals, a.var_names, genes)
            marker_rows.append(
                {
                    "domain": "DLPFC",
                    "sample": sample,
                    "field": module,
                    "n_predeclared": len(genes),
                    "n_measured": len(present),
                    "measured_genes": ";".join(present),
                }
            )
        dominance = zscore(scores["excitatory"] - scores["inhibitory"])
        coords = np.asarray(a.obsm["spatial"], dtype=float)
        donor = DLPFC_DONOR[sample]
        results.append(
            run_field_sample(
                domain="DLPFC",
                sample=sample,
                replication_unit="section_nested_in_donor",
                coords=coords,
                dominance=dominance,
                astro=scores["astrocyte"],
                vascular=scores["vascular"],
                k_values=k_values,
                n_shifts=n_shifts,
                metadata={"donor": donor},
            )
        )
        audit_rows.append(
            {
                "domain": "DLPFC",
                "sample": sample,
                "path": str(path),
                "n_obs": int(a.n_obs),
                "n_vars": int(a.n_vars),
                "coordinate_span_x": float(np.ptp(coords[:, 0])),
                "coordinate_span_y": float(np.ptp(coords[:, 1])),
                "count_status": "integer_raw_counts",
                "sct_scvi_used": False,
                "independent_biological_unit": donor,
            }
        )
        del a, counts, scores
        gc.collect()
    return results, marker_rows, audit_rows


def analyse_merfish(
    path: Path,
    k_values: np.ndarray,
    n_shifts: int,
) -> tuple[list[SampleResult], list[dict[str, object]]]:
    a = ad.read_h5ad(path)
    results: list[SampleResult] = []
    audit_rows: list[dict[str, object]] = []
    for batch in sorted(a.obs["batch"].astype(str).unique(), key=lambda x: int(float(x))):
        mask = a.obs["batch"].astype(str).to_numpy() == batch
        obs = a.obs.loc[mask]
        coords = np.asarray(a.obsm["spatial"])[mask].astype(float)
        labels = obs["Cell_class"].astype(str).to_numpy()
        neuron_codes = np.full(len(obs), -1, dtype=np.int8)
        neuron_codes[labels == "Excitatory"] = 0
        neuron_codes[labels == "Inhibitory"] = 1
        astro = labels == "Astrocyte"
        vascular = np.char.startswith(labels.astype(str), "Endothelial") | (labels == "Pericytes")
        bregma_values = obs["Bregma"].astype(str).unique().tolist()
        bregma = bregma_values[0] if len(bregma_values) == 1 else ";".join(bregma_values)
        results.append(
            run_label_sample(
                domain="MERFISH",
                sample=f"batch_{batch}",
                replication_unit="section_same_animal",
                coords=coords,
                neuron_codes=neuron_codes,
                astro_mask=astro,
                vascular_mask=vascular,
                k_values=k_values,
                n_shifts=n_shifts,
                metadata={"animal": "Animal_1", "bregma": bregma},
            )
        )
        audit_rows.append(
            {
                "domain": "MERFISH",
                "sample": f"batch_{batch}",
                "path": str(path),
                "n_obs": int(len(obs)),
                "n_vars": int(a.n_vars),
                "coordinate_span_x": float(np.ptp(coords[:, 0])),
                "coordinate_span_y": float(np.ptp(coords[:, 1])),
                "count_status": "normalized_noninteger_topology_only",
                "sct_scvi_used": False,
                "independent_biological_unit": "Animal_1",
                "n_neuron": int(np.sum(neuron_codes >= 0)),
                "n_astrocyte": int(np.sum(astro)),
                "n_vascular": int(np.sum(vascular)),
            }
        )
    del a
    gc.collect()
    return results, audit_rows


def analyse_slideseq(
    path: Path,
    k_values: np.ndarray,
    n_shifts: int,
) -> tuple[list[SampleResult], list[dict[str, object]]]:
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], dtype=float)
    domains = a.obs["domain_truth"].astype(str).to_numpy()
    neuron_codes = np.full(a.n_obs, -1, dtype=np.int8)
    neuron_codes[domains == "hippocampus_pyramidal"] = 0
    neuron_codes[domains == "neuron_interneuron"] = 1
    astro = domains == "glia_astrocyte"
    vascular = domains == "vascular"
    result = run_label_sample(
        domain="Slide-seqV2",
        sample="single_puck",
        replication_unit="single_puck_unknown_animal",
        coords=coords,
        neuron_codes=neuron_codes,
        astro_mask=astro,
        vascular_mask=vascular,
        k_values=k_values,
        n_shifts=n_shifts,
        metadata={"animal": "not_encoded"},
    )
    audit = {
        "domain": "Slide-seqV2",
        "sample": "single_puck",
        "path": str(path),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        "coordinate_span_x": float(np.ptp(coords[:, 0])),
        "coordinate_span_y": float(np.ptp(coords[:, 1])),
        "count_status": "normalized_noninteger_topology_only",
        "sct_scvi_used": False,
        "independent_biological_unit": "not_encoded",
        "n_neuron": int(np.sum(neuron_codes >= 0)),
        "n_astrocyte": int(np.sum(astro)),
        "n_vascular": int(np.sum(vascular)),
    }
    del a
    gc.collect()
    return [result], [audit]


def upper_p(actual: float, null: np.ndarray) -> float:
    finite = np.asarray(null, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float((1 + np.sum(finite >= actual)) / (len(finite) + 1)) if len(finite) else math.nan


def summarise_samples(results: list[SampleResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        for process in PROCESSES:
            observed = result.observed[process]
            null = result.null[process]
            center = np.nanmedian(null, axis=0)
            spread = np.nanstd(null, axis=0, ddof=1)
            for j, k in enumerate(result.k_values):
                rows.append(
                    {
                        "domain": result.domain,
                        "sample": result.sample,
                        "replication_unit": result.replication_unit,
                        "process": process,
                        "k": int(k),
                        "radius_coordinate": float(result.radius_coord[j]),
                        "radius_native_nn": float(result.radius_native[j]),
                        "observed": float(observed[j]),
                        "null_median": float(center[j]),
                        "null_sd": float(spread[j]),
                        "excess": float(observed[j] - center[j]),
                        "z_vs_shift": float((observed[j] - center[j]) / spread[j])
                        if spread[j] > 0
                        else math.nan,
                        "p_shift_upper": upper_p(float(observed[j]), null[:, j]),
                        "n_obs": result.n_obs,
                        **result.metadata,
                    }
                )
    return pd.DataFrame(rows)


def summarise_domains(
    results: list[SampleResult],
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    domain_null: dict[tuple[str, str], np.ndarray] = {}
    for domain in DOMAINS:
        items = [result for result in results if result.domain == domain]
        for process in PROCESSES:
            observed = np.mean(np.vstack([x.observed[process] for x in items]), axis=0)
            null = np.mean(np.stack([x.null[process] for x in items], axis=0), axis=0)
            domain_null[(domain, process)] = null
            center = np.median(null, axis=0)
            radius_native = np.median(np.vstack([x.radius_native for x in items]), axis=0)
            radius_coord = np.median(np.vstack([x.radius_coord for x in items]), axis=0)
            for j, k in enumerate(items[0].k_values):
                rows.append(
                    {
                        "domain": domain,
                        "process": process,
                        "k": int(k),
                        "radius_coordinate_median": float(radius_coord[j]),
                        "radius_native_nn": float(radius_native[j]),
                        "observed_equal_sample_mean": float(observed[j]),
                        "null_median": float(center[j]),
                        "excess": float(observed[j] - center[j]),
                        "p_shift_upper": upper_p(float(observed[j]), null[:, j]),
                        "n_samples": len(items),
                        "n_independent_biological_units": (3 if domain == "DLPFC" else 1),
                    }
                )
    return pd.DataFrame(rows), domain_null


def interpolate_half_decay(
    radius: np.ndarray,
    effect: np.ndarray,
    peak_index: int,
) -> tuple[float, bool]:
    peak = float(effect[peak_index])
    if not np.isfinite(peak) or peak <= 0:
        return math.nan, False
    half = 0.5 * peak
    for j in range(peak_index + 1, len(effect)):
        if effect[j] <= half:
            x0, x1 = float(radius[j - 1]), float(radius[j])
            y0, y1 = float(effect[j - 1]), float(effect[j])
            if y1 == y0:
                return x1, False
            fraction = (half - y0) / (y1 - y0)
            return float(x0 + fraction * (x1 - x0)), False
    return float(radius[-1]), True


def scale_summaries(
    domain_curves: pd.DataFrame,
    domain_null: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        for process in PROCESSES:
            curve = domain_curves[
                (domain_curves["domain"] == domain) & (domain_curves["process"] == process)
            ].sort_values("k")
            effect = curve["excess"].to_numpy(dtype=float)
            radius = curve["radius_native_nn"].to_numpy(dtype=float)
            observed = curve["observed_equal_sample_mean"].to_numpy(dtype=float)
            null = domain_null[(domain, process)]
            null_center = np.median(null, axis=0)
            peak_index = int(np.nanargmax(effect))
            peak_effect = float(effect[peak_index])
            half_radius, censored = interpolate_half_decay(radius, effect, peak_index)
            null_max_t = np.nanmax(null - null_center[None, :], axis=1)
            max_t_p = upper_p(peak_effect, null_max_t)
            positive_area = float(np.trapz(np.maximum(effect, 0), x=np.log2(radius)))
            rows.append(
                {
                    "domain": domain,
                    "process": process,
                    "peak_k": int(curve.iloc[peak_index]["k"]),
                    "peak_radius_native_nn": float(radius[peak_index]),
                    "peak_effect": peak_effect,
                    "peak_pointwise_p": upper_p(float(observed[peak_index]), null[:, peak_index]),
                    "maxT_p_across_scales": max_t_p,
                    "half_decay_radius_native_nn": half_radius,
                    "half_decay_right_censored": bool(censored),
                    "positive_log2_radius_auc": positive_area,
                    "positive_direction": bool(peak_effect > 0),
                }
            )
    return pd.DataFrame(rows)


def make_decision(summary: pd.DataFrame, n_shifts: int) -> dict[str, object]:
    domain_checks: dict[str, dict[str, object]] = {}
    decay_directions: list[int] = []
    peak_directions: list[int] = []
    for domain in DOMAINS:
        local = summary[summary["domain"] == domain].set_index("process")
        neuron = local.loc["neuronal_homotypy"]
        cross = local.loc["astrocyte_vascular"]
        peak_delta = float(neuron["peak_radius_native_nn"] - cross["peak_radius_native_nn"])
        decay_delta = float(
            neuron["half_decay_radius_native_nn"] - cross["half_decay_radius_native_nn"]
        )
        peak_directions.append(int(np.sign(peak_delta)))
        decay_directions.append(int(np.sign(decay_delta)))
        both_uncensored = not bool(neuron["half_decay_right_censored"]) and not bool(
            cross["half_decay_right_censored"]
        )
        both_positive = bool(neuron["positive_direction"] and cross["positive_direction"])
        both_max_t = bool(
            neuron["maxT_p_across_scales"] <= 0.05 and cross["maxT_p_across_scales"] <= 0.05
        )
        ordered = bool(peak_delta >= 0 and decay_delta > 0 and both_uncensored)
        domain_checks[domain] = {
            "neuronal_minus_astrovascular_peak_radius_nn": peak_delta,
            "neuronal_minus_astrovascular_half_decay_radius_nn": decay_delta,
            "both_processes_positive": both_positive,
            "both_processes_maxT_p_le_0.05": both_max_t,
            "both_half_decays_resolved": both_uncensored,
            "predicted_order_neuronal_longer_than_astrovascular": ordered,
            "domain_pass": bool(both_positive and both_max_t and ordered),
        }
    decay_rank_consistent = bool(all(x > 0 for x in decay_directions))
    peak_rank_consistent = bool(all(x >= 0 for x in peak_directions))
    all_pass = bool(all(x["domain_pass"] for x in domain_checks.values()))
    replication_gate = False  # MERFISH is one animal and Slide-seq is one puck.
    if all_pass:
        verdict = "GO_TO_INDEPENDENT_REPLICATION_ONLY"
        rationale = (
            "The pre-specified topology hierarchy is consistent in the cached data, "
            "but independent-animal replication is absent for MERFISH and Slide-seqV2."
        )
    else:
        verdict = "NO_GO_FOR_CONSERVED_SCALE_HIERARCHY"
        rationale = (
            "At least one domain failed direction, max-T significance, resolved decay, "
            "or the pre-specified scale ordering."
        )
    return {
        "verdict": verdict,
        "rationale": rationale,
        "candidate_only": True,
        "discovery_claim_allowed": False,
        "n_spatial_shifts": n_shifts,
        "smallest_attainable_shift_p": 1.0 / (n_shifts + 1),
        "pre_specified_order": "astrocyte-vascular shorter than neuronal homotypy",
        "peak_rank_consistent": peak_rank_consistent,
        "half_decay_rank_consistent": decay_rank_consistent,
        "all_topology_domains_pass": all_pass,
        "independent_replication_gate_pass": replication_gate,
        "domain_checks": domain_checks,
        "limitations": [
            "DLPFC uses marker-score fields whereas MERFISH and Slide-seqV2 use labels.",
            "Species, brain region, and platform are confounded.",
            "MERFISH sections all come from one animal.",
            "Slide-seqV2 has one puck and no encoded animal identifier.",
            "MERFISH and Slide-seqV2 cached matrices are normalized, not valid SCT/scVI inputs.",
            "A toroidal shift null controls broad spatial autocorrelation but cannot model all tissue-boundary geometry.",  # noqa: E501
        ],
    }


def load_figure_helper() -> object:
    helper_path = Path(
        r"C:\Users\13264\.agents\skills\scientific-figure-pro\scripts\scientific_figure_pro.py"
    )
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load figure helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_figure(
    sample_curves: pd.DataFrame,
    domain_curves: pd.DataFrame,
    summary: pd.DataFrame,
    decision: dict[str, object],
    out_dir: Path,
) -> None:
    figpro = load_figure_helper()
    figpro.apply_publication_style(figpro.FigureStyle(font_size=11, axes_linewidth=1.6))
    fig, axes = figpro.create_subplots(2, 3, figsize=(13.5, 7.6))
    colors = {
        "neuronal_homotypy": figpro.PALETTE["blue_main"],
        "astrocyte_vascular": figpro.PALETTE["red_strong"],
    }
    labels = {
        "neuronal_homotypy": "Neuronal homotypy",
        "astrocyte_vascular": "Astrocyte–vascular",
    }

    for ax, domain, panel in zip(axes[:3], DOMAINS, ("A", "B", "C"), strict=False):
        for process in PROCESSES:
            curve = domain_curves[
                (domain_curves["domain"] == domain) & (domain_curves["process"] == process)
            ].sort_values("k")
            radius = curve["radius_native_nn"].to_numpy(dtype=float)
            effect = curve["excess"].to_numpy(dtype=float)
            peak = float(np.max(effect))
            relative = effect / peak if peak > 0 else effect
            ax.plot(
                radius,
                relative,
                color=colors[process],
                marker="o",
                markersize=4,
                linewidth=2.2,
                label=labels[process],
            )
            local = sample_curves[
                (sample_curves["domain"] == domain) & (sample_curves["process"] == process)
            ].copy()
            matrices: list[np.ndarray] = []
            for _, sample_frame in local.groupby("sample"):
                sample_frame = sample_frame.sort_values("k")
                values = sample_frame["excess"].to_numpy(dtype=float)
                sample_peak = float(np.max(values))
                if sample_peak > 0:
                    matrices.append(values / sample_peak)
            if len(matrices) > 1:
                matrix = np.vstack(matrices)
                ax.fill_between(
                    radius,
                    np.quantile(matrix, 0.25, axis=0),
                    np.quantile(matrix, 0.75, axis=0),
                    color=colors[process],
                    alpha=0.13,
                    linewidth=0,
                )
        ax.axhline(0, color="#777777", linewidth=0.9, linestyle="--")
        ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Radius (native NN units)")
        ax.set_ylabel("Shift-null excess / peak")
        ax.set_title(f"{panel}  {domain}")
        ax.set_ylim(-0.3, 1.15)
        if domain == "DLPFC":
            ax.legend(loc="best")

    # Half-decay scales.
    ax = axes[3]
    x = np.arange(len(DOMAINS), dtype=float)
    for offset, process in ((-0.12, "neuronal_homotypy"), (0.12, "astrocyte_vascular")):
        local = summary[summary["process"] == process].set_index("domain").loc[list(DOMAINS)]
        values = local["half_decay_radius_native_nn"].to_numpy(dtype=float)
        censored = local["half_decay_right_censored"].to_numpy(dtype=bool)
        ax.scatter(x + offset, values, s=58, color=colors[process], label=labels[process], zorder=3)
        for xx, yy, is_censored in zip(x + offset, values, censored, strict=False):
            if is_censored:
                ax.annotate(">", (xx, yy), xytext=(5, 0), textcoords="offset points", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, rotation=20, ha="right")
    ax.set_ylabel("Half-decay radius (native NN)")
    ax.set_title("D  Decay-scale comparison")
    ax.legend(loc="best")

    # Within-domain radius differences.
    ax = axes[4]
    deltas = [
        decision["domain_checks"][domain]["neuronal_minus_astrovascular_half_decay_radius_nn"]
        for domain in DOMAINS
    ]
    bars = ax.bar(
        x,
        deltas,
        color=[
            figpro.PALETTE["green_3"] if value > 0 else figpro.PALETTE["red_strong"]
            for value in deltas
        ],
        edgecolor="black",
        linewidth=0.9,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, rotation=20, ha="right")
    ax.set_ylabel("Neuronal − astrovascular\nhalf-decay radius")
    ax.set_title("E  Pre-specified ordering")
    for bar, value in zip(bars, deltas, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    # Explicit evidence/decision matrix.
    ax = axes[5]
    checks = (
        "both_processes_positive",
        "both_processes_maxT_p_le_0.05",
        "both_half_decays_resolved",
        "predicted_order_neuronal_longer_than_astrovascular",
    )
    check_labels = ("Positive", "max-T p≤0.05", "Decay resolved", "Expected order")
    matrix = np.asarray(
        [
            [float(decision["domain_checks"][domain][check]) for domain in DOMAINS]
            for check in checks
        ]
    )
    cmap = plt.matplotlib.colors.ListedColormap(
        [figpro.PALETTE["red_2"], figpro.PALETTE["green_2"]]
    )
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(check_labels)))
    ax.set_yticklabels(check_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "PASS" if matrix[i, j] else "FAIL", ha="center", va="center", fontsize=8)
    ax.set_title("F  Go/no-go evidence")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Candidate screen: conserved spatial-scale hierarchy across brain platforms",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figpro.finalize_figure(
        fig,
        out_dir / "figure_scale_hierarchy",
        formats=["png", "pdf", "svg"],
        dpi=600,
        pad=0.04,
    )


def write_method_note(out_dir: Path, decision: dict[str, object]) -> None:
    note = {
        "analysis_type": "topology-only candidate screen",
        "hypothesis": (
            "The astrocyte-vascular association peaks/decays at a shorter native-NN "
            "scale than broad neuronal homotypy in all three datasets."
        ),
        "neuronal_definition": {
            "DLPFC": "predeclared excitatory-minus-inhibitory marker-score field",
            "MERFISH": "same broad Excitatory/Inhibitory Cell_class among neuronal neighbours",
            "Slide-seqV2": "same broad pyramidal/interneuron domain among neuronal neighbours",
        },
        "astrocyte_vascular_definition": {
            "DLPFC": "symmetric cross-lag of predeclared astrocyte and vascular marker fields",
            "MERFISH": "symmetric Astrocyte-to-(Endothelial/Pericyte) neighbour fraction",
            "Slide-seqV2": "symmetric glia_astrocyte-to-vascular neighbour fraction",
        },
        "oligodendrocytes_included": False,
        "radius": "median kth-neighbour distance divided by median first-neighbour distance",
        "null": "2-D toroidal field translation with nearest-observation remapping",
        "scale_selection_control": "max-T over all tested radii",
        "decision": decision,
    }
    (out_dir / "method_and_decision.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo_root = script.parents[3]
    workspace = repo_root.parent
    staging = (
        workspace / "Biomni_lab_downloads_20260714_164953" / "histoweave_upgrade" / "datasets_cache"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--merfish",
        type=Path,
        default=staging / "merfish" / "merfish_mouse_hypothalamus.h5ad",
    )
    parser.add_argument(
        "--slideseq",
        type=Path,
        default=staging / "slideseqv2" / "slideseqv2_mouse_hippocampus.h5ad",
    )
    parser.add_argument("--out-dir", type=Path, default=script.parent / "results")
    parser.add_argument("--n-shifts", type=int, default=99)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_shifts < 19:
        raise ValueError("use at least 19 spatial shifts")
    k_values = np.asarray(sorted(set(args.k)), dtype=int)
    if np.any(k_values < 1):
        raise ValueError("all k values must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dlpfc, markers, dlpfc_audit = analyse_dlpfc(args.repo_root, k_values, args.n_shifts)
    merfish, merfish_audit = analyse_merfish(args.merfish, k_values, args.n_shifts)
    slideseq, slideseq_audit = analyse_slideseq(args.slideseq, k_values, args.n_shifts)
    results = [*dlpfc, *merfish, *slideseq]

    sample_curves = summarise_samples(results)
    domain_curves, domain_null = summarise_domains(results)
    summary = scale_summaries(domain_curves, domain_null)
    decision = make_decision(summary, args.n_shifts)

    sample_curves.to_csv(args.out_dir / "sample_scale_curves.csv", index=False)
    domain_curves.to_csv(args.out_dir / "domain_scale_curves.csv", index=False)
    summary.to_csv(args.out_dir / "scale_summary.csv", index=False)
    pd.DataFrame(markers).to_csv(args.out_dir / "dlpfc_marker_availability.csv", index=False)
    pd.DataFrame([*dlpfc_audit, *merfish_audit, *slideseq_audit]).to_csv(
        args.out_dir / "input_audit.csv", index=False
    )
    write_method_note(args.out_dir, decision)
    draw_figure(sample_curves, domain_curves, summary, decision, args.out_dir)

    _log(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
