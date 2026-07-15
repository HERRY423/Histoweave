#!/usr/bin/env python3
"""Directional Slide-seq check of the weak DLPFC vascular-barrier candidate.

The gene set is read unchanged from the DLPFC ``module_spec.tsv``. Mouse symbol
capitalization is matched case-insensitively, but missing genes are not replaced
by paralogs or additional markers.  The same deterministic 4,000-bead raw-count
pilot is evaluated through log-normalized, SCT, and scVI views.

This is a single-puck technical directional check, not biological validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
from sklearn.neighbors import NearestNeighbors

SEED = 20_011_508
K = 6
DEFAULT_SHIFTS = 999


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("slideseq_vascular_external_hypothesis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            output_dir / "vascular_external_hypothesis.log", mode="w", encoding="utf-8"
        ),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def as_csr(matrix: Any) -> sp.csr_matrix:
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)


def zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = np.std(array, ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-12:
        return np.zeros_like(array)
    return (array - np.mean(array)) / standard_deviation


def bh_qvalues(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def align_view(path: Path, obs_names: pd.Index, genes: list[str]) -> np.ndarray:
    view = ad.read_h5ad(path)
    obs_positions = view.obs_names.get_indexer(obs_names)
    if np.any(obs_positions < 0):
        raise ValueError(f"{path.name} is missing pilot barcodes")
    lookup = {str(gene).upper(): index for index, gene in enumerate(view.var_names)}
    gene_positions = [lookup[gene.upper()] for gene in genes]
    values = view.X[obs_positions][:, gene_positions]
    if sp.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} has non-finite values in the requested module")
    return values


def score_module(values: np.ndarray) -> np.ndarray:
    standardized = np.column_stack([zscore(values[:, column]) for column in range(values.shape[1])])
    return np.mean(standardized, axis=1)


def neighborhood_exposure(
    coords: np.ndarray,
    labels: np.ndarray,
    vascular: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=K + 1, algorithm="kd_tree", n_jobs=-1)
    distances, indices = model.fit(coords).kneighbors(coords)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    categories = sorted(set(labels))
    probabilities = np.column_stack(
        [np.mean(labels[indices] == category, axis=1) for category in categories]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probability = np.where(probabilities > 0, np.log(probabilities), 0.0)
    entropy = -np.sum(probabilities * log_probability, axis=1) / math.log(min(K, len(categories)))
    spacing = np.mean(distances, axis=1)
    local_vascular_abundance = np.mean(vascular[indices], axis=1)
    return entropy, spacing, local_vascular_abundance, indices


def nuisance_matrix(
    log_depth: np.ndarray,
    spacing: np.ndarray,
    labels: np.ndarray,
    vascular_abundance: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    columns = [
        np.ones(len(labels), dtype=float),
        zscore(log_depth),
        zscore(spacing),
        zscore(vascular_abundance),
    ]
    names = ["intercept", "z_log_library_depth", "z_local_spacing", "z_local_vascular_abundance"]
    categories = sorted(set(labels))
    for category in categories[1:]:
        columns.append((labels == category).astype(float))
        names.append(f"label[{category}]")
    matrix = np.column_stack(columns)
    rank = np.linalg.matrix_rank(matrix)
    if rank != matrix.shape[1]:
        raise ValueError(f"Nuisance matrix is rank deficient ({rank}/{matrix.shape[1]})")
    return matrix, names


def residualize(values: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    return values - nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]


def observed_effect(
    outcome: np.ndarray, exposure: np.ndarray, nuisance: np.ndarray
) -> dict[str, float]:
    y = zscore(outcome)
    x = zscore(exposure)
    residual_y = residualize(y, nuisance)
    residual_x = residualize(x, nuisance)
    denominator = float(residual_x @ residual_x)
    if denominator < 1e-12:
        raise ValueError("Exposure has no residual variance after nuisance adjustment")
    beta = float(residual_x @ residual_y / denominator)
    residual = residual_y - beta * residual_x
    degrees_freedom = len(y) - np.linalg.matrix_rank(nuisance) - 1
    residual_variance = float(residual @ residual / degrees_freedom)
    standard_error = math.sqrt(residual_variance / denominator)
    statistic = beta / standard_error
    analytic_p = float(2 * student_t.sf(abs(statistic), degrees_freedom))
    partial_r = float(np.corrcoef(residual_x, residual_y)[0, 1])
    nuisance_r2 = 1.0 - float(residual_y @ residual_y) / float(np.sum((y - y.mean()) ** 2))
    full_residual = residual_y - beta * residual_x
    full_r2 = 1.0 - float(full_residual @ full_residual) / float(np.sum((y - y.mean()) ** 2))
    return {
        "beta_entropy": beta,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "analytic_p": analytic_p,
        "partial_r": partial_r,
        "delta_r2": full_r2 - nuisance_r2,
        "degrees_freedom": degrees_freedom,
    }


def toroidal_shift_mappings(coords: np.ndarray, n_shifts: int, seed: int) -> list[np.ndarray]:
    low = np.min(coords, axis=0)
    span = np.ptp(coords, axis=0)
    if np.any(span <= 0):
        raise ValueError("Coordinates have zero spatial span")
    base = coords - low
    tree = cKDTree(coords)
    generator = np.random.default_rng(seed)
    mappings = []
    for _ in range(n_shifts):
        shift = generator.uniform(0.20, 0.80, size=2) * span
        query = np.mod(base + shift, span) + low
        mappings.append(tree.query(query, k=1)[1].astype(np.int64))
    return mappings


def shift_null(
    outcome: np.ndarray,
    entropy: np.ndarray,
    nuisance: np.ndarray,
    mappings: list[np.ndarray],
) -> np.ndarray:
    residual_y = residualize(zscore(outcome), nuisance)
    null = np.empty(len(mappings), dtype=np.float64)
    for index, mapping in enumerate(mappings):
        residual_x = residualize(zscore(entropy[mapping]), nuisance)
        denominator = float(residual_x @ residual_x)
        null[index] = (
            float(residual_x @ residual_y / denominator) if denominator > 1e-12 else np.nan
        )
    return null


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent / "dlpfc_sct_scvi" / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-pilot",
        type=Path,
        default=script_dir / "results" / "Puck_200115_08_raw_counts_sct_pilot_input.h5ad",
    )
    parser.add_argument(
        "--sct",
        type=Path,
        default=script_dir / "results" / "Puck_200115_08_sct_v2_pearson_pilot.h5ad",
    )
    parser.add_argument(
        "--scvi",
        type=Path,
        default=script_dir / "results" / "Puck_200115_08_scvi_40epoch_smoke.h5ad",
    )
    parser.add_argument("--module-spec", type=Path, default=root / "module_spec.tsv")
    parser.add_argument("--dlpfc-effects", type=Path, default=root / "overall_effects.csv")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--shifts", type=int, default=DEFAULT_SHIFTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logger = configure_logging(output_dir)
    for path in (args.raw_pilot, args.sct, args.scvi, args.module_spec, args.dlpfc_effects):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.shifts < 99:
        raise ValueError("Use at least 99 two-dimensional shifts")

    specification = pd.read_csv(args.module_spec, sep="\t")
    requested = (
        specification.loc[specification["module"] == "vascular_barrier", "gene"]
        .astype(str)
        .tolist()
    )
    if requested != ["CLDN5", "PECAM1", "VWF", "EMCN", "KDR", "RAMP2", "SLC2A1", "MFSD2A", "ABCB1"]:
        raise ValueError("DLPFC vascular_barrier gene specification changed unexpectedly")

    raw = ad.read_h5ad(args.raw_pilot)
    counts = as_csr(raw.layers["counts"])
    stored = counts.data
    if (
        not np.isfinite(stored).all()
        or np.any(stored < 0)
        or not np.allclose(stored, np.rint(stored), atol=0, rtol=0)
    ):
        raise ValueError("Raw pilot failed integer count contract")
    raw_lookup = {str(gene).upper(): str(gene) for gene in raw.var_names}
    used = [gene for gene in requested if gene in raw_lookup]
    missing = [gene for gene in requested if gene not in raw_lookup]
    if len(used) < 2:
        raise ValueError("Fewer than two pre-specified vascular_barrier genes are present")
    positions = [raw.var_names.get_loc(raw_lookup[gene]) for gene in used]
    raw_module_counts = counts[:, positions].toarray().astype(np.float64)
    depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    lognorm = np.log1p(raw_module_counts * (10_000.0 / depth[:, None]))
    sct = align_view(args.sct, raw.obs_names, used)
    scvi_values = np.log1p(align_view(args.scvi, raw.obs_names, used))
    scores = {
        "lognorm": score_module(lognorm),
        "SCT": score_module(sct),
        "scVI": score_module(scvi_values),
    }

    coords = np.asarray(raw.obsm["spatial"], dtype=np.float64)
    labels = raw.obs["cluster"].astype(str).to_numpy()
    vascular = raw.obs["domain_truth"].astype(str).to_numpy() == "vascular"
    entropy, spacing, vascular_abundance, indices = neighborhood_exposure(coords, labels, vascular)
    nuisance, nuisance_names = nuisance_matrix(np.log1p(depth), spacing, labels, vascular_abundance)
    mappings = toroidal_shift_mappings(coords, args.shifts, SEED + 991)

    dlpfc = pd.read_csv(args.dlpfc_effects)
    dlpfc = dlpfc.loc[dlpfc["module"] == "vascular_barrier"].set_index("branch")
    rows: list[dict[str, Any]] = []
    nulls: dict[str, np.ndarray] = {}
    for branch, outcome in scores.items():
        result = observed_effect(outcome, entropy, nuisance)
        null = shift_null(outcome, entropy, nuisance, mappings)
        nulls[branch] = null
        shift_p = (
            1 + int(np.sum(np.abs(null[np.isfinite(null)]) >= abs(result["beta_entropy"])))
        ) / (1 + int(np.isfinite(null).sum()))
        dlpfc_beta = float(dlpfc.loc[branch, "mean_beta_entropy"])
        rows.append(
            {
                "branch": branch,
                **result,
                "spatial_shift_p": shift_p,
                "null_median": float(np.nanmedian(null)),
                "null_sd": float(np.nanstd(null, ddof=1)),
                "direction": "positive" if result["beta_entropy"] > 0 else "negative",
                "dlpfc_beta_reference": dlpfc_beta,
                "direction_matches_dlpfc": bool(
                    np.sign(result["beta_entropy"]) == np.sign(dlpfc_beta)
                ),
                "n_beads": raw.n_obs,
                "n_module_genes_used": len(used),
            }
        )
    effects = pd.DataFrame(rows)
    effects["spatial_shift_q_bh"] = bh_qvalues(effects["spatial_shift_p"].to_numpy())
    effects.to_csv(output_dir / "vascular_external_hypothesis_effects.csv", index=False)
    np.savez_compressed(output_dir / "vascular_external_hypothesis_shift_nulls.npz", **nulls)

    coverage = pd.DataFrame(
        {
            "module": "vascular_barrier",
            "requested_gene": requested,
            "present_in_fixed_2024_gene_pilot": [gene in used for gene in requested],
            "matched_mouse_symbol": [raw_lookup.get(gene, "") for gene in requested],
            "substitution_used": False,
        }
    )
    coverage.to_csv(output_dir / "vascular_external_hypothesis_gene_coverage.csv", index=False)
    score_frame = pd.DataFrame(scores, index=raw.obs_names)
    score_frame.insert(0, "cluster", labels)
    score_frame.insert(0, "barcode", raw.obs_names.astype(str))
    score_frame.to_csv(output_dir / "vascular_external_hypothesis_scores.csv", index=False)
    score_frame[list(scores)].corr().to_csv(
        output_dir / "vascular_external_hypothesis_branch_correlations.csv"
    )

    exposure_qc = pd.DataFrame(
        {
            "barcode": raw.obs_names.astype(str),
            "cluster": labels,
            "domain_truth": raw.obs["domain_truth"].astype(str).to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "annotation_entropy_k6": entropy,
            "local_spacing_k6": spacing,
            "local_vascular_abundance_k6": vascular_abundance,
            "log_library_depth": np.log1p(depth),
        }
    )
    exposure_qc.to_csv(output_dir / "vascular_external_hypothesis_exposure_qc.csv", index=False)

    scvi_row = effects.set_index("branch").loc["scVI"]
    report = {
        "status": "technical_directional_check_complete",
        "hypothesis": "The weak positive DLPFC scVI vascular_barrier association recurs directionally with local annotation entropy in Slide-seqV2.",  # noqa: E501
        "pre_specified_source": str(args.module_spec.resolve()),
        "module_spec_sha256": sha256(args.module_spec),
        "requested_genes": requested,
        "used_genes": used,
        "missing_without_substitution": missing,
        "coverage": len(used) / len(requested),
        "exposure": "six-nearest-neighbor Shannon entropy of cached hippocampal cluster annotations",  # noqa: E501
        "controls": nuisance_names,
        "vascular_abundance_definition": "fraction of six neighbors with cached domain_truth == vascular",  # noqa: E501
        "spatial_null": {
            "method": "2-D toroidal translation with nearest-observation remapping",
            "n_shifts": args.shifts,
            "seed": SEED + 991,
            "smallest_attainable_p": 1 / (args.shifts + 1),
        },
        "scvi_prior_direction_passed": bool(scvi_row["beta_entropy"] > 0),
        "scvi_shift_p_le_0_05": bool(scvi_row["spatial_shift_p"] <= 0.05),
        "all_branch_directions_match_dlpfc": bool(effects["direction_matches_dlpfc"].all()),
        "normalization_robust_same_direction": bool(len(set(effects["direction"])) == 1),
        "biological_n": 1,
        "independent_replication": False,
        "biological_validation": False,
        "claim_boundary": "Technical consistency or inconsistency only; this single puck cannot upgrade the DLPFC weak candidate to a biological finding.",  # noqa: E501
        "limitations": [
            "Only five of nine pre-specified genes are present in the fixed pilot; no paralog substitution was allowed.",  # noqa: E501
            "The 4,000 beads are a deterministic stratified subset, so kNN spacing is controlled but does not recreate the full-puck graph.",  # noqa: E501
            "Transferred labels and coordinates come from the same public puck and are not independent validation.",  # noqa: E501
            "Toroidal remapping controls broad 2-D autocorrelation but not all irregular tissue-boundary geometry.",  # noqa: E501
        ],
        "effects": effects.to_dict(orient="records"),
    }
    write_json(output_dir / "vascular_external_hypothesis.json", report)
    logger.info(
        "COMPLETE single-puck directional check; scVI beta=%.5f shift_p=%.4f; biological_validation=false",  # noqa: E501
        scvi_row["beta_entropy"],
        scvi_row["spatial_shift_p"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
