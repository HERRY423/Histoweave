"""DLPFC spatial-neighborhood robustness pilot across lognorm, SCT, and scVI.

This is a deliberately narrow, donor-aware experiment.  It asks whether a
predeclared glial-support program is associated with local layer mixing in
DLPFC, and whether the sign and magnitude survive three genuinely different
normalization/modeling branches:

* library-size log normalization from integer UMI counts;
* SCTransform v2 Pearson residuals fit independently in each section;
* scVI trained directly on the same integer UMI counts, with section as batch.

SCTransform residuals are never supplied to scVI.  The default ``pilot`` scope
audits all twelve sections but chooses one representative (median-size) section
per donor without looking at the biological outcome.  ``--scope all`` runs all
twelve sections.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

SEED = 20260715
T_975_DF2 = 4.3026527297

DLPFC_DONOR: dict[str, str] = {
    **{str(section): "donor_1" for section in range(151507, 151511)},
    **{str(section): "donor_2" for section in range(151669, 151673)},
    **{str(section): "donor_3" for section in range(151673, 151677)},
}

MODULES: dict[str, tuple[str, ...]] = {
    "astro_ion": (
        "AQP4",
        "SLC1A2",
        "SLC1A3",
        "KCNJ10",
        "GLUL",
        "ATP1A2",
        "GJA1",
        "ALDOC",
        "GFAP",
    ),
    "oligo_myelin": (
        "MBP",
        "PLP1",
        "MOG",
        "MAG",
        "CNP",
        "CLDN11",
        "MAL",
        "SOX10",
        "OLIG1",
        "OLIG2",
    ),
    "vascular_barrier": (
        "CLDN5",
        "PECAM1",
        "VWF",
        "EMCN",
        "KDR",
        "RAMP2",
        "SLC2A1",
        "MFSD2A",
        "ABCB1",
    ),
    "neuronal_synaptic_control": (
        "SNAP25",
        "SYT1",
        "RBFOX3",
        "CAMK2A",
        "SLC17A7",
        "GAD1",
        "GAD2",
    ),
    "housekeeping_control": (
        "RPLP0",
        "RPL13A",
        "ACTB",
        "GAPDH",
        "TUBB",
        "EEF1A1",
    ),
}
PRIMARY_MODULES = ("astro_ion", "oligo_myelin", "vascular_barrier", "GEI")
GEI_COMPONENTS = ("astro_ion", "oligo_myelin", "vascular_barrier")
BRANCHES = ("lognorm", "SCT", "scVI")
LAYER_LEVELS = ("Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6", "WM")


def configure_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dlpfc_sct_scvi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def matrix_as_csr(matrix: object) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix, dtype=np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def audit_integer_counts(matrix: object) -> dict[str, object]:
    csr = matrix_as_csr(matrix)
    values = csr.data
    finite = bool(np.isfinite(values).all())
    nonnegative = bool(values.size > 0 and np.min(values) >= -1e-6)
    max_fractional = float(np.max(np.abs(values - np.rint(values)), initial=0.0))
    integer_like = bool(finite and nonnegative and max_fractional <= 1e-6)
    return {
        "finite": finite,
        "nonnegative": nonnegative,
        "integer_like": integer_like,
        "max_fractional_error": max_fractional,
        "nnz": int(csr.nnz),
    }


def matrices_identical(left: object, right: object) -> bool:
    a = matrix_as_csr(left)
    b = matrix_as_csr(right)
    if a.shape != b.shape:
        return False
    difference = a - b
    return bool(difference.nnz == 0)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd < 1e-10:
        return np.zeros(len(values), dtype=float)
    return (values - mean) / sd


def marker_membership() -> dict[str, list[str]]:
    membership: dict[str, list[str]] = defaultdict(list)
    for module, genes in MODULES.items():
        for gene in genes:
            membership[gene].append(module)
    return dict(membership)


def audit_and_select_features(
    files: list[Path],
    n_hvg: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Audit every section and select global lognorm-dispersion HVGs."""
    audit_rows: list[dict[str, object]] = []
    gene_names: np.ndarray | None = None
    log_sum: np.ndarray | None = None
    log_sq_sum: np.ndarray | None = None
    nonzero: np.ndarray | None = None
    n_total = 0

    for path in files:
        section = path.stem.rsplit("_", 1)[-1]
        if section not in DLPFC_DONOR:
            raise RuntimeError(f"section has no explicit donor mapping: {section}")
        logger.info("Auditing raw counts: %s", section)
        a = ad.read_h5ad(path)
        if "counts" not in a.layers:
            raise RuntimeError(f"{section}: missing layers['counts']")
        if "domain_truth" not in a.obs or "spatial" not in a.obsm:
            raise RuntimeError(f"{section}: missing domain_truth or spatial coordinates")
        counts = matrix_as_csr(a.layers["counts"])
        count_check = audit_integer_counts(counts)
        x_matches = matrices_identical(a.X, counts)
        current_genes = np.asarray(a.var_names.astype(str))
        if gene_names is None:
            gene_names = current_genes
            log_sum = np.zeros(len(gene_names), dtype=np.float64)
            log_sq_sum = np.zeros(len(gene_names), dtype=np.float64)
            nonzero = np.zeros(len(gene_names), dtype=np.int64)
        elif not np.array_equal(gene_names, current_genes):
            raise RuntimeError(f"{section}: gene order differs from the other sections")
        if not all(count_check[key] for key in ("finite", "nonnegative", "integer_like")):
            raise RuntimeError(f"{section}: raw-count gate failed: {count_check}")
        totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        if np.any(totals <= 0):
            raise RuntimeError(f"{section}: zero-depth spots detected")
        normalized = counts.astype(np.float64).multiply((1e4 / totals)[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
        squared = normalized.copy()
        squared.data **= 2
        assert log_sum is not None and log_sq_sum is not None and nonzero is not None
        log_sum += np.asarray(normalized.sum(axis=0)).ravel()
        log_sq_sum += np.asarray(squared.sum(axis=0)).ravel()
        nonzero += counts.getnnz(axis=0)
        n_total += int(a.n_obs)
        labels = a.obs["domain_truth"].astype(str)
        label_valid = labels.isin(LAYER_LEVELS)
        audit_rows.append(
            {
                "section": section,
                "donor": DLPFC_DONOR[section],
                "path": str(path),
                "n_spots": int(a.n_obs),
                "n_genes": int(a.n_vars),
                "n_valid_layer_spots": int(label_valid.sum()),
                "counts_layer_present": True,
                "x_identical_to_counts": x_matches,
                **count_check,
            }
        )
        del a, counts, normalized, squared
        gc.collect()

    if gene_names is None or log_sum is None or log_sq_sum is None or nonzero is None:
        raise RuntimeError("no DLPFC files were found")
    mean = log_sum / n_total
    variance = np.maximum(log_sq_sum / n_total - mean**2, 0.0)
    dispersion = variance / np.maximum(mean, 1e-8)
    expressed = nonzero >= max(20, int(math.ceil(0.001 * n_total)))

    feature = pd.DataFrame(
        {
            "gene": gene_names,
            "lognorm_mean": mean,
            "lognorm_variance": variance,
            "dispersion": dispersion,
            "n_nonzero_spots": nonzero,
            "passes_expression_filter": expressed,
        }
    )
    feature["mean_bin"] = -1
    eligible_indices = np.flatnonzero(expressed & np.isfinite(dispersion))
    ranks = pd.Series(mean[eligible_indices]).rank(method="first", pct=True).to_numpy()
    bins = np.minimum((ranks * 20).astype(int), 19)
    feature.loc[eligible_indices, "mean_bin"] = bins
    feature["hvg_score"] = -np.inf
    for mean_bin in range(20):
        indices = eligible_indices[bins == mean_bin]
        if len(indices) == 0:
            continue
        values = dispersion[indices]
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center))) * 1.4826
        if mad < 1e-10:
            mad = float(np.std(values)) or 1.0
        feature.loc[indices, "hvg_score"] = (values - center) / mad
    top = (
        feature.loc[feature["passes_expression_filter"]]
        .sort_values(["hvg_score", "dispersion", "gene"], ascending=[False, False, True])
        .head(n_hvg)["gene"]
        .tolist()
    )
    marker_map = marker_membership()
    measured_markers = sorted(set(marker_map).intersection(set(gene_names)))
    selected = list(top)
    selected_set = set(selected)
    for gene in measured_markers:
        if gene not in selected_set:
            selected.append(gene)
            selected_set.add(gene)
    feature["is_hvg"] = feature["gene"].isin(top)
    feature["is_predeclared_marker"] = feature["gene"].isin(measured_markers)
    feature["marker_modules"] = feature["gene"].map(
        lambda gene: ";".join(marker_map.get(str(gene), []))
    )
    feature["selected_for_models"] = feature["gene"].isin(selected)
    feature["selection_rank"] = pd.NA
    rank_map = {gene: rank + 1 for rank, gene in enumerate(selected)}
    mask = feature["selected_for_models"]
    feature.loc[mask, "selection_rank"] = feature.loc[mask, "gene"].map(rank_map)
    feature = feature.sort_values(
        ["selected_for_models", "selection_rank", "gene"],
        ascending=[False, True, True],
    )
    logger.info(
        "Raw-count gate passed for all %d sections; selected %d HVGs + markers = %d genes",
        len(files),
        n_hvg,
        len(selected),
    )
    return pd.DataFrame(audit_rows), feature, selected


def choose_scope(audit: pd.DataFrame, scope: str) -> list[str]:
    if scope == "all":
        return sorted(audit["section"].astype(str).tolist())
    selected: list[str] = []
    for _donor, group in audit.groupby("donor", sort=True):
        median_size = float(group["n_spots"].median())
        ranked = group.assign(
            distance=(group["n_spots"] - median_size).abs(),
            section_string=group["section"].astype(str),
        ).sort_values(["distance", "section_string"])
        selected.append(str(ranked.iloc[0]["section"]))
    return sorted(selected)


def subset_raw_section(
    path: Path,
    section: str,
    selected_genes: list[str],
    intermediate_dir: Path,
    logger: logging.Logger,
) -> Path:
    output = intermediate_dir / f"raw_selected_{section}.h5ad"
    a = ad.read_h5ad(path)
    indices = a.var_names.get_indexer(selected_genes)
    if np.any(indices < 0):
        missing = np.asarray(selected_genes)[indices < 0].tolist()
        raise RuntimeError(f"{section}: selected genes missing: {missing[:10]}")
    sub = a[:, indices].copy()
    raw = matrix_as_csr(sub.layers["counts"])
    check = audit_integer_counts(raw)
    if not check["integer_like"]:
        raise RuntimeError(f"{section}: selected raw matrix is not integer-like")
    sub.X = raw.copy()
    sub.layers["counts"] = raw.copy()
    sub.obs["section"] = section
    sub.obs["donor"] = DLPFC_DONOR[section]
    sub.obs["source_barcode"] = sub.obs_names.astype(str)
    sub.obs_names = pd.Index([f"{section}:{barcode}" for barcode in sub.obs_names])
    sub.uns["raw_count_gate"] = {
        "integer_like": True,
        "source_layer": "counts",
        "section": section,
        "donor": DLPFC_DONOR[section],
    }
    sub.write_h5ad(output, compression="gzip")
    logger.info("Prepared raw selected matrix: %s (%d x %d)", section, sub.n_obs, sub.n_vars)
    del a, sub, raw
    gc.collect()
    return output


def run_sct(
    raw_path: Path,
    output_path: Path,
    rscript: Path,
    bridge: Path,
    n_cells: int,
    logger: logging.Logger,
) -> None:
    command = [
        str(rscript),
        str(bridge),
        str(raw_path),
        str(output_path),
        "layer=counts",
        "vst_flavor=v2",
        "residual_type=pearson",
        "min_cells=5",
        f"n_cells={n_cells}",
    ]
    logger.info("Running SCTransform: %s", raw_path.name)
    started = time.time()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.stdout.strip():
        logger.info("SCT stdout | %s", process.stdout.strip().replace("\n", " | "))
    if process.stderr.strip():
        logger.info("SCT stderr | %s", process.stderr.strip().replace("\n", " | "))
    if process.returncode != 0:
        raise RuntimeError(f"SCTransform failed ({process.returncode}): {' '.join(command)}")
    logger.info("SCTransform completed in %.1f s: %s", time.time() - started, output_path.name)


def neighborhood_frame(a: ad.AnnData, k: int = 6) -> pd.DataFrame:
    coords = np.asarray(a.obsm["spatial"], dtype=float)
    labels = a.obs["domain_truth"].astype(str).to_numpy()
    valid = np.isin(labels, LAYER_LEVELS)
    if not valid.all():
        raise RuntimeError("pilot expects every selected DLPFC spot to have a valid layer label")
    requested = min(k + 1, len(coords))
    neighbors = NearestNeighbors(n_neighbors=requested, algorithm="kd_tree", n_jobs=-1)
    distances, indices = neighbors.fit(coords).kneighbors(coords)
    distances, indices = distances[:, 1:], indices[:, 1:]
    codes = pd.Categorical(labels, categories=LAYER_LEVELS).codes
    neighbor_codes = codes[indices]
    probabilities = np.zeros((len(coords), len(LAYER_LEVELS)), dtype=np.float32)
    for code in range(len(LAYER_LEVELS)):
        probabilities[:, code] = np.mean(neighbor_codes == code, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probabilities > 0, np.log(probabilities), 0.0)
    entropy = -np.sum(probabilities * logp, axis=1) / math.log(min(k, len(LAYER_LEVELS)))
    discordance = np.mean(neighbor_codes != codes[:, None], axis=1)
    counts = matrix_as_csr(a.layers["counts"])
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    return pd.DataFrame(
        {
            "spot_id": a.obs_names.astype(str),
            "section": a.obs["section"].astype(str).to_numpy(),
            "donor": a.obs["donor"].astype(str).to_numpy(),
            "layer": labels,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "total_counts": totals,
            "log_total_counts": np.log1p(totals),
            "local_spacing": np.mean(distances, axis=1),
            "layer_entropy": entropy,
            "neighbor_discordance": discordance,
        }
    )


def dense_gene_values(matrix: object, indices: list[int]) -> np.ndarray:
    selected = matrix[:, indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float64)


def score_modules(
    expression: np.ndarray,
    genes: list[str],
    branch: str,
    section: str,
    eligible_genes: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    lookup = {str(gene): index for index, gene in enumerate(genes)}
    score_data: dict[str, np.ndarray] = {}
    coverage: list[dict[str, object]] = []
    for module, requested in MODULES.items():
        present = [gene for gene in requested if gene in lookup]
        used = [gene for gene in present if eligible_genes is None or gene in eligible_genes]
        if len(used) < 2:
            score = np.full(expression.shape[0], np.nan, dtype=float)
        else:
            values = expression[:, [lookup[gene] for gene in used]]
            standardized = np.column_stack(
                [zscore(values[:, index]) for index in range(values.shape[1])]
            )
            score = np.nanmean(standardized, axis=1)
        score_data[module] = score
        coverage.append(
            {
                "branch": branch,
                "section": section,
                "module": module,
                "n_requested": len(requested),
                "n_present": len(present),
                "n_used": len(used),
                "genes_used": ";".join(used),
            }
        )
    components = [zscore(score_data[module]) for module in GEI_COMPONENTS]
    score_data["GEI"] = np.nanmean(np.column_stack(components), axis=1)
    coverage.append(
        {
            "branch": branch,
            "section": section,
            "module": "GEI",
            "n_requested": sum(len(MODULES[module]) for module in GEI_COMPONENTS),
            "n_present": sum(
                len([gene for gene in MODULES[module] if gene in lookup])
                for module in GEI_COMPONENTS
            ),
            "n_used": sum(
                len(
                    [
                        gene
                        for gene in MODULES[module]
                        if gene in lookup and (eligible_genes is None or gene in eligible_genes)
                    ]
                )
                for module in GEI_COMPONENTS
            ),
            "genes_used": "component module z-scores",
        }
    )
    return pd.DataFrame(score_data), coverage


def lognorm_scores(a: ad.AnnData, section: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    counts = matrix_as_csr(a.layers["counts"])
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    marker_genes = sorted(set(marker_membership()).intersection(set(a.var_names.astype(str))))
    indices = a.var_names.get_indexer(marker_genes).tolist()
    values = dense_gene_values(counts, indices)
    values = np.log1p(values * (1e4 / totals[:, None]))
    return score_modules(values, marker_genes, "lognorm", section)


def sct_scores(a: ad.AnnData, section: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    marker_genes = sorted(set(marker_membership()).intersection(set(a.var_names.astype(str))))
    indices = a.var_names.get_indexer(marker_genes).tolist()
    residuals = dense_gene_values(a.X, indices)
    if "sctransform_modeled" in a.var:
        modeled = set(a.var_names[a.var["sctransform_modeled"].astype(bool)].astype(str))
    else:
        modeled = set(marker_genes)
    return score_modules(residuals, marker_genes, "SCT", section, eligible_genes=modeled)


def train_scvi_and_score(
    raw_adatas: list[ad.AnnData],
    sections: list[str],
    out_dir: Path,
    max_epochs: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    import scvi
    import torch

    scvi.settings.seed = SEED
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    combined = ad.concat(raw_adatas, axis=0, join="inner", merge="same", index_unique=None)
    combined.obs["section"] = combined.obs["section"].astype("category")
    combined.obs["donor"] = combined.obs["donor"].astype("category")
    check = audit_integer_counts(combined.layers["counts"])
    if not check["integer_like"]:
        raise RuntimeError("combined scVI input failed integer raw-count gate")
    scvi.model.SCVI.setup_anndata(combined, layer="counts", batch_key="section")
    model = scvi.model.SCVI(
        combined,
        n_hidden=128,
        n_latent=15,
        n_layers=2,
        dispersion="gene-batch",
        gene_likelihood="nb",
    )
    logger.info(
        "Training scVI 1.3.3 on raw counts only: %d spots x %d genes, %d batches",
        combined.n_obs,
        combined.n_vars,
        len(sections),
    )
    started = time.time()
    model.train(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        train_size=0.90,
        batch_size=256,
        early_stopping=True,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )
    logger.info("scVI training completed in %.1f s", time.time() - started)
    model.save(str(out_dir / "scvi_model"), overwrite=True, save_anndata=False)

    history_rows: list[dict[str, object]] = []
    for metric, table in model.history.items():
        frame = pd.DataFrame(table)
        if frame.empty:
            continue
        for epoch, value in zip(frame.index, frame.iloc[:, 0], strict=False):
            history_rows.append({"metric": str(metric), "epoch": int(epoch), "value": float(value)})
    history = pd.DataFrame(history_rows)

    latent = model.get_latent_representation(combined)
    latent_frame = pd.DataFrame(latent, index=combined.obs_names)
    latent_frame.columns = [f"scVI_latent_{index + 1}" for index in range(latent.shape[1])]
    latent_frame.insert(0, "donor", combined.obs["donor"].astype(str).to_numpy())
    latent_frame.insert(0, "section", combined.obs["section"].astype(str).to_numpy())
    latent_frame.insert(0, "spot_id", combined.obs_names.astype(str))

    marker_genes = sorted(
        set(marker_membership()).intersection(set(combined.var_names.astype(str)))
    )
    logger.info(
        "Decoding %d marker genes while averaging counterfactual expression over %d section batches",  # noqa: E501
        len(marker_genes),
        len(sections),
    )
    normalized = model.get_normalized_expression(
        combined,
        transform_batch=sections,
        gene_list=marker_genes,
        library_size=1e4,
        n_samples=1,
        batch_size=512,
        return_mean=True,
        return_numpy=True,
        silent=False,
    )
    normalized = np.log1p(np.asarray(normalized, dtype=float))
    score_frames: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    section_values = combined.obs["section"].astype(str).to_numpy()
    for section in sections:
        mask = section_values == section
        scored, rows = score_modules(normalized[mask], marker_genes, "scVI", section)
        scored.insert(0, "spot_id", combined.obs_names[mask].astype(str))
        score_frames.append(scored)
        coverage.extend(rows)
    return pd.concat(score_frames, ignore_index=True), coverage, history, latent_frame


def nuisance_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = [
        np.ones(len(frame), dtype=float),
        zscore(frame["log_total_counts"].to_numpy()),
        zscore(frame["local_spacing"].to_numpy()),
    ]
    labels = frame["layer"].astype(str).to_numpy()
    for level in LAYER_LEVELS[1:]:
        columns.append((labels == level).astype(float))
    return np.column_stack(columns)


def spatial_order(coords: np.ndarray) -> np.ndarray:
    centered = np.asarray(coords, dtype=float) - np.mean(coords, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return np.argsort(centered @ vt[0])


def partial_effect(
    outcome: np.ndarray,
    entropy: np.ndarray,
    nuisance: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    y = zscore(outcome)
    x = zscore(entropy)
    q, _ = np.linalg.qr(nuisance, mode="reduced")
    residual_y = y - q @ (q.T @ y)
    residual_x = x - q @ (q.T @ x)
    denominator = float(residual_x @ residual_x)
    beta = float(residual_x @ residual_y / denominator) if denominator > 1e-12 else math.nan
    partial_r = float(np.corrcoef(residual_x, residual_y)[0, 1])
    nuisance_fit = nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    full_fit = nuisance_fit + beta * residual_x
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r2_nuisance = 1.0 - float(np.sum((y - nuisance_fit) ** 2)) / ss_total
    r2_full = 1.0 - float(np.sum((y - full_fit) ** 2)) / ss_total
    return beta, partial_r, r2_full - r2_nuisance, residual_y, q


def section_effects_and_nulls(
    spot_scores: pd.DataFrame,
    selected_sections: list[str],
    n_permutations: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    nulls: dict[tuple[str, str, str], np.ndarray] = {}
    for section in selected_sections:
        section_frame = spot_scores.loc[spot_scores["section"] == section].copy()
        entropy = section_frame["layer_entropy"].to_numpy(dtype=float)
        nuisance = nuisance_matrix(section_frame)
        order = spatial_order(section_frame[["x", "y"]].to_numpy())
        n = len(section_frame)
        generator = np.random.default_rng(SEED + int(section))
        offsets = generator.integers(max(1, int(0.10 * n)), max(2, int(0.90 * n)), n_permutations)
        for branch in BRANCHES:
            for module in MODULES.keys() | {"GEI"}:
                column = f"{branch}__{module}"
                outcome = section_frame[column].to_numpy(dtype=float)
                finite = np.isfinite(outcome) & np.isfinite(entropy)
                if not finite.all():
                    raise RuntimeError(f"non-finite score in {section} {branch} {module}")
                beta, partial_r, delta_r2, residual_y, q = partial_effect(
                    outcome, entropy, nuisance
                )
                null = np.empty(n_permutations, dtype=float)
                for index, offset in enumerate(offsets):
                    shifted = np.empty(n, dtype=float)
                    shifted[order] = np.roll(entropy[order], int(offset))
                    shifted_z = zscore(shifted)
                    residual_x = shifted_z - q @ (q.T @ shifted_z)
                    denominator = float(residual_x @ residual_x)
                    null[index] = (
                        float(residual_x @ residual_y / denominator)
                        if denominator > 1e-12
                        else math.nan
                    )
                spatial_p = (1 + int(np.sum(np.abs(null) >= abs(beta)))) / (n_permutations + 1)
                nulls[(section, branch, module)] = null
                rows.append(
                    {
                        "section": section,
                        "donor": DLPFC_DONOR[section],
                        "branch": branch,
                        "module": module,
                        "beta_entropy": beta,
                        "partial_r": partial_r,
                        "delta_r2": delta_r2,
                        "spatial_shift_p_section": spatial_p,
                        "n_spots": n,
                    }
                )
    return pd.DataFrame(rows), nulls


def bh_qvalues(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def aggregate_effects(
    section_effects: pd.DataFrame,
    nulls: dict[tuple[str, str, str], np.ndarray],
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_effects = section_effects.groupby(["donor", "branch", "module"], as_index=False).agg(
        beta_entropy=("beta_entropy", "mean"),
        partial_r=("partial_r", "mean"),
        delta_r2=("delta_r2", "mean"),
        n_sections=("section", "nunique"),
        n_spots=("n_spots", "sum"),
    )
    overall_rows: list[dict[str, object]] = []
    donors = sorted(donor_effects["donor"].unique())
    for branch in BRANCHES:
        for module in MODULES.keys() | {"GEI"}:
            subset = donor_effects.loc[
                (donor_effects["branch"] == branch) & (donor_effects["module"] == module)
            ].sort_values("donor")
            effects = subset["beta_entropy"].to_numpy(dtype=float)
            mean = float(np.mean(effects))
            sd = float(np.std(effects, ddof=1)) if len(effects) > 1 else math.nan
            half = T_975_DF2 * sd / math.sqrt(len(effects)) if len(effects) == 3 else math.nan
            donor_nulls: list[np.ndarray] = []
            for donor in donors:
                sections = (
                    section_effects.loc[
                        (section_effects["donor"] == donor)
                        & (section_effects["branch"] == branch)
                        & (section_effects["module"] == module),
                        "section",
                    ]
                    .astype(str)
                    .tolist()
                )
                donor_nulls.append(
                    np.mean(
                        np.vstack([nulls[(section, branch, module)] for section in sections]),
                        axis=0,
                    )
                )
            overall_null = np.mean(np.vstack(donor_nulls), axis=0)
            spatial_p = (1 + int(np.sum(np.abs(overall_null) >= abs(mean)))) / (n_permutations + 1)
            sign = np.sign(mean)
            sign_fraction = float(np.mean(np.sign(effects) == sign)) if sign != 0 else 0.0
            overall_rows.append(
                {
                    "branch": branch,
                    "module": module,
                    "mean_beta_entropy": mean,
                    "ci95_low_donor_t": mean - half if np.isfinite(half) else math.nan,
                    "ci95_high_donor_t": mean + half if np.isfinite(half) else math.nan,
                    "sd_between_donors": sd,
                    "n_donors": len(effects),
                    "donor_sign_fraction": sign_fraction,
                    "mean_partial_r": float(subset["partial_r"].mean()),
                    "mean_delta_r2": float(subset["delta_r2"].mean()),
                    "spatial_shift_p": spatial_p,
                    "n_permutations": n_permutations,
                }
            )
    overall = pd.DataFrame(overall_rows)
    overall["spatial_shift_q_bh"] = bh_qvalues(overall["spatial_shift_p"])
    return donor_effects, overall


def lodo_prediction(spot_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    donors = sorted(spot_scores["donor"].unique())
    layer_columns = list(LAYER_LEVELS[1:])
    for branch in BRANCHES:
        for module in MODULES.keys() | {"GEI"}:
            outcome_column = f"{branch}__{module}"
            frame = spot_scores.copy()
            frame["outcome"] = frame.groupby("section")[outcome_column].transform(
                lambda values: zscore(values.to_numpy())
            )
            frame["z_depth"] = frame.groupby("section")["log_total_counts"].transform(
                lambda values: zscore(values.to_numpy())
            )
            frame["z_spacing"] = frame.groupby("section")["local_spacing"].transform(
                lambda values: zscore(values.to_numpy())
            )
            frame["z_entropy"] = frame.groupby("section")["layer_entropy"].transform(
                lambda values: zscore(values.to_numpy())
            )
            for level in layer_columns:
                frame[level] = (frame["layer"].astype(str) == level).astype(float)
            baseline_columns = ["z_depth", "z_spacing", *layer_columns]
            full_columns = [*baseline_columns, "z_entropy"]
            for held_out in donors:
                train = frame.loc[frame["donor"] != held_out]
                test = frame.loc[frame["donor"] == held_out]
                y_train = train["outcome"].to_numpy(dtype=float)
                y_test = test["outcome"].to_numpy(dtype=float)
                result: dict[str, object] = {
                    "branch": branch,
                    "module": module,
                    "held_out_donor": held_out,
                    "n_train_spots": len(train),
                    "n_test_spots": len(test),
                }
                for name, columns in (("baseline", baseline_columns), ("full", full_columns)):
                    x_train = np.column_stack(
                        [np.ones(len(train)), train[columns].to_numpy(dtype=float)]
                    )
                    x_test = np.column_stack(
                        [np.ones(len(test)), test[columns].to_numpy(dtype=float)]
                    )
                    coefficient = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
                    prediction = x_test @ coefficient
                    denominator = float(np.sum((y_test - np.mean(y_test)) ** 2))
                    r2 = 1.0 - float(np.sum((y_test - prediction) ** 2)) / denominator
                    result[f"r2_{name}"] = r2
                result["delta_r2_heldout"] = float(result["r2_full"] - result["r2_baseline"])
                rows.append(result)
    return pd.DataFrame(rows)


def concordance_tables(
    donor_effects: pd.DataFrame,
    overall: pd.DataFrame,
    lodo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    module_rows: list[dict[str, object]] = []
    for module in PRIMARY_MODULES:
        current = overall.loc[overall["module"] == module].set_index("branch")
        means = current.loc[list(BRANCHES), "mean_beta_entropy"].to_numpy(dtype=float)
        donor_ok = current.loc[list(BRANCHES), "donor_sign_fraction"].to_numpy(dtype=float) >= (
            2 / 3
        )
        q_ok = current.loc[list(BRANCHES), "spatial_shift_q_bh"].to_numpy(dtype=float) <= 0.10
        held = lodo.loc[lodo["module"] == module]
        lodo_fraction = held.groupby("branch")["delta_r2_heldout"].apply(lambda x: np.mean(x > 0))
        lodo_ok = np.array([lodo_fraction.get(branch, 0.0) >= (2 / 3) for branch in BRANCHES])
        sign_concordant = bool(np.all(np.sign(means) == np.sign(means[0])) and means[0] != 0)
        module_rows.append(
            {
                "module": module,
                **{f"mean_beta_{branch}": means[index] for index, branch in enumerate(BRANCHES)},
                "branch_sign_concordant": sign_concordant,
                "minimum_absolute_branch_effect": float(np.min(np.abs(means))),
                "all_branches_donor_direction_ge_2_of_3": bool(np.all(donor_ok)),
                "all_branches_spatial_q_le_0_10": bool(np.all(q_ok)),
                "all_branches_lodo_positive_ge_2_of_3": bool(np.all(lodo_ok)),
                "pilot_robust_candidate": bool(
                    sign_concordant
                    and np.min(np.abs(means)) >= 0.05
                    and np.all(donor_ok)
                    and np.all(q_ok)
                    and np.all(lodo_ok)
                ),
            }
        )
    module_concordance = pd.DataFrame(module_rows)

    primary_donor = donor_effects.loc[donor_effects["module"].isin(PRIMARY_MODULES)]
    pivot = primary_donor.pivot_table(
        index=["donor", "module"], columns="branch", values="beta_entropy"
    )
    pair_rows: list[dict[str, object]] = []
    for index, left in enumerate(BRANCHES):
        for right in BRANCHES[index + 1 :]:
            values = pivot[[left, right]].dropna()
            pearson = pearsonr(values[left], values[right])
            spearman = spearmanr(values[left], values[right])
            pair_rows.append(
                {
                    "branch_left": left,
                    "branch_right": right,
                    "n_donor_module_pairs": len(values),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p_descriptive": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_descriptive": float(spearman.pvalue),
                }
            )
    return module_concordance, pd.DataFrame(pair_rows)


def load_figure_helper():
    path = Path(
        r"C:\Users\13264\.agents\skills\scientific-figure-pro\scripts\scientific_figure_pro.py"
    )
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load figure helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_figures(
    out_dir: Path,
    overall: pd.DataFrame,
    donor_effects: pd.DataFrame,
    concordance: pd.DataFrame,
    spot_scores: pd.DataFrame,
    representative_section: str,
) -> None:
    sfp = load_figure_helper()
    sfp.apply_publication_style(sfp.FigureStyle(font_size=11, axes_linewidth=1.4))
    colors = {"lognorm": "#777777", "SCT": "#1B9E77", "scVI": "#2166AC"}

    fig, axes = sfp.create_subplots(2, 2, figsize=(12, 9))
    ax = axes[0]
    x = np.arange(len(PRIMARY_MODULES))
    offsets = {"lognorm": -0.22, "SCT": 0.0, "scVI": 0.22}
    for branch in BRANCHES:
        rows = (
            overall.loc[(overall["branch"] == branch) & overall["module"].isin(PRIMARY_MODULES)]
            .set_index("module")
            .loc[list(PRIMARY_MODULES)]
        )
        mean = rows["mean_beta_entropy"].to_numpy()
        low = mean - rows["ci95_low_donor_t"].to_numpy()
        high = rows["ci95_high_donor_t"].to_numpy() - mean
        ax.errorbar(
            x + offsets[branch],
            mean,
            yerr=np.vstack([low, high]),
            fmt="o",
            color=colors[branch],
            capsize=3,
            lw=1.5,
            ms=6,
            label=branch,
        )
    ax.axhline(0, color="#B2182B", ls="--", lw=1)
    ax.set_xticks(x, PRIMARY_MODULES, rotation=20, ha="right")
    ax.set_ylabel("Partial beta: layer-neighborhood entropy")
    ax.set_title("A  Donor-level effects (95% t CI, n=3 donors)")
    ax.legend(fontsize=8)

    ax = axes[1]
    gei = (
        donor_effects.loc[donor_effects["module"] == "GEI"]
        .pivot(index="donor", columns="branch", values="beta_entropy")
        .reindex(columns=BRANCHES)
    )
    limit = max(0.1, float(np.nanmax(np.abs(gei.to_numpy()))))
    image = ax.imshow(gei.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(BRANCHES)), BRANCHES)
    ax.set_yticks(np.arange(len(gei)), gei.index)
    ax.set_title("B  GEI direction in each biological donor")
    for row in range(gei.shape[0]):
        for column in range(gei.shape[1]):
            ax.text(
                column, row, f"{gei.iloc[row, column]:.2f}", ha="center", va="center", fontsize=9
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="partial beta")

    ax = axes[2]
    module_means = (
        overall.loc[overall["module"].isin(PRIMARY_MODULES)]
        .pivot(index="module", columns="branch", values="mean_beta_entropy")
        .loc[list(PRIMARY_MODULES)]
    )
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.axvline(0, color="#BBBBBB", lw=0.8)
    ax.scatter(module_means["lognorm"], module_means["SCT"], s=55, color=colors["SCT"], label="SCT")
    ax.scatter(
        module_means["lognorm"],
        module_means["scVI"],
        s=55,
        color=colors["scVI"],
        marker="s",
        label="scVI",
    )
    for module in PRIMARY_MODULES:
        ax.text(
            module_means.loc[module, "lognorm"],
            module_means.loc[module, "scVI"],
            module,
            fontsize=8,
        )
    bounds = np.asarray([ax.get_xlim(), ax.get_ylim()])
    lower, upper = float(np.min(bounds)), float(np.max(bounds))
    ax.plot([lower, upper], [lower, upper], color="#444444", ls=":", lw=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("lognorm effect")
    ax.set_ylabel("SCT / scVI effect")
    ax.set_title("C  Normalization-branch concordance")
    ax.legend(fontsize=8)

    ax = axes[3]
    gate_columns = [
        "branch_sign_concordant",
        "all_branches_donor_direction_ge_2_of_3",
        "all_branches_spatial_q_le_0_10",
        "all_branches_lodo_positive_ge_2_of_3",
        "pilot_robust_candidate",
    ]
    gate = concordance.set_index("module").loc[list(PRIMARY_MODULES), gate_columns].astype(float)
    ax.imshow(gate.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        np.arange(len(gate_columns)),
        ["branch sign", "donor sign", "spatial q", "LODO delta R2", "robust"],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(gate)), gate.index)
    ax.set_title("D  Predeclared pilot evidence gates")
    for row in range(gate.shape[0]):
        for column in range(gate.shape[1]):
            ax.text(
                column,
                row,
                "PASS" if gate.iloc[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
    fig.suptitle(
        "DLPFC neighborhood-conditioned glial programs: preprocessing robustness",
        y=1.01,
        fontsize=14,
    )
    sfp.finalize_figure(
        fig, out_dir / "figure1_dlpfc_robustness", formats=["png", "pdf", "svg"], dpi=600
    )

    spatial = spot_scores.loc[spot_scores["section"] == representative_section].copy()
    fig2, axes2 = sfp.create_subplots(1, 2, figsize=(10, 4.4))
    for ax, column, title, cmap in (
        (axes2[0], "layer_entropy", "Layer-neighborhood entropy", "magma"),
        (axes2[1], "scVI__GEI", "scVI-decoded GEI score", "coolwarm"),
    ):
        scatter = ax.scatter(
            spatial["x"],
            spatial["y"],
            c=spatial[column],
            s=7,
            cmap=cmap,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{representative_section}: {title}")
        fig2.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03)
    sfp.finalize_figure(
        fig2, out_dir / "figure2_representative_spatial", formats=["png", "pdf", "svg"], dpi=600
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--n-hvg", type=int, default=1900)
    parser.add_argument("--scvi-epochs", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--rscript", type=Path, default=Path(r"C:\R1\R-4.5.3\bin\Rscript.exe"))
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    data_dir = repo / "datasets_cache" / "dlpfc"
    out_dir = here / "results"
    intermediate_dir = out_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir)
    bridge = repo / "workflows" / "containers" / "histoweave-r" / "histoweave-sctransform.R"
    files = sorted(data_dir.glob("dlpfc_*.h5ad"))
    if len(files) != 12:
        raise RuntimeError(f"expected exactly 12 DLPFC sections; found {len(files)}")
    if not args.rscript.exists() or not bridge.exists():
        raise FileNotFoundError(f"missing Rscript or SCT bridge: {args.rscript}, {bridge}")

    logger.info("Starting DLPFC SCT/scVI robustness experiment; scope=%s", args.scope)
    audit, feature, selected_genes = audit_and_select_features(files, args.n_hvg, logger)
    audit.to_csv(out_dir / "audit_all_12_sections.csv", index=False)
    feature.to_csv(out_dir / "feature_selection.csv", index=False)
    selected_sections = choose_scope(audit, args.scope)
    scope_payload = {
        "scope": args.scope,
        "all_section_to_donor": DLPFC_DONOR,
        "selected_sections": selected_sections,
        "selection_rule": (
            "all sections"
            if args.scope == "all"
            else "one section per donor closest to donor-median spot count; ties by section id"
        ),
        "biological_n": 3,
        "section_is_not_independent_n": True,
    }
    (out_dir / "analysis_scope.json").write_text(
        json.dumps(scope_payload, indent=2), encoding="utf-8"
    )
    logger.info("Selected sections: %s", ", ".join(selected_sections))

    raw_paths: dict[str, Path] = {}
    sct_paths: dict[str, Path] = {}
    paths_by_section = {path.stem.rsplit("_", 1)[-1]: path for path in files}
    for section in selected_sections:
        raw_path = subset_raw_section(
            paths_by_section[section], section, selected_genes, intermediate_dir, logger
        )
        raw_paths[section] = raw_path
        sct_path = intermediate_dir / f"sct_{section}.h5ad"
        n_spots = int(audit.loc[audit["section"].astype(str) == section, "n_spots"].iloc[0])
        run_sct(raw_path, sct_path, args.rscript, bridge, min(3000, n_spots), logger)
        sct_paths[section] = sct_path

    metadata_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    raw_adatas: list[ad.AnnData] = []
    for section in selected_sections:
        raw = ad.read_h5ad(raw_paths[section])
        sct = ad.read_h5ad(sct_paths[section])
        metadata_frames.append(neighborhood_frame(raw))
        log_scores, log_coverage = lognorm_scores(raw, section)
        sct_score, sct_coverage = sct_scores(sct, section)
        log_scores.insert(0, "spot_id", raw.obs_names.astype(str))
        sct_score.insert(0, "spot_id", sct.obs_names.astype(str))
        merged = log_scores.merge(sct_score, on="spot_id", suffixes=("__lognorm", "__SCT"))
        renamed: dict[str, str] = {}
        for module in MODULES.keys() | {"GEI"}:
            renamed[f"{module}__lognorm"] = f"lognorm__{module}"
            renamed[f"{module}__SCT"] = f"SCT__{module}"
        score_frames.append(merged.rename(columns=renamed))
        coverage_rows.extend(log_coverage)
        coverage_rows.extend(sct_coverage)
        raw_adatas.append(raw)
        del sct
        gc.collect()

    scvi_scores, scvi_coverage, history, latent = train_scvi_and_score(
        raw_adatas, selected_sections, out_dir, args.scvi_epochs, logger
    )
    coverage_rows.extend(scvi_coverage)
    if not history.empty:
        history.to_csv(out_dir / "scvi_training_history.csv", index=False)
    latent.to_csv(out_dir / "scvi_latent.csv.gz", index=False, compression="gzip")

    metadata = pd.concat(metadata_frames, ignore_index=True)
    branch_scores = pd.concat(score_frames, ignore_index=True)
    spot_scores = metadata.merge(branch_scores, on="spot_id", validate="one_to_one")
    scvi_renamed = scvi_scores.rename(
        columns={module: f"scVI__{module}" for module in MODULES.keys() | {"GEI"}}
    )
    spot_scores = spot_scores.merge(scvi_renamed, on="spot_id", validate="one_to_one")
    spot_scores.to_csv(out_dir / "spot_module_scores.csv.gz", index=False, compression="gzip")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out_dir / "module_gene_coverage.csv", index=False)

    section_effects, nulls = section_effects_and_nulls(
        spot_scores, selected_sections, args.permutations
    )
    donor_effects, overall = aggregate_effects(section_effects, nulls, args.permutations)
    lodo = lodo_prediction(spot_scores)
    concordance, correlations = concordance_tables(donor_effects, overall, lodo)
    section_effects.to_csv(out_dir / "section_effects.csv", index=False)
    donor_effects.to_csv(out_dir / "donor_effects.csv", index=False)
    overall.to_csv(out_dir / "overall_effects.csv", index=False)
    lodo.to_csv(out_dir / "leave_one_donor_out_prediction.csv", index=False)
    concordance.to_csv(out_dir / "module_branch_concordance.csv", index=False)
    correlations.to_csv(out_dir / "branch_pair_correlations.csv", index=False)
    np.savez_compressed(
        out_dir / "spatial_shift_nulls.npz",
        **{
            f"{section}__{branch}__{module}": values
            for (section, branch, module), values in nulls.items()
        },
    )

    representative = selected_sections[-1]
    make_figures(out_dir, overall, donor_effects, concordance, spot_scores, representative)
    result = {
        "analysis": "DLPFC neighborhood-conditioned glial program preprocessing robustness",
        "scope": scope_payload,
        "seed": SEED,
        "n_hvg_requested": args.n_hvg,
        "n_model_genes": len(selected_genes),
        "scvi_max_epochs": args.scvi_epochs,
        "spatial_permutations": args.permutations,
        "raw_count_contract": {
            "all_12_sections_pass": bool(audit["integer_like"].all()),
            "scvi_input": "layers['counts'] integer UMI only",
            "sct_input": "layers['counts'] integer UMI only",
            "sct_residuals_supplied_to_scvi": False,
        },
        "primary_module_concordance": concordance.to_dict(orient="records"),
        "overall_primary_effects": overall.loc[overall["module"].isin(PRIMARY_MODULES)].to_dict(
            orient="records"
        ),
        "interpretation": (
            "A donor-aware DLPFC pilot, not cross-tissue validation. Biological n is three donors; "
            "sections are nested technical/anatomical replicates. A PASS identifies a candidate for "  # noqa: E501
            "independent cross-region testing, not a Nature Methods-level discovery by itself."
        ),
        "software": {
            "python": platform.python_version(),
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Completed. Primary concordance:\n%s", concordance.to_string(index=False))


if __name__ == "__main__":
    main()
