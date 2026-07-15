#!/usr/bin/env python3
"""Recover raw Slide-seqV2 counts, attach annotations, and run a real SCT pilot.

The cached Squidpy H5AD is opened in backed mode and contributes only ``obs``,
``obsm['spatial']``, and ``var_names``.  Its normalized expression matrix is
never read or copied.  All count-based calculations originate in the SCP815
``Puck_200115_08_count_location.RData`` integer sparse matrix.

This is a technical-validity and candidate-generation workflow for one puck
(biological n=1).  It cannot establish replication or cross-tissue transfer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread, mmwrite

SEED = 20_011_508
COORDINATE_TOLERANCE = 1e-6
DEFAULT_PILOT_CELLS = 4_000
DEFAULT_HVG = 2_000

MARKER_MODULES: dict[str, tuple[str, ...]] = {
    "astrocyte_homeostasis": (
        "Aldoc",
        "Aqp4",
        "Gfap",
        "Slc1a2",
        "Slc1a3",
        "Glul",
        "Kcnj10",
        "Atp1a2",
        "Slc4a4",
        "Gja1",
    ),
    "vascular_endothelial": (
        "Cldn5",
        "Pecam1",
        "Kdr",
        "Klf2",
        "Klf4",
        "Slco1a4",
        "Abcb1a",
        "Col4a1",
        "Col4a2",
    ),
    "vascular_mural": ("Rgs5", "Pdgfrb", "Cspg4", "Vtn", "Acta2", "Des"),
    "oligodendrocyte": (
        "Mbp",
        "Plp1",
        "Mog",
        "Mag",
        "Cnp",
        "Mobp",
        "Opalin",
        "Mal",
    ),
    "microglia_homeostasis": (
        "C1qa",
        "C1qb",
        "C1qc",
        "P2ry12",
        "Tmem119",
        "Csf1r",
        "Cx3cr1",
    ),
    "lipid_metabolic_support": (
        "Apoe",
        "Lpl",
        "Abca1",
        "Mertk",
        "Fabp7",
        "Acsl6",
        "Angptl4",
    ),
    "excitatory_neuron": ("Slc17a7", "Slc17a6", "Camk2a", "Satb2", "Tbr1"),
    "inhibitory_neuron": ("Gad1", "Gad2", "Slc6a1", "Slc32a1"),
}


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("slideseq_raw_recovery")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def run_logged(command: list[str], logger: logging.Logger) -> None:
    logger.info("RUN %s", subprocess.list2cmdline(command))
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout:
        for line in completed.stdout.splitlines():
            logger.info("[subprocess] %s", line)
    if completed.stderr:
        for line in completed.stderr.splitlines():
            logger.warning("[subprocess] %s", line)
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def export_rdata(
    rscript: Path,
    exporter: Path,
    rdata: Path,
    intermediate: Path,
    logger: logging.Logger,
    force: bool,
) -> None:
    required = (
        intermediate / "counts_genes_by_beads.mtx.gz",
        intermediate / "genes.txt",
        intermediate / "barcodes.txt",
        intermediate / "coordinates.tsv",
    )
    if all(path.exists() for path in required) and not force:
        logger.info("Reusing complete raw export in %s", intermediate)
        return
    intermediate.mkdir(parents=True, exist_ok=True)
    run_logged([str(rscript), str(exporter), str(rdata), str(intermediate)], logger)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"R exporter completed but files are missing: {missing}")


def load_raw_export(
    intermediate: Path, logger: logging.Logger
) -> tuple[sp.csr_matrix, list[str], list[str], np.ndarray]:
    genes = read_lines(intermediate / "genes.txt")
    barcodes = read_lines(intermediate / "barcodes.txt")
    if len(set(genes)) != len(genes):
        raise ValueError("Raw gene identifiers are not unique")
    if len(set(barcodes)) != len(barcodes):
        raise ValueError("Raw bead barcodes are not unique")

    coordinates = pd.read_csv(intermediate / "coordinates.tsv", sep="\t")
    expected_columns = ["barcode", "x", "y"]
    if coordinates.columns.tolist() != expected_columns:
        raise ValueError(f"Coordinate columns must be {expected_columns}")
    if coordinates["barcode"].tolist() != barcodes:
        raise ValueError("Coordinate barcode order differs from exported matrix order")
    xy = coordinates[["x", "y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("Coordinates contain non-finite values")

    matrix_path = intermediate / "counts_genes_by_beads.mtx.gz"
    logger.info("Reading raw Matrix Market export %s", matrix_path)
    with gzip.open(matrix_path, "rb") as handle:
        genes_by_beads = sp.coo_matrix(mmread(handle))
    if genes_by_beads.shape != (len(genes), len(barcodes)):
        raise ValueError(
            f"Matrix shape {genes_by_beads.shape} differs from "
            f"{len(genes)} genes x {len(barcodes)} barcodes"
        )
    data = np.asarray(genes_by_beads.data)
    if not data.size or not np.any(data > 0):
        raise ValueError("Raw matrix contains no positive counts")
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("Raw counts must be finite and non-negative")
    if not np.allclose(data, np.rint(data), atol=1e-8, rtol=0):
        raise ValueError("Raw matrix violates the integer-like count contract")
    if data.max() > np.iinfo(np.int32).max:
        raise OverflowError("Raw count exceeds signed int32")
    genes_by_beads.data = np.rint(data).astype(np.int32)
    counts = genes_by_beads.T.tocsr()
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    logger.info(
        "Loaded raw counts: beads=%d genes=%d nnz=%d",
        counts.shape[0],
        counts.shape[1],
        counts.nnz,
    )
    return counts, genes, barcodes, xy


def attach_annotations(
    annotation_path: Path,
    raw_barcodes: list[str],
    raw_xy: np.ndarray,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], set[str]]:
    logger.info("Opening annotation H5AD in backed mode; cached expression is not read")
    source = ad.read_h5ad(annotation_path, backed="r")
    try:
        source_names = pd.Index(source.obs_names.astype(str))
        source_obs = source.obs.copy()
        source_xy = np.asarray(source.obsm["spatial"], dtype=np.float64)
        source_genes = set(source.var_names.astype(str))
    finally:
        source.file.close()

    raw_index = pd.Index(raw_barcodes)
    raw_positions = raw_index.get_indexer(source_names)
    barcode_matched = raw_positions >= 0
    deltas = np.full(len(source_names), np.nan, dtype=np.float64)
    deltas[barcode_matched] = np.linalg.norm(
        raw_xy[raw_positions[barcode_matched]] - source_xy[barcode_matched], axis=1
    )
    coordinate_matched = barcode_matched & (deltas <= COORDINATE_TOLERANCE)
    trusted = coordinate_matched

    obs = pd.DataFrame(index=raw_index)
    obs.index.name = "barcode"
    obs["barcode"] = raw_barcodes
    obs["x"] = raw_xy[:, 0]
    obs["y"] = raw_xy[:, 1]
    obs["source_annotation_matched"] = False
    obs["source_coordinate_delta"] = np.nan
    for column in ("leiden", "cluster", "domain_truth"):
        obs[column] = "unannotated"

    transfer_columns = [
        column for column in ("leiden", "cluster", "domain_truth") if column in source_obs
    ]
    trusted_source_indices = np.flatnonzero(trusted)
    trusted_raw_positions = raw_positions[trusted]
    obs.iloc[trusted_raw_positions, obs.columns.get_loc("source_annotation_matched")] = True
    obs.iloc[trusted_raw_positions, obs.columns.get_loc("source_coordinate_delta")] = deltas[
        trusted
    ]
    for column in transfer_columns:
        values = source_obs.iloc[trusted_source_indices][column].astype(str).to_numpy()
        obs.iloc[trusted_raw_positions, obs.columns.get_loc(column)] = values

    match_table = pd.DataFrame(
        {
            "source_barcode": source_names,
            "raw_index": raw_positions,
            "barcode_exact_match": barcode_matched,
            "coordinate_delta": deltas,
            "coordinate_within_tolerance": coordinate_matched,
        }
    )
    for column in transfer_columns:
        match_table[column] = source_obs[column].astype(str).to_numpy()

    report = {
        "annotation_source": str(annotation_path.resolve()),
        "cached_expression_read": False,
        "matching_method": "exact barcode followed by Euclidean coordinate audit",
        "coordinate_tolerance": COORDINATE_TOLERANCE,
        "n_raw_beads": len(raw_barcodes),
        "n_source_beads": len(source_names),
        "n_exact_barcode_matches": int(barcode_matched.sum()),
        "source_barcode_match_rate": float(barcode_matched.mean()),
        "raw_annotated_rate": float(barcode_matched.sum() / len(raw_barcodes)),
        "n_coordinate_matches": int(coordinate_matched.sum()),
        "source_coordinate_match_rate": float(coordinate_matched.mean()),
        "coordinate_delta_max": float(np.nanmax(deltas)),
        "coordinate_delta_median": float(np.nanmedian(deltas)),
        "n_trusted_annotation_transfers": int(trusted.sum()),
        "transferred_columns": transfer_columns,
        "n_source_genes": len(source_genes),
    }
    logger.info(
        "Annotation match: %d/%d source beads (%.3f%%); %.3f%% of raw beads; max coordinate delta %.6g",  # noqa: E501
        report["n_exact_barcode_matches"],
        report["n_source_beads"],
        100 * report["source_barcode_match_rate"],
        100 * report["raw_annotated_rate"],
        report["coordinate_delta_max"],
    )
    return obs, match_table, report, source_genes


def count_contract(counts: sp.csr_matrix, genes: list[str], barcodes: list[str]) -> dict[str, Any]:
    data = counts.data
    counts_layer = counts.copy()
    exact_layer = (
        np.array_equal(counts.indptr, counts_layer.indptr)
        and np.array_equal(counts.indices, counts_layer.indices)
        and np.array_equal(counts.data, counts_layer.data)
    )
    checks = {
        "sparse_csr": sp.isspmatrix_csr(counts),
        "finite_stored_values": bool(np.isfinite(data).all()),
        "nonnegative_stored_values": bool(np.all(data >= 0)),
        "integer_dtype": bool(np.issubdtype(counts.dtype, np.integer)),
        "integer_like_stored_values": bool(np.allclose(data, np.rint(data), atol=0, rtol=0)),
        "at_least_one_positive": bool(np.any(data > 0)),
        "unique_genes": len(set(genes)) == len(genes),
        "unique_barcodes": len(set(barcodes)) == len(barcodes),
        "x_counts_layer_exact_before_write": bool(exact_layer),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "shape_beads_by_genes": list(counts.shape),
        "nnz": int(counts.nnz),
        "dtype": str(counts.dtype),
        "stored_min": int(data.min()),
        "stored_max": int(data.max()),
        "total_umi": int(counts.sum(dtype=np.int64)),
    }


def add_raw_qc(
    counts: sp.csr_matrix, genes: list[str], obs: pd.DataFrame, var: pd.DataFrame
) -> None:
    obs["raw_total_counts"] = np.asarray(counts.sum(axis=1, dtype=np.int64)).ravel()
    obs["raw_n_genes_by_counts"] = np.asarray((counts > 0).sum(axis=1)).ravel().astype(np.int32)
    var["raw_total_counts"] = np.asarray(counts.sum(axis=0, dtype=np.int64)).ravel()
    var["raw_n_beads_by_counts"] = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int32)
    var["is_mitochondrial"] = [gene.lower().startswith("mt-") for gene in genes]
    mitochondrial = np.flatnonzero(var["is_mitochondrial"].to_numpy())
    if mitochondrial.size:
        mt_counts = np.asarray(counts[:, mitochondrial].sum(axis=1, dtype=np.int64)).ravel()
    else:
        mt_counts = np.zeros(counts.shape[0], dtype=np.int64)
    obs["raw_mt_counts"] = mt_counts
    denominator = np.maximum(obs["raw_total_counts"].to_numpy(), 1)
    obs["raw_pct_mt"] = 100.0 * mt_counts / denominator


def write_full_h5ad(
    path: Path,
    counts: sp.csr_matrix,
    genes: list[str],
    obs: pd.DataFrame,
    raw_xy: np.ndarray,
    source_genes: set[str],
    contract: dict[str, Any],
    match_report: dict[str, Any],
    rdata: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    var["in_squidpy_4000"] = [gene in source_genes for gene in genes]
    add_raw_qc(counts, genes, obs, var)
    dataset = ad.AnnData(X=counts, obs=obs, var=var)
    dataset.layers["counts"] = counts
    dataset.obsm["spatial"] = raw_xy
    dataset.uns["provenance"] = {
        "raw_source": str(rdata.resolve()),
        "raw_source_sha256": sha256(rdata),
        "puck": "Puck_200115_08",
        "study": "SCP815",
        "tissue": "mouse hippocampus",
        "expression_origin": "raw integer countmat from RData",
        "annotation_expression_used": False,
    }
    dataset.uns["count_contract"] = contract
    dataset.uns["annotation_match"] = match_report
    dataset.uns["claim_scope"] = {
        "biological_n": 1,
        "replication": False,
        "allowed": "technical validity and candidate generation",
        "forbidden": "replicated biological discovery or cross-tissue generalization",
    }
    logger.info("Writing full raw-count H5AD %s", path)
    dataset.write_h5ad(path, compression="gzip")
    return var


def stratified_sample(labels: np.ndarray, target: int, seed: int) -> np.ndarray:
    if target >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = {label: np.flatnonzero(labels == label) for label in sorted(set(labels))}
    counts = {label: len(indices) for label, indices in groups.items()}
    quotas = {
        label: min(counts[label], max(1, int(np.floor(target * counts[label] / len(labels)))))
        for label in groups
    }
    while sum(quotas.values()) < target:
        candidates = [label for label in groups if quotas[label] < counts[label]]
        label = max(
            candidates,
            key=lambda item: (
                target * counts[item] / len(labels) - quotas[item],
                counts[item],
                item,
            ),
        )
        quotas[label] += 1
    while sum(quotas.values()) > target:
        candidates = [label for label in groups if quotas[label] > 1]
        label = min(
            candidates,
            key=lambda item: (
                target * counts[item] / len(labels) - quotas[item],
                -counts[item],
                item,
            ),
        )
        quotas[label] -= 1
    selected = np.concatenate(
        [rng.choice(groups[label], size=quotas[label], replace=False) for label in groups]
    )
    return np.sort(selected.astype(np.int64))


def select_pilot(
    counts: sp.csr_matrix,
    genes: list[str],
    obs: pd.DataFrame,
    n_cells: int,
    n_hvg: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    trusted = obs["source_annotation_matched"].to_numpy(dtype=bool)
    annotated_positions = np.flatnonzero(trusted)
    labels = obs.iloc[annotated_positions]["cluster"].astype(str).to_numpy()
    relative = stratified_sample(labels, min(n_cells, len(labels)), SEED)
    cell_positions = annotated_positions[relative]
    pilot_all = counts[cell_positions]

    library_size = np.asarray(pilot_all.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library_size <= 0):
        raise ValueError("Pilot selection includes a zero-library bead")
    normalized = pilot_all.astype(np.float64).multiply((10_000.0 / library_size)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean**2, 0)
    detected = np.asarray((pilot_all > 0).sum(axis=0)).ravel()
    score = variance / (mean + 1e-3)
    mitochondrial = np.array([gene.lower().startswith("mt-") for gene in genes])
    eligible = (detected >= 20) & (~mitochondrial) & np.isfinite(score)
    eligible_positions = np.flatnonzero(eligible)
    order = eligible_positions[np.argsort(score[eligible_positions], kind="stable")[::-1]]
    hvg_positions = order[: min(n_hvg, len(order))]

    lookup = {gene.lower(): index for index, gene in enumerate(genes)}
    marker_to_module = {
        marker.lower(): module for module, markers in MARKER_MODULES.items() for marker in markers
    }
    marker_positions = np.array(
        sorted({lookup[marker] for marker in marker_to_module if marker in lookup}), dtype=np.int64
    )
    gene_positions = np.array(sorted(set(hvg_positions) | set(marker_positions)), dtype=np.int64)

    hvg_set = set(hvg_positions)
    marker_set = set(marker_positions)
    selection = []
    marker_module = []
    for position in gene_positions:
        in_hvg = position in hvg_set
        in_marker = position in marker_set
        selection.append("hvg+marker" if in_hvg and in_marker else "hvg" if in_hvg else "marker")
        marker_module.append(marker_to_module.get(genes[position].lower(), ""))
    pilot_var = pd.DataFrame(
        {
            "selection": selection,
            "marker_module": marker_module,
            "pilot_detection_beads": detected[gene_positions].astype(np.int32),
            "pilot_log_normalized_variability": score[gene_positions],
        },
        index=pd.Index([genes[position] for position in gene_positions], name="gene"),
    )
    report = {
        "seed": SEED,
        "selection_scope": "trusted exact-barcode-and-coordinate annotated beads only",
        "n_available_annotated_beads": int(trusted.sum()),
        "n_pilot_beads": len(cell_positions),
        "n_requested_hvg": n_hvg,
        "n_selected_hvg": len(hvg_positions),
        "n_predefined_markers_present": len(marker_positions),
        "n_union_genes": len(gene_positions),
        "cluster_counts": Counter(obs.iloc[cell_positions]["cluster"].astype(str)),
        "hvg_method": "variance/mean ranking after raw-count library-size normalization and log1p; detection>=20; mitochondrial genes excluded",  # noqa: E501
    }
    logger.info(
        "Pilot: %d beads, %d HVG, %d markers, %d union genes",
        len(cell_positions),
        len(hvg_positions),
        len(marker_positions),
        len(gene_positions),
    )
    return cell_positions, gene_positions, pilot_var, report


def write_pilot_inputs(
    intermediate: Path,
    pilot_h5ad: Path,
    counts: sp.csr_matrix,
    obs: pd.DataFrame,
    pilot_var: pd.DataFrame,
    raw_xy: np.ndarray,
    cell_positions: np.ndarray,
    gene_positions: np.ndarray,
    report: dict[str, Any],
    logger: logging.Logger,
) -> sp.csr_matrix:
    pilot_counts = counts[cell_positions][:, gene_positions].tocsr()
    pilot_obs = obs.iloc[cell_positions].copy()
    pilot = ad.AnnData(X=pilot_counts, obs=pilot_obs, var=pilot_var.copy())
    pilot.layers["counts"] = pilot_counts
    pilot.obsm["spatial"] = raw_xy[cell_positions]
    pilot.uns["pilot_selection"] = report
    pilot.uns["count_contract_passed"] = True
    pilot.uns["expression_origin"] = "raw integer countmat from SCP815 RData"
    logger.info("Writing raw-count SCT pilot input %s", pilot_h5ad)
    pilot.write_h5ad(pilot_h5ad, compression="gzip")

    intermediate.mkdir(parents=True, exist_ok=True)
    matrix_path = intermediate / "pilot_counts_genes_by_beads.mtx"
    matrix_gz_path = matrix_path.with_suffix(matrix_path.suffix + ".gz")
    mmwrite(matrix_path, pilot_counts.T.tocoo(), symmetry="general")
    with (
        matrix_path.open("rb") as source,
        gzip.open(matrix_gz_path, "wb", compresslevel=6) as target,
    ):
        shutil.copyfileobj(source, target)
    matrix_path.unlink()
    (intermediate / "pilot_genes.txt").write_text(
        "\n".join(pilot_var.index.astype(str)) + "\n", encoding="utf-8"
    )
    (intermediate / "pilot_barcodes.txt").write_text(
        "\n".join(pilot_obs.index.astype(str)) + "\n", encoding="utf-8"
    )
    return pilot_counts


def correlations_with_depth(matrix: np.ndarray, log_depth: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(log_depth, dtype=np.float64)
    y = y - y.mean()
    centered = x - x.mean(axis=0, keepdims=True)
    numerator = centered.T @ y
    denominator = np.sqrt(np.sum(centered**2, axis=0) * np.sum(y**2))
    return np.divide(numerator, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 0)


def summarize_absolute(values: np.ndarray) -> dict[str, float]:
    finite = np.abs(values[np.isfinite(values)])
    return {
        "median_abs_correlation": float(np.median(finite)),
        "p90_abs_correlation": float(np.quantile(finite, 0.9)),
        "max_abs_correlation": float(np.max(finite)),
    }


def finalize_sct(
    sct_dir: Path,
    output_h5ad: Path,
    pilot_counts: sp.csr_matrix,
    pilot_obs: pd.DataFrame,
    pilot_var: pd.DataFrame,
    raw_xy: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    modeled_genes = read_lines(sct_dir / "sct_modeled_genes.txt")
    sct_barcodes = read_lines(sct_dir / "sct_barcodes.txt")
    if sct_barcodes != pilot_obs.index.astype(str).tolist():
        raise ValueError("SCT output barcode order differs from pilot input")
    gene_positions = pilot_var.index.get_indexer(modeled_genes)
    if np.any(gene_positions < 0):
        raise ValueError("SCT output contains a gene absent from pilot input")
    n_genes = len(modeled_genes)
    n_cells = len(sct_barcodes)
    flat = np.fromfile(sct_dir / "sct_residuals_float32.bin", dtype="<f4")
    if flat.size != n_genes * n_cells:
        raise ValueError(
            f"SCT residual binary has {flat.size} floats; expected {n_genes * n_cells}"
        )
    residuals = flat.reshape((n_genes, n_cells), order="F").T
    if not np.isfinite(residuals).all():
        raise ValueError("SCT residual binary contains non-finite values")

    modeled_counts = pilot_counts[:, gene_positions].tocsr()
    modeled_var = pilot_var.iloc[gene_positions].copy()
    gene_attr = pd.read_csv(sct_dir / "sct_gene_attributes.csv").set_index("gene")
    for column in gene_attr.columns:
        modeled_var[f"sct_{column}"] = gene_attr.reindex(modeled_genes)[column].to_numpy()

    result = ad.AnnData(X=residuals, obs=pilot_obs.copy(), var=modeled_var)
    result.layers["counts"] = modeled_counts
    result.obsm["spatial"] = raw_xy
    metadata = {}
    for line in read_lines(sct_dir / "sct_metadata.txt"):
        key, value = line.split("=", 1)
        metadata[key] = value
    result.uns["sctransform"] = {
        **metadata,
        "real_sctransform_vst": True,
        "raw_integer_count_input": True,
        "cached_normalized_expression_used": False,
        "scope": "single-puck technical validation and candidate generation",
    }
    logger.info("Writing SCT residual H5AD %s", output_h5ad)
    result.write_h5ad(output_h5ad, compression="gzip")

    depth = np.asarray(modeled_counts.sum(axis=1)).ravel().astype(np.float64)
    log_depth = np.log1p(depth)
    raw_dense = np.log1p(modeled_counts.toarray().astype(np.float32))
    normalized = modeled_counts.astype(np.float64).multiply((10_000.0 / depth)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    normalized_dense = normalized.toarray().astype(np.float32)
    raw_correlation = correlations_with_depth(raw_dense, log_depth)
    normalized_correlation = correlations_with_depth(normalized_dense, log_depth)
    sct_correlation = correlations_with_depth(residuals, log_depth)
    depth_table = pd.DataFrame(
        {
            "gene": modeled_genes,
            "raw_log1p_correlation_with_log_depth": raw_correlation,
            "library_normalized_log1p_correlation_with_log_depth": normalized_correlation,
            "sct_residual_correlation_with_log_depth": sct_correlation,
        }
    )
    depth_table.to_csv(output_dir / "sct_depth_correlations.csv", index=False)

    marker_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    cluster_values = pilot_obs["cluster"].astype(str).to_numpy()
    modeled_lookup = {gene.lower(): position for position, gene in enumerate(modeled_genes)}
    for module, marker_names in MARKER_MODULES.items():
        module_positions = [
            modeled_lookup[name.lower()] for name in marker_names if name.lower() in modeled_lookup
        ]
        if not module_positions:
            continue
        module_score = residuals[:, module_positions].mean(axis=1)
        for cluster in sorted(set(cluster_values)):
            mask = cluster_values == cluster
            module_rows.append(
                {
                    "module": module,
                    "cluster": cluster,
                    "n_beads": int(mask.sum()),
                    "n_modeled_markers": len(module_positions),
                    "mean_sct_module_score": float(module_score[mask].mean()),
                    "median_sct_module_score": float(np.median(module_score[mask])),
                }
            )
        for marker_name in marker_names:
            position = modeled_lookup.get(marker_name.lower())
            if position is None:
                continue
            raw_gene = modeled_counts[:, position].toarray().ravel()
            for cluster in sorted(set(cluster_values)):
                mask = cluster_values == cluster
                marker_rows.append(
                    {
                        "module": module,
                        "gene": modeled_genes[position],
                        "cluster": cluster,
                        "n_beads": int(mask.sum()),
                        "raw_detection_fraction": float(np.mean(raw_gene[mask] > 0)),
                        "mean_raw_count": float(raw_gene[mask].mean()),
                        "mean_sct_residual": float(residuals[mask, position].mean()),
                        "median_sct_residual": float(np.median(residuals[mask, position])),
                    }
                )
    pd.DataFrame(marker_rows).to_csv(output_dir / "sct_marker_cluster_summary.csv", index=False)
    pd.DataFrame(module_rows).to_csv(output_dir / "sct_module_cluster_summary.csv", index=False)

    metrics = {
        "status": "passed",
        "method": "sctransform::vst",
        "vst_flavor": "v2",
        "residual_type": "pearson",
        "n_pilot_beads": n_cells,
        "n_modeled_genes": n_genes,
        "residuals_finite": bool(np.isfinite(residuals).all()),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
        "raw_log1p_depth_dependence": summarize_absolute(raw_correlation),
        "library_normalized_log1p_depth_dependence": summarize_absolute(normalized_correlation),
        "sct_residual_depth_dependence": summarize_absolute(sct_correlation),
        "interpretation": "technical depth-dependence diagnostic, not a biological endpoint",
    }
    write_json(output_dir / "sct_technical_metrics.json", metrics)
    return metrics


def grouped_qc(obs: pd.DataFrame, output_dir: Path) -> None:
    for grouping in ("cluster", "domain_truth"):
        summary = (
            obs.groupby(grouping, observed=True)
            .agg(
                n_beads=("barcode", "size"),
                median_raw_total_counts=("raw_total_counts", "median"),
                median_raw_n_genes=("raw_n_genes_by_counts", "median"),
                median_raw_pct_mt=("raw_pct_mt", "median"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / f"raw_qc_by_{grouping}.csv", index=False)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    histoweave_root = script_dir.parents[2]
    workspace_root = histoweave_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rdata",
        type=Path,
        default=script_dir.parent / "Puck_200115_08_count_location.RData",
    )
    parser.add_argument(
        "--annotation-h5ad",
        type=Path,
        default=workspace_root
        / "Biomni_lab_downloads_20260714_164953"
        / "histoweave_upgrade"
        / "datasets_cache"
        / "slideseqv2"
        / "slideseqv2_mouse_hippocampus.h5ad",
    )
    parser.add_argument("--rscript", type=Path, default=Path(r"C:\R1\R-4.5.3\bin\Rscript.exe"))
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--pilot-cells", type=int, default=DEFAULT_PILOT_CELLS)
    parser.add_argument("--hvg", type=int, default=DEFAULT_HVG)
    parser.add_argument("--skip-sct", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logger = configure_logging(output_dir)
    script_dir = Path(__file__).resolve().parent
    intermediate = script_dir / "intermediate"
    sct_intermediate = intermediate / "sct"

    for required in (args.rdata, args.annotation_h5ad, args.rscript):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.pilot_cells < 100:
        raise ValueError("--pilot-cells must be at least 100")
    if args.hvg < 100:
        raise ValueError("--hvg must be at least 100")

    logger.info("Starting raw Slide-seqV2 recovery; biological_n=1")
    export_rdata(
        args.rscript,
        script_dir / "export_raw_rdata.R",
        args.rdata,
        intermediate,
        logger,
        args.force_export,
    )
    counts, genes, barcodes, raw_xy = load_raw_export(intermediate, logger)
    contract = count_contract(counts, genes, barcodes)
    if not contract["passed"]:
        raise RuntimeError(f"Raw count contract failed: {contract}")
    obs, match_table, match_report, source_genes = attach_annotations(
        args.annotation_h5ad, barcodes, raw_xy, logger
    )
    match_report["n_source_genes_present_in_raw"] = sum(gene in set(genes) for gene in source_genes)
    match_report["source_gene_match_rate"] = (
        match_report["n_source_genes_present_in_raw"] / match_report["n_source_genes"]
    )
    match_table.to_csv(output_dir / "annotation_match.csv", index=False)
    write_json(output_dir / "annotation_match.json", match_report)
    write_json(output_dir / "count_contract.json", contract)

    full_h5ad = output_dir / "Puck_200115_08_raw_counts_annotated.h5ad"
    write_full_h5ad(
        full_h5ad,
        counts,
        genes,
        obs,
        raw_xy,
        source_genes,
        contract,
        match_report,
        args.rdata,
        logger,
    )
    grouped_qc(obs, output_dir)

    cell_positions, gene_positions, pilot_var, pilot_report = select_pilot(
        counts, genes, obs, args.pilot_cells, args.hvg, logger
    )
    pilot_dir = intermediate / "pilot"
    pilot_counts_h5ad = output_dir / "Puck_200115_08_raw_counts_sct_pilot_input.h5ad"
    pilot_counts = write_pilot_inputs(
        pilot_dir,
        pilot_counts_h5ad,
        counts,
        obs,
        pilot_var,
        raw_xy,
        cell_positions,
        gene_positions,
        pilot_report,
        logger,
    )
    write_json(output_dir / "pilot_selection.json", pilot_report)

    sct_metrics: dict[str, Any]
    if args.skip_sct:
        sct_metrics = {"status": "not_run", "reason": "--skip-sct requested"}
    else:
        sct_intermediate.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                str(args.rscript),
                str(script_dir / "run_sct_pilot.R"),
                str(pilot_dir),
                str(sct_intermediate),
            ],
            logger,
        )
        pilot_obs = obs.iloc[cell_positions].copy()
        sct_metrics = finalize_sct(
            sct_intermediate,
            output_dir / "Puck_200115_08_sct_v2_pearson_pilot.h5ad",
            pilot_counts,
            pilot_obs,
            pilot_var,
            raw_xy[cell_positions],
            output_dir,
            logger,
        )

    scvi_status = {
        "status": "not_run",
        "reason": "scvi-tools and torch are absent from the audited local environment",
        "substitution_used": False,
        "required_input": "raw integer counts (now recovered)",
        "next_gate": "install an approved isolated scvi-tools environment before validation",
    }
    write_json(output_dir / "scvi_status.json", scvi_status)
    guardrails = {
        "biological_n": 1,
        "independent_pucks": 1,
        "replication_gate_passed": False,
        "cross_tissue_generalization_gate_passed": False,
        "allowed_claim": "raw-count/SCT technical validity and single-puck candidates",
        "prohibited_claims": [
            "replicated hippocampal discovery",
            "cross-tissue conserved program",
            "Nature Methods-level validation",
        ],
        "annotation_circularity_note": "cluster/domain labels are transferred from the same public puck and are descriptive strata, not independent validation",  # noqa: E501
    }
    write_json(output_dir / "interpretation_guardrails.json", guardrails)

    summary = {
        "status": "complete",
        "raw_h5ad": str(full_h5ad),
        "raw_count_contract": contract,
        "annotation_match": match_report,
        "pilot_selection": pilot_report,
        "sctransform": sct_metrics,
        "scvi": scvi_status,
        "claim_guardrails": guardrails,
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    write_json(output_dir / "run_summary.json", summary)
    logger.info("COMPLETE raw-count recovery and SCT pilot; outputs=%s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
