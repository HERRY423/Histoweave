#!/usr/bin/env python3
"""Deterministic 40-epoch scVI technical smoke test on one raw-count puck.

No synthetic batch labels are created.  The single observed puck is modeled as
one dataset, so its latent space cannot demonstrate batch correction,
replication, or cross-tissue generalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi
import torch

SEED = 20_011_508
EPOCHS = 40

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
    "oligodendrocyte": ("Mbp", "Plp1", "Mog", "Mag", "Cnp", "Mobp", "Opalin", "Mal"),
    "microglia_homeostasis": ("C1qa", "C1qb", "C1qc", "P2ry12", "Tmem119", "Csf1r", "Cx3cr1"),
    "lipid_metabolic_support": ("Apoe", "Lpl", "Abca1", "Mertk", "Fabp7", "Acsl6", "Angptl4"),
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
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("slideseq_scvi_smoke")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "scvi_run.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def stored_values(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.data)
    return np.asarray(matrix)


def validate_raw_counts(adata: ad.AnnData) -> sp.csr_matrix:
    if "counts" not in adata.layers:
        raise ValueError("Input H5AD has no layers['counts']")
    counts = sp.csr_matrix(adata.layers["counts"])
    values = stored_values(counts)
    checks = {
        "csr": sp.isspmatrix_csr(counts),
        "finite": bool(np.isfinite(values).all()),
        "nonnegative": bool(np.all(values >= 0)),
        "integer_like": bool(np.allclose(values, np.rint(values), atol=0, rtol=0)),
        "positive": bool(np.any(values > 0)),
    }
    if not all(checks.values()):
        raise ValueError(f"Raw-count gate failed: {checks}")
    counts.data = np.rint(counts.data).astype(np.int32)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    if sp.issparse(adata.X):
        x = sp.csr_matrix(adata.X)
        exact = (
            np.array_equal(x.indptr, counts.indptr)
            and np.array_equal(x.indices, counts.indices)
            and np.array_equal(x.data, counts.data)
        )
        if not exact:
            raise ValueError("Input X and layers['counts'] are not exact")
    return counts


def column_correlations(matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    centered_y = np.asarray(y, dtype=np.float64) - np.mean(y)
    centered_x = x - x.mean(axis=0, keepdims=True)
    numerator = centered_x.T @ centered_y
    denominator = np.sqrt(np.sum(centered_x**2, axis=0) * np.sum(centered_y**2))
    return np.divide(numerator, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 0)


def summarize_abs(values: np.ndarray) -> dict[str, float]:
    finite = np.abs(values[np.isfinite(values)])
    return {
        "median_abs_correlation": float(np.median(finite)),
        "p90_abs_correlation": float(np.quantile(finite, 0.9)),
        "max_abs_correlation": float(np.max(finite)),
    }


def flatten_history(history: dict[str, Any]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for key, value in history.items():
        if isinstance(value, pd.DataFrame):
            if value.shape[1] == 1:
                columns[key] = value.iloc[:, 0].reset_index(drop=True)
            else:
                for subcolumn in value.columns:
                    columns[f"{key}_{subcolumn}"] = value[subcolumn].reset_index(drop=True)
        elif isinstance(value, pd.Series):
            columns[key] = value.reset_index(drop=True)
        else:
            array = np.asarray(value).ravel()
            columns[key] = pd.Series(array)
    frame = pd.DataFrame(columns)
    frame.index.name = "record"
    return frame.reset_index()


def finite_endpoint(frame: pd.DataFrame, token: str) -> dict[str, float]:
    candidates = [column for column in frame if token in column.lower()]
    result: dict[str, float] = {}
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values):
            result[f"{column}_first"] = float(values.iloc[0])
            result[f"{column}_last"] = float(values.iloc[-1])
    return result


def module_summary(
    normalized: np.ndarray,
    genes: list[str],
    clusters: np.ndarray,
) -> pd.DataFrame:
    lookup = {gene.lower(): index for index, gene in enumerate(genes)}
    rows: list[dict[str, Any]] = []
    for module, markers in MARKER_MODULES.items():
        positions = [lookup[marker.lower()] for marker in markers if marker.lower() in lookup]
        if not positions:
            continue
        score = np.log1p(normalized[:, positions]).mean(axis=1)
        for cluster in sorted(set(clusters)):
            mask = clusters == cluster
            rows.append(
                {
                    "module": module,
                    "cluster": cluster,
                    "n_beads": int(mask.sum()),
                    "n_markers": len(positions),
                    "mean_log1p_scvi_normalized_module": float(score[mask].mean()),
                    "median_log1p_scvi_normalized_module": float(np.median(score[mask])),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "results" / "Puck_200115_08_raw_counts_sct_pilot_input.h5ad",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-latent", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logger = configure_logging(output_dir)
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.epochs != 40:
        logger.warning("Non-default epoch count requested: %d", args.epochs)

    scvi.settings.seed = SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    logger.info("Reading deterministic raw-count pilot %s", args.input)
    adata = ad.read_h5ad(args.input)
    counts = validate_raw_counts(adata)
    if adata.n_obs != 4_000 or adata.n_vars != 2_024:
        raise ValueError(f"Expected deterministic 4000 x 2024 pilot; got {adata.shape}")
    if (
        "source_annotation_matched" in adata.obs
        and not adata.obs["source_annotation_matched"].all()
    ):
        raise ValueError("Pilot contains an annotation-unmatched bead")

    # Deliberately omit batch_key: there is one observed puck and no legitimate
    # batch factor to regress.  scVI's internal registry therefore has one
    # default category rather than fabricated biological replication.
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(
        adata,
        n_hidden=128,
        n_latent=args.n_latent,
        n_layers=2,
        dropout_rate=0.1,
        dispersion="gene",
        gene_likelihood="nb",
        latent_distribution="normal",
    )
    logger.info(
        "Training scVI %d epochs on CPU; no batch_key and no synthetic batches",
        args.epochs,
    )
    model.train(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        train_size=0.9,
        validation_size=0.1,
        shuffle_set_split=True,
        batch_size=args.batch_size,
        early_stopping=False,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        deterministic=True,
    )

    history = flatten_history(model.history)
    history.to_csv(output_dir / "scvi_training_history.csv", index=False)
    latent = np.asarray(model.get_latent_representation(give_mean=True), dtype=np.float32)
    normalized = np.asarray(
        model.get_normalized_expression(
            library_size=10_000,
            n_samples=1,
            return_mean=True,
            return_numpy=True,
            batch_size=args.batch_size,
        ),
        dtype=np.float32,
    )
    if latent.shape != (adata.n_obs, args.n_latent):
        raise RuntimeError(f"Unexpected latent shape {latent.shape}")
    if normalized.shape != adata.shape:
        raise RuntimeError(f"Unexpected normalized-expression shape {normalized.shape}")
    if not np.isfinite(latent).all() or not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise RuntimeError("scVI output failed finite/non-negative QC")

    barcodes = adata.obs_names.astype(str).tolist()
    genes = adata.var_names.astype(str).tolist()
    latent_table = pd.DataFrame(
        latent,
        index=pd.Index(barcodes, name="barcode"),
        columns=[f"scvi_latent_{index + 1}" for index in range(latent.shape[1])],
    )
    latent_table.to_csv(output_dir / "scvi_latent.csv")

    depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    log_depth = np.log1p(depth)
    raw_log = np.log1p(counts.toarray().astype(np.float32))
    library_normalized = counts.astype(np.float64).multiply((10_000.0 / depth)[:, None]).tocsr()
    library_normalized.data = np.log1p(library_normalized.data)
    raw_corr = column_correlations(raw_log, log_depth)
    library_corr = column_correlations(library_normalized.toarray().astype(np.float32), log_depth)
    scvi_corr = column_correlations(np.log1p(normalized), log_depth)
    gene_qc = pd.DataFrame(
        {
            "gene": genes,
            "mean_scvi_normalized": normalized.mean(axis=0),
            "variance_scvi_normalized": normalized.var(axis=0),
            "raw_log1p_correlation_with_log_depth": raw_corr,
            "library_normalized_log1p_correlation_with_log_depth": library_corr,
            "scvi_log1p_normalized_correlation_with_log_depth": scvi_corr,
        }
    )
    gene_qc.to_csv(output_dir / "scvi_normalized_expression_qc.csv", index=False)

    latent_corr = column_correlations(latent, log_depth)
    latent_depth = pd.DataFrame(
        {
            "latent_dimension": latent_table.columns,
            "correlation_with_log_depth": latent_corr,
        }
    )
    latent_depth.to_csv(output_dir / "scvi_latent_depth_correlations.csv", index=False)

    cell_sums = normalized.sum(axis=1)
    cell_qc = adata.obs[
        [column for column in ("cluster", "domain_truth") if column in adata.obs]
    ].copy()
    cell_qc["raw_total_counts"] = depth.astype(np.int64)
    cell_qc["scvi_normalized_sum"] = cell_sums
    cell_qc.to_csv(output_dir / "scvi_cell_qc.csv")
    clusters = adata.obs["cluster"].astype(str).to_numpy()
    module_summary(normalized, genes, clusters).to_csv(
        output_dir / "scvi_module_cluster_summary.csv", index=False
    )

    result = ad.AnnData(X=normalized, obs=adata.obs.copy(), var=adata.var.copy())
    result.layers["counts"] = counts
    result.obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    result.obsm["X_scVI"] = latent
    result.uns["scvi_smoke"] = {
        "technical_validation": True,
        "raw_integer_count_input": True,
        "epochs": args.epochs,
        "seed": SEED,
        "n_latent": args.n_latent,
        "batch_key": "none",
        "observed_pucks": 1,
        "synthetic_batches_created": False,
        "normalized_expression_library_size": 10_000,
        "claim_scope": "single-puck technical validity and candidate generation",
    }
    output_h5ad = output_dir / "Puck_200115_08_scvi_40epoch_smoke.h5ad"
    logger.info("Writing scVI normalized-expression/latent H5AD %s", output_h5ad)
    result.write_h5ad(output_h5ad, compression="gzip")
    model_dir = output_dir / "scvi_model_40epoch"
    model.save(model_dir, overwrite=True, save_anndata=False)

    metrics = {
        "status": "passed",
        "method": "scvi.model.SCVI",
        "scvi_version": scvi.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": SEED,
        "deterministic_algorithms": True,
        "epochs_requested": args.epochs,
        "n_beads": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_latent": args.n_latent,
        "raw_count_gate_passed": True,
        "batch_key": None,
        "observed_batches": 1,
        "synthetic_batches_created": False,
        "latent_finite": bool(np.isfinite(latent).all()),
        "normalized_expression_finite_nonnegative": bool(
            np.isfinite(normalized).all() and np.all(normalized >= 0)
        ),
        "normalized_expression_min": float(normalized.min()),
        "normalized_expression_max": float(normalized.max()),
        "normalized_cell_sum_median": float(np.median(cell_sums)),
        "normalized_cell_sum_min": float(np.min(cell_sums)),
        "normalized_cell_sum_max": float(np.max(cell_sums)),
        "raw_log1p_depth_dependence": summarize_abs(raw_corr),
        "library_normalized_log1p_depth_dependence": summarize_abs(library_corr),
        "scvi_normalized_depth_dependence": summarize_abs(scvi_corr),
        "scvi_latent_depth_dependence": summarize_abs(latent_corr),
        "training_endpoints": finite_endpoint(history, "elbo"),
        "claim_scope": {
            "biological_n": 1,
            "replication_gate_passed": False,
            "cross_tissue_generalization_gate_passed": False,
            "allowed": "technical smoke test and candidate generation",
        },
    }
    write_json(output_dir / "scvi_technical_metrics.json", metrics)
    write_json(
        output_dir / "scvi_status.json",
        {
            "status": "passed_technical_smoke",
            "raw_integer_counts": True,
            "epochs": args.epochs,
            "batch_key": None,
            "synthetic_batches_created": False,
            "biological_n": 1,
            "generalization_claim": False,
        },
    )

    run_summary_path = output_dir / "run_summary.json"
    if run_summary_path.exists():
        summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        summary["scvi"] = metrics
        summary["output_files"] = sorted(path.name for path in output_dir.iterdir())
        write_json(run_summary_path, summary)
    logger.info("COMPLETE scVI 40-epoch raw-count smoke test; biological_n=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
