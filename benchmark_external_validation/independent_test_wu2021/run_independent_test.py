"""Run the preregistered Wu et al. 2021 study-level independent test.

The frozen deployment policy is ``spectral``. Other common-panel methods are
evaluated only to calculate selection regret and cannot change the policy or
the 0.02 ARI success margin. BANKSY-Python is a separate diagnostic comparator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = ROOT / "datasets_cache" / "raw_sources" / "wu2021_breast"
EXTRACTED = RAW / "extracted"
CACHE = ROOT / "datasets_cache" / "wu2021_breast"
EMBEDDINGS = CACHE / "embeddings"
CHECKPOINTS = CACHE / "checkpoints"
PROTOCOL_PATH = HERE / "preregistered_protocol.json"

ADAPTER_ROOT = ROOT / "5x15_spatial_aware"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from adapters import banksy_py_adapter  # noqa: E402

from histoweave._math import kmeans as histoweave_kmeans  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins.builtin._sklearn_base import _spatial_embedding  # noqa: E402

SAMPLES = ("1142243F", "1160920F", "CID4290", "CID4465", "CID44971", "CID4535")
COMMON_METHODS = (
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "gaussian_mixture",
    "kmeans",
    "minibatch_kmeans",
    "spectral",
)
DIAGNOSTIC_METHODS = ("banksy_py",)
SEEDS = (42, 1, 2)
FROZEN_METHOD = "spectral"
SUCCESS_MARGIN = 0.02
N_BOOT = 10000
BOOT_SEED = 20260722
NONBIOLOGICAL_LABELS = {"Artefact", "Uncertain"}

LOG = logging.getLogger("histoweave.independent_test_wu2021")


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lines(path: Path) -> list[str]:
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]


def _sample_paths(sample: str) -> dict[str, Path]:
    matrix_dir = EXTRACTED / "filtered_count_matrices" / f"{sample}_filtered_count_matrix"
    return {
        "matrix": matrix_dir / "matrix.mtx.gz",
        "features": matrix_dir / "features.tsv.gz",
        "barcodes": matrix_dir / "barcodes.tsv.gz",
        "metadata": EXTRACTED / "metadata" / f"{sample}_metadata.csv",
        "positions": EXTRACTED / "spatial" / f"{sample}_spatial" / "tissue_positions_list.csv",
    }


def _load_sample(sample: str, *, sensitivity: bool = False) -> dict[str, Any]:
    paths = _sample_paths(sample)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing deposited files for {sample}: {missing}")

    barcodes = _read_lines(paths["barcodes"])
    genes = _read_lines(paths["features"])
    # The deposit uses a `.gz` suffix for plain-text MatrixMarket files.
    # Passing a file handle prevents SciPy from inferring gzip from the suffix.
    with paths["matrix"].open("rb") as handle:
        matrix = mmread(handle)
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    matrix = matrix.tocsr()
    if matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsr()
    if matrix.shape != (len(barcodes), len(genes)):
        raise RuntimeError(
            f"Matrix shape {matrix.shape} does not match "
            f"barcodes={len(barcodes)}, genes={len(genes)} for {sample}"
        )

    metadata = pd.read_csv(paths["metadata"], index_col=0)
    positions = pd.read_csv(
        paths["positions"],
        header=None,
        names=("barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"),
        index_col=0,
    )
    barcode_index = pd.Index(barcodes)
    keep = barcode_index.isin(metadata.index) & barcode_index.isin(positions.index)
    aligned_meta = metadata.reindex(barcode_index)
    aligned_pos = positions.reindex(barcode_index)
    keep &= aligned_meta["Classification"].notna().to_numpy()
    keep &= aligned_pos["in_tissue"].fillna(0).to_numpy(dtype=int) == 1
    if sensitivity:
        keep &= ~aligned_meta["Classification"].isin(NONBIOLOGICAL_LABELS).to_numpy()
    idx = np.flatnonzero(keep)
    truth = aligned_meta.iloc[idx]["Classification"].astype(str).to_numpy()
    coords = aligned_pos.iloc[idx][["pixel_col", "pixel_row"]].to_numpy(dtype=np.float32)
    counts = matrix[idx].astype(np.float32).tocsr()
    return {
        "sample": sample,
        "counts": counts,
        "coords": coords,
        "truth": truth,
        "genes": genes,
        "barcodes": barcode_index[idx].astype(str).tolist(),
        "n_original": len(barcodes),
        "n_domains": int(pd.Series(truth).nunique()),
        "label_counts": pd.Series(truth).value_counts().sort_index().to_dict(),
        "paths": paths,
    }


def _normalise_sparse(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    normed = counts.multiply((1e4 / totals)[:, None]).tocsr()
    normed.data = np.log1p(normed.data)
    return normed


def _embedding(sample_data: dict[str, Any], *, sensitivity: bool = False) -> np.ndarray:
    suffix = "sensitivity" if sensitivity else "primary"
    path = EMBEDDINGS / f"{sample_data['sample']}__{suffix}.npy"
    if path.is_file():
        return np.load(path)
    dense = _normalise_sparse(sample_data["counts"]).toarray().astype(np.float32)
    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical(sample_data["truth"])},
        index=sample_data["barcodes"],
    )
    table = SpatialTable(
        X=dense,
        obs=obs,
        var=pd.DataFrame(index=sample_data["genes"]),
        obsm={"spatial": sample_data["coords"]},
    )
    embedding = _spatial_embedding(
        table,
        n_pcs=15,
        n_neighbors=12,
        spatial_weight=0.3,
        random_state=0,
    ).astype(np.float32)
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding)
    return embedding


def _fit_common(method: str, embedding: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=k).fit_predict(embedding)
    if method == "birch":
        return Birch(n_clusters=k, threshold=0.5, branching_factor=50).fit_predict(embedding)
    if method == "bisecting_kmeans":
        return BisectingKMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embedding)
    if method == "gaussian_mixture":
        return GaussianMixture(n_components=k, n_init=5, random_state=seed).fit_predict(embedding)
    if method == "kmeans":
        return histoweave_kmeans(embedding, k, random_state=seed)
    if method == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=k,
            batch_size=256,
            n_init=10,
            random_state=seed,
        ).fit_predict(embedding)
    if method == "spectral":
        return SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            n_neighbors=min(12, len(embedding) - 1),
            random_state=seed,
        ).fit_predict(embedding)
    raise KeyError(method)


def _checkpoint_path(sample: str, method: str, seed: int, *, sensitivity: bool) -> Path:
    suffix = "sensitivity" if sensitivity else "primary"
    return CHECKPOINTS / f"{sample}__{method}__seed{seed}__{suffix}.json"


def _run_cell(
    sample_data: dict[str, Any],
    embedding: np.ndarray,
    method: str,
    seed: int,
    *,
    sensitivity: bool,
    run_banksy: bool,
) -> dict[str, Any]:
    checkpoint = _checkpoint_path(sample_data["sample"], method, seed, sensitivity=sensitivity)
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if method == "banksy_py":
            if not run_banksy:
                raise RuntimeError("BANKSY diagnostic disabled")
            labels = banksy_py_adapter.run(
                sample_data["counts"],
                sample_data["coords"],
                seed=seed,
                n_domains=sample_data["n_domains"],
            )
        else:
            labels = _fit_common(method, embedding, sample_data["n_domains"], seed)
        payload = {
            "sample": sample_data["sample"],
            "method": method,
            "seed": seed,
            "ari": float(adjusted_rand_score(sample_data["truth"], labels)),
            "seconds": float(time.perf_counter() - started),
            "status": "success",
            "sensitivity": sensitivity,
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "sample": sample_data["sample"],
            "method": method,
            "seed": seed,
            "ari": None,
            "seconds": float(time.perf_counter() - started),
            "status": "failed",
            "error": str(exc)[:500],
            "sensitivity": sensitivity,
        }
    _json_write(checkpoint, payload)
    return payload


def _study_summary(rows: list[dict[str, Any]], sample_meta: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "success" and not row["sensitivity"]]
    means: dict[str, dict[str, float]] = {}
    sample_rows: list[dict[str, Any]] = []
    for sample in SAMPLES:
        means[sample] = {}
        for method in COMMON_METHODS + DIAGNOSTIC_METHODS:
            values = [
                float(row["ari"])
                for row in successful
                if row["sample"] == sample and row["method"] == method
            ]
            if values:
                means[sample][method] = float(np.mean(values))
        common = {
            method: means[sample][method] for method in COMMON_METHODS if method in means[sample]
        }
        if FROZEN_METHOD not in common or len(common) != len(COMMON_METHODS):
            continue
        oracle_method = max(common, key=common.get)
        regret = common[oracle_method] - common[FROZEN_METHOD]
        sample_rows.append(
            {
                "sample": sample,
                "frozen_method": FROZEN_METHOD,
                "frozen_mean_ari": common[FROZEN_METHOD],
                "oracle_method": oracle_method,
                "oracle_mean_ari": common[oracle_method],
                "frozen_regret": float(regret),
                "frozen_top1": bool(regret <= 1e-12),
                "n_domains": next(
                    item["n_domains"] for item in sample_meta if item["sample"] == sample
                ),
            }
        )
    regrets = np.asarray([row["frozen_regret"] for row in sample_rows], dtype=float)
    if len(regrets):
        rng = np.random.default_rng(BOOT_SEED)
        boot = regrets[rng.integers(0, len(regrets), size=(N_BOOT, len(regrets)))].mean(axis=1)
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        mean_regret = float(regrets.mean())
    else:
        ci_low = ci_high = mean_regret = float("nan")

    method_means = {
        method: float(np.mean([means[s][method] for s in SAMPLES if method in means[s]]))
        for method in COMMON_METHODS + DIAGNOSTIC_METHODS
        if any(method in means[s] for s in SAMPLES)
    }
    passed = len(sample_rows) >= 4 and mean_regret <= SUCCESS_MARGIN
    return {
        "schema_version": "histoweave.independent_test.result.v1",
        "protocol": "histoweave.independent_test.wu2021_breast.v1",
        "test_set": "Wu et al. 2021 six-patient breast-cancer Visium cohort",
        "independent_unit": "external_study with patient/section-level resampling",
        "n_evaluable_sections": len(sample_rows),
        "frozen_policy": FROZEN_METHOD,
        "confirmatory_endpoint": "study-level mean regret of frozen spectral policy",
        "success_margin_ari": SUCCESS_MARGIN,
        "mean_frozen_policy_regret": mean_regret,
        "bootstrap_ci": {
            "level": 0.95,
            "low": float(ci_low),
            "high": float(ci_high),
            "n_boot": N_BOOT,
            "resampling_unit": "patient/section",
            "seed": BOOT_SEED,
        },
        "success": bool(passed),
        "decision": "independent_test_pass" if passed else "independent_test_fail",
        "top1_frequency": float(np.mean([row["frozen_top1"] for row in sample_rows]))
        if sample_rows
        else 0.0,
        "sample_results": sample_rows,
        "method_mean_ari_across_sections": method_means,
        "claim_boundary": (
            "A pass supports transport of the already-frozen global spectral policy "
            "to this independent study. It does not establish personalised-method "
            "superiority. A fail is retained as a negative external result."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    successful = [row for row in rows if row["status"] == "success" and not row["sensitivity"]]
    methods = list(COMMON_METHODS) + [
        m for m in DIAGNOSTIC_METHODS if m in summary["method_mean_ari_across_sections"]
    ]
    matrix = np.full((len(SAMPLES), len(methods)), np.nan)
    for i, sample in enumerate(SAMPLES):
        for j, method in enumerate(methods):
            values = [
                row["ari"]
                for row in successful
                if row["sample"] == sample and row["method"] == method
            ]
            if values:
                matrix[i, j] = float(np.mean(values))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    image = ax.imshow(
        matrix, aspect="auto", cmap="viridis", vmin=-0.05, vmax=max(0.4, np.nanmax(matrix))
    )
    ax.set_yticks(range(len(SAMPLES)))
    ax.set_yticklabels(SAMPLES)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=42, ha="right")
    ax.set_title("A  Frozen independent-study ARI")
    for i in range(len(SAMPLES)):
        for j in range(len(methods)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if matrix[i, j] > 0.2 else "black",
                )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Mean ARI")

    ax = axes[1]
    sample_results = summary["sample_results"]
    regrets = [row["frozen_regret"] for row in sample_results]
    colors = ["#2A9D8F" if value <= SUCCESS_MARGIN else "#E76F51" for value in regrets]
    ax.bar(range(len(regrets)), regrets, color=colors)
    ax.axhline(
        SUCCESS_MARGIN, color="#555555", linestyle="--", linewidth=1, label="Locked 0.02 margin"
    )
    ax.axhline(
        summary["mean_frozen_policy_regret"], color="#264653", linewidth=1.2, label="Study mean"
    )
    ax.set_xticks(range(len(regrets)))
    ax.set_xticklabels([row["sample"] for row in sample_results], rotation=42, ha="right")
    ax.set_ylabel("Frozen spectral regret (ARI)")
    ax.set_title(f"B  Independent test: {'PASS' if summary['success'] else 'FAIL'}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7)
    fig.text(
        0.01,
        0.01,
        "Policy and margin locked before outcome download. Sections are equally weighted; "
        "BANKSY is diagnostic and excluded from confirmatory regret.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "fig_independent_test_wu2021.svg", bbox_inches="tight")
    fig.savefig(HERE / "fig_independent_test_wu2021.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _independence_audit() -> dict[str, Any]:
    sources = {
        "strict_training_landscape": ROOT
        / "independent_personalisation_results"
        / "independent_unit_landscape.json",
        "external_development_manifest": ROOT
        / "benchmark_external_validation"
        / "dataset_manifest.json",
        "tls_discovery_summary": ROOT
        / "research"
        / "phaseB_tls_consensus"
        / "tables"
        / "discovery_summary.json",
    }
    identifiers = (*SAMPLES, "4739739", "Wu et al.")
    matches: dict[str, list[str]] = {}
    for name, path in sources.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        matches[name] = [
            identifier for identifier in identifiers if identifier.lower() in text.lower()
        ]
    return {
        "schema_version": "histoweave.independent_test.audit.v1",
        "test_study": "Wu et al. 2021 / Zenodo 4739739",
        "training_sources_checked": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "test_identifiers_found_in_training_sources": matches,
        "all_identifier_matches_empty": all(not value for value in matches.values()),
        "study_level_independence": all(not value for value in matches.values()),
        "note": (
            "The test study is absent from model-selection sources. The prior 10x FFPE "
            "breast TLS sample is a different public dataset and study."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="*", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--skip-banksy", action="store_true")
    parser.add_argument(
        "--sensitivity", action="store_true", help="Also exclude Artefact/Uncertain labels"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["frozen_policy"]["selected_method"] != FROZEN_METHOD:
        raise RuntimeError("Code and preregistered frozen policy disagree")
    if float(protocol["confirmatory_endpoint"]["success_margin_ari"]) != SUCCESS_MARGIN:
        raise RuntimeError("Code and preregistered success margin disagree")

    all_rows: list[dict[str, Any]] = []
    sample_meta: list[dict[str, Any]] = []
    for sample in args.samples:
        LOG.info("Loading frozen test sample %s", sample)
        data = _load_sample(sample, sensitivity=False)
        embedding = _embedding(data, sensitivity=False)
        sample_meta.append(
            {
                "sample": sample,
                "n_spots": len(data["truth"]),
                "n_original": data["n_original"],
                "n_genes": len(data["genes"]),
                "n_domains": data["n_domains"],
                "label_counts": data["label_counts"],
            }
        )
        for seed in SEEDS:
            for method in COMMON_METHODS + DIAGNOSTIC_METHODS:
                row = _run_cell(
                    data,
                    embedding,
                    method,
                    seed,
                    sensitivity=False,
                    run_banksy=not args.skip_banksy,
                )
                all_rows.append(row)
                LOG.info("%s %s seed=%s ARI=%s", sample, method, seed, row.get("ari"))

        if args.sensitivity:
            sensitivity_data = _load_sample(sample, sensitivity=True)
            sensitivity_embedding = _embedding(sensitivity_data, sensitivity=True)
            for seed in SEEDS:
                for method in COMMON_METHODS:
                    all_rows.append(
                        _run_cell(
                            sensitivity_data,
                            sensitivity_embedding,
                            method,
                            seed,
                            sensitivity=True,
                            run_banksy=False,
                        )
                    )

    if tuple(args.samples) != SAMPLES:
        LOG.info("Partial run complete; final study summary requires all six samples")
        return

    summary = _study_summary(all_rows, sample_meta)
    audit = _independence_audit()
    _write_csv(HERE / "benchmark_long.csv", all_rows)
    _write_csv(HERE / "sample_regret.csv", summary["sample_results"])
    _json_write(HERE / "independent_test_summary.json", summary)
    _json_write(HERE / "dataset_manifest.json", {"samples": sample_meta})
    _json_write(HERE / "independence_audit.json", audit)
    _plot(summary, all_rows)

    report = f"""# Frozen independent test: Wu et al. 2021 breast-cancer cohort

## Independence and lock

This is a one-shot test on six primary breast cancers from an external study
(Zenodo DOI 10.5281/zenodo.4739739). Test identifiers were absent from the
training landscape, five-study external development benchmark, and prior TLS
discovery summary. The policy (`spectral`), seven-method comparator panel,
oracle-K task contract, and 0.02 ARI regret margin were locked before outcome
download in `preregistered_protocol.json`.

## Confirmatory result

- Evaluable patient/sections: **{summary["n_evaluable_sections"]}**.
- Frozen spectral-policy mean regret: **{summary["mean_frozen_policy_regret"]:.4f} ARI**.
- Patient/section bootstrap 95% CI:
  **[{summary["bootstrap_ci"]["low"]:.4f}, {summary["bootstrap_ci"]["high"]:.4f}]**.
- Locked success margin: **{SUCCESS_MARGIN:.2f} ARI**.
- Test decision: **{summary["decision"]}**.
- Spectral top-1 frequency: **{summary["top1_frequency"]:.1%}**.

The decision uses the preregistered point-estimate rule. The confidence interval
is descriptive because this is one external study with six patient/sections.
Regardless of pass/fail, the result does not support personalised superiority.

![Independent test](fig_independent_test_wu2021.png)

## Per-section results

| Section | Frozen spectral ARI | Oracle method | Oracle ARI | Regret |
|---|---:|---|---:|---:|
"""
    for row in summary["sample_results"]:
        report += (
            f"| {row['sample']} | {row['frozen_mean_ari']:.4f} | {row['oracle_method']} | "
            f"{row['oracle_mean_ari']:.4f} | {row['frozen_regret']:.4f} |\n"
        )
    report += """

## Reproduction

```bash
python benchmark_external_validation/independent_test_wu2021/run_independent_test.py
```

Raw files are not redistributed by HistoWeave; download them from the DOI above.
"""
    (HERE / "REPORT_independent_test_wu2021.md").write_text(report, encoding="utf-8")

    inputs = [
        RAW / "metadata.tar.gz",
        RAW / "filtered_count_matrices.tar.gz",
        RAW / "spatial.tar.gz",
        PROTOCOL_PATH,
    ]
    artifacts = [
        HERE / "benchmark_long.csv",
        HERE / "sample_regret.csv",
        HERE / "independent_test_summary.json",
        HERE / "dataset_manifest.json",
        HERE / "independence_audit.json",
        HERE / "fig_independent_test_wu2021.svg",
        HERE / "fig_independent_test_wu2021.png",
        HERE / "REPORT_independent_test_wu2021.md",
    ]
    manifest = {
        "protocol": protocol["protocol"],
        "inputs": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in artifacts
        },
    }
    _json_write(HERE / "manifest.json", manifest)
    LOG.info("%s", summary["decision"])


if __name__ == "__main__":
    main()
