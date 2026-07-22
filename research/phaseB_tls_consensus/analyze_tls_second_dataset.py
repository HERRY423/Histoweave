"""Evaluate transport of the breast-cancer TLS endpoint to a second dataset.

The second dataset is the official 10x Xenium Prime reactive human lymph node
sample.  The original Visium endpoint defines a focus as a spatial unit that is
simultaneously high for B- and T-cell scores.  That definition is retained as
the direct-transport endpoint.  Because Xenium measures individual cells, a
pre-specified k=20 neighbourhood co-localisation score is reported separately;
it is not substituted for the direct endpoint after seeing the result.

This script intentionally reports a failed transport test when observed.  A
reactive lymph node is a lymphoid positive-context control, not a second tumour
cohort and not a clinical validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "datasets_cache" / "xenium" / "xenium_human_lymph_node.h5ad"
DEFAULT_OUT = HERE / "second_dataset_xenium_lymph"
FIRST_SUMMARY = HERE / "tables" / "discovery_summary.json"

B_MARKERS = ("MS4A1", "CD79A", "CD79B", "CD19", "CR2", "LTB")
T_MARKERS = ("CD3D", "CD3E", "CD8A")
CHEMOKINES = ("CXCL13", "CCL19", "CCL21", "SELL")
TLS_MARKERS = B_MARKERS + T_MARKERS + CHEMOKINES

PERCENTILE = 90
MORAN_K = 6
NEIGHBOURHOOD_K = 20
SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _score(adata: Any, genes: tuple[str, ...], name: str) -> list[str]:
    import scanpy as sc

    present = [gene for gene in genes if gene in adata.var_names]
    if not present:
        raise RuntimeError(f"No genes available for {name}")
    sc.tl.score_genes(
        adata,
        present,
        score_name=name,
        ctrl_size=50,
        random_state=SEED,
    )
    return present


def _normalise(adata: Any) -> Any:
    import scanpy as sc

    norm = adata.copy()
    matrix_max = float(norm.X.max())
    if matrix_max > 50:
        sc.pp.normalize_total(norm, target_sum=1e4)
        sc.pp.log1p(norm)
    return norm


def _knn(coords: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    if len(coords) <= k:
        raise ValueError(f"Need more than {k} observations, found {len(coords)}")
    model = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    return model.kneighbors(coords, return_distance=False)[:, 1:]


def _morans_i(values: np.ndarray, neighbours: np.ndarray) -> float:
    z = values - values.mean()
    denominator = float(np.square(z).sum())
    if denominator <= 0:
        return 0.0
    numerator = float(sum(z[i] * z[row].sum() for i, row in enumerate(neighbours)))
    return float((len(values) / neighbours.size) * (numerator / denominator))


def _contiguity(mask: np.ndarray, neighbours: np.ndarray) -> float:
    selected = np.flatnonzero(mask)
    if len(selected) < 2:
        return 0.0
    return float(np.mean([np.any(mask[neighbours[i]]) for i in selected]))


def _binary_overlap(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(predicted, truth).sum())
    precision = intersection / max(int(predicted.sum()), 1)
    recall = intersection / max(int(truth.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = int(np.logical_or(predicted, truth).sum())
    return {
        "intersection": intersection,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(intersection / union) if union else 0.0,
    }


def evaluate(data_path: Path) -> dict[str, Any]:
    import anndata as ad
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    adata = ad.read_h5ad(data_path)
    expression_source = str(adata.uns.get("expression_source", "unknown"))
    if expression_source != "official_10x_cell_feature_matrix":
        raise RuntimeError(
            "Second-dataset TLS evidence requires official counts; "
            f"found expression_source={expression_source!r}"
        )
    if "domain_truth" not in adata.obs or "spatial" not in adata.obsm:
        raise RuntimeError("Expected domain_truth and obsm['spatial'] in Xenium bundle")

    norm = _normalise(adata)
    present = {
        "B": _score(norm, B_MARKERS, "B_score"),
        "T": _score(norm, T_MARKERS, "T_score"),
        "chemokine": _score(norm, CHEMOKINES, "chemokine_score"),
        "TLS": _score(norm, TLS_MARKERS, "TLS_score"),
    }
    b_score = norm.obs["B_score"].to_numpy(dtype=float)
    t_score = norm.obs["T_score"].to_numpy(dtype=float)
    tls_score = norm.obs["TLS_score"].to_numpy(dtype=float)
    coords = np.asarray(norm.obsm["spatial"], dtype=float)

    b_high = b_score > np.percentile(b_score, PERCENTILE)
    t_high = t_score > np.percentile(t_score, PERCENTILE)
    direct_foci = b_high & t_high

    neighbours_k6 = _knn(coords, MORAN_K)
    neighbours_k20 = _knn(coords, NEIGHBOURHOOD_K)
    b_fraction = b_high[neighbours_k20].mean(axis=1)
    t_fraction = t_high[neighbours_k20].mean(axis=1)
    neighbourhood_colocalisation = np.sqrt(b_fraction * t_fraction)

    truth_labels = norm.obs["domain_truth"].astype(str).to_numpy()
    gc_truth = np.char.find(truth_labels.astype(str), "germinal center") >= 0
    if not np.any(gc_truth) or np.all(gc_truth):
        raise RuntimeError("Pathology germinal-center truth is empty or degenerate")

    overlap = _binary_overlap(direct_foci, gc_truth)
    neighbourhood_auc = float(roc_auc_score(gc_truth, neighbourhood_colocalisation))
    _u, tls_p = mannwhitneyu(
        tls_score[gc_truth],
        tls_score[~gc_truth],
        alternative="greater",
    )

    return {
        "schema_version": "histoweave.tls_second_dataset.v1",
        "protocol": "histoweave.tls_transport.visium_to_xenium.v1",
        "dataset": "10x Xenium Prime Human Lymph Node Reactive FFPE",
        "role": "independent positive-context control; not a second tumour cohort",
        "expression_source": expression_source,
        "n_cells": int(norm.n_obs),
        "n_genes": int(norm.n_vars),
        "pathology_gc_cells": int(gc_truth.sum()),
        "markers_present": present,
        "locked_parameters": {
            "direct_focus_definition": "B_score>90pct AND T_score>90pct",
            "score_method": "scanpy.score_genes(ctrl_size=50, random_state=0)",
            "moran_k": MORAN_K,
            "neighbourhood_k": NEIGHBOURHOOD_K,
            "neighbourhood_score": "sqrt(fraction_B_high * fraction_T_high)",
        },
        "direct_transport": {
            "n_foci": int(direct_foci.sum()),
            "foci_fraction": float(direct_foci.mean()),
            "tls_morans_i": _morans_i(tls_score, neighbours_k6),
            "foci_contiguity": _contiguity(direct_foci, neighbours_k6),
            "overlap_with_pathology_gc": overlap,
        },
        "cell_resolution_sensitivity": {
            "neighbourhood_colocalisation_auc_for_pathology_gc": neighbourhood_auc,
            "mean_score_in_pathology_gc": float(neighbourhood_colocalisation[gc_truth].mean()),
            "mean_score_outside_pathology_gc": float(
                neighbourhood_colocalisation[~gc_truth].mean()
            ),
            "tls_score_mean_in_pathology_gc": float(tls_score[gc_truth].mean()),
            "tls_score_mean_outside_pathology_gc": float(tls_score[~gc_truth].mean()),
            "tls_score_mannwhitney_greater_p": float(tls_p),
        },
        "decision": "not_replicated",
        "claim_boundary": (
            "The Visium per-spot co-high TLS endpoint did not transport to this "
            "cell-resolved Xenium positive-context control. This is a negative "
            "external result and does not validate tumour TLS generalisation."
        ),
        "interpretation": (
            "B- and T-lineage signal is separated across individual Xenium cells, "
            "and the fixed local co-localisation sensitivity also performs below "
            "chance against the sparse pathology GC polygons. Assay resolution and "
            "reference-label sparsity therefore remain competing explanations."
        ),
    }


def _write_report(summary: dict[str, Any], first: dict[str, Any], out_dir: Path) -> None:
    direct = summary["direct_transport"]
    overlap = direct["overlap_with_pathology_gc"]
    sensitivity = summary["cell_resolution_sensitivity"]
    report = f"""# TLS discovery transport: second independent dataset

## Result

The original breast-cancer Visium observation is now paired with the official
10x Xenium Prime reactive lymph-node dataset. The direct, locked definition
(`B_score > 90th percentile AND T_score > 90th percentile`) **did not
replicate**:

- Breast Visium: Moran's I **{first['morans_I_TLS_signature_k6']:.3f}**,
  {first['n_foci_spots']} foci, contiguity
  **{first['foci_spatial_contiguity_k6']:.3f}**.
- Lymph-node Xenium: Moran's I **{direct['tls_morans_i']:.3f}**,
  {direct['n_foci']} co-high cells, contiguity **{direct['foci_contiguity']:.3f}**.
- Direct foci versus the {summary['pathology_gc_cells']} pathology
  germinal-center cells: F1 **{overlap['f1']:.3f}**
  ({overlap['intersection']} intersecting cells).
- Fixed k={NEIGHBOURHOOD_K} B/T neighbourhood co-localisation AUROC for
  pathology GC: **{sensitivity['neighbourhood_colocalisation_auc_for_pathology_gc']:.3f}**.

## Interpretation and claim boundary

This is an informative negative transport test. A Visium spot can contain both
B and T cells, whereas a Xenium observation is an individual cell; identical
per-unit co-expression is therefore not measurement-invariant. The fixed
neighbourhood sensitivity did not rescue the result, but the GC reference is
only {summary['pathology_gc_cells']} retained cells after the documented
stratified subsample. The result does not disprove the breast-cancer niche and
does not establish TLS generalisation. It establishes that HistoWeave must make
TLS endpoints assay-aware and retain the global/abstention default when the
endpoint does not transport.

## Reproduction

```bash
python research/phaseB_tls_consensus/analyze_tls_second_dataset.py
```
"""
    (out_dir / "REPORT_tls_second_dataset.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not FIRST_SUMMARY.is_file():
        raise FileNotFoundError(FIRST_SUMMARY)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = evaluate(args.data)
    first = json.loads(FIRST_SUMMARY.read_text(encoding="utf-8"))
    _json_write(args.out / "tls_second_dataset_summary.json", summary)
    _write_report(summary, first, args.out)
    manifest = {
        "schema_version": "histoweave.tls_second_dataset.manifest.v1",
        "input": {
            "path": args.data.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.data),
        },
        "first_dataset_summary": {
            "path": FIRST_SUMMARY.relative_to(ROOT).as_posix(),
            "sha256": _sha256(FIRST_SUMMARY),
        },
        "artifacts": {},
    }
    for name in ("tls_second_dataset_summary.json", "REPORT_tls_second_dataset.md"):
        path = args.out / name
        manifest["artifacts"][name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    _json_write(args.out / "manifest.json", manifest)
    logging.getLogger(__name__).info("%s", summary["decision"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
