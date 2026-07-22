"""Build the task-stratified strict external panel v2.

The panel registry contains ten independent biological/study units. Nine have
strict spatial-domain ground truth and enter the common seven-method LOOCV.
Two have TLS discovery evidence (breast Visium and lymph-node Xenium), with the
lymph node shared across both task strata. SOTA results are aligned to the same
unit registry and missing cells remain explicit; they are not imputed into the
confirmatory LOOCV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from histoweave.benchmark.independent_personalisation import (
    evaluate_personalisation_policies,
    summarise_policies,
)
from histoweave.benchmark.landscape import LandscapeResult

PROTOCOL = "histoweave.external_validation.strict_task_stratified.v2"
SEED = 20260722
N_BOOT = 10000
NONINFERIOR_MARGIN = 0.02
K_NEIGHBOURS = 3
MIN_TRAINING = 4
CONFIDENCE_FLOOR = 0.20
PROXY_ADVANTAGE = 0.02

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "independent_personalisation_results" / "independent_unit_landscape.json"
EXT_LONG = ROOT / "benchmark_external_validation" / "benchmark_long.csv"
DLPFC_SOTA = ROOT / "5x15_spatial_aware" / "performance_matrix_mean_full.csv"
BANKSY_LYMPH = HERE / "banksy_lymph_summary.json"
TLS_BREAST = ROOT / "research" / "phaseB_tls_consensus" / "tables" / "discovery_summary.json"
TLS_LYMPH = (
    ROOT
    / "research"
    / "phaseB_tls_consensus"
    / "second_dataset_xenium_lymph"
    / "tls_second_dataset_summary.json"
)

DOMAIN_UNITS = (
    "dlpfc_donor_Br5292",
    "dlpfc_donor_Br5595",
    "dlpfc_donor_Br8100",
    "visium_hd_crc",
    "xenium_lung_cancer",
    "xenium_ovarian_cancer",
    "visium_mouse_brain",
    "allen_merfish_brain_section",
    "xenium_human_lymph_node",
)
TLS_ONLY_UNIT = "visium_breast_cancer_tls"
ALL_UNITS = DOMAIN_UNITS + (TLS_ONLY_UNIT,)
METHODS = (
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "gaussian_mixture",
    "kmeans",
    "minibatch_kmeans",
    "spectral",
)
SOTA_METHODS = ("banksy_py", "spagcn", "stagate", "graphst", "bayesspace")

DLPFC_MEMBERS = {
    "dlpfc_donor_Br5292": ("151507",),
    "dlpfc_donor_Br5595": ("151669", "151670"),
    "dlpfc_donor_Br8100": ("151673", "151674"),
}
DISPLAY = {
    "dlpfc_donor_Br5292": "DLPFC Br5292",
    "dlpfc_donor_Br5595": "DLPFC Br5595",
    "dlpfc_donor_Br8100": "DLPFC Br8100",
    "visium_hd_crc": "CRC Visium HD",
    "xenium_lung_cancer": "Lung Xenium",
    "xenium_ovarian_cancer": "Ovary Xenium",
    "visium_mouse_brain": "Mouse brain Visium",
    "allen_merfish_brain_section": "Mouse brain MERFISH",
    "xenium_human_lymph_node": "Reactive lymph node Xenium",
    TLS_ONLY_UNIT: "Breast cancer Visium TLS",
}


def _json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_landscape() -> tuple[LandscapeResult, dict[str, Any]]:
    raw = _json_read(SOURCE)
    meta = raw.get("dataset_meta", {})
    missing = [unit for unit in DOMAIN_UNITS if unit not in raw["performance"]]
    if missing:
        raise RuntimeError(f"Strict domain units missing: {missing}")
    bad_truth = [
        unit
        for unit in DOMAIN_UNITS
        if meta.get(unit, {}).get("ground_truth_kind") != "spatial_domain"
    ]
    if bad_truth:
        raise RuntimeError(f"Non-spatial-domain truth in confirmatory stratum: {bad_truth}")

    performance = {
        unit: {method: float(raw["performance"][unit][method]) for method in METHODS}
        for unit in DOMAIN_UNITS
    }
    features = {unit: np.asarray(raw["features"][unit], dtype=float) for unit in DOMAIN_UNITS}
    timings = {
        unit: {method: raw.get("timings", {}).get(unit, {}).get(method) for method in METHODS}
        for unit in DOMAIN_UNITS
    }
    best = {
        unit: max(performance[unit], key=lambda method: performance[unit][method])
        for unit in DOMAIN_UNITS
    }
    niches = {method: [unit for unit in DOMAIN_UNITS if best[unit] == method] for method in METHODS}
    return (
        LandscapeResult(
            performance=performance,
            features=features,
            embedding={},
            best_method=best,
            niches=niches,
            timings=timings,
            feature_order=list(raw["feature_order"]),
            method_count=len(METHODS),
            dataset_count=len(DOMAIN_UNITS),
            task="spatial_domain",
            metric="ARI",
            higher_is_better=True,
            dataset_meta={unit: dict(meta[unit]) for unit in DOMAIN_UNITS},
        ),
        raw,
    )


def _registry_rows(source_raw: dict[str, Any]) -> list[dict[str, Any]]:
    meta = source_raw["dataset_meta"]
    rows: list[dict[str, Any]] = []
    for unit in DOMAIN_UNITS:
        record = meta[unit]
        rows.append(
            {
                "unit_id": unit,
                "display": DISPLAY[unit],
                "independence_class": record["independence_class"],
                "platform": record["platform"],
                "study": record.get("study", record.get("study_group", unit)),
                "domain_loocv_eligible": True,
                "tls_evidence_eligible": unit == "xenium_human_lymph_node",
                "ground_truth": "spatial_domain",
            }
        )
    rows.append(
        {
            "unit_id": TLS_ONLY_UNIT,
            "display": DISPLAY[TLS_ONLY_UNIT],
            "independence_class": "external_study",
            "platform": "visium_ffpe",
            "study": "10x Visium FFPE Human Breast Cancer",
            "domain_loocv_eligible": False,
            "tls_evidence_eligible": True,
            "ground_truth": "marker_and_spatial_statistics_only",
        }
    )
    return rows


def _sota_scores() -> list[dict[str, Any]]:
    with DLPFC_SOTA.open(newline="", encoding="utf-8") as handle:
        dlpfc_rows = {row["dataset"]: row for row in csv.DictReader(handle)}

    external_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with EXT_LONG.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method in SOTA_METHODS and row.get("status", "success") == "success":
                try:
                    external_values[(row["dataset"], method)].append(float(row["ari"]))
                except (TypeError, ValueError):
                    pass

    lymph = _json_read(BANKSY_LYMPH)
    rows: list[dict[str, Any]] = []
    for unit in ALL_UNITS:
        for method in SOTA_METHODS:
            score: float | None = None
            status = "missing"
            source = ""
            completeness = "none"
            if unit in DLPFC_MEMBERS:
                values = [
                    float(dlpfc_rows[sid][method])
                    for sid in DLPFC_MEMBERS[unit]
                    if sid in dlpfc_rows and dlpfc_rows[sid].get(method, "") not in {"", None}
                ]
                if values:
                    score = float(np.mean(values))
                    status = "available"
                    source = "5x15_spatial_aware/performance_matrix_mean_full.csv"
                    completeness = "partial_donor_slices"
            elif unit in {
                "visium_hd_crc",
                "xenium_lung_cancer",
                "xenium_ovarian_cancer",
                "visium_mouse_brain",
                "allen_merfish_brain_section",
            }:
                values = external_values.get((unit, method), [])
                if values:
                    score = float(np.mean(values))
                    status = "available"
                    source = "benchmark_external_validation/benchmark_long.csv"
                    completeness = "complete_unit_three_seed"
            elif unit == "xenium_human_lymph_node" and method == "banksy_py":
                score = float(lymph["mean_ari"])
                status = "available"
                source = (
                    "benchmark_external_validation/strict_external_panel_v2/"
                    "banksy_lymph_summary.json"
                )
                completeness = "complete_unit_three_seed"

            rows.append(
                {
                    "unit_id": unit,
                    "method": method,
                    "mean_ari": "" if score is None else f"{score:.9f}",
                    "status": status,
                    "completeness": completeness,
                    "source": source,
                }
            )
    return rows


def _sota_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in SOTA_METHODS:
        selected = [row for row in rows if row["method"] == method]
        available = [row for row in selected if row["status"] == "available"]
        complete = [row for row in available if row["completeness"] == "complete_unit_three_seed"]
        partial = [row for row in available if row["completeness"] == "partial_donor_slices"]
        summary[method] = {
            "available_units": len(available),
            "strict_complete_units": len(complete),
            "partial_donor_units": len(partial),
            "total_registry_units": len(ALL_UNITS),
            "domain_stratum_units": len(DOMAIN_UNITS),
        }
    return {
        "schema_version": "histoweave.strict_external_panel.sota_coverage.v2",
        "methods": summary,
        "confirmatory_loocv_inclusion": [],
        "reason": (
            "No SOTA method has complete, protocol-matched coverage across all nine "
            "domain units. BANKSY-Python is available for all nine domain units, "
            "but the three DLPFC donor scores cover only selected slices rather than "
            "all donor-member slices; the remaining SOTA methods cover DLPFC only."
        ),
        "missing_cells_imputed": False,
    }


def _tls_summary() -> dict[str, Any]:
    breast = _json_read(TLS_BREAST)
    lymph = _json_read(TLS_LYMPH)
    return {
        "schema_version": "histoweave.strict_external_panel.tls.v2",
        "n_independent_datasets": 2,
        "datasets": [
            {
                "unit_id": TLS_ONLY_UNIT,
                "platform": "Visium FFPE",
                "role": "exploratory tumour discovery",
                "tls_morans_i": breast["morans_I_TLS_signature_k6"],
                "n_foci": breast["n_foci_spots"],
                "foci_contiguity": breast["foci_spatial_contiguity_k6"],
                "decision": "discovery_supported_within_sample",
            },
            {
                "unit_id": "xenium_human_lymph_node",
                "platform": "Xenium Prime",
                "role": "independent positive-context transport control",
                "tls_morans_i": lymph["direct_transport"]["tls_morans_i"],
                "n_foci": lymph["direct_transport"]["n_foci"],
                "foci_contiguity": lymph["direct_transport"]["foci_contiguity"],
                "pathology_gc_f1": lymph["direct_transport"]["overlap_with_pathology_gc"]["f1"],
                "neighbourhood_auc": lymph["cell_resolution_sensitivity"][
                    "neighbourhood_colocalisation_auc_for_pathology_gc"
                ],
                "decision": lymph["decision"],
            },
        ],
        "cross_dataset_decision": "not_replicated",
        "claim_boundary": (
            "The breast-cancer TLS observation remains a single-sample discovery. "
            "The second, cell-resolved dataset is a negative transport result and "
            "motivates assay-aware endpoints rather than a general TLS claim."
        ),
    }


def _plot(sota_rows: list[dict[str, Any]], tls: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    status = np.zeros((len(SOTA_METHODS), len(ALL_UNITS)), dtype=int)
    for row in sota_rows:
        i = SOTA_METHODS.index(row["method"])
        j = ALL_UNITS.index(row["unit_id"])
        if row["completeness"] == "partial_donor_slices":
            status[i, j] = 1
        elif row["completeness"] == "complete_unit_three_seed":
            status[i, j] = 2

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    cmap = ListedColormap(["#ECECEC", "#F1B44C", "#2878B5"])
    ax.imshow(status, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_yticks(range(len(SOTA_METHODS)))
    ax.set_yticklabels([m.replace("_py", "") for m in SOTA_METHODS])
    ax.set_xticks(range(len(ALL_UNITS)))
    ax.set_xticklabels([DISPLAY[u] for u in ALL_UNITS], rotation=42, ha="right")
    ax.set_title("A  SOTA coverage on the shared unit registry")
    for i in range(len(SOTA_METHODS)):
        for j in range(len(ALL_UNITS)):
            ax.text(
                j,
                i,
                {0: "—", 1: "P", 2: "✓"}[status[i, j]],
                ha="center",
                va="center",
                fontsize=8,
            )

    ax = axes[1]
    datasets = tls["datasets"]
    x = np.arange(2)
    moran = [float(row["tls_morans_i"]) for row in datasets]
    contig = [float(row["foci_contiguity"]) for row in datasets]
    width = 0.34
    ax.bar(x - width / 2, moran, width, label="TLS Moran's I", color="#2878B5")
    ax.bar(x + width / 2, contig, width, label="Foci contiguity", color="#E07A5F")
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Breast Visium\ndiscovery", "Lymph-node Xenium\ntransport"], rotation=0)
    ax.set_ylabel("Observed spatial statistic")
    ax.set_ylim(0, 0.82)
    ax.set_title("B  TLS endpoint does not transport")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)

    fig.text(
        0.01,
        0.01,
        "✓ complete three-seed unit; P partial DLPFC donor slices; — not run. "
        "SOTA missingness is not imputed. TLS datasets use the locked B/T co-high endpoint.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "fig_strict_external_panel_v2.svg", bbox_inches="tight")
    fig.savefig(HERE / "fig_strict_external_panel_v2.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    required = (SOURCE, EXT_LONG, DLPFC_SOTA, BANKSY_LYMPH, TLS_BREAST, TLS_LYMPH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required panel inputs: {missing}")

    landscape, source_raw = _load_landscape()
    policy_rows = evaluate_personalisation_policies(
        landscape,
        methods=list(METHODS),
        k_neighbours=K_NEIGHBOURS,
        min_training=MIN_TRAINING,
        confidence_floor=CONFIDENCE_FLOOR,
        proxy_advantage=PROXY_ADVANTAGE,
    )
    summary = summarise_policies(
        policy_rows,
        noninferior_margin=NONINFERIOR_MARGIN,
        min_queries=8,
        n_boot=N_BOOT,
        seed=SEED,
    )
    summary.update(
        {
            "protocol": PROTOCOL,
            "n_registry_units": len(ALL_UNITS),
            "n_domain_eligible_units": len(DOMAIN_UNITS),
            "n_tls_datasets": 2,
            "task_contract": {
                "task": "spatial_domain",
                "metric": "ARI",
                "ground_truth_kind": "spatial_domain",
                "proxy_label_units_excluded": True,
                "sota_missing_cells_imputed": False,
            },
            "locked_parameters": {
                "methods": list(METHODS),
                "k_neighbours": K_NEIGHBOURS,
                "min_training": MIN_TRAINING,
                "confidence_floor": CONFIDENCE_FLOOR,
                "proxy_advantage": PROXY_ADVANTAGE,
                "noninferior_margin": NONINFERIOR_MARGIN,
                "n_boot": N_BOOT,
                "seed": SEED,
            },
            "decision": (
                "Retain the global default: the n=9 common-panel policy is "
                "non-inferior but not superior. SOTA cells remain a coverage "
                "audit until all nine units are protocol-matched."
            ),
        }
    )

    registry = _registry_rows(source_raw)
    sota_rows = _sota_scores()
    sota = _sota_summary(sota_rows)
    tls = _tls_summary()

    _json_write(HERE / "loocv_summary.json", summary)
    policy_dicts = [asdict(row) for row in policy_rows]
    _json_write(HERE / "loocv_rows.json", policy_dicts)
    _write_csv(HERE / "loocv_rows.csv", policy_dicts)
    _write_csv(HERE / "strict_external_units.csv", registry)
    _write_csv(HERE / "sota_coverage.csv", sota_rows)
    _json_write(HERE / "sota_coverage_summary.json", sota)
    _json_write(HERE / "tls_two_dataset_summary.json", tls)
    _plot(sota_rows, tls)

    bank = sota["methods"]["banksy_py"]
    report = f"""# Strict task-stratified external panel v2

## Panel definition

The registry contains **{len(ALL_UNITS)} independent units**. Nine units have
anatomical/pathology spatial-domain ground truth and enter the common-panel
LOOCV; two datasets carry TLS evidence, with the reactive lymph-node Xenium
unit shared between strata. Task-ineligible cells remain explicit and are not
converted into pseudo-ground truth.

## Spatial-domain decision endpoint (n=9)

- Gated-policy mean regret: **{summary['mean_gated_regret']:.4f} ARI**.
- Training-fold global-best mean regret: **{summary['mean_global_best_regret']:.4f} ARI**.
- Non-inferior at margin {NONINFERIOR_MARGIN:.2f}: **{summary['primary_noninferior']}**.
- Superior to global best: **{summary['primary_superior']}**.

The lymph-node pathology unit increases the independent-unit endpoint from
n=8 to n=9. It does not unlock personalisation; the global default remains the
deployment policy.

## SOTA coverage on the same registry

BANKSY-Python has scores for {bank['available_units']}/{bank['total_registry_units']}
registry units and all nine domain units, but its three DLPFC donor cells use
selected slices rather than every donor-member slice. SpaGCN, STAGATE, GraphST,
and BayesSpace remain DLPFC-only. Consequently, **no SOTA method enters the
confirmatory n=9 LOOCV**, and no missing cell is imputed. The coverage matrix is
an audit and a precise execution backlog, not a completed SOTA comparison.

## TLS second dataset

The breast Visium TLS signal (Moran's I {tls['datasets'][0]['tls_morans_i']:.3f},
contiguity {tls['datasets'][0]['foci_contiguity']:.3f}) did not transport to the
cell-resolved reactive lymph-node Xenium dataset (Moran's I
{tls['datasets'][1]['tls_morans_i']:.3f}, contiguity
{tls['datasets'][1]['foci_contiguity']:.3f}; pathology-GC F1
{tls['datasets'][1]['pathology_gc_f1']:.3f}). This is a negative external result.
It motivates assay-aware neighbourhood endpoints and preserves the discovery
claim as single-sample rather than general TLS validation.

![Strict panel v2](fig_strict_external_panel_v2.png)

## Reproduction

```bash
python research/phaseB_tls_consensus/analyze_tls_second_dataset.py
python benchmark_external_validation/evaluate_banksy_lymph.py
python benchmark_external_validation/strict_external_panel_v2/build_strict_external_panel_v2.py
```
"""
    (HERE / "REPORT_strict_external_panel_v2.md").write_text(report, encoding="utf-8")

    artifacts = [
        HERE / "loocv_summary.json",
        HERE / "loocv_rows.json",
        HERE / "loocv_rows.csv",
        HERE / "strict_external_units.csv",
        HERE / "sota_coverage.csv",
        HERE / "sota_coverage_summary.json",
        HERE / "tls_two_dataset_summary.json",
        HERE / "fig_strict_external_panel_v2.svg",
        HERE / "fig_strict_external_panel_v2.png",
        HERE / "REPORT_strict_external_panel_v2.md",
    ]
    manifest = {
        "protocol": PROTOCOL,
        "builder": Path(__file__).name,
        "inputs": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in required
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    _json_write(HERE / "manifest.json", manifest)
    logging.getLogger(__name__).info(
        "strict panel v2: registry=%d domain_n=%d tls_n=%d",
        len(ALL_UNITS),
        len(policy_rows),
        tls["n_independent_datasets"],
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
