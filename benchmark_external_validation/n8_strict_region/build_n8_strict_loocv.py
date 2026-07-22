"""Build the strict n=8 independent-unit LOOCV and tissue-condition flip report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from histoweave.benchmark.independent_personalisation import (
    evaluate_personalisation_policies,
    summarise_policies,
)
from histoweave.benchmark.landscape import LandscapeResult

PROTOCOL = "histoweave.external_validation.recommender_loocv.n8_strict_region.v1"
SEED = 20260721
N_BOOT = 10000
NONINFERIOR_MARGIN = 0.02
K_NEIGHBOURS = 3
MIN_TRAINING = 4
CONFIDENCE_FLOOR = 0.20
PROXY_ADVANTAGE = 0.02

STRICT_UNITS = (
    "dlpfc_donor_Br5292",
    "dlpfc_donor_Br5595",
    "dlpfc_donor_Br8100",
    "visium_hd_crc",
    "xenium_lung_cancer",
    "xenium_ovarian_cancer",
    "visium_mouse_brain",
    "allen_merfish_brain_section",
)
METHODS = (
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "gaussian_mixture",
    "kmeans",
    "minibatch_kmeans",
    "spectral",
)
CONDITIONS = {
    "dlpfc_donor_Br5292": "human_cerebral_cortex",
    "dlpfc_donor_Br5595": "human_cerebral_cortex",
    "dlpfc_donor_Br8100": "human_cerebral_cortex",
    "visium_hd_crc": "human_tumor",
    "xenium_lung_cancer": "human_tumor",
    "xenium_ovarian_cancer": "human_tumor",
    "visium_mouse_brain": "mouse_brain",
    "allen_merfish_brain_section": "mouse_brain",
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
}
CONDITION_DISPLAY = {
    "human_cerebral_cortex": "Human cerebral cortex",
    "human_tumor": "Human tumour",
    "mouse_brain": "Mouse brain",
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "independent_personalisation_results" / "independent_unit_landscape.json"
N5_SOURCE = ROOT / "benchmark_external_validation" / "recommendation_loocv.json"


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_landscape() -> tuple[LandscapeResult, dict[str, Any]]:
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    missing = [unit for unit in STRICT_UNITS if unit not in raw["performance"]]
    if missing:
        raise RuntimeError(f"Strict units missing from source landscape: {missing}")

    meta = raw.get("dataset_meta", {})
    bad_gt = [
        unit
        for unit in STRICT_UNITS
        if meta.get(unit, {}).get("ground_truth_kind") != "spatial_domain"
    ]
    if bad_gt:
        raise RuntimeError(f"Non-spatial-domain ground truth in strict panel: {bad_gt}")

    performance = {
        unit: {method: float(raw["performance"][unit][method]) for method in METHODS}
        for unit in STRICT_UNITS
    }
    features = {unit: np.asarray(raw["features"][unit], dtype=float) for unit in STRICT_UNITS}
    timings = {
        unit: {method: raw.get("timings", {}).get(unit, {}).get(method) for method in METHODS}
        for unit in STRICT_UNITS
    }
    best = {
        unit: max(performance[unit], key=lambda method: performance[unit][method])
        for unit in STRICT_UNITS
    }
    niches = {method: [unit for unit in STRICT_UNITS if best[unit] == method] for method in METHODS}
    landscape = LandscapeResult(
        performance=performance,
        features=features,
        embedding={},
        best_method=best,
        niches=niches,
        timings=timings,
        feature_order=list(raw["feature_order"]),
        method_count=len(METHODS),
        dataset_count=len(STRICT_UNITS),
        task="spatial_domain",
        metric="ARI",
        higher_is_better=True,
        dataset_meta={unit: dict(meta[unit]) for unit in STRICT_UNITS},
    )
    return landscape, raw


def _condition_loocv(landscape: LandscapeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held in STRICT_UNITS:
        condition = CONDITIONS[held]
        training = [unit for unit in STRICT_UNITS if unit != held and CONDITIONS[unit] == condition]
        means = {
            method: float(np.mean([landscape.performance[unit][method] for unit in training]))
            for method in METHODS
        }
        selected = max(METHODS, key=lambda method: (means[method], method))
        held_scores = landscape.performance[held]
        oracle_method = max(METHODS, key=lambda method: (held_scores[method], method))
        oracle_score = held_scores[oracle_method]
        rows.append(
            {
                "held_out": held,
                "condition": condition,
                "n_training_same_condition": len(training),
                "selected_method": selected,
                "oracle_method": oracle_method,
                "oracle_score": oracle_score,
                "selected_score": held_scores[selected],
                "regret": oracle_score - held_scores[selected],
            }
        )
    return rows


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    denominator = math.comb(total, row1)

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / denominator

    lo = max(0, row1 - (total - col1))
    hi = min(row1, col1)
    observed = probability(a)
    return min(
        1.0,
        sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + 1e-15),
    )


def _build_flip_rows(landscape: LandscapeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in STRICT_UNITS:
        scores = landscape.performance[unit]
        oracle = max(METHODS, key=lambda method: (scores[method], method))
        meta = landscape.dataset_meta[unit]
        finite_features = int(np.isfinite(landscape.features[unit]).sum())
        rows.append(
            {
                "unit_id": unit,
                "display_name": DISPLAY[unit],
                "condition": CONDITIONS[unit],
                "independence_class": meta.get("independence_class"),
                "platform": meta.get("platform"),
                "oracle_method": oracle,
                "oracle_ari": scores[oracle],
                "spectral_ari": scores["spectral"],
                "gaussian_mixture_ari": scores["gaussian_mixture"],
                "oracle_flips_from_global_spectral": oracle != "spectral",
                "n_finite_target_free_features": finite_features,
                "n_target_free_features": len(landscape.feature_order),
            }
        )
    return rows


def _condition_summary(flip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flip_rows:
        grouped[row["condition"]].append(row)
    result = []
    for condition in CONDITIONS.values():
        if any(item["condition"] == condition for item in result):
            continue
        rows = grouped[condition]
        wins = Counter(row["oracle_method"] for row in rows)
        result.append(
            {
                "condition": condition,
                "label": CONDITION_DISPLAY[condition],
                "n_independent_units": len(rows),
                "oracle_wins": dict(sorted(wins.items())),
                "spectral_win_fraction": wins["spectral"] / len(rows),
                "gaussian_mixture_win_fraction": wins["gaussian_mixture"] / len(rows),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    landscape: LandscapeResult,
    policy_rows: list[Any],
    condition_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
        }
    )
    colors = {
        "spectral": "#0072B2",
        "gaussian_mixture": "#D55E00",
        "global": "#0072B2",
        "knn": "#000000",
        "condition": "#E69F00",
    }
    x = np.arange(len(STRICT_UNITS), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

    ax = axes[0]
    for i, unit in enumerate(STRICT_UNITS):
        spectral = landscape.performance[unit]["spectral"]
        gmm = landscape.performance[unit]["gaussian_mixture"]
        ax.plot([i, i], [spectral, gmm], color="#B7B7B7", linewidth=0.8, zorder=1)
    ax.scatter(
        x,
        [landscape.performance[u]["spectral"] for u in STRICT_UNITS],
        color=colors["spectral"],
        marker="s",
        s=28,
        label="Spectral",
        zorder=3,
    )
    ax.scatter(
        x,
        [landscape.performance[u]["gaussian_mixture"] for u in STRICT_UNITS],
        facecolors="white",
        edgecolors=colors["gaussian_mixture"],
        marker="o",
        linewidths=1.2,
        s=32,
        label="Gaussian mixture",
        zorder=3,
    )
    for i, unit in enumerate(STRICT_UNITS):
        winner = max(METHODS, key=lambda method: landscape.performance[unit][method])
        ax.text(
            i,
            max(
                landscape.performance[unit]["spectral"],
                landscape.performance[unit]["gaussian_mixture"],
            )
            + 0.025,
            "GMM" if winner == "gaussian_mixture" else "SP",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )
    ax.set_ylabel("ARI")
    ax.set_title("A  Oracle winner flips in human cerebral cortex (exploratory)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_ylim(0.1, 0.76)

    ax = axes[1]
    policy_map = {row.held_out: row for row in policy_rows}
    condition_map = {row["held_out"]: row for row in condition_rows}
    ax.scatter(
        x - 0.08,
        [policy_map[u].global_best_regret for u in STRICT_UNITS],
        color=colors["global"],
        marker="s",
        s=27,
        label="Global-best LOOCV",
    )
    ax.scatter(
        x,
        [policy_map[u].knn_regret for u in STRICT_UNITS],
        color=colors["knn"],
        marker="x",
        s=32,
        linewidths=1.2,
        label="k-NN LOOCV",
    )
    ax.scatter(
        x + 0.08,
        [condition_map[u]["regret"] for u in STRICT_UNITS],
        facecolors="white",
        edgecolors=colors["condition"],
        marker="^",
        s=35,
        linewidths=1.2,
        label="Within-condition LOOCV",
    )
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.set_ylabel("Selection regret (ARI)")
    ax.set_title("B  n=8 independent-unit LOOCV: k-NN ties global; naive tissue rule worsens")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY[unit] + "\n" + CONDITION_DISPLAY[CONDITIONS[unit]] for unit in STRICT_UNITS],
        rotation=28,
        ha="right",
    )

    fig.text(
        0.01,
        0.01,
        "Independent unit = donor or external study; no cell-level resampling. "
        "SP = spectral, GMM = Gaussian mixture. Tissue grouping is exploratory.",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "fig_tissue_condition_flip.svg", bbox_inches="tight")
    fig.savefig(HERE / "fig_tissue_condition_flip.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    landscape, source_raw = _load_landscape()
    policy_rows = evaluate_personalisation_policies(
        landscape,
        methods=list(METHODS),
        k_neighbours=K_NEIGHBOURS,
        min_training=MIN_TRAINING,
        confidence_floor=CONFIDENCE_FLOOR,
        proxy_advantage=PROXY_ADVANTAGE,
    )
    if len(policy_rows) != len(STRICT_UNITS):
        raise RuntimeError(f"Expected 8 LOOCV rows, found {len(policy_rows)}")

    summary = summarise_policies(
        policy_rows,
        noninferior_margin=NONINFERIOR_MARGIN,
        min_queries=8,
        n_boot=N_BOOT,
        seed=SEED,
    )
    summary["protocol"] = PROTOCOL
    summary["confirmatory_endpoint"] = (
        "mean gated-policy selection regret versus leave-one-independent-unit-out "
        "training-set global-best regret"
    )
    summary["independent_unit_definition"] = "one biological donor or one external study"
    summary["task_contract"] = {
        "task": "spatial_domain",
        "metric": "ARI",
        "ground_truth_kind": "spatial_domain",
        "proxy_label_units_excluded": True,
    }
    summary["locked_parameters"] = {
        "methods": list(METHODS),
        "k_neighbours": K_NEIGHBOURS,
        "min_training": MIN_TRAINING,
        "confidence_floor": CONFIDENCE_FLOOR,
        "proxy_advantage": PROXY_ADVANTAGE,
        "noninferior_margin": NONINFERIOR_MARGIN,
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    summary["top1_accuracy"] = float(
        np.mean([(row.knn_regret or 0.0) <= 1e-12 for row in policy_rows])
    )
    summary["global_best_top1_accuracy"] = float(
        np.mean([(row.global_best_regret or 0.0) <= 1e-12 for row in policy_rows])
    )
    summary["regret_reduction_vs_random"] = (
        1.0 - summary["mean_gated_regret"] / summary["mean_random_regret"]
    )
    summary["regret_reduction_vs_global_best"] = (
        summary["mean_global_best_regret"] - summary["mean_gated_regret"]
    )
    if N5_SOURCE.exists():
        n5 = json.loads(N5_SOURCE.read_text(encoding="utf-8"))
        n5_summary = n5.get("summary", {})
        summary["n5_reference"] = {
            "path": N5_SOURCE.relative_to(ROOT).as_posix(),
            "n_queries": n5_summary.get("n_queries"),
            "mean_selection_regret": n5_summary.get("mean_selection_regret"),
            "mean_global_best_regret": n5_summary.get("global_best_mean_regret"),
        }

    flip_rows = _build_flip_rows(landscape)
    condition_summary = _condition_summary(flip_rows)
    condition_rows = _condition_loocv(landscape)
    condition_mean_regret = float(np.mean([row["regret"] for row in condition_rows]))
    cortex_flips = sum(
        row["oracle_flips_from_global_spectral"]
        for row in flip_rows
        if row["condition"] == "human_cerebral_cortex"
    )
    other_flips = sum(
        row["oracle_flips_from_global_spectral"]
        for row in flip_rows
        if row["condition"] != "human_cerebral_cortex"
    )
    cortex_n = sum(row["condition"] == "human_cerebral_cortex" for row in flip_rows)
    other_n = len(flip_rows) - cortex_n
    tissue = {
        "protocol": "histoweave.tissue_condition_flip.exploratory.v1",
        "status": "exploratory_not_deployment_ready",
        "global_training_default": "spectral",
        "condition_summary": condition_summary,
        "flip_table": {
            "human_cerebral_cortex": {
                "flips": cortex_flips,
                "non_flips": cortex_n - cortex_flips,
            },
            "other_conditions": {
                "flips": other_flips,
                "non_flips": other_n - other_flips,
            },
        },
        "fisher_exact_two_sided_p": _fisher_two_sided(
            cortex_flips,
            cortex_n - cortex_flips,
            other_flips,
            other_n - other_flips,
        ),
        "global_loocv_mean_regret": summary["mean_global_best_regret"],
        "within_condition_loocv_mean_regret": condition_mean_regret,
        "within_condition_minus_global_regret": (
            condition_mean_regret - summary["mean_global_best_regret"]
        ),
        "interpretation": (
            "The oracle winner changes in two of three human cerebral-cortex donors, "
            "whereas spectral wins all tumour and mouse-brain units. This is a "
            "small-n tissue interaction signal, not a validated selection rule."
        ),
        "negative_control": (
            "A naive leave-one-out within-condition selector increases mean regret, "
            "so tissue labels must not unlock personalised deployment."
        ),
    }

    decision_validation = {
        "schema_version": 2,
        "protocol": "dataset_grouped_holdout",
        "source_protocol": PROTOCOL,
        "task": "spatial_domain",
        "ground_truth_kind": "spatial_domain",
        "n_queries": len(policy_rows),
        "independent_unit": "biological_donor_or_external_study",
        "beats_global_best": False,
        "noninferior_to_global_best": bool(summary["primary_noninferior"]),
        "superior_to_global_best": bool(summary["primary_superior"]),
        "metrics": {
            "mean_gated_regret": summary["mean_gated_regret"],
            "mean_global_best_regret": summary["mean_global_best_regret"],
            "mean_random_regret": summary["mean_random_regret"],
            "gated_minus_global": summary["gated_minus_global_point"],
            "top1_accuracy": summary["top1_accuracy"],
        },
        "decision": (
            "Retain the global spectral default. The n=8 policy is non-inferior "
            "but not superior and therefore does not unlock personalisation."
        ),
        "limitations": [
            "Only eight independent units are available; tissue strata contain 2-3 units.",
            "The three DLPFC units are donor aggregates from one study.",
            "Five external-study feature vectors have partial target-free feature coverage.",
            "The common panel is restricted to seven sklearn methods available on all units.",
        ],
    }

    policy_dicts = [asdict(row) for row in policy_rows]
    _json_write(HERE / "loocv_summary.json", summary)
    _json_write(HERE / "loocv_rows.json", policy_dicts)
    _write_csv(HERE / "loocv_rows.csv", policy_dicts)
    _json_write(HERE / "tissue_condition_flip.json", tissue)
    _write_csv(HERE / "tissue_condition_flip.csv", flip_rows)
    _write_csv(HERE / "within_condition_loocv.csv", condition_rows)
    _json_write(HERE / "decision_validation.json", decision_validation)
    _plot(landscape, policy_rows, condition_rows)

    report = f"""# Strict spatial-domain external validation: n=8

## Confirmatory result

The leave-one-independent-unit-out panel contains **8 units**: three DLPFC
biological donors and five external studies. Every unit uses anatomical,
pathology, or manually curated spatial-region ground truth; cell-type and
Leiden proxy labels are excluded.

- Gated/k-NN mean selection regret: **{summary["mean_gated_regret"]:.4f} ARI**
- Training-fold global-best mean regret: **{summary["mean_global_best_regret"]:.4f} ARI**
- Paired difference: **{summary["gated_minus_global_point"]["mean_delta"]:.4f}**
  (95% bootstrap CI [{summary["gated_minus_global_point"]["ci_low"]:.4f},
  {summary["gated_minus_global_point"]["ci_high"]:.4f}])
- Non-inferior at the predeclared {NONINFERIOR_MARGIN:.2f} margin:
  **{summary["primary_noninferior"]}**
- Superior to the global best: **{summary["primary_superior"]}**
- Regret reduction versus a uniform random method: **{summary["regret_reduction_vs_random"]:.1%}**

This extends the earlier n=5 tie to n=8, but it does not demonstrate incremental
value over the global spectral default. The deployment decision therefore
remains global-default.

## Exploratory tissue-condition flip

The oracle winner is Gaussian mixture in 2/3 human cerebral-cortex donors and
spectral in 1/3. Spectral wins 3/3 human tumour studies and 2/2 mouse-brain
studies. Fisher exact p={tissue["fisher_exact_two_sided_p"]:.3f}; the strata are
too small for a confirmatory tissue interaction claim.

The negative control is decisive: a naive same-condition LOOCV rule has mean
regret **{condition_mean_regret:.4f}**, worse than the global rule
(**{summary["mean_global_best_regret"]:.4f}**). The condition flip is therefore
a hypothesis for mechanism and future sampling, not a production selector.

![Tissue-condition flip](fig_tissue_condition_flip.png)

## Scope and limitations

- Independent n counts donors or external studies, never cells or repeated slices.
- The three DLPFC donors come from one study and are not three laboratories.
- External-study feature vectors are partially observed; see the per-unit CSV.
- The strict common method panel has seven sklearn methods. SOTA method outputs
  remain in the broader benchmark but do not yet have complete n=8 coverage.
- All inferential claims use the strict spatial-domain task contract. Proxy-label
  imaging and cell-type datasets are excluded.
"""
    (HERE / "report_n8_strict_loocv.md").write_text(report, encoding="utf-8")

    artifacts = [
        HERE / "loocv_summary.json",
        HERE / "loocv_rows.json",
        HERE / "loocv_rows.csv",
        HERE / "tissue_condition_flip.json",
        HERE / "tissue_condition_flip.csv",
        HERE / "within_condition_loocv.csv",
        HERE / "decision_validation.json",
        HERE / "fig_tissue_condition_flip.svg",
        HERE / "fig_tissue_condition_flip.png",
        HERE / "report_n8_strict_loocv.md",
    ]
    manifest = {
        "protocol": PROTOCOL,
        "builder": str(Path(__file__).name),
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": _sha256(SOURCE),
            "source_protocol": source_raw.get("protocol"),
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in artifacts
        },
    }
    _json_write(HERE / "manifest.json", manifest)


if __name__ == "__main__":
    main()
