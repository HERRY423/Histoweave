"""Rebuild the HistoWeave submission-freeze v1 artifacts.

The script is intentionally conservative: it regenerates the five external
validation figures and rebuilds the manuscript-facing summary files from
tracked benchmark artifacts. It does not rerun external SOTA methods whose
environments live outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "submission_freeze_v1"
EXT = ROOT / "benchmark_external_validation"
FIG = EXT / "figures"
N8 = EXT / "n8_strict_region"
STRICT_V2 = EXT / "strict_external_panel_v2"
INDEPENDENT = EXT / "independent_test_wu2021"
PROTOCOL = ROOT / "protocol_endpoints_results"
SOTA = ROOT / "5x15_spatial_aware"


MAIN_FIGURES = [
    {
        "figure": "Figure 1",
        "slug": "fig1_performance_heatmap",
        "title": "External spatial-domain performance heatmap",
        "source_data": [
            "benchmark_external_validation/performance_matrix_mean.csv",
            "benchmark_external_validation/performance_matrix_std.csv",
        ],
        "generator": "benchmark_external_validation/make_figures.py",
        "caption_short": (
            "Mean ARI across five external spatial-domain datasets and the shared method panel."
        ),
    },
    {
        "figure": "Figure 2",
        "slug": "fig2_method_boxplot",
        "title": "External ARI distribution by method",
        "source_data": ["benchmark_external_validation/benchmark_long.csv"],
        "generator": "benchmark_external_validation/make_figures.py",
        "caption_short": "Per-method ARI variation across datasets and random seeds.",
    },
    {
        "figure": "Figure 3",
        "slug": "fig3_landscape_embedding",
        "title": "Dataset-feature landscape embedding",
        "source_data": [
            "benchmark_external_validation/dataset_manifest.json",
            "benchmark_external_validation/performance_matrix_mean.csv",
        ],
        "generator": "benchmark_external_validation/make_figures.py",
        "caption_short": (
            "Target-free dataset features reveal heterogeneous external benchmark regimes."
        ),
    },
    {
        "figure": "Figure 4",
        "slug": "fig4_recommender_regret",
        "title": "Recommender regret against baselines",
        "source_data": ["benchmark_external_validation/recommendation_loocv.json"],
        "generator": "benchmark_external_validation/make_figures.py",
        "caption_short": (
            "LOOCV selection regret matches the training-fold global-best baseline "
            "and improves over random choice."
        ),
    },
    {
        "figure": "Figure 5",
        "slug": "selective_regret_coverage",
        "title": "Selective regret-coverage",
        "source_data": ["protocol_endpoints_results/selective_regret_coverage.json"],
        "generator": "benchmark_external_validation/make_figures.py",
        "caption_short": (
            "Abstention prevents higher-regret personalisation; the global default "
            "has lower regret than always personalising."
        ),
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _regenerate_external_figures() -> None:
    subprocess.run(
        [sys.executable, str(EXT / "make_figures.py")],
        cwd=ROOT,
        check=True,
    )


def _regenerate_strict_panel() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "research" / "phaseB_tls_consensus" / "analyze_tls_second_dataset.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(EXT / "evaluate_banksy_lymph.py")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(STRICT_V2 / "build_strict_external_panel_v2.py")],
        cwd=ROOT,
        check=True,
    )


def _regenerate_independent_test() -> None:
    subprocess.run(
        [sys.executable, str(INDEPENDENT / "run_independent_test.py")],
        cwd=ROOT,
        check=True,
    )


def _figure_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fig in MAIN_FIGURES:
        formats: dict[str, dict[str, Any]] = {}
        for suffix in ("svg", "png"):
            path = FIG / f"{fig['slug']}.{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
            formats[suffix] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        records.append({**fig, "formats": formats})
    return records


def _best_sota_row() -> dict[str, str]:
    with (SOTA / "sota_method_means.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return max(rows, key=lambda row: float(row["grand_mean"]))


def _count_full_matrix_methods() -> int:
    with (SOTA / "performance_matrix_mean_full.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return len(header) - 1


def _sota_success_counts() -> tuple[int, int]:
    with (SOTA / "sota_method_means.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        sum(int(row["n_success"]) for row in rows),
        sum(int(row["n_total"]) for row in rows),
    )


def _write_supplement_table() -> list[dict[str, str]]:
    loocv5 = _read_json(EXT / "recommendation_loocv.json")["summary"]
    strict_v2 = _read_json(STRICT_V2 / "loocv_summary.json")
    independent = _read_json(INDEPENDENT / "independent_test_summary.json")
    selective = _read_json(PROTOCOL / "selective_regret_coverage.json")
    curve = selective["curve"]
    first = curve[0]
    best_sota = _best_sota_row()
    sota_success, sota_total = _sota_success_counts()

    federation_tests = sorted(ROOT.glob("tests/test_federation_*.py"))

    rows = [
        {
            "endpoint": "External LOOCV recommendation",
            "design": "5 held-out external datasets; spatial-domain task",
            "n": str(loocv5["n_queries"]),
            "primary_metric": "mean selection regret",
            "observed_value": f"{loocv5['mean_selection_regret']:.6f}",
            "comparator": "training-fold global-best",
            "comparator_value": f"{loocv5['global_best_mean_regret']:.6f}",
            "decision": "ties global default; no superiority claim",
            "source": "benchmark_external_validation/recommendation_loocv.json",
        },
        {
            "endpoint": "Strict task-stratified external panel v2",
            "design": "10-unit registry; 9 domain LOOCV units; 2 TLS datasets",
            "n": str(strict_v2["n_queries"]),
            "primary_metric": "mean gated-policy regret",
            "observed_value": f"{strict_v2['mean_gated_regret']:.6f}",
            "comparator": "training-fold global-best",
            "comparator_value": f"{strict_v2['mean_global_best_regret']:.6f}",
            "decision": (
                "non-inferior, not superior; TLS transport not replicated; "
                "BANKSY 9/10 registry units available"
            ),
            "source": (
                "benchmark_external_validation/strict_external_panel_v2/"
                "REPORT_strict_external_panel_v2.md"
            ),
        },
        {
            "endpoint": "Frozen independent study test",
            "design": (
                "Wu 2021; six unseen breast-cancer patients/sections; one-shot preregistration"
            ),
            "n": str(independent["n_evaluable_sections"]),
            "primary_metric": "mean frozen-policy regret",
            "observed_value": f"{independent['mean_frozen_policy_regret']:.6f}",
            "comparator": "preregistered success margin",
            "comparator_value": f"{independent['success_margin_ari']:.6f}",
            "decision": ("external validation failed; test cohort remains excluded from training"),
            "source": (
                "benchmark_external_validation/independent_test_wu2021/"
                "REPORT_independent_test_wu2021.md"
            ),
        },
        {
            "endpoint": "Selective regret-coverage",
            "design": "20 study-grouped queries across confidence thresholds",
            "n": str(selective["n_queries"]),
            "primary_metric": "always-personalised regret",
            "observed_value": f"{first['mean_regret_always_personalised']:.6f}",
            "comparator": "always-global regret",
            "comparator_value": f"{first['mean_regret_always_global']:.6f}",
            "decision": "full abstention/global default selected",
            "source": "protocol_endpoints_results/selective_regret_coverage.json",
        },
        {
            "endpoint": "DLPFC SOTA benchmark",
            "design": "5 DLPFC slices x 3 seeds; unified real-data harness",
            "n": (
                f"{_count_full_matrix_methods()} methods; "
                f"{sota_success}/{sota_total} new SOTA runs successful"
            ),
            "primary_metric": "top method mean ARI",
            "observed_value": f"{best_sota['method']}={float(best_sota['grand_mean']):.4f}",
            "comparator": "sklearn and spatial-aware method panel",
            "comparator_value": "see performance_matrix_mean_full.csv",
            "decision": "SOTA included; no single method universally wins",
            "source": "5x15_spatial_aware/report_sota_5x20.md",
        },
        {
            "endpoint": "Federated evidence network",
            "design": "signed scalar evidence bundles; privacy gate; consensus tests",
            "n": f"{len(federation_tests)} federation test files",
            "primary_metric": "contract test status",
            "observed_value": "covered by targeted pytest suite",
            "comparator": "no federation files present",
            "comparator_value": "additive fallback remains supported",
            "decision": "reference implementation; supplementary infrastructure",
            "source": "federation/PROTOCOL.md",
        },
    ]

    out = FREEZE_DIR / "supplement_benchmark_table.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_figure_lock(figure_records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": "histoweave.submission_freeze.figures.v1",
        "freeze_version": "v1",
        "target_journal": "Bioinformatics",
        "n_main_figures": len(figure_records),
        "figures": figure_records,
    }
    (FREEZE_DIR / "main_figures.lock.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(figure_records: list[dict[str, Any]], table_rows: list[dict[str, str]]) -> None:
    independent = _read_json(INDEPENDENT / "independent_test_summary.json")
    independent_paths = [
        "benchmark_external_validation/independent_test_wu2021/preregistered_protocol.json",
        "benchmark_external_validation/independent_test_wu2021/independence_audit.json",
        "benchmark_external_validation/independent_test_wu2021/independent_test_summary.json",
        "benchmark_external_validation/independent_test_wu2021/sample_regret.csv",
        "benchmark_external_validation/independent_test_wu2021/fig_independent_test_wu2021.svg",
        "benchmark_external_validation/independent_test_wu2021/fig_independent_test_wu2021.png",
        "benchmark_external_validation/independent_test_wu2021/REPORT_independent_test_wu2021.md",
    ]
    tracked_outputs = [
        "submission_freeze_v1/README.md",
        "submission_freeze_v1/DATA_CODE_AVAILABILITY.md",
        "submission_freeze_v1/reproduce_submission_freeze.py",
        "submission_freeze_v1/main_figures.lock.json",
        "submission_freeze_v1/supplement_benchmark_table.csv",
    ]
    manifest = {
        "schema_version": "histoweave.submission_freeze.v1",
        "freeze_version": "v1",
        "target_journal": "Bioinformatics",
        "target_article_type": "Original Paper or Application Note",
        "claim_boundary": (
            "HistoWeave supports evidence-governed method decisions, fallback, "
            "and abstention. The frozen Wu 2021 test failed, so current evidence "
            "does not support transport of the global spectral policy or "
            "personalised recommendation superiority."
        ),
        "main_figures": [record["figure"] for record in figure_records],
        "supplement_table": {
            "path": "submission_freeze_v1/supplement_benchmark_table.csv",
            "rows": len(table_rows),
        },
        "independent_test": {
            "policy": independent["frozen_policy"],
            "n_evaluable_sections": independent["n_evaluable_sections"],
            "mean_regret": independent["mean_frozen_policy_regret"],
            "decision": independent["decision"],
            "training_exclusion_locked": True,
            "artifacts": [
                {
                    "path": path,
                    "sha256": _sha256(ROOT / path),
                    "bytes": (ROOT / path).stat().st_size,
                }
                for path in independent_paths
            ],
        },
        "locked_outputs": [
            {
                "path": path,
                "sha256": _sha256(ROOT / path),
                "bytes": (ROOT / path).stat().st_size,
            }
            for path in tracked_outputs
            if (ROOT / path).exists()
        ],
        "primary_source_reports": [
            "benchmark_external_validation/report_external_validation.md",
            "benchmark_external_validation/n8_strict_region/report_n8_strict_loocv.md",
            "benchmark_external_validation/strict_external_panel_v2/REPORT_strict_external_panel_v2.md",
            "benchmark_external_validation/independent_test_wu2021/REPORT_independent_test_wu2021.md",
            "research/phaseB_tls_consensus/second_dataset_xenium_lymph/REPORT_tls_second_dataset.md",
            "5x15_spatial_aware/report_sota_5x20.md",
            "protocol_endpoints_results/protocol_endpoints_report.md",
            "federation/PROTOCOL.md",
        ],
    }
    (FREEZE_DIR / "submission_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-figures",
        action="store_true",
        help=(
            "Rerun benchmark_external_validation/make_figures.py before hashing. "
            "Requires prepared dataset caches such as datasets_cache/*.h5ad."
        ),
    )
    parser.add_argument(
        "--regenerate-strict-panel",
        action="store_true",
        help=(
            "Rerun the second TLS dataset, aligned lymph-node BANKSY cell, and "
            "strict external panel v2. Requires the official local Xenium bundle."
        ),
    )
    parser.add_argument(
        "--regenerate-independent-test",
        action="store_true",
        help=(
            "Rerun the preregistered Wu 2021 six-patient independent test. "
            "Requires the official local Zenodo raw bundle."
        ),
    )
    args = parser.parse_args()

    FREEZE_DIR.mkdir(exist_ok=True)
    if args.regenerate_figures:
        _regenerate_external_figures()
    if args.regenerate_strict_panel:
        _regenerate_strict_panel()
    if args.regenerate_independent_test:
        _regenerate_independent_test()
    figure_records = _figure_records()
    table_rows = _write_supplement_table()
    _write_figure_lock(figure_records)
    _write_manifest(figure_records, table_rows)
    logging.getLogger(__name__).info(
        "wrote %s with %d figures",
        FREEZE_DIR.relative_to(ROOT),
        len(figure_records),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
