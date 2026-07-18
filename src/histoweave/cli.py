"""HistoWeave command-line interface.

Everything scriptable is reproducible; nothing essential is GUI-only. The CLI is a thin
driver over the same SDK functions, so a run from the terminal and a run from a notebook
are identical.

Commands
--------
    histoweave version
    histoweave list-methods [--category qc] [--assay xenium] [--json]
    histoweave run --demo [--out report.html] [--manifest manifest.json]
    histoweave benchmark [--task domain_detection] [--json]
    histoweave recommend --in data.ttab --knowledge-base landscape.json
                      [--k-neighbours 3] [--top 3] [--json]
    histoweave ask "Find spatial domains" --in data.ttab [--model mock] [--yes]
    histoweave digital-twin --in data.ttab [--out-dir DIR] [--methods kmeans,spectral]
    histoweave automl "Find spatial domains for my Visium data"
                   --in data.ttab --knowledge-base landscape.json [--out-dir DIR]
    histoweave failure-fingerprint [--methods kmeans,spectral] [--out-dir DIR]
    histoweave calibrate-recommender --in data.ttab --knowledge-base landscape.json
    histoweave sota [--dry-run] [--out-dir DIR]
    histoweave stats-review --landscape landscape.json [--out stats.json]
    histoweave discovery (run|cohort|bootstrap-ci|panel|if-package|if-analyze) [options]


The next three drive the per-stage Nextflow pipeline: each reads and writes a portable
bundle directory (see ``histoweave.io.bundle``) so stages can run as isolated processes.

    histoweave ingest (--demo | --input DIR --assay ASSAY) --out data.ttab
    histoweave step CATEGORY --method NAME --in IN.ttab --out OUT.ttab [--param k=v ...]
    histoweave report --in data.ttab --out report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from .logging import configure_logging, get_logger, log_event

_LOGGER = get_logger("histoweave.cli")


def _emit(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Emit stable CLI text and record the output as a structured event."""
    message = sep.join(str(value) for value in values)
    stream = file or sys.stdout
    stream.write(message + end)
    if flush:
        stream.flush()
    channel = "stderr" if stream is sys.stderr else "stdout"
    log_event(_LOGGER, logging.DEBUG, "cli.output", message, channel=channel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="histoweave", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log level for operational diagnostics (stderr). Default: WARNING.",
    )
    parser.add_argument(
        "--log-format",
        default="text",
        choices=("text", "json"),
        help="Log format: 'text' for human-readable or 'json' for structured. Default: text.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print the HistoWeave version.")

    p_list = sub.add_parser("list-methods", help="List registered analysis methods.")
    p_list.add_argument("--category", help="Filter by method category (e.g. qc, domain_detection).")
    p_list.add_argument("--assay", help="Filter by assay applicability (e.g. xenium).")
    p_list.add_argument("--json", action="store_true", help="Emit JSON.")
    p_list.add_argument(
        "--all-versions",
        action="store_true",
        help="Include deprecated and superseded method releases.",
    )

    p_run = sub.add_parser("run", help="Run the default pipeline and write a report.")
    p_run.add_argument("--demo", action="store_true", help="Use the synthetic demo dataset.")
    p_run.add_argument("--input", help="Vendor data path (needs --assay and [spatial] extra).")
    p_run.add_argument("--assay", help="Assay for --input, e.g. visium/xenium/stereo_seq.")
    p_run.add_argument("--out", default="histoweave_report.html", help="Output HTML report.")
    p_run.add_argument("--manifest", help="Optional path to write the run manifest JSON.")
    p_run.add_argument("--seed", type=int, default=0, help="Seed for the demo dataset.")
    p_run.add_argument(
        "--n-domains",
        type=int,
        help="Domain count for methods that require it (required for real data with kmeans).",
    )
    p_run.add_argument(
        "--annotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable marker annotation (auto: enabled only when marker genes exist).",
    )

    p_bench = sub.add_parser("benchmark", help="Run a benchmark task and print a leaderboard.")
    p_bench.add_argument("--task", default="domain_detection", help="Benchmark task name.")
    p_bench.add_argument(
        "--suite",
        choices=("figure3", "phenomenology"),
        help="Run a predefined multi-dataset or spatial-phenomenology benchmark suite.",
    )
    p_bench.add_argument(
        "--out-dir",
        default=None,
        help="Artifact directory (defaults to a suite-specific directory).",
    )
    p_bench.add_argument(
        "--phenomena",
        default="all",
        help="Comma-separated phenomenon primitives for --suite phenomenology.",
    )
    p_bench.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated observation conditions for --suite phenomenology.",
    )
    p_bench.add_argument(
        "--methods",
        default="all-release",
        help="Comma-separated frozen release methods for --suite phenomenology.",
    )
    p_bench.add_argument(
        "--track",
        choices=("locked", "tuned", "both"),
        default="locked",
        help="Parameter track for --suite phenomenology.",
    )
    p_bench.add_argument(
        "--seeds",
        default="1729,2718,3141,5772,8111",
        help="Comma-separated paired evaluation seeds.",
    )
    p_bench.add_argument("--workers", type=int, default=1, help="Concurrent isolated runs.")
    p_bench.add_argument(
        "--standard-timeout", type=float, default=600.0, help="Standard per-run budget seconds."
    )
    p_bench.add_argument(
        "--heavy-timeout", type=float, default=1800.0, help="Heavy per-run budget seconds."
    )
    p_bench.add_argument(
        "--memory-limit-gb", type=float, default=16.0, help="Per-run RSS limit in GiB."
    )
    p_bench.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse content-addressed run checkpoints.",
    )
    p_bench.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze manifests and report counts without executing methods.",
    )
    p_bench.add_argument(
        "--tiny",
        action="store_true",
        help="Use 60 observations, 64 genes and one seed for a CI smoke run.",
    )
    p_bench.add_argument("--seed", type=int, default=42, help="Synthetic suite seed.")
    p_bench.add_argument("--json", action="store_true", help="Emit JSON.")
    p_bench.add_argument("--out", help="Persist the machine-readable benchmark result.")
    p_bench.add_argument(
        "--min-score", type=float, help="Fail if the best score is below this regression threshold."
    )
    p_bench.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Fail when any candidate method raises an exception.",
    )
    p_bench.add_argument(
        "--stats",
        action="store_true",
        help="Attach statistical review (cell-bootstrap ARI CIs on domain tasks).",
    )
    p_bench.add_argument(
        "--n-boot",
        type=int,
        default=200,
        help="Bootstrap resamples when --stats is set (default 200).",
    )
    p_bench.add_argument(
        "--k-policy",
        choices=("estimate", "oracle", "fixed"),
        default="estimate",
        help="Domain-count policy for multi-dataset landscape suites (default: estimate).",
    )
    p_bench.add_argument(
        "--allow-oracle-k",
        action="store_true",
        help="Permit k-policy=oracle (true domain count). Off by default.",
    )

    p_boundary = sub.add_parser(
        "benchmark-boundary",
        help="Map each method's failure boundary by sweeping one synthetic-data knob.",
    )
    p_boundary.add_argument(
        "--task",
        action="append",
        choices=("domain_detection", "deconvolution", "svg"),
        help="Restrict to a task (repeatable). Default: all tasks.",
    )
    p_boundary.add_argument(
        "--axis",
        action="append",
        dest="axes",
        metavar="PARAM",
        help="Restrict to a sweep axis by parameter name (repeatable). Default: all axes.",
    )
    p_boundary.add_argument(
        "--methods",
        help="Comma-separated method names to evaluate. Default: all runnable methods.",
    )
    p_boundary.add_argument(
        "--tau", type=float, default=0.7, help="Absolute acceptability threshold. Default: 0.7."
    )
    p_boundary.add_argument("--seeds", type=int, default=5, help="Replicate seeds. Default: 5.")
    p_boundary.add_argument(
        "--out",
        help="Directory to write boundary_long.csv + safe_operating_cards.{csv,json,md}.",
    )
    p_boundary.add_argument("--json", action="store_true", help="Emit the cards as JSON.")
    p_boundary.add_argument(
        "--no-fingerprints",
        action="store_true",
        help="Skip the failure-fingerprint atlas (how methods fail).",
    )

    p_fp = sub.add_parser(
        "failure-fingerprint",
        help=(
            "Build a method failure fingerprint atlas: 4-mode vectors "
            "(fragmentation / merge / noise / structural) from easy vs hard probes."
        ),
    )
    p_fp.add_argument(
        "--methods",
        help="Comma-separated domain-detection methods. Default: all runnable.",
    )
    p_fp.add_argument("--seeds", type=int, default=3, help="Replicate seeds. Default: 3.")
    p_fp.add_argument("--tau", type=float, default=0.7, help="Acceptability threshold.")
    p_fp.add_argument(
        "--out-dir",
        default="failure_fingerprints",
        help="Artifact directory. Default: failure_fingerprints.",
    )
    p_fp.add_argument("--json", action="store_true", help="Emit atlas JSON to stdout.")

    p_cal = sub.add_parser(
        "calibrate-recommender",
        help=(
            "When the recommender fails to beat global-best, propose dataset×method "
            "pairs that maximise expected information gain (evidence-acquisition todo)."
        ),
    )
    p_cal.add_argument("--in", dest="in_path", required=True, help="Input bundle.")
    p_cal.add_argument(
        "--knowledge-base", required=True, help="Landscape knowledge-base JSON."
    )
    p_cal.add_argument("--dataset-name", default="user_dataset")
    p_cal.add_argument("--k-neighbours", type=int, default=3)
    p_cal.add_argument("--top", type=int, default=10, help="Max evidence tasks to list.")
    p_cal.add_argument("--json", action="store_true", help="Emit calibration plan JSON.")
    p_cal.add_argument("--out", help="Persist calibration plan JSON.")

    p_scale = sub.add_parser("scale", help="Run an isolated computational scaling sweep.")
    p_scale.add_argument("--scales", default="1000,10000,100000,500000,1000000")
    p_scale.add_argument("--genes", type=int, default=2000)
    p_scale.add_argument("--density", type=float, default=0.05)
    p_scale.add_argument(
        "--methods", default="all-compute", help="all-compute or category:method pairs."
    )
    p_scale.add_argument("--timeout", type=float, default=1800.0)
    p_scale.add_argument("--mem-cap", type=float, default=58.0)
    p_scale.add_argument("--seed", type=int, default=42)
    p_scale.add_argument("--out-dir", default="scalability_proof")
    p_scale.add_argument("--quick", action="store_true", help="Use a small smoke-test sweep.")
    p_recommend = sub.add_parser(
        "recommend",
        help="Recommend methods for a bundle from benchmark evidence.",
    )
    p_recommend.add_argument("--in", dest="in_path", required=True, help="Input bundle.")
    p_recommend.add_argument(
        "--knowledge-base", required=True, help="Versioned landscape knowledge-base JSON."
    )
    p_recommend.add_argument("--dataset-name", default="user_dataset")
    p_recommend.add_argument("--k-neighbours", type=int, default=3)
    p_recommend.add_argument("--top", type=int, default=3)
    p_recommend.add_argument("--json", action="store_true", help="Emit JSON.")
    p_recommend.add_argument("--out", help="Persist recommendation JSON.")

    p_causal = sub.add_parser(
        "causal-landscape",
        help="Estimate causal effects of a data feature on method performance via intervention.",
    )
    p_causal.add_argument(
        "--knob",
        default="marker_gene_lift",
        help="Generator knob to intervene on (do). Default: marker_gene_lift.",
    )
    p_causal.add_argument("--lift-lo", type=float, default=2.0, help="Low intervention anchor.")
    p_causal.add_argument("--lift-hi", type=float, default=12.0, help="High intervention anchor.")
    p_causal.add_argument("--levels", type=int, default=5, help="Grid levels from lo to hi.")
    p_causal.add_argument("--seeds", type=int, default=10, help="Synthetic replicates per level.")
    p_causal.add_argument("--n-cells", type=int, default=500, help="Cells per synthetic dataset.")
    p_causal.add_argument(
        "--n-domains", type=int, default=4, help="Ground-truth domains (held fixed)."
    )
    p_causal.add_argument(
        "--noise", type=float, default=0.25, help="Expression noise (held fixed)."
    )
    p_causal.add_argument(
        "--methods", default="all", help="'all' or comma-separated domain-detection method names."
    )
    p_causal.add_argument(
        "--out-dir", default="causal_landscape", help="Artifact directory for the causal landscape."
    )
    p_causal.add_argument("--json", action="store_true", help="Emit result JSON to stdout.")

    p_ask = sub.add_parser(
        "ask",
        help="Compile a natural-language question into an executable spatial pipeline.",
    )
    p_ask.add_argument("question", help="Natural-language spatial analysis question.")
    p_ask.add_argument("--in", dest="in_path", required=True, help="Input bundle directory.")
    p_ask.add_argument(
        "--out", default="histoweave_compiled_report.html", help="Report or hand-off path."
    )
    p_ask.add_argument(
        "--model",
        help="LiteLLM model id; use 'mock' for deterministic offline compilation.",
    )
    p_ask.add_argument(
        "--executor",
        choices=("in-process", "nextflow"),
        default="in-process",
        help="Execution backend. Nextflow emits a params hand-off without spawning Nextflow.",
    )
    p_ask.add_argument("--plan-only", action="store_true", help="Compile and validate only.")
    p_ask.add_argument(
        "--plan-out",
        help="Persist the validated v1 plan JSON for review or later SDK loading.",
    )
    p_ask.add_argument(
        "--timeout",
        type=float,
        help="Model request timeout in seconds (1-600).",
    )
    p_ask.add_argument(
        "--max-repair-attempts",
        type=int,
        default=1,
        help="Schema/registry repair retries (0-3). Default: 1.",
    )
    p_ask.add_argument("--json", action="store_true", help="Emit the compiled plan as JSON.")
    p_ask.add_argument("--yes", action="store_true", help="Execute without confirmation.")
    p_ask.add_argument(
        "--gaps-file",
        default="docs/COMPILER_GAPS.md",
        help="Markdown audit log for approximated capabilities.",
    )

    p_twin = sub.add_parser(
        "digital-twin",
        help=(
            "Build a feature-matched synthetic twin of a real sample, benchmark "
            "methods on planted truth, and return the ranking as a prediction."
        ),
    )
    p_twin.add_argument("--in", dest="in_path", required=True, help="Input bundle directory.")
    p_twin.add_argument(
        "--out-dir",
        default="digital_twin_out",
        help="Artifact directory (JSON + HTML report). Default: digital_twin_out.",
    )
    p_twin.add_argument("--dataset-name", default="user_dataset")
    p_twin.add_argument(
        "--methods",
        help="Comma-separated domain-detection methods. Default: harness release defaults.",
    )
    p_twin.add_argument("--seed", type=int, default=0)
    p_twin.add_argument("--n-domains", type=int, help="Planted domain count (default: estimated).")
    p_twin.add_argument("--max-cells", type=int, default=1500)
    p_twin.add_argument("--max-genes", type=int, default=500)
    p_twin.add_argument("--n-trials", type=int, default=12, help="Twin match search budget.")
    p_twin.add_argument(
        "--k-policy",
        choices=("oracle", "estimate", "fixed"),
        default="oracle",
        help="Domain-count policy on the twin (default: oracle — planted labels).",
    )
    p_twin.add_argument("--stats", action="store_true", help="Attach cell-bootstrap ARI CIs.")
    p_twin.add_argument("--json", action="store_true", help="Emit full validation JSON.")
    p_twin.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation.",
    )

    p_automl = sub.add_parser(
        "automl",
        help=(
            "Spatial AutoML compiler: extract features, retrieve landscape neighbours, "
            "auto-run top-k methods, compare on a Pareto front, emit HTML report."
        ),
    )
    p_automl.add_argument(
        "question",
        nargs="?",
        default="Find spatial domains for my data.",
        help='Natural-language request, e.g. "Find spatial domains for my Visium HCC data."',
    )
    p_automl.add_argument("--in", dest="in_path", required=True, help="Input bundle directory.")
    p_automl.add_argument(
        "--knowledge-base",
        required=True,
        help="Landscape knowledge-base JSON (from a prior landscape run).",
    )
    p_automl.add_argument(
        "--out-dir",
        default="automl_out",
        help="Artifact directory (JSON + HTML). Default: automl_out.",
    )
    p_automl.add_argument("--dataset-name", default="user_dataset")
    p_automl.add_argument("--top", type=int, default=3, help="Top-k methods to auto-run.")
    p_automl.add_argument("--k-neighbours", type=int, default=3)
    p_automl.add_argument(
        "--methods",
        help="Optional comma-separated method override (skips recommender top-k).",
    )
    p_automl.add_argument("--n-domains", type=int, help="Domain count override.")
    p_automl.add_argument("--platform", help="Platform prior (visium, xenium, …).")
    p_automl.add_argument("--seed", type=int, default=0)
    p_automl.add_argument(
        "--model",
        default="mock",
        help="Compiler model id for the advisory NL plan (default: mock).",
    )
    p_automl.add_argument(
        "--no-compiler",
        action="store_true",
        help="Skip the natural-language compiler plan.",
    )
    p_automl.add_argument("--json", action="store_true", help="Emit full AutoML JSON.")
    p_automl.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation.",
    )

    p_ingest = sub.add_parser("ingest", help="Read vendor data (or the demo) into a bundle.")
    p_ingest.add_argument("--demo", action="store_true", help="Use the synthetic demo dataset.")
    p_ingest.add_argument("--input", help="Vendor output directory.")
    p_ingest.add_argument("--assay", help="Assay for --input: visium/xenium/stereo_seq.")
    p_ingest.add_argument("--engine", default="native", help="Reader engine: native|spatialdata.")
    p_ingest.add_argument("--seed", type=int, default=0, help="Seed for the demo dataset.")
    p_ingest.add_argument("--out", default="data.ttab", help="Output bundle directory.")
    p_ingest.add_argument("--force", action="store_true", help="Replace an existing bundle.")

    p_step = sub.add_parser("step", help="Run one analysis method over a bundle.")
    p_step.add_argument("category", help="Method category, e.g. qc or domain_detection.")
    p_step.add_argument("--method", required=True, help="Registered method name.")
    p_step.add_argument("--method-version", help="Exact registered wrapper version.")
    p_step.add_argument("--in", dest="in_path", required=True, help="Input bundle directory.")
    p_step.add_argument("--out", required=True, help="Output bundle directory.")
    p_step.add_argument("--force", action="store_true", help="Replace an existing bundle.")
    p_step.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Method parameter (repeatable); values are coerced to int/float/bool/str.",
    )

    p_report = sub.add_parser("report", help="Render an HTML report from a bundle.")
    p_report.add_argument("--in", dest="in_path", required=True, help="Input bundle directory.")
    p_report.add_argument("--out", default="report.html", help="Output HTML report.")

    p_validate = sub.add_parser(
        "validate-bundle", help="Validate a bundle manifest and all artifact checksums."
    )
    p_validate.add_argument("bundle", help="Bundle directory to validate.")
    p_validate.add_argument("--json", action="store_true", help="Emit metadata as JSON.")

    p_doctor = sub.add_parser("doctor", help="Diagnose runtime, optional extras, and plugins.")
    p_doctor.add_argument("--json", action="store_true", help="Emit diagnostics as JSON.")

    p_sota = sub.add_parser(
        "sota",
        help="Probe / run the official SOTA DLPFC reproduction grid (P2).",
    )
    p_sota.add_argument(
        "--methods",
        default="banksy_py,spagcn,graphst,stagate,bayesspace",
        help="Comma-separated methods (default: all SOTA + banksy_py).",
    )
    p_sota.add_argument(
        "--slices",
        default="151673,151674,151507,151669,151670",
        help="Comma-separated DLPFC slice ids.",
    )
    p_sota.add_argument("--seeds", default="42,1,2", help="Comma-separated seeds.")
    p_sota.add_argument(
        "--out-dir",
        default="5x15_spatial_aware",
        help="Output directory for sota_benchmark_long.csv and throughput JSON.",
    )
    p_sota.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe backends only; write skipped status grid.",
    )
    p_sota.add_argument(
        "--force",
        action="store_true",
        help="Ignore per-cell checkpoints.",
    )
    p_sota.add_argument(
        "--all-methods",
        action="store_true",
        help="Attempt methods even when the probe fails.",
    )
    p_sota.add_argument(
        "--show-contract",
        action="store_true",
        help="Print the environment contract JSON and exit.",
    )
    p_sota.add_argument("--json", action="store_true", help="Emit the report summary as JSON.")

    p_stats = sub.add_parser(
        "stats-review",
        help="Independent statistical review of a multi-dataset performance landscape.",
    )
    p_stats.add_argument(
        "--landscape",
        help="Landscape JSON (schema with performance[dataset][method] scores).",
    )
    p_stats.add_argument(
        "--performance-json",
        help="Raw performance dict JSON {dataset: {method: score}} (alternative to --landscape).",
    )
    p_stats.add_argument("--n-boot", type=int, default=500, help="Dataset bootstrap resamples.")
    p_stats.add_argument("--n-perm", type=int, default=1000, help="Paired permutation draws.")
    p_stats.add_argument("--seed", type=int, default=0)
    p_stats.add_argument(
        "--fdr-method",
        default="bh",
        choices=("bh", "by", "holm", "bonferroni"),
        help="Multiple-testing correction for pairwise tests.",
    )
    p_stats.add_argument("--alpha", type=float, default=0.05)
    p_stats.add_argument("--out", help="Write StatsReviewReport JSON.")
    p_stats.add_argument("--json", action="store_true", help="Print full report JSON.")

    p_pareto = sub.add_parser(
        "pareto",
        help="Multi-objective Pareto frontier over accuracy/speed/memory/robustness.",
    )
    p_pareto.add_argument(
        "--benchmark-long",
        help="benchmark_long.csv with per-seed ari/seconds (preferred source; enables "
        "the robustness axis).",
    )
    p_pareto.add_argument(
        "--landscape",
        help="Landscape JSON (accuracy + timings only; no robustness axis). Used when "
        "--benchmark-long is absent.",
    )
    p_pareto.add_argument(
        "--scaling-dir",
        help="Directory with scaling_metrics.csv to attach the memory (GB) axis.",
    )
    p_pareto.add_argument(
        "--dataset",
        help="Restrict the text summary to a single dataset id (JSON still covers all).",
    )
    p_pareto.add_argument(
        "--eps-rel",
        type=float,
        default=1e-3,
        help="Relative per-axis tolerance for dominance ties (default 1e-3).",
    )
    p_pareto.add_argument(
        "--n-boot",
        type=int,
        default=200,
        help="Bootstrap resamples for the ARI CI-width robustness axis.",
    )
    p_pareto.add_argument("--seed", type=int, default=0)
    p_pareto.add_argument("--out", help="Write ParetoReport JSON.")
    p_pareto.add_argument(
        "--svg",
        help="Write a per-dataset Pareto SVG (requires --dataset, or writes the first).",
    )
    p_pareto.add_argument("--json", action="store_true", help="Print full report JSON.")

    p_isus = sub.add_parser(
        "isus",
        help="Information-theoretic Spatial Utility Score: does this dataset need a "
        "spatial method?",
    )
    p_isus.add_argument(
        "--in",
        dest="in_path",
        help="Input bundle / dataset with expression, spatial coords and domain labels.",
    )
    p_isus.add_argument(
        "--calibrate",
        help="Benchmark directory (needs benchmark_long.csv + per-slice .h5ad in data/) "
        "to correlate ISUS against the observed spatial ARI gain.",
    )
    p_isus.add_argument(
        "--domain-key",
        default="domain_truth",
        help="obs column holding ground-truth domain labels (default: domain_truth).",
    )
    p_isus.add_argument("--n-pcs", type=int, default=20, help="Expression PCA components.")
    p_isus.add_argument("--k", type=int, default=3, help="kNN neighbours for the MI estimator.")
    p_isus.add_argument("--seed", type=int, default=0)
    p_isus.add_argument("--out", help="Write ISUS result JSON.")
    p_isus.add_argument("--json", action="store_true", help="Print full JSON.")

    p_disc = sub.add_parser(
        "discovery",
        help="Run cryptic-niche discovery / panel / cohort / donor-bootstrap workflows.",
    )
    disc_sub = p_disc.add_subparsers(dest="discovery_command")
    p_disc_run = disc_sub.add_parser(
        "run", help="Multi-method uncertainty discovery on configured DLPFC slices."
    )
    p_disc_run.add_argument(
        "--repo-root",
        help="Repository root containing research/discovery_uncertainty_niches "
        "(default: auto-detect).",
    )
    p_disc_cohort = disc_sub.add_parser(
        "cohort", help="12-slice DLPFC cohort discovery + pure L3/L6 panel scoring."
    )
    p_disc_cohort.add_argument(
        "--slices",
        help="Comma-separated dlpfc_* slice ids (default: all 12).",
    )
    p_disc_cohort.add_argument(
        "--force-discovery",
        action="store_true",
        help="Recompute uncertainty maps even when cached.",
    )
    p_disc_cohort.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Only score panels on slices with existing maps.",
    )
    p_disc_cohort.add_argument("--repo-root", help="Repository root (auto-detect default).")
    p_disc_boot = disc_sub.add_parser(
        "bootstrap-ci",
        help="Donor-stratified bootstrap CIs for direction_ok L3 components.",
    )
    p_disc_boot.add_argument(
        "--panel-csv",
        help="cohort_component_panel.csv path (default: research/.../cohort_component_panel.csv).",
    )
    p_disc_boot.add_argument("--n-boot", type=int, default=2000)
    p_disc_boot.add_argument("--seed", type=int, default=0)
    p_disc_boot.add_argument("--out", help="Write DonorBootstrapResult JSON.")
    p_disc_boot.add_argument("--json", action="store_true", help="Print full JSON.")
    p_disc_boot.add_argument("--repo-root", help="Repository root (auto-detect default).")
    p_disc_panel = disc_sub.add_parser(
        "panel",
        help="Pre-registered ENC1/HOPX/MBP panel validation + IF ROI export (pilot slices).",
    )
    p_disc_panel.add_argument("--repo-root", help="Repository root (auto-detect default).")
    p_disc_if = disc_sub.add_parser(
        "if-package",
        help="Build wet-lab IF package for 151508 L3+L6 (+ optional 151669 L3).",
    )
    p_disc_if.add_argument("--repo-root", help="Repository root (auto-detect default).")
    p_disc_ifret = disc_sub.add_parser(
        "if-analyze",
        help="Score IF return tables and upgrade claim ladder (or --simulate-from-rna).",
    )
    p_disc_ifret.add_argument(
        "--simulate-from-rna",
        action="store_true",
        help="Dry-run analyzer using RNA as fake IF (NOT protein validation).",
    )
    p_disc_ifret.add_argument("--repo-root", help="Repository root (auto-detect default).")
    p_disc_ln = disc_sub.add_parser(
        "xenium-lymph",
        help="Run cryptic-niche discovery on Xenium human lymph node (second tissue).",
    )
    p_disc_ln.add_argument("--repo-root", help="Repository root (auto-detect default).")
    p_disc_ln.add_argument(
        "--swap-official",
        action="store_true",
        help=(
            "Prefer official matrix (download if needed), re-run discovery, "
            "compare AUROC/panel Δ, GC deep-dive."
        ),
    )
    p_disc_ln.add_argument(
        "--gc-deep-dive",
        action="store_true",
        help="Only run GC-enriched component deep-dive (requires prior discovery results).",
    )
    p_disc_ln.add_argument(
        "--skip-gc", action="store_true", help="With --swap-official, skip GC deep-dive."
    )
    p_disc_ln.add_argument(
        "--force-synthetic",
        action="store_true",
        help="With --swap-official, force synthetic counts.",
    )
    p_disc_ln.add_argument(
        "--no-download", action="store_true", help="With --swap-official, do not hit 10x CDN."
    )

    args = parser.parse_args(argv)

    configure_logging(level=args.log_level, log_format=args.log_format)
    _LOGGER.debug("CLI invoked: %s", " ".join(argv) if argv else "histoweave")

    if args.version or args.command == "version":
        return _cmd_version()
    if args.command == "list-methods":
        return _cmd_list_methods(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "benchmark-boundary":
        return _cmd_benchmark_boundary(args)
    if args.command == "failure-fingerprint":
        return _cmd_failure_fingerprint(args)
    if args.command == "calibrate-recommender":
        return _cmd_calibrate_recommender(args)
    if args.command == "scale":
        return _cmd_scale(args)
    if args.command == "recommend":
        return _cmd_recommend(args)
    if args.command == "causal-landscape":
        return _cmd_causal_landscape(args)
    if args.command == "ask":
        return _cmd_ask(args)
    if args.command == "digital-twin":
        return _cmd_digital_twin(args)
    if args.command == "automl":
        return _cmd_automl(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "step":
        return _cmd_step(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "validate-bundle":
        return _cmd_validate_bundle(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "sota":
        return _cmd_sota(args)
    if args.command == "stats-review":
        return _cmd_stats_review(args)
    if args.command == "pareto":
        return _cmd_pareto(args)
    if args.command == "isus":
        return _cmd_isus(args)
    if args.command == "discovery":
        return _cmd_discovery(args)

    parser.print_help()
    return 0


def _cmd_version() -> int:
    from . import __version__

    _emit(f"histoweave {__version__}")
    return 0


def _cmd_causal_landscape(args: argparse.Namespace) -> int:
    """Run the interventional (do) causal performance landscape and write artifacts."""
    import csv
    import json
    from pathlib import Path

    import numpy as np

    from .benchmark.causal import causal_graph_svg, run_causal_landscape

    grid = tuple(
        float(x) for x in np.linspace(args.lift_lo, args.lift_hi, max(2, int(args.levels)))
    )
    methods = (
        None if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    )

    result = run_causal_landscape(
        knob=args.knob,
        grid=grid,
        n_seeds=int(args.seeds),
        fixed_params={
            "n_cells": int(args.n_cells),
            "n_domains": int(args.n_domains),
            "noise": float(args.noise),
        },
        methods=methods,
        progress=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "causal_landscape.json").write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "causal_graph.svg").write_text(causal_graph_svg(result), encoding="utf-8")

    with (out_dir / "ace_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "method",
                "ace",
                "ci_low",
                "ci_high",
                "significant",
                "ari_lo",
                "ari_hi",
                "support_lo",
                "support_hi",
            ]
        )
        for e in result.ranked_effects():
            writer.writerow(
                [
                    e.method,
                    e.ace,
                    e.ci_low,
                    e.ci_high,
                    e.significant,
                    e.ari_lo,
                    e.ari_hi,
                    e.support_lo,
                    e.support_hi,
                ]
            )

    with (out_dir / "feature_displacement.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature"] + [f"lift={lv:g}" for lv in result.grid])
        for feature in result.feature_order:
            row = [feature] + [
                result.feature_displacement.get(lv, {}).get(feature, {}).get("mean", float("nan"))
                for lv in result.grid
            ]
            writer.writerow(row)

    if args.json:
        _emit(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    else:
        _emit(result.summary())
        _emit(f"\nArtifacts written to {out_dir}/")
    return 0


def _cmd_ask(args) -> int:
    from .compiler import CompilerValidationError, run_compiled, save_plan
    from .compiler import compile as compile_question
    from .io import read_bundle
    from .workflow import PipelineExecutionError

    try:
        data = read_bundle(args.in_path)
        plan = compile_question(
            args.question,
            data=data,
            provider=args.model,
            executor=args.executor,
            gaps_path=args.gaps_file,
            timeout=args.timeout,
            max_repair_attempts=args.max_repair_attempts,
        )
        if args.plan_out:
            save_plan(plan, args.plan_out)
    except (OSError, ValueError, CompilerValidationError, RuntimeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _emit(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    else:
        _emit(f"Compiled rationale: {plan.rationale}")
        _emit(f"Plan ID: {plan.plan_id}")
        for number, step in enumerate(plan.steps, start=1):
            _emit(
                f"  {number}. {step.category}:{step.method} "
                f"{json.dumps(step.params, ensure_ascii=False)} — {step.purpose}"
            )
        for gap in plan.gaps:
            _emit(f"  approximation: {gap.concept} -> {gap.degraded_to}", file=sys.stderr)
        if args.plan_out:
            _emit(f"Plan written to {Path(args.plan_out).resolve()}")

    if args.plan_only:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            _emit("Plan validated; not executed in a non-interactive session (use --yes).")
            return 0
        answer = input("Execute this pipeline? [y/N] ").strip().casefold()
        if answer not in {"y", "yes"}:
            _emit("Plan validated; execution cancelled.")
            return 0

    try:
        result = run_compiled(plan, data=data, out=args.out, confirmed=True)
    except (PipelineExecutionError, OSError, ValueError, RuntimeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 1
    if isinstance(result, dict):
        _emit(json.dumps(result, indent=2))
    else:
        _emit(f"Report written to {Path(args.out).resolve()}")
    return 0


def _cmd_digital_twin(args) -> int:
    """Feature-matched twin + planted-truth benchmark → predicted ranking."""
    from pathlib import Path

    from .benchmark import run_digital_twin_validation
    from .io import read_bundle

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        if not methods:
            _emit("error: --methods is empty", file=sys.stderr)
            return 2
    try:
        data = read_bundle(args.in_path)
        result = run_digital_twin_validation(
            data,
            methods=methods,
            dataset_name=args.dataset_name,
            seed=args.seed,
            n_domains=args.n_domains,
            max_cells=args.max_cells,
            max_genes=args.max_genes,
            n_trials=args.n_trials,
            k_policy=args.k_policy,
            stats=args.stats,
            out_dir=args.out_dir,
            write_report=not args.no_report,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False, default=str))
    else:
        _emit(result.summary())
        _emit(f"Artifacts written to {Path(args.out_dir).resolve()}")
    return 0


def _cmd_automl(args) -> int:
    """Spatial AutoML: recommend → run top-k → Pareto HTML report."""
    from pathlib import Path

    from .automl import run_spatial_automl
    from .io import read_bundle

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        if not methods:
            _emit("error: --methods is empty", file=sys.stderr)
            return 2
    if args.top < 1:
        _emit("error: --top must be at least 1", file=sys.stderr)
        return 2
    try:
        data = read_bundle(args.in_path)
        result = run_spatial_automl(
            data,
            args.question,
            knowledge_base=args.knowledge_base,
            dataset_name=args.dataset_name,
            top_k=args.top,
            k_neighbours=args.k_neighbours,
            methods=methods,
            n_domains=args.n_domains,
            use_compiler=not args.no_compiler,
            compiler_model=args.model,
            seed=args.seed,
            out_dir=args.out_dir,
            write_report=not args.no_report,
            platform=args.platform,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _emit(json.dumps(result.to_dict(), indent=2, allow_nan=False, default=str))
    else:
        _emit(result.summary())
        _emit(f"Artifacts written to {Path(args.out_dir).resolve()}")
    return 0


def _cmd_list_methods(args) -> int:
    from .plugins import list_methods

    methods = list_methods(
        category=args.category,
        assay=args.assay,
        all_versions=args.all_versions,
    )
    if args.json:
        _emit(json.dumps(methods, indent=2))
        return 0
    if not methods:
        _emit("No methods match the filter.")
        return 0
    width = max(len(m["name"]) for m in methods)
    # Show benchmark scores when any method in the listing carries them.
    has_bench = any(m["benchmark"] for m in methods)
    header = f"{'CATEGORY':<20} {'METHOD':<{width}} {'VER':<8}"
    sep = "-" * (40 + width)
    if has_bench:
        header += " TASK              SCORE  RANK"
        sep += "------ ------ ----"
    header += " SUMMARY"
    _emit(header)
    _emit(sep)
    for m in methods:
        line = f"{m['category']:<20} {m['name']:<{width}} {m['version']:<8}"
        if has_bench:
            if m["benchmark"]:
                # Show the first (best-scoring) task entry.
                task_name, entry = next(iter(m["benchmark"].items()))
                score = entry.get("score", "n/a")
                is_finite_number = isinstance(score, int | float) and score not in (
                    float("inf"),
                    float("-inf"),
                )
                score_str = f"{score:.4f}" if is_finite_number else str(score)
                rank = str(entry.get("rank", "-"))
                line += f" {task_name:<16} {score_str:<6} {rank:<4}"
            else:
                line += " -                -      -   "
        lifecycle = " [DEPRECATED]" if m["deprecated"] else ""
        line += f"{lifecycle} {m['summary']}"
        _emit(line)
    return 0


def _cmd_run(args) -> int:
    from . import (
        PipelineExecutionError,
        PipelineStep,
        build_report,
        datasets,
        default_pipeline,
        run_pipeline,
    )

    if args.input:
        from .io import read

        if not args.assay:
            _emit("error: --assay is required with --input", file=sys.stderr)
            return 2
        _emit(f"Reading {args.assay} data from {args.input} ...")
        data = read(args.assay, args.input)
    else:
        if not args.demo:
            _emit("note: no --input given; using --demo synthetic dataset", file=sys.stderr)
        data = datasets.make_synthetic(seed=args.seed)

    if args.n_domains is None and not data.uns.get("n_domains"):
        _emit(
            "error: --n-domains is required because this dataset has no known domain count",
            file=sys.stderr,
        )
        return 2

    steps = []
    for step in default_pipeline():
        params = dict(step.params)
        if step.method == "kmeans" and args.n_domains is not None:
            params["n_domains"] = args.n_domains
        if getattr(step.category, "value", step.category) == "annotation":
            has_markers = bool(data.uns.get("marker_genes"))
            should_annotate = args.annotation if args.annotation is not None else has_markers
            if should_annotate and not has_markers:
                _emit(
                    "error: annotation was requested but no marker_genes are present; "
                    "provide markers or use --no-annotation",
                    file=sys.stderr,
                )
                return 2
            if not should_annotate:
                _emit("note: marker annotation skipped (no marker genes)", file=sys.stderr)
                continue
        steps.append(PipelineStep(step.category, step.method, params, step.name))

    _emit(f"Input: {data!r}")
    _emit("Running pipeline:")
    try:
        result = run_pipeline(data, steps, verbose=True)
    except PipelineExecutionError as exc:
        if args.manifest:
            _write_manifest(Path(args.manifest), exc.manifest.to_dict())
        _emit(f"error: {exc}", file=sys.stderr)
        return 1

    out = build_report(result, args.out)
    _emit(f"Report written to {out.resolve()}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        _write_manifest(manifest_path, result.uns.get("run_manifest", {}))
        _emit(f"Manifest written to {manifest_path.resolve()}")
    return 0


def _cmd_benchmark(args) -> int:
    from .benchmark import deconvolution_task, domain_detection_task, run_benchmark

    if args.suite == "figure3":
        return _cmd_figure3_benchmark(args)
    if args.suite == "phenomenology":
        return _cmd_phenomenology_benchmark(args)

    tasks = {
        "domain_detection": domain_detection_task,
        "deconvolution": deconvolution_task,
    }
    if args.task not in tasks:
        _emit(f"error: unknown task '{args.task}'. Available: {sorted(tasks)}", file=sys.stderr)
        return 2

    # Single-dataset smoke harness: pass oracle-free method params when possible.
    # Domain methods without n_domains still fall back to uns['n_domains'] from
    # the synthetic factory (which records the generative K).  For a fully
    # non-oracle path use landscape with k_policy='estimate'.
    result = run_benchmark(
        tasks[args.task](),
        stats=bool(getattr(args, "stats", False)),
        n_boot=int(getattr(args, "n_boot", 200)),
        seed=int(getattr(args, "seed", 0)),
        k_policy=str(getattr(args, "k_policy", "estimate")),
        allow_oracle_k=bool(getattr(args, "allow_oracle_k", False)),
    )
    json_rows = []
    for row in result.leaderboard:
        json_row = dict(row)
        if json_row.get("score") in (float("inf"), float("-inf")):
            json_row["score"] = None
        json_rows.append(json_row)
    payload = {
        "schema_version": 1,
        "task": result.task,
        "leaderboard": json_rows,
        "stats": result.stats,
    }
    if args.out:
        _write_json_atomic(Path(args.out), payload)
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
    else:
        _emit(f"Benchmark: {result.task}  (metric: {result.metric}, higher is better)\n")
        _emit(f"{'RANK':<5} {'METHOD':<18} {'SCORE':<8} {'TIME(s)':<8}")
        _emit("-" * 40)
        for row in result.leaderboard:
            score = row["score"]
            score_str = f"{score:.4f}" if score not in (float("inf"), float("-inf")) else "n/a"
            secs = str(row.get("seconds", "n/a"))
            _emit(f"{row['rank']:<5} {row['method']:<18} {score_str:<8} {secs:<8}")
            if "error" in row:
                _emit(f"      -> {row['error']}")
            if "ari_ci_low" in row and "ari_ci_high" in row:
                _emit(f"      ARI 95% CI [{row['ari_ci_low']:.4f}, {row['ari_ci_high']:.4f}]")
        best = result.best()
        if best and best["score"] not in (float("inf"), float("-inf")):
            _emit(f"\nRecommended: {best['method']} (score {best['score']:.4f})")
        if result.stats:
            _emit(
                "\nStatistical review: cell-bootstrap ARI CIs attached "
                f"(n_boot={result.stats.get('n_boot')}). "
                "Multi-dataset rank FDR requires review_landscape()."
            )
        if args.out:
            _emit(f"\nBenchmark result written to {Path(args.out).resolve()}")

    best = result.best()
    if args.fail_on_error and any("error" in row for row in result.leaderboard):
        _emit("error: one or more benchmark candidates failed", file=sys.stderr)
        return 1
    if args.min_score is not None and (best is None or best["score"] < args.min_score):
        actual = "no valid score" if best is None else best["score"]
        _emit(
            f"error: best benchmark score {actual} is below threshold {args.min_score}",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_phenomenology_benchmark(args) -> int:
    """Plan or run the phenomenon-centred capability matrix."""
    from collections import Counter

    from .benchmark import (
        BenchmarkExecutionConfig,
        ParameterTrack,
        RunStatus,
        build_suite_plan,
        execute_suite,
        write_suite_plan,
    )
    from .datasets import ObservationCondition, SpatialPhenomenon

    smoke_methods = (
        "basic_qc",
        "log1p_cp10k",
        "kmeans",
        "marker_deconv",
        "marker_score",
        "morans_i",
        "spatial_graph",
        "combat",
    )

    def parse_selection(raw: str, enum_type, all_token: str = "all"):
        if raw == all_token:
            return tuple(enum_type)
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
        if not values:
            raise ValueError(f"selection must be '{all_token}' or a comma-separated list")
        return tuple(enum_type(item) for item in values)

    try:
        phenomena = parse_selection(args.phenomena, SpatialPhenomenon)
        conditions = parse_selection(args.conditions, ObservationCondition)
        methods = None
        if args.methods != "all-release":
            methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
            if not methods:
                raise ValueError("--methods must be 'all-release' or a comma-separated list")
        elif args.tiny:
            methods = smoke_methods
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
        if args.tiny:
            seeds = seeds[:1]
        tracks = (
            (ParameterTrack.LOCKED, ParameterTrack.TUNED)
            if args.track == "both"
            else (ParameterTrack(args.track),)
        )
        output_dir = Path(args.out_dir or "phenomenology_benchmark")
        plan = build_suite_plan(
            phenomena=phenomena,
            conditions=conditions,
            methods=methods,
            seeds=seeds,
            tracks=tracks,
            n_obs=60 if args.tiny else 600,
            n_genes=64 if args.tiny else 256,
            image_size=64 if args.tiny else 256,
        )
        if args.dry_run:
            paths = write_suite_plan(plan, output_dir)
            payload = plan.to_dict()
            payload["dry_run"] = True
            payload["artifacts"] = {name: str(path) for name, path in paths.items()}
        else:
            standard_timeout = (
                min(args.standard_timeout, 30.0) if args.tiny else args.standard_timeout
            )
            heavy_timeout = min(args.heavy_timeout, 30.0) if args.tiny else args.heavy_timeout
            memory_limit = min(args.memory_limit_gb, 4.0) if args.tiny else args.memory_limit_gb
            config = BenchmarkExecutionConfig(
                standard_budget_seconds=standard_timeout,
                heavy_budget_seconds=heavy_timeout,
                memory_limit_gb=memory_limit,
                checkpoint_dir=str(output_dir / "checkpoints"),
                resume=args.resume,
            )
            outcomes, paths = execute_suite(
                plan,
                config=config,
                output_dir=output_dir,
                workers=args.workers,
            )
            statuses = Counter(outcome.run.status for outcome in outcomes)
            payload = plan.to_dict()
            payload.update(
                {
                    "dry_run": False,
                    "status_counts": dict(sorted(statuses.items())),
                    "artifacts": {name: str(path) for name, path in paths.items()},
                }
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
    else:
        mode = "planned" if args.dry_run else "complete"
        _emit(f"Phenomenology benchmark {mode}")
        _emit(
            f"Methods: {payload['method_count']}; scenarios: {payload['scenario_count']}; "
            f"design units/track: {payload['design_unit_count_per_track']}; "
            f"run records: {payload['run_record_count']}"
        )
        _emit(f"Method manifest: {payload['method_manifest_hash']}")
        if not args.dry_run:
            _emit(f"Statuses: {payload['status_counts']}")
        _emit(f"Artifacts: {output_dir.resolve()}")

    if not args.dry_run and args.fail_on_error:
        failed = {
            RunStatus.INVALID_INPUT.value,
            RunStatus.METHOD_ERROR.value,
            RunStatus.TIMEOUT.value,
            RunStatus.OOM.value,
            RunStatus.BUDGET_EXCEEDED.value,
        }
        if any(payload["status_counts"].get(status, 0) for status in failed):
            return 1
    return 0


def _cmd_figure3_benchmark(args) -> int:
    """Run the fixed 10-method x 3-dataset Figure 3 experiment."""
    from .benchmark import run_figure3_experiment

    try:
        result = run_figure3_experiment(args.out_dir or "benchmark_results", seed=args.seed)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 1

    payload = result.to_dict()
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
        return 0

    summary = result.summary
    _emit("Figure 3 benchmark complete: 10 methods x 3 datasets")
    _emit(
        f"Recommendation accuracy: top-1={summary['top1_accuracy']:.3f}, "
        f"top-3={summary['top3_accuracy']:.3f}"
    )
    _emit(
        f"Selection regret: mean={summary['mean_selection_regret']:.4f}, "
        f"max={summary['max_selection_regret']:.4f}"
    )
    _emit(f"Landscape: {result.landscape_path}")
    _emit(f"Performance matrix: {result.performance_matrix_path}")
    _emit(f"Figure 3 data: {result.figure3_data_path}")
    _emit(f"Validation: {result.validation_path}")
    return 0


def _cmd_benchmark_boundary(args) -> int:
    """Sweep one synthetic-data knob per axis and locate each method's tau crossing."""
    import pandas as pd

    from .benchmark import run_boundary_study, write_study_outputs

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    try:
        result = run_boundary_study(
            tasks=args.task,
            params=args.axes,
            methods=methods,
            tau=args.tau,
            n_seeds=args.seeds,
            progress=not args.json,
            include_fingerprints=not args.no_fingerprints,
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        paths = write_study_outputs(result, args.out)

    if args.json:
        _emit(json.dumps(result.to_bundle(), indent=2, allow_nan=False, default=float))
        return 0

    cards = result.cards_dataframe()
    _emit(
        f"Failure-boundary map  (tau={result.tau}, {len(result.seeds)} seeds, "
        f"{len(cards)} method x axis cards)\n"
    )
    _emit(f"{'TASK':<17} {'AXIS':<17} {'METHOD':<34} {'VERDICT':<18} {'x*':<9} SAFE RANGE")
    _emit("-" * 118)
    for _, r in cards.iterrows():
        x_star = "\u2014" if pd.isna(r["x_star"]) else f"{float(r['x_star']):.3g}"
        lo, hi = r.get("safe_low"), r.get("safe_high")
        safe = "\u2014" if (pd.isna(lo) and pd.isna(hi)) else f"{_num(lo)}\u2013{_num(hi)}"
        _emit(
            f"{r['task']:<17} {r['param']:<17} {r['method']:<34} "
            f"{r['verdict']:<18} {x_star:<9} {safe}"
        )
    n_found = int((cards["verdict"] == "boundary_found").sum())
    n_always = int((cards["verdict"] == "always_acceptable").sum())
    n_never = int((cards["verdict"] == "never_acceptable").sum())
    _emit(f"\nboundary_found={n_found}  always_acceptable={n_always}  never_acceptable={n_never}")
    if args.out:
        _emit(f"\nWritten to {Path(args.out).resolve()}:")
        for name, p in paths.items():
            _emit(f"  {name}: {p}")
    return 0


def _num(x) -> str:
    import math

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "\u2014"
    return f"{float(x):.3g}"


def _cmd_recommend(args) -> int:
    """Run target-free feature extraction, k-NN retrieval, and method ranking."""
    from .benchmark import MethodRecommender
    from .io import read_bundle

    if args.top < 1:
        _emit("error: --top must be at least 1", file=sys.stderr)
        return 2
    try:
        data = read_bundle(args.in_path)
        recommender = MethodRecommender(
            args.knowledge_base,
            k_neighbours=args.k_neighbours,
        )
        recommendation = recommender.recommend(
            data,
            dataset_name=args.dataset_name,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    payload = recommendation.to_dict()
    payload["ranked_methods"] = payload["ranked_methods"][: args.top]
    if args.out:
        _write_json_atomic(Path(args.out), payload)
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
        return 0

    _emit(f"Recommendation: {recommendation.dataset_name} [{recommendation.task}]")
    _emit(f"{'RANK':<5} {'METHOD':<20} {'SCORE':<9} {'UNCERTAINTY':<12} SUPPORT")
    _emit("-" * 66)
    for rank, method in enumerate(recommendation.top(args.top), 1):
        _emit(
            f"{rank:<5} {method.method:<20} {method.score:<9.4f} "
            f"{method.uncertainty:<12.4f} "
            f"{method.support}/{len(recommendation.neighbours)}"
        )
    if recommendation.neighbours:
        evidence = ", ".join(
            f"{item['name']} (similarity={item['similarity']:.3f})"
            for item in recommendation.neighbours
        )
        _emit(f"Nearest evidence: {evidence}")
    _emit(f"Ensemble suggestion: {recommendation.ensemble_strategy}")
    if recommendation.beats_global_best_baseline is False:
        _emit(
            "Baseline: does NOT beat global-best "
            f"({recommendation.global_best_method}); "
            "see evidence-acquisition todo below."
        )
        if recommendation.evidence_todo:
            _emit("Evidence-acquisition todo (maximise expected information gain):")
            for item in recommendation.evidence_todo[:10]:
                _emit(
                    f"  {item.get('priority', '?'):>2}. "
                    f"{item.get('dataset')} × {item.get('method')}  "
                    f"EIG={item.get('expected_information_gain')}  "
                    f"— {item.get('reason')}"
                )
        else:
            _emit(
                "  (no missing cells near the query; expand the landscape "
                "with new reference datasets)"
            )
    if args.out:
        _emit(f"Recommendation written to {Path(args.out).resolve()}")
    return 0


def _cmd_failure_fingerprint(args) -> int:
    """Probe easy/hard synthetic conditions and emit failure fingerprints."""
    from pathlib import Path

    from .benchmark import run_failure_fingerprint_probe, write_fingerprint_atlas

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        if not methods:
            _emit("error: --methods is empty", file=sys.stderr)
            return 2
    try:
        atlas = run_failure_fingerprint_probe(
            methods,
            seeds=tuple(range(max(1, int(args.seeds)))),
            tau=float(args.tau),
            progress=not args.json,
        )
        paths = write_fingerprint_atlas(atlas, args.out_dir)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _emit(json.dumps(atlas.to_dict(), indent=2, allow_nan=False))
    else:
        _emit(atlas.summary())
        _emit(f"Artifacts written to {Path(args.out_dir).resolve()}")
        for name, path in paths.items():
            _emit(f"  {name}: {path}")
    return 0


def _cmd_calibrate_recommender(args) -> int:
    """Propose evidence-acquisition experiments for recommender calibration."""
    from pathlib import Path

    from .benchmark import MethodRecommender, propose_evidence_acquisition
    from .io import read_bundle

    if args.top < 1:
        _emit("error: --top must be at least 1", file=sys.stderr)
        return 2
    try:
        data = read_bundle(args.in_path)
        recommender = MethodRecommender(
            args.knowledge_base,
            k_neighbours=args.k_neighbours,
        )
        # recommend() already attaches calibration; recompute with explicit top_n.
        recommendation = recommender.recommend(
            data,
            dataset_name=args.dataset_name,
        )
        plan = propose_evidence_acquisition(
            recommender,
            recommendation,
            top_n=args.top,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    payload = plan.to_dict()
    if args.out:
        _write_json_atomic(Path(args.out), payload)
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    _emit(plan.summary())
    if args.out:
        _emit(f"Calibration plan written to {Path(args.out).resolve()}")
    return 0


def _coerce(value: str):
    """Coerce a ``--param`` string to int, then float, then bool, else leave as str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    return value


def _cmd_ingest(args) -> int:
    from . import RunManifest, datasets
    from .io import read, write_bundle

    if args.input:
        if not args.assay:
            _emit("error: --assay is required with --input", file=sys.stderr)
            return 2
        _emit(f"Reading {args.assay} data from {args.input} (engine={args.engine}) ...")
        data = read(args.assay, args.input, engine=args.engine)
    else:
        if not args.demo:
            _emit("note: no --input given; using --demo synthetic dataset", file=sys.stderr)
        data = datasets.make_synthetic(seed=args.seed)

    # Seed the run manifest so downstream `step` commands can append to it.
    data.uns.setdefault("run_manifest", RunManifest().to_dict())

    try:
        out = write_bundle(data, args.out, overwrite=args.force)
    except (FileExistsError, OSError, RuntimeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(f"Ingested {data!r} -> {out}")
    return 0


def _cmd_step(args) -> int:
    from .io import read_bundle, write_bundle
    from .workflow import PipelineStep, PipelineStepError, RunManifest, execute_step

    params = {}
    for item in args.param:
        if "=" not in item:
            _emit(f"error: --param must be KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, raw = item.split("=", 1)
        params[key] = _coerce(raw)

    try:
        data = read_bundle(args.in_path)
        result, record = execute_step(
            data,
            PipelineStep(
                args.category,
                args.method,
                params,
                method_version=args.method_version,
            ),
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, PipelineStepError):
            error_type = exc.record.get("error", {}).get("type")
            return 2 if error_type in {"KeyError", "TypeError", "ValueError"} else 1
        return 2

    # Append this step to the pipeline manifest so the final report displays the
    # full chain — same shape the in-process `run_pipeline` produces.
    manifest = dict(result.uns.get("run_manifest", {}))
    if not manifest:
        # First step after an ingest that predates manifest seeding.
        manifest = RunManifest().to_dict()
    steps = list(manifest.get("steps", []))
    steps.append(record)
    manifest["steps"] = steps
    manifest["status"] = "success"
    manifest["finished"] = record["finished"]
    result.uns["run_manifest"] = manifest

    try:
        out = write_bundle(result, args.out, overwrite=args.force)
    except (FileExistsError, OSError, RuntimeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(
        f"[{args.category}:{args.method}] {data.n_obs} -> {result.n_obs} obs  "
        f"({record['seconds']:.3f}s)  wrote {out}"
    )
    return 0


def _cmd_report(args) -> int:
    from .io import read_bundle
    from .report import build_report

    data = read_bundle(args.in_path)
    out = build_report(data, args.out)
    _emit(f"Report written to {out.resolve()}")
    return 0


def _cmd_validate_bundle(args) -> int:
    from .io import inspect_bundle

    try:
        metadata = inspect_bundle(args.bundle, verify=True)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _emit(json.dumps(metadata, indent=2))
    else:
        shape = metadata.get("table", {}).get("shape", "unknown")
        _emit(
            f"valid bundle: schema={metadata['schema_version']} shape={shape} "
            f"artifacts={metadata['artifact_count']} checksums=verified"
        )
    return 0


def _cmd_doctor(args) -> int:
    from importlib.util import find_spec
    from shutil import which

    from . import __version__
    from .io import BUNDLE_SCHEMA_VERSION
    from .plugins import list_methods, list_plugin_failures

    dependencies = {}
    for name in ("numpy", "pandas", "jinja2", "h5py", "pyarrow", "sklearn", "spatialdata"):
        dependencies[name] = "available" if find_spec(name) is not None else "missing"

    plugin_failures = list_plugin_failures()
    core_missing = [
        name for name in ("numpy", "pandas", "jinja2") if dependencies[name] == "missing"
    ]
    status = "ok" if not core_missing and not plugin_failures else "degraded"
    diagnostic = {
        "status": status,
        "histoweave_version": __version__,
        "python": sys.version.split()[0],
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "registered_methods": len(list_methods()),
        "dependencies": dependencies,
        "rscript": which("Rscript"),
        "plugin_failures": plugin_failures,
    }

    if args.json:
        _emit(json.dumps(diagnostic, indent=2, allow_nan=False))
    else:
        _emit(f"HistoWeave doctor: {status}")
        _emit(f"  version: {__version__}  python: {diagnostic['python']}")
        _emit(f"  registered methods: {diagnostic['registered_methods']}")
        for name, state in dependencies.items():
            _emit(f"  {name:<12} {state}")
        _emit(f"  Rscript      {diagnostic['rscript'] or 'missing'}")
        for failure in plugin_failures:
            _emit(
                f"  plugin error {failure['entry_point']}: "
                f"{failure['error_type']}: {failure['message']}"
            )
    return 0 if status == "ok" else 1


def _cmd_sota(args) -> int:
    from .benchmark.sota_pipeline import env_contract, run_sota_benchmark

    if args.show_contract:
        _emit(json.dumps(env_contract(), indent=2, allow_nan=False))
        return 0

    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    slices = [item.strip() for item in str(args.slices).split(",") if item.strip()]
    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    report = run_sota_benchmark(
        methods=methods,
        slices=slices,
        seeds=seeds,
        out_dir=Path(args.out_dir),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        only_available=not bool(args.all_methods),
    )
    summary = report.throughput_summary()
    if args.json:
        _emit(json.dumps(report.to_dict(), indent=2, allow_nan=False))
    else:
        _emit(
            f"SOTA grid dry_run={summary['dry_run']} "
            f"success={summary['n_success']} failed={summary['n_failed']} "
            f"skipped={summary['n_skipped']}"
        )
        _emit(f"  available: {', '.join(summary['available_methods']) or '(none)'}")
        _emit(f"  out-dir: {args.out_dir}")
    return 0


def _load_performance_dict(args) -> dict[str, dict[str, float]]:
    """Load performance matrix from --landscape or --performance-json."""
    if args.landscape:
        payload = json.loads(Path(args.landscape).read_text(encoding="utf-8"))
        if "performance" in payload:
            raw = payload["performance"]
        elif "landscape" in payload and "performance" in payload["landscape"]:
            raw = payload["landscape"]["performance"]
        else:
            raise ValueError(
                "landscape JSON must contain a top-level 'performance' mapping "
                "(dataset → method → score)"
            )
    elif args.performance_json:
        raw = json.loads(Path(args.performance_json).read_text(encoding="utf-8"))
    else:
        raise ValueError("provide --landscape or --performance-json")
    # Coerce to float, keep NaN for missing
    out: dict[str, dict[str, float]] = {}
    for ds, row in raw.items():
        out[str(ds)] = {str(m): float(v) if v is not None else float("nan") for m, v in row.items()}
    return out


def _cmd_stats_review(args) -> int:
    from .benchmark.stats_review import review_landscape

    try:
        performance = _load_performance_dict(args)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2
    report = review_landscape(
        performance,
        n_boot=int(args.n_boot),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        fdr_method=args.fdr_method,
        alpha=float(args.alpha),
    )
    payload = report.to_dict()
    if args.out:
        _write_json_atomic(Path(args.out), payload)
        _emit(f"stats review written to {Path(args.out).resolve()}")
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
    else:
        _emit(
            f"Stats review: {report.n_datasets} datasets × {report.n_methods} methods  "
            f"(n_boot={report.n_boot}, n_perm={report.n_perm}, fdr={report.fdr_method})"
        )
        _emit("Rank summary (best mean-rank first):")
        for row in report.rank_summary[:10]:
            _emit(
                f"  {row['method']:<24} mean_rank={row['mean_rank']:.2f}  "
                f"p_best={row['p_best']:.3f}  mean_score={row['mean_score']:.4f}"
            )
        pair = report.pairwise
        _emit(
            f"Pairwise FDR: {pair.get('n_significant', 0)}/{pair.get('n_tests', 0)} "
            f"significant at alpha={report.alpha}"
        )
    return 0


def _cmd_pareto(args) -> int:
    """Multi-objective Pareto frontier over accuracy/speed/memory/robustness."""
    from .benchmark.pareto import analyze_dataset, build_report, pareto_svg
    from .benchmark.pareto_io import (
        objective_tables_from_landscape,
        objective_tables_from_long_csv,
    )

    try:
        if args.benchmark_long:
            tables = objective_tables_from_long_csv(
                args.benchmark_long,
                scaling_dir=args.scaling_dir,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        elif args.landscape:
            tables = objective_tables_from_landscape(
                args.landscape, scaling_dir=args.scaling_dir
            )
        else:
            _emit(
                "error: provide --benchmark-long (preferred) or --landscape",
                file=sys.stderr,
            )
            return 2
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    if not tables:
        _emit("error: no datasets found in the provided source", file=sys.stderr)
        return 2

    report = build_report(tables, eps_rel=float(args.eps_rel))
    payload = report.to_dict()

    if args.out:
        _write_json_atomic(Path(args.out), payload)
        _emit(f"pareto report written to {Path(args.out).resolve()}")

    if args.svg:
        target = args.dataset or tables[0].dataset
        chosen = next((t for t in tables if t.dataset == target), None)
        if chosen is None:
            _emit(f"error: dataset {target!r} not found for --svg", file=sys.stderr)
            return 2
        svg = pareto_svg(analyze_dataset(chosen, eps_rel=float(args.eps_rel)), report.directions)
        Path(args.svg).write_text(svg, encoding="utf-8")
        _emit(f"pareto SVG ({target}) written to {Path(args.svg).resolve()}")

    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
        return 0

    _emit(
        f"Pareto frontier: {report.n_datasets} datasets  objectives={report.objectives}  "
        f"(eps_rel={report.eps_rel})"
    )
    shown = [args.dataset] if args.dataset else list(report.datasets)
    for ds in shown:
        info = report.datasets.get(ds)
        if info is None:
            _emit(f"  {ds}: (not found)")
            continue
        _emit(
            f"\nDataset {ds}: {info['n_frontier']}/{info['n_configs']} non-dominated  "
            f"knee={info['knee']}"
        )
        for cfg in sorted(
            info["frontier"],
            key=lambda c: -(info["table"][c].get("accuracy") or 0.0),
        ):
            row = info["table"][cfg]
            bits = []
            for obj in info["objectives"]:
                val = row.get(obj)
                bits.append(f"{obj}={'NA' if val is None else f'{val:.4g}'}")
            _emit(f"  {cfg:<24} " + "  ".join(bits))
    if not args.dataset:
        _emit("\nFrontier frequency (configs non-dominated across datasets):")
        for cfg, count in list(report.frontier_frequency.items())[:12]:
            _emit(f"  {cfg:<24} {count}/{report.n_datasets}")
    return 0


def _cmd_isus(args) -> int:
    """Information-theoretic Spatial Utility Score for a dataset (or calibration)."""
    from .benchmark.isus import compute_isus_from_table

    if args.calibrate:
        return _isus_calibrate(args)

    if not args.in_path:
        _emit("error: provide --in <bundle> (or --calibrate <dir>)", file=sys.stderr)
        return 2

    from .io import read_bundle

    try:
        data = read_bundle(args.in_path)
        result = compute_isus_from_table(
            data,
            domain_key=args.domain_key,
            n_pcs=int(args.n_pcs),
            k=int(args.k),
            seed=int(args.seed),
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    if args.out:
        _write_json_atomic(Path(args.out), payload)
        _emit(f"ISUS written to {Path(args.out).resolve()}")
    if args.json:
        _emit(json.dumps(payload, indent=2, allow_nan=False))
        return 0

    isus_str = "NA" if payload["isus"] is None else f"{payload['isus']:.3f}"
    _emit(f"ISUS[{result.dataset}] = {isus_str}   band: {result.band}")
    _emit(
        f"  I(D;E)={result.i_d_e:.4f}  I(D;[S,E])={result.i_d_se:.4f}  "
        f"I(D;S|E)={result.i_d_s_given_e:.4f}"
    )
    _emit(
        f"  (n_obs={result.n_obs}, domains={result.n_domains}, n_pcs={result.n_pcs}, "
        f"k={result.k}, estimator={result.estimator})"
    )
    _emit(
        "  bands: <0.1 expression-sufficient | 0.1-0.3 modest | >0.3 spatial-critical "
        "(provisional heuristics)"
    )
    for flag in result.flags:
        _emit(f"  ! {flag}")
    return 0


def _isus_calibrate(args) -> int:
    """Correlate ISUS against the observed spatial ARI gain over benchmark slices."""
    import csv as _csv

    import numpy as np

    from .benchmark.isus import compute_isus

    root = Path(args.calibrate)
    long_csv = root / "benchmark_long.csv"
    data_dir = root / "data"
    if not long_csv.is_file():
        _emit(f"error: {long_csv} not found", file=sys.stderr)
        return 2
    try:
        import anndata as ad
    except ImportError:
        _emit("error: --calibrate requires anndata to read per-slice .h5ad", file=sys.stderr)
        return 2

    # Observed spatial benefit per dataset: mean over methods of best(sw>0) - sw0.0.
    rows: dict[str, dict[str, list[float]]] = {}
    with long_csv.open(encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            ds = str(row["dataset"])
            base = str(row.get("method") or row["config"].split("@", 1)[0])
            cfg = str(row["config"])
            try:
                ari = float(row["ari"])
            except (TypeError, ValueError):
                continue
            rows.setdefault(ds, {}).setdefault(f"{base}|{cfg}", []).append(ari)

    def _spatial_gain(ds: str) -> float:
        means = {k: float(np.mean(v)) for k, v in rows[ds].items()}
        per_method: list[float] = []
        bases = {k.split("|", 1)[0] for k in means}
        for base in bases:
            sw0 = means.get(f"{base}|{base}@sw0.0")
            pos = [
                means[k]
                for k in means
                if k.startswith(f"{base}|") and not k.endswith("@sw0.0")
            ]
            if sw0 is not None and pos:
                per_method.append(max(pos) - sw0)
        return float(np.mean(per_method)) if per_method else float("nan")

    records: list[dict[str, Any]] = []
    for ds in sorted(rows):
        h5 = data_dir / f"{ds}.h5ad"
        if not h5.is_file():
            continue
        adata = ad.read_h5ad(str(h5))
        coords = adata.obsm.get("spatial")
        if coords is None or args.domain_key not in adata.obs.columns:
            continue
        res = compute_isus(
            adata.X,
            np.asarray(coords),
            np.asarray(adata.obs[args.domain_key]),
            dataset=ds,
            n_pcs=int(args.n_pcs),
            k=int(args.k),
            seed=int(args.seed),
        )
        records.append(
            {
                "dataset": ds,
                "isus": res.isus,
                "i_d_e": res.i_d_e,
                "i_d_s_given_e": res.i_d_s_given_e,
                "band": res.band,
                "spatial_ari_gain": _spatial_gain(ds),
            }
        )

    if len(records) < 2:
        _emit(
            "error: need >=2 slices with matching .h5ad in <dir>/data/ to calibrate",
            file=sys.stderr,
        )
        return 2

    isus_arr = np.array([r["isus"] for r in records], dtype=float)
    gain_arr = np.array([r["spatial_ari_gain"] for r in records], dtype=float)
    ok = ~np.isnan(isus_arr) & ~np.isnan(gain_arr)
    rho = float("nan")
    if ok.sum() >= 2 and np.unique(isus_arr[ok]).size > 1:
        from scipy.stats import spearmanr

        rho = float(spearmanr(isus_arr[ok], gain_arr[ok]).correlation)

    summary = {
        "n_slices": len(records),
        "spearman_rho_isus_vs_spatial_gain": None if np.isnan(rho) else rho,
        "per_slice": records,
        "note": (
            "ISUS measures the spatial fraction of domain information (validated by "
            "coordinate-shuffle control); thresholds are provisional heuristics, not "
            "validated predictors of a specific method's ARI gain."
        ),
    }
    if args.out:
        _write_json_atomic(Path(args.out), summary)
        _emit(f"ISUS calibration written to {Path(args.out).resolve()}")
    if args.json:
        _emit(json.dumps(summary, indent=2, allow_nan=False))
        return 0

    _emit(f"ISUS calibration on {len(records)} slices:")
    _emit(f"{'DATASET':<12} {'ISUS':<8} {'I(D;E)':<9} {'I(D;S|E)':<10} {'ARI_gain':<9} BAND")
    _emit("-" * 62)
    for r in records:
        isus_v = "NA" if np.isnan(r["isus"]) else f"{r['isus']:.3f}"
        gain_v = "NA" if np.isnan(r["spatial_ari_gain"]) else f"{r['spatial_ari_gain']:.3f}"
        _emit(
            f"{r['dataset']:<12} {isus_v:<8} {r['i_d_e']:<9.4f} "
            f"{r['i_d_s_given_e']:<10.4f} {gain_v:<9} {r['band']}"
        )
    rho_str = "NA" if np.isnan(rho) else f"{rho:.3f}"
    _emit(f"\nSpearman(ISUS, spatial ARI gain) = {rho_str}  (n={int(ok.sum())})")
    return 0


def _discovery_repo_root(explicit: str | None = None) -> Path | None:
    """Locate the git checkout that contains research/discovery_uncertainty_niches."""
    if explicit:
        root = Path(explicit).resolve()
        if (root / "research" / "discovery_uncertainty_niches").is_dir():
            return root
        return None
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],  # src/
        Path(__file__).resolve().parents[3]
        if len(Path(__file__).resolve().parents) > 3
        else Path.cwd(),
    ]
    # Also walk parents of CWD (when invoked from a subdir).
    here = Path.cwd().resolve()
    candidates.extend(here.parents[:6])
    for root in candidates:
        if (root / "research" / "discovery_uncertainty_niches").is_dir():
            return root
    return None


def _run_discovery_script(repo: Path, script_name: str, extra_args: list[str] | None = None) -> int:
    """Execute a research discovery script with the same interpreter."""
    import subprocess

    script = repo / "research" / "discovery_uncertainty_niches" / script_name
    if not script.is_file():
        _emit(f"error: discovery script not found: {script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script), *(extra_args or [])]
    _emit(f"running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(repo), check=False)
    return int(proc.returncode)


def _cmd_discovery(args) -> int:
    sub = getattr(args, "discovery_command", None)
    if not sub:
        _emit(
            "usage: histoweave discovery {run,cohort,bootstrap-ci,panel} …",
            file=sys.stderr,
        )
        return 2

    if sub == "bootstrap-ci":
        # Prefer in-package implementation so CI works without research scripts.
        from .benchmark.donor_bootstrap import (
            donor_stratified_bootstrap_l3,
            load_cohort_panel,
        )

        repo = _discovery_repo_root(getattr(args, "repo_root", None))
        default_csv = None
        if repo is not None:
            default_csv = (
                repo
                / "research"
                / "discovery_uncertainty_niches"
                / "results"
                / "cohort"
                / "cohort_component_panel.csv"
            )
        csv_path = Path(args.panel_csv) if args.panel_csv else default_csv
        if csv_path is None or not csv_path.is_file():
            _emit(
                "error: cohort panel CSV not found; pass --panel-csv or run "
                "`histoweave discovery cohort` first",
                file=sys.stderr,
            )
            return 2
        frame = load_cohort_panel(csv_path)
        result = donor_stratified_bootstrap_l3(frame, n_boot=int(args.n_boot), seed=int(args.seed))
        payload = result.to_dict()
        if args.out:
            _write_json_atomic(Path(args.out), payload)
            _emit(f"donor bootstrap written to {Path(args.out).resolve()}")
        # Always also write next to the CSV when possible
        if csv_path.parent.is_dir():
            side = csv_path.parent / "donor_bootstrap_l3.json"
            _write_json_atomic(side, payload)
        if args.json:
            _emit(json.dumps(payload, indent=2, allow_nan=False))
        else:
            p, ci = result.point, result.ci
            _emit(
                f"Donor-stratified bootstrap (n_boot={result.n_boot}, "
                f"components={result.n_components}, donors={result.n_donors})"
            )
            _emit(
                f"  L3 Δrest:     {p['l3_delta_rest']:.4f}  "
                f"95% CI [{ci['l3_delta_rest']['ci_low']:.4f}, "
                f"{ci['l3_delta_rest']['ci_high']:.4f}]"
            )
            _emit(
                f"  Myelin Δrest: {p['myelin_delta_rest']:.4f}  "
                f"95% CI [{ci['myelin_delta_rest']['ci_low']:.4f}, "
                f"{ci['myelin_delta_rest']['ci_high']:.4f}]"
            )
            _emit(
                f"  Direction rate: {p['direction_rate']:.3f}  "
                f"95% CI [{ci['direction_rate']['ci_low']:.3f}, "
                f"{ci['direction_rate']['ci_high']:.3f}]"
            )
            for donor, means in result.donor_means.items():
                _emit(
                    f"  {donor}: L3={means['l3_delta_rest']:.3f}  "
                    f"myelin={means['myelin_delta_rest']:.3f}  "
                    f"n_comp={means['n_components']}"
                )
        return 0

    repo = _discovery_repo_root(getattr(args, "repo_root", None))
    if repo is None:
        _emit(
            "error: could not locate research/discovery_uncertainty_niches "
            "(pass --repo-root PATH to the histoweave checkout)",
            file=sys.stderr,
        )
        return 2

    if sub == "run":
        return _run_discovery_script(repo, "run_discovery.py")
    if sub == "panel":
        return _run_discovery_script(repo, "validate_panel_and_rois.py")
    if sub == "if-package":
        return _run_discovery_script(repo, "prepare_if_lab_package.py")
    if sub == "if-analyze":
        extra_if = ["--simulate-from-rna"] if getattr(args, "simulate_from_rna", False) else []
        return _run_discovery_script(repo, "analyze_if_return.py", extra_if)
    if sub == "xenium-lymph":
        # Second tissue context lives under research/discovery_xenium_lymph/
        ln_dir = repo / "research" / "discovery_xenium_lymph"
        import subprocess

        if getattr(args, "gc_deep_dive", False):
            script = ln_dir / "analyze_gc_components.py"
            if not script.is_file():
                _emit(f"error: missing {script}", file=sys.stderr)
                return 2
            _emit(f"running: {sys.executable} {script}")
            return int(
                subprocess.run([sys.executable, str(script)], cwd=str(repo), check=False).returncode
            )

        if getattr(args, "swap_official", False):
            script = ln_dir / "swap_and_rerun.py"
            if not script.is_file():
                _emit(f"error: missing {script}", file=sys.stderr)
                return 2
            extra = []
            if getattr(args, "skip_gc", False):
                extra.append("--skip-gc")
            if getattr(args, "force_synthetic", False):
                extra.append("--force-synthetic")
            if getattr(args, "no_download", False):
                extra.append("--no-download")
            _emit(f"running: {sys.executable} {script} {' '.join(extra)}")
            return int(
                subprocess.run(
                    [sys.executable, str(script), *extra], cwd=str(repo), check=False
                ).returncode
            )

        script = ln_dir / "run_discovery_ln.py"
        if not script.is_file():
            _emit(f"error: missing {script}", file=sys.stderr)
            return 2
        _emit(f"running: {sys.executable} {script}")
        return int(
            subprocess.run([sys.executable, str(script)], cwd=str(repo), check=False).returncode
        )
    if sub == "cohort":
        extra = []
        if getattr(args, "slices", None):
            extra.extend(["--slices", str(args.slices)])
        if getattr(args, "force_discovery", False):
            extra.append("--force-discovery")
        if getattr(args, "skip_discovery", False):
            extra.append("--skip-discovery")
        return _run_discovery_script(repo, "run_cohort_panel.py", extra)

    _emit(f"error: unknown discovery subcommand {sub!r}", file=sys.stderr)
    return 2


def _write_manifest(path: Path, manifest: dict) -> None:
    _write_json_atomic(path, manifest)


def _write_json_atomic(path: Path, value: dict) -> None:
    from uuid import uuid4

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cmd_scale(args) -> int:
    """Run the reproducible pyramid benchmark and persist its machine-readable artifacts."""
    from .benchmark import (
        DEFAULT_COMPUTE_METHODS,
        ScalingConfig,
        run_scaling,
        write_scaling_artifacts,
    )

    try:
        if args.methods == "all-compute":
            methods = DEFAULT_COMPUTE_METHODS
        else:
            methods = tuple(tuple(item.split(":", 1)) for item in args.methods.split(","))
            if any(len(item) != 2 or not all(item) for item in methods):
                raise ValueError(
                    "methods must be all-compute or comma-separated category:method pairs"
                )
        scales = tuple(int(item) for item in args.scales.split(","))
        if args.quick:
            scales = (200, 500)
            methods = methods if args.methods != "all-compute" else (("qc", "basic_qc"),)
        config = ScalingConfig(
            scales=scales,
            n_genes=min(args.genes, 250) if args.quick else args.genes,
            density=args.density,
            methods=methods,
            per_method_timeout_s=args.timeout,
            per_method_mem_cap_gb=args.mem_cap,
            seed=args.seed,
        )
        result = run_scaling(config)
        artifacts = write_scaling_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        _emit(f"error: {exc}", file=sys.stderr)
        return 2
    summary = result.summary()
    _emit(
        f"Scaling sweep: {len(result.records)} cells measured; "
        f"{summary['n_methods_reaching_max_scale']} methods reached {summary['max_scale']:,} cells."
    )
    for name, path in artifacts.items():
        _emit(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
