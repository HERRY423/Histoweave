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
from typing import TextIO

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
        choices=("figure3",),
        help="Run a predefined multi-dataset benchmark suite.",
    )
    p_bench.add_argument(
        "--out-dir",
        default="figure3_results",
        help="Artifact directory for --suite figure3.",
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
    p_step.add_argument("--in", dest="in_path", required=True, help="Input bundle directory.")
    p_step.add_argument("--out", required=True, help="Output bundle directory.")
    p_step.add_argument("--force", action="store_true", help="Replace an existing bundle.")
    p_step.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
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
    if args.command == "recommend":
        return _cmd_recommend(args)
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

    parser.print_help()
    return 0


def _cmd_version() -> int:
    from . import __version__

    _emit(f"histoweave {__version__}")
    return 0


def _cmd_list_methods(args) -> int:
    from .plugins import list_methods

    methods = list_methods(category=args.category, assay=args.assay)
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
        line += f" {m['summary']}"
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

    tasks = {
        "domain_detection": domain_detection_task,
        "deconvolution": deconvolution_task,
    }
    if args.task not in tasks:
        _emit(f"error: unknown task '{args.task}'. Available: {sorted(tasks)}", file=sys.stderr)
        return 2

    result = run_benchmark(tasks[args.task]())
    json_rows = []
    for row in result.leaderboard:
        json_row = dict(row)
        if json_row.get("score") in (float("inf"), float("-inf")):
            json_row["score"] = None
        json_rows.append(json_row)
    payload = {"schema_version": 1, "task": result.task, "leaderboard": json_rows}
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
        best = result.best()
        if best and best["score"] not in (float("inf"), float("-inf")):
            _emit(f"\nRecommended: {best['method']} (score {best['score']:.4f})")
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


def _cmd_figure3_benchmark(args) -> int:
    """Run the fixed 10-method x 3-dataset Figure 3 experiment."""
    from .benchmark import run_figure3_experiment

    try:
        result = run_figure3_experiment(args.out_dir, seed=args.seed)
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
    if args.out:
        _emit(f"Recommendation written to {Path(args.out).resolve()}")
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
            PipelineStep(args.category, args.method, params),
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
