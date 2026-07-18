"""Digital-twin synthetic validation — rank methods when real GT is absent.

Given a real, typically unlabelled spatial sample:

1. Build a feature-matched synthetic twin
   (:func:`~histoweave.datasets.digital_twin.make_digital_twin`).
2. Benchmark domain-detection methods on the twin against **planted** labels.
3. Return the twin leaderboard as the predicted ranking for the real sample,
   together with match quality, timings, and an optional HTML report.

This is *not* a claim that twin ARI equals real-world accuracy.  It is a
**proxy ranking** under the assumption that methods that recover planted structure
on a statistically matched twin are more likely to succeed on the real sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..data import SpatialTable
from ..datasets.digital_twin import (
    DIGITAL_TWIN_SCHEMA_VERSION,
    DigitalTwinResult,
    make_digital_twin,
)
from .harness import BenchmarkResult, domain_detection_task, run_benchmark

VALIDATION_SCHEMA_VERSION = 1


@dataclass
class DigitalTwinValidationResult:
    """Full output of :func:`run_digital_twin_validation`."""

    twin_result: DigitalTwinResult
    leaderboard: list[dict[str, Any]]
    predicted_ranking: list[str]
    task: str = "domain_detection"
    metric: str = "ARI"
    methods: list[str] = field(default_factory=list)
    dataset_name: str = "user_dataset"
    warnings: list[str] = field(default_factory=list)
    schema_version: int = VALIDATION_SCHEMA_VERSION
    benchmark_stats: dict[str, Any] | None = None

    def best_method(self) -> str | None:
        return self.predicted_ranking[0] if self.predicted_ranking else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digital_twin_schema_version": DIGITAL_TWIN_SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "task": self.task,
            "metric": self.metric,
            "methods": list(self.methods),
            "predicted_ranking": list(self.predicted_ranking),
            "best_method": self.best_method(),
            "leaderboard": list(self.leaderboard),
            "twin": self.twin_result.to_dict(),
            "match": self.twin_result.match.to_dict(),
            "warnings": list(self.warnings),
            "benchmark_stats": self.benchmark_stats,
        }

    def summary(self) -> str:
        lines = [
            f"Digital-twin validation for {self.dataset_name!r} [{self.task}]",
            f"  Twin match: L2={self.twin_result.match.match_l2:.4f}  "
            f"cosine={self.twin_result.match.match_cosine:.4f}",
            f"  Predicted ranking ({self.metric} on twin):",
        ]
        for row in self.leaderboard[:10]:
            score = row.get("score")
            score_s = f"{score:.4f}" if isinstance(score, int | float) and score > -1e100 else "n/a"
            lines.append(
                f"    {row.get('rank', '?'):>2}. {row.get('method', '?'):<24} "
                f"score={score_s}  t={row.get('seconds', 'n/a')}s"
            )
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def run_digital_twin_validation(
    data: SpatialTable,
    *,
    methods: list[str] | None = None,
    dataset_name: str = "user_dataset",
    seed: int = 0,
    n_domains: int | None = None,
    max_cells: int | None = 1500,
    max_genes: int | None = 500,
    n_trials: int = 12,
    k_policy: str = "oracle",
    allow_oracle_k: bool = True,
    stats: bool = False,
    out_dir: str | Path | None = None,
    write_report: bool = True,
) -> DigitalTwinValidationResult:
    """Generate a digital twin, benchmark methods on it, return predicted ranks.

    Parameters
    ----------
    data
        Real sample (ground truth optional / unused for ranking).
    methods
        Domain-detection methods to evaluate.  Default: harness release defaults.
    k_policy
        Domain-count policy on the **twin**.  Default ``oracle`` is safe here
        because the twin *has* planted labels; set ``estimate`` to simulate a
        fully blind k selection.
    out_dir
        When set, write ``twin_match.json``, ``leaderboard.json``, and
        optionally ``digital_twin_report.html``.
    """
    twin_pack = make_digital_twin(
        data,
        seed=seed,
        n_domains=n_domains,
        max_cells=max_cells,
        max_genes=max_genes,
        n_trials=n_trials,
    )
    twin = twin_pack.twin
    task = domain_detection_task(twin, truth_key="domain_truth")

    # On the twin, oracle k is scientifically justified (planted labels).
    if k_policy == "oracle" and not allow_oracle_k:
        allow_oracle_k = True

    bench: BenchmarkResult = run_benchmark(
        task,
        methods=methods,
        k_policy=k_policy,
        allow_oracle_k=allow_oracle_k,
        stats=stats,
        seed=seed,
    )

    leaderboard = [dict(row) for row in bench.leaderboard]
    # Sanitize non-JSON scores.
    for row in leaderboard:
        score = row.get("score")
        if score is not None and not _finite(score):
            row["score"] = None
            row.setdefault("error", row.get("error", "non-finite score"))

    ranking = [
        str(row["method"])
        for row in leaderboard
        if row.get("score") is not None and "error" not in row
    ]
    # Append failed methods at the end to keep a complete ranking.
    failed = [
        str(row["method"])
        for row in leaderboard
        if str(row["method"]) not in ranking
    ]
    predicted = ranking + failed

    warnings: list[str] = []
    if twin_pack.match.match_cosine < 0.85:
        warnings.append(
            f"Twin feature match is modest (cosine={twin_pack.match.match_cosine:.3f}). "
            "Interpret predicted rankings cautiously."
        )
    if twin_pack.match.match_l2 > 2.0:
        warnings.append(
            f"Twin feature L2 distance is high ({twin_pack.match.match_l2:.3f}). "
            "Consider increasing n_trials or max_cells."
        )
    if not ranking:
        warnings.append("No method produced a finite score on the twin.")

    result = DigitalTwinValidationResult(
        twin_result=twin_pack,
        leaderboard=leaderboard,
        predicted_ranking=predicted,
        methods=list(methods) if methods else [str(r["method"]) for r in leaderboard],
        dataset_name=dataset_name,
        warnings=warnings,
        benchmark_stats=bench.stats if isinstance(bench.stats, dict) else None,
    )

    if out_dir is not None:
        write_digital_twin_artifacts(result, out_dir, write_report=write_report)
    return result


def write_digital_twin_artifacts(
    result: DigitalTwinValidationResult,
    out_dir: str | Path,
    *,
    write_report: bool = True,
) -> dict[str, Path]:
    """Persist JSON artifacts and optional HTML report; return written paths."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    payload = result.to_dict()
    paths["validation"] = _write_json(root / "digital_twin_validation.json", payload)
    paths["match"] = _write_json(root / "twin_match.json", result.twin_result.match.to_dict())
    paths["leaderboard"] = _write_json(
        root / "leaderboard.json",
        {
            "predicted_ranking": result.predicted_ranking,
            "leaderboard": result.leaderboard,
            "metric": result.metric,
            "task": result.task,
        },
    )
    if write_report:
        from .digital_twin_report import build_digital_twin_report

        paths["report"] = build_digital_twin_report(result, root / "digital_twin_report.html")
    return paths


def _finite(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False, default=_json_default),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
