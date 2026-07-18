"""Automated spatial AutoML compiler.

Pipeline
--------
1. (Optional) Compile the natural-language question with :func:`histoweave.compiler.compile`
   to extract task intent and a proposed plan.
2. Extract target-free features and query :class:`MethodRecommender`.
3. Auto-run the top-*k* recommended domain methods (plus normalization prep).
4. Score each run with multi-objective proxies (spatial coherence, silhouette,
   consensus agreement, runtime).
5. Compute the Pareto front and emit a structured result + HTML report.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..benchmark.features import (
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)
from ..benchmark.recommend import MethodRecommender, Recommendation
from ..data import SpatialTable
from ..plugins import MethodCategory, create_method
from ..workflow import PipelineStep

AUTOML_SCHEMA_VERSION = 1


@dataclass
class MethodRunResult:
    """One method execution on the user sample."""

    method: str
    success: bool
    seconds: float | None
    n_domains: int | None = None
    spatial_coherence: float | None = None
    silhouette: float | None = None
    consensus_agreement: float | None = None
    quality_score: float | None = None  # higher is better scalar proxy
    error: str | None = None
    labels_key: str = "domain"
    recommendation_rank: int | None = None
    recommendation_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float) and not _finite(value):
                payload[key] = None
        return payload


@dataclass
class ParetoPoint:
    """One point on the multi-objective front."""

    method: str
    objectives: dict[str, float]  # all maximised after sign flip for runtime
    is_pareto: bool
    pareto_rank: int  # 1 = non-dominated front
    scalarised_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        cleaned_obj: dict[str, float] = {}
        for key, value in (payload.get("objectives") or {}).items():
            if isinstance(value, float) and _finite(value):
                cleaned_obj[key] = value
            elif isinstance(value, int | float) and _finite(float(value)):
                cleaned_obj[key] = float(value)
        payload["objectives"] = cleaned_obj
        if isinstance(payload.get("scalarised_score"), float) and not _finite(
            payload["scalarised_score"]
        ):
            payload["scalarised_score"] = None
        return payload


@dataclass
class AutoMLResult:
    """Complete AutoML compiler output."""

    question: str
    dataset_name: str
    task: str
    platform: str | None
    recommendation: dict[str, Any]
    neighbours: list[dict[str, Any]]
    feature_order: list[str]
    feature_vector: list[float | None]
    method_runs: list[MethodRunResult]
    pareto: list[ParetoPoint]
    ranked_methods: list[str]
    compiled_plan: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    schema_version: int = AUTOML_SCHEMA_VERSION
    label_columns: dict[str, str] = field(default_factory=dict)

    def best_method(self) -> str | None:
        return self.ranked_methods[0] if self.ranked_methods else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question": self.question,
            "dataset_name": self.dataset_name,
            "task": self.task,
            "platform": self.platform,
            "recommendation": self.recommendation,
            "neighbours": self.neighbours,
            "feature_order": self.feature_order,
            "feature_vector": self.feature_vector,
            "method_runs": [m.to_dict() for m in self.method_runs],
            "pareto": [p.to_dict() for p in self.pareto],
            "ranked_methods": list(self.ranked_methods),
            "best_method": self.best_method(),
            "compiled_plan": self.compiled_plan,
            "warnings": list(self.warnings),
            "label_columns": dict(self.label_columns),
        }

    def summary(self) -> str:
        lines = [
            f"Spatial AutoML for {self.dataset_name!r}",
            f"  Question: {self.question!r}",
            f"  Task={self.task}  platform={self.platform or 'unknown'}",
            "  Neighbours: "
            + ", ".join(
                f"{n.get('name')}({n.get('similarity', 0):.2f})" for n in self.neighbours[:5]
            ),
            "  Pareto-ranked methods:",
        ]
        for i, name in enumerate(self.ranked_methods, 1):
            run = next((r for r in self.method_runs if r.method == name), None)
            if run is None:
                lines.append(f"    {i}. {name}")
                continue
            lines.append(
                f"    {i}. {name:<24} quality={_fmt(run.quality_score)}  "
                f"coherence={_fmt(run.spatial_coherence)}  "
                f"silhouette={_fmt(run.silhouette)}  "
                f"t={_fmt(run.seconds)}s"
            )
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def run_spatial_automl(
    data: SpatialTable,
    question: str = "Find spatial domains for my data.",
    *,
    knowledge_base: str | Path | Any,
    dataset_name: str = "user_dataset",
    top_k: int = 3,
    k_neighbours: int = 3,
    methods: list[str] | None = None,
    n_domains: int | None = None,
    use_compiler: bool = True,
    compiler_model: str = "mock",
    seed: int = 0,
    out_dir: str | Path | None = None,
    write_report: bool = True,
    platform: str | None = None,
) -> AutoMLResult:
    """Run the full spatial AutoML compiler loop.

    Parameters
    ----------
    data
        User sample (no ground truth required).
    question
        Natural-language analysis request, e.g.
        ``"Find spatial domains for my Visium liver cancer data."``
    knowledge_base
        Landscape JSON path or :class:`~histoweave.benchmark.landscape.LandscapeResult`.
    top_k
        Number of recommended methods to auto-run (default 3).
    use_compiler
        When *True*, also compile ``question`` via the LLM compiler (mock by
        default) and attach the plan for auditability.  Recommendation still
        drives method selection.
    methods
        Optional explicit method list (overrides recommender top-k).
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    question = question.strip()

    warnings: list[str] = []
    compiled_plan: dict[str, Any] | None = None
    if use_compiler:
        try:
            from ..compiler import compile as compile_question

            plan = compile_question(
                question,
                data=data,
                provider=compiler_model,
                dry_run=True,
                max_repair_attempts=1,
            )
            compiled_plan = plan.to_dict()
        except Exception as exc:  # compiler is advisory here
            warnings.append(f"Compiler plan unavailable: {type(exc).__name__}: {exc}")

    recommender = MethodRecommender(knowledge_base, k_neighbours=k_neighbours)
    rec: Recommendation = recommender.recommend(
        data,
        dataset_name=dataset_name,
        platform=platform,
    )
    warnings.extend(rec.warnings)

    if methods is None:
        selected = [m.method for m in rec.top(top_k)]
        # Strip @policy suffixes for execution when present.
        selected = [_base_method(name) for name in selected]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        selected_unique: list[str] = []
        for name in selected:
            if name not in seen:
                seen.add(name)
                selected_unique.append(name)
        selected = selected_unique
    else:
        selected = list(methods)

    if not selected:
        warnings.append("No methods selected; falling back to kmeans.")
        selected = ["kmeans"]

    rec_score_map = { _base_method(m.method): m.score for m in rec.ranked_methods }
    rec_rank_map = {
        _base_method(m.method): i for i, m in enumerate(rec.ranked_methods, start=1)
    }

    # Estimate k once for all methods.
    estimated_k = n_domains
    if estimated_k is None:
        try:
            from ..benchmark.k_selection import estimate_n_domains

            estimated_k = estimate_n_domains(data, random_state=seed).k
        except Exception:
            estimated_k = int(data.uns.get("n_domains") or 4)
            warnings.append(
                f"Domain-count estimation failed; using n_domains={estimated_k}."
            )

    prepared = _normalize(data)
    runs: list[MethodRunResult] = []
    label_map: dict[str, np.ndarray] = {}
    label_columns: dict[str, str] = {}

    for name in selected:
        run = _run_domain_method(
            prepared,
            method=name,
            n_domains=int(estimated_k),
            seed=seed,
            recommendation_rank=rec_rank_map.get(name),
            recommendation_score=rec_score_map.get(name),
        )
        runs.append(run)
        if run.success and name in prepared.obs.columns:
            # Labels stored under method-specific column after run helper.
            pass
        col = f"domain_{name}"
        if col in prepared.obs.columns:
            label_map[name] = prepared.obs[col].to_numpy()
            label_columns[name] = col

    # Consensus agreement (pairwise ARI) when ≥2 methods succeed.
    _attach_consensus(runs, label_map)
    _attach_quality(runs)

    pareto = compute_pareto_front(runs)
    ranked = [p.method for p in sorted(pareto, key=lambda p: (p.pareto_rank, -p.scalarised_score))]

    feats = extract_features(data, include_domain=False)
    order = list(RECOMMENDATION_FEATURE_ORDER)
    vec = feature_vector(feats, order=order).tolist()

    result = AutoMLResult(
        question=question,
        dataset_name=dataset_name,
        task=rec.task,
        platform=rec.platform,
        recommendation=rec.to_dict(),
        neighbours=list(rec.neighbours),
        feature_order=order,
        feature_vector=[float(x) if _finite(x) else None for x in vec],  # type: ignore[misc]
        method_runs=runs,
        pareto=pareto,
        ranked_methods=ranked,
        compiled_plan=compiled_plan,
        warnings=warnings,
        label_columns=label_columns,
    )

    if out_dir is not None:
        write_automl_artifacts(result, prepared, out_dir, write_report=write_report)
    return result


def compute_pareto_front(runs: list[MethodRunResult]) -> list[ParetoPoint]:
    """Multi-objective ranking: maximise quality proxies, minimise runtime.

    Objectives (all oriented so higher is better):
    - ``quality`` — composite of spatial coherence, silhouette, consensus
    - ``speed`` — ``1 / (1 + seconds)``
    - ``recommendation`` — landscape recommendation score (when available)
    """
    successful = [r for r in runs if r.success and r.quality_score is not None]
    if not successful:
        return [
            ParetoPoint(
                method=r.method,
                objectives={},
                is_pareto=False,
                pareto_rank=99,
                scalarised_score=float("-inf"),
            )
            for r in runs
        ]

    objectives: dict[str, dict[str, float]] = {}
    for r in successful:
        seconds = float(r.seconds) if r.seconds is not None else 1e6
        rec = float(r.recommendation_score) if r.recommendation_score is not None else 0.0
        objectives[r.method] = {
            "quality": float(r.quality_score or 0.0),
            "speed": 1.0 / (1.0 + max(seconds, 0.0)),
            "recommendation": rec,
        }

    names = list(objectives)
    obj_keys = ("quality", "speed", "recommendation")
    mats = np.array([[objectives[n][k] for k in obj_keys] for n in names])
    # Non-dominated sorting (higher is better on each objective).
    ranks = _non_dominated_ranks(mats)
    points: list[ParetoPoint] = []
    for i, name in enumerate(names):
        # Scalarisation: equal weights on z-scored objectives.
        z = (mats[i] - mats.mean(axis=0)) / (mats.std(axis=0) + 1e-8)
        scalar = float(z.mean())
        points.append(
            ParetoPoint(
                method=name,
                objectives=dict(objectives[name]),
                is_pareto=bool(ranks[i] == 1),
                pareto_rank=int(ranks[i]),
                scalarised_score=scalar,
            )
        )

    # Include failed methods at the bottom.
    present = {p.method for p in points}
    for r in runs:
        if r.method not in present:
            points.append(
                ParetoPoint(
                    method=r.method,
                    objectives={},
                    is_pareto=False,
                    pareto_rank=99,
                    scalarised_score=float("-inf"),
                )
            )
    return points


def write_automl_artifacts(
    result: AutoMLResult,
    data: SpatialTable | None,
    out_dir: str | Path,
    *,
    write_report: bool = True,
) -> dict[str, Path]:
    """Persist AutoML JSON + HTML artifacts."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["result"] = _write_json(root / "automl_result.json", result.to_dict())
    paths["pareto"] = _write_json(
        root / "pareto_front.json",
        {"ranked_methods": result.ranked_methods, "pareto": [p.to_dict() for p in result.pareto]},
    )
    if write_report:
        from .report import build_automl_report

        paths["report"] = build_automl_report(result, root / "automl_report.html", data=data)
    return paths


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _base_method(name: str) -> str:
    return name.split("@", 1)[0].strip()


def _normalize(data: SpatialTable) -> SpatialTable:
    try:
        step = PipelineStep(MethodCategory.NORMALIZATION, "log1p_cp10k")
        method = create_method(step.category, step.method, **step.params)
        return method.run(data.copy())
    except Exception:
        return data.copy()


def _run_domain_method(
    data: SpatialTable,
    *,
    method: str,
    n_domains: int,
    seed: int,
    recommendation_rank: int | None,
    recommendation_score: float | None,
) -> MethodRunResult:
    col = f"domain_{method}"
    t0 = time.perf_counter()
    try:
        probe = create_method(MethodCategory.DOMAIN_DETECTION, method)
        params: dict[str, Any] = {}
        if "n_domains" in probe.params:
            params["n_domains"] = n_domains
        if "random_state" in probe.params:
            params["random_state"] = seed
        if "key_added" in probe.params:
            params["key_added"] = col
        else:
            col = "domain"

        runner = create_method(MethodCategory.DOMAIN_DETECTION, method, **params)
        result = runner.run(data.copy())
        seconds = time.perf_counter() - t0

        if col in result.obs.columns:
            labels = result.obs[col].to_numpy()
        elif "domain" in result.obs.columns:
            labels = result.obs["domain"].to_numpy()
            col = f"domain_{method}"
        else:
            raise RuntimeError(f"{method} did not write domain labels")

        # Store labels on the shared working table for consensus comparison.
        data.obs[col] = np.asarray(labels)

        n_dom = int(len({str(x) for x in labels}))
        coherence = _spatial_coherence(result.spatial, labels)
        sil = _spatial_expression_silhouette(result, labels, seed=seed)
        return MethodRunResult(
            method=method,
            success=True,
            seconds=round(float(seconds), 4),
            n_domains=n_dom,
            spatial_coherence=coherence,
            silhouette=sil,
            recommendation_rank=recommendation_rank,
            recommendation_score=recommendation_score,
            labels_key=col,
        )
    except Exception as exc:
        seconds = time.perf_counter() - t0
        return MethodRunResult(
            method=method,
            success=False,
            seconds=round(float(seconds), 4),
            error=f"{type(exc).__name__}: {exc}",
            recommendation_rank=recommendation_rank,
            recommendation_score=recommendation_score,
        )


def _spatial_coherence(coords: np.ndarray | None, labels: np.ndarray, k: int = 8) -> float:
    """Fraction of kNN neighbours that share the same label (higher = smoother)."""
    if coords is None or len(labels) < 3:
        return float("nan")
    from .._math import knn_indices

    coords = np.asarray(coords, dtype=float)
    n = len(labels)
    kk = min(k, n - 1)
    if kk < 1:
        return float("nan")
    nbrs = knn_indices(coords, kk + 1)
    labels = np.asarray(labels)
    agree = 0
    total = 0
    for i in range(n):
        row = nbrs[i]
        for j in row:
            if j == i:
                continue
            total += 1
            if str(labels[j]) == str(labels[i]):
                agree += 1
    return float(agree / total) if total else float("nan")


def _spatial_expression_silhouette(
    data: SpatialTable,
    labels: np.ndarray,
    *,
    seed: int,
    n_pcs: int = 10,
) -> float:
    """Silhouette on PCA features (expression); higher is better, range [-1, 1]."""
    labels = np.asarray(labels)
    unique = np.unique(labels.astype(str))
    if len(unique) < 2 or data.n_obs < 10:
        return float("nan")
    try:
        from .._math import pca, zscore

        X = np.asarray(data.X, dtype=float)
        if X.ndim != 2:
            return float("nan")
        # Subsample for speed on large tables.
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        if n > 800:
            idx = rng.choice(n, size=800, replace=False)
            X = X[idx]
            labels = labels[idx]
        feats = zscore(X)
        n_comp = min(n_pcs, feats.shape[0] - 1, feats.shape[1])
        if n_comp < 2:
            return float("nan")
        scores = pca(feats, n_comp, random_state=seed)
        return _silhouette_numpy(scores, labels.astype(str))
    except Exception:
        return float("nan")


def _silhouette_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient without sklearn."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")
    n = X.shape[0]
    # Pairwise squared Euclidean (adequate ranking signal).
    # For n<=800 this is fine.
    gram = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    dist = np.sqrt(np.maximum(sq + sq.T - 2 * gram, 0.0))
    sil = np.zeros(n, dtype=float)
    for i in range(n):
        same = labels == labels[i]
        # a: mean intra-cluster distance
        same_idx = np.where(same)[0]
        if len(same_idx) <= 1:
            sil[i] = 0.0
            continue
        a = float(dist[i, same_idx].sum() / (len(same_idx) - 1))
        # b: mean distance to nearest other cluster
        b = float("inf")
        for c in classes:
            if c == labels[i]:
                continue
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            b = min(b, float(dist[i, idx].mean()))
        if not np.isfinite(b):
            sil[i] = 0.0
            continue
        denom = max(a, b)
        sil[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(np.mean(sil))


def _attach_consensus(runs: list[MethodRunResult], label_map: dict[str, np.ndarray]) -> None:
    if len(label_map) < 2:
        for r in runs:
            if r.success:
                r.consensus_agreement = 1.0  # single method: trivial consensus
        return
    from .._math import adjusted_rand_index

    names = list(label_map)
    for name in names:
        aris: list[float] = []
        for other in names:
            if other == name:
                continue
            try:
                aris.append(float(adjusted_rand_index(label_map[name], label_map[other])))
            except Exception:
                continue
        mean_ari = float(np.mean(aris)) if aris else float("nan")
        for r in runs:
            if r.method == name:
                r.consensus_agreement = mean_ari


def _attach_quality(runs: list[MethodRunResult]) -> None:
    """Composite quality in [0, 1] from coherence, silhouette, consensus."""
    for r in runs:
        if not r.success:
            r.quality_score = None
            continue
        parts: list[float] = []
        if r.spatial_coherence is not None and _finite(r.spatial_coherence):
            parts.append(float(np.clip(r.spatial_coherence, 0.0, 1.0)))
        if r.silhouette is not None and _finite(r.silhouette):
            parts.append(float(np.clip((r.silhouette + 1.0) / 2.0, 0.0, 1.0)))
        if r.consensus_agreement is not None and _finite(r.consensus_agreement):
            # ARI can be slightly negative.
            parts.append(float(np.clip((r.consensus_agreement + 1.0) / 2.0, 0.0, 1.0)))
        r.quality_score = float(np.mean(parts)) if parts else 0.0


def _non_dominated_ranks(
    objectives: np.ndarray,
    *,
    sig_digits: int = 4,
) -> np.ndarray:
    """Non-dominated sorting ranks (1 = Pareto front). Higher objective is better.

    Parameters
    ----------
    sig_digits
        Significant digits to retain before dominance comparison.  Raw objective
        values can differ by machine-epsilon-level noise (e.g. two methods with
        identical quality but runtimes differing by <1e-12 s); rounding to
        *sig_digits* makes the front numerically stable without changing
        the biological interpretation.
    """
    n = objectives.shape[0]
    ranks = np.zeros(n, dtype=int)
    remaining = set(range(n))
    current_rank = 1
    while remaining:
        front: list[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if _dominates(objectives[j], objectives[i], sig_digits=sig_digits):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        if not front:
            # Numerical stalemate — assign remaining together.
            for i in remaining:
                ranks[i] = current_rank
            break
        for i in front:
            ranks[i] = current_rank
            remaining.discard(i)
        current_rank += 1
    return ranks


def _dominates(a: np.ndarray, b: np.ndarray, *, sig_digits: int = 4) -> bool:
    """Return True if *a* Pareto-dominates *b* (all ``≥``, one ``>``).

    Objective values are rounded to *sig_digits* significant digits before
    comparison so that machine-epsilon noise (e.g. runtimes within
    femtoseconds) does not create spurious dominance edges.  Four digits
    means 0.1234 and 0.12345 are treated as equal — well below any
    biological or operational threshold.
    """
    a_rounded = _round_significant(a, sig_digits)
    b_rounded = _round_significant(b, sig_digits)
    return bool(np.all(a_rounded >= b_rounded) and np.any(a_rounded > b_rounded))


def _round_significant(values: np.ndarray, digits: int) -> np.ndarray:
    """Round *values* to *digits* significant decimal digits element-wise."""
    out = np.empty_like(values, dtype=float)
    for idx in range(values.size):
        v = float(values.flat[idx])
        if v == 0.0 or not np.isfinite(v):
            out.flat[idx] = v
            continue
        try:
            scale = int(np.floor(np.log10(abs(v))))
        except (ValueError, OverflowError):
            out.flat[idx] = v
            continue
        factor = 10.0 ** (digits - 1 - scale)
        out.flat[idx] = float(np.round(v * factor) / factor)
    return out


def _finite(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: float | None) -> str:
    if value is None or not _finite(value):
        return "n/a"
    return f"{float(value):.3f}"


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
