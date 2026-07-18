"""Method recommendation engine v2 — selection under uncertainty.

HistoWeave does **not** claim a universal best method.  Given a new dataset the
recommender:

1. Extracts a *target-free* feature vector (no domain / cell-type labels).
2. Finds the *k* most similar reference datasets (cosine similarity).
3. Applies **task** and **platform** priors so spatial-domain evidence is not
   transferred onto cell-type problems (and vice versa).
4. Ranks ``method`` or ``method@policy`` configurations by similarity-weighted
   performance, reporting regret vs a global-best baseline.
5. Surfaces applicability warnings when the knowledge base is too narrow or the
   recommender fails to beat strong defaults.

Negative results are first-class outputs: if the engine cannot beat
always-pick-global-best, the recommendation says so instead of overselling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..data import SpatialTable
from .features import (
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)
from .landscape import LandscapeResult
from .task_contract import (
    AnalysisTask,
    classify_platform,
    default_spatial_context_policy,
    split_method_policy,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class MethodScore:
    """One ranked method (or method@policy) with recommendation metadata."""

    method: str
    score: float  # similarity-weighted mean metric (higher = better by default)
    confidence: float  # 0‑1 support × coverage × effective-n factor
    wins: int  # how many neighbour datasets this config won
    neighbour_scores: dict[str, float]  # dataset_name → score on that neighbour

    uncertainty: float = 0.0
    support: int = 0
    coverage: float = 0.0
    base_method: str = ""
    spatial_context_policy: str | None = None
    prior_boost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    """Complete recommendation for one task on one dataset."""

    task: str
    dataset_name: str  # caller-supplied label (or "user_dataset")
    ranked_methods: list[MethodScore]
    neighbours: list[dict[str, Any]]  # reference datasets that informed the rec
    feature_vector: list[float] = field(default_factory=list)
    ensemble_strategy: str = ""

    feature_order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    platform: str | None = None
    spatial_context_policy: str | None = None
    # Diagnostics vs strong defaults (recommend only when useful).
    global_best_method: str | None = None
    global_best_score: float | None = None
    selection_regret_vs_oracle_neighbours: float | None = None
    selection_regret_vs_global_best: float | None = None
    beats_global_best_baseline: bool | None = None
    # Active-learning calibration (filled when global-best is not beaten).
    evidence_todo: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, Any] | None = None
    schema_version: int = 3

    def top(self, n: int = 3) -> list[MethodScore]:
        return self.ranked_methods[:n]

    def best(self) -> MethodScore | None:
        return self.ranked_methods[0] if self.ranked_methods else None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "task": self.task,
            "dataset_name": self.dataset_name,
            "platform": self.platform,
            "spatial_context_policy": self.spatial_context_policy,
            "ranked_methods": [m.to_dict() for m in self.ranked_methods],
            "neighbours": _json_safe(self.neighbours),
            "feature_order": self.feature_order,
            "feature_vector": _json_safe(self.feature_vector),
            "warnings": self.warnings,
            "ensemble_suggestion": self.ensemble_strategy,
            "baselines": {
                "global_best_method": self.global_best_method,
                "global_best_score": self.global_best_score,
                "selection_regret_vs_oracle_neighbours": (
                    self.selection_regret_vs_oracle_neighbours
                ),
                "selection_regret_vs_global_best": self.selection_regret_vs_global_best,
                "beats_global_best_baseline": self.beats_global_best_baseline,
            },
            "evidence_todo": list(self.evidence_todo),
            "calibration": self.calibration,
        }
        return payload

    def summary(self) -> str:
        lines = [
            f"Recommendation for {self.dataset_name!r} "
            f"[task={self.task}, platform={self.platform or 'unknown'}]:"
        ]
        for i, m in enumerate(self.top(3), 1):
            policy = m.spatial_context_policy or "default"
            lines.append(
                f"  {i}. {m.method:<28} score={m.score:.3f}  "
                f"policy={policy}  uncertainty={m.uncertainty:.3f}  "
                f"support={m.support}/{len(self.neighbours)}"
            )
        if self.global_best_method is not None:
            flag = (
                "yes"
                if self.beats_global_best_baseline
                else "no — prefer global default or ensemble"
            )
            lines.append(
                f"  Global-best baseline: {self.global_best_method} "
                f"(score={self.global_best_score}); beats baseline: {flag}"
            )
        if self.ensemble_strategy:
            lines.append(f"  Ensemble suggestion: {self.ensemble_strategy}")
        if self.warnings:
            lines.append("  Warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        if self.neighbours:
            nbr_names = [n["name"] for n in self.neighbours]
            lines.append(f"  Based on: {', '.join(nbr_names)}")
        return "\n".join(lines)


class MethodRecommender:
    """Recommend methods / configurations under explicit task and platform priors.

    Parameters
    ----------
    knowledge_base
        A :class:`LandscapeResult` from a prior landscape run, or a JSON path.
    k_neighbours
        Number of nearest reference datasets to consult (default 3).
    platform_prior_weight
        Multiplier applied to neighbour similarity when platforms match
        (default 1.35).  Values near 1.0 disable the prior.
    task_prior_weight
        Multiplier when reference task matches the query task (default 1.5).
    """

    def __init__(
        self,
        knowledge_base: LandscapeResult | str | Path,
        *,
        k_neighbours: int = 3,
        platform_prior_weight: float = 1.35,
        task_prior_weight: float = 1.5,
    ) -> None:
        if k_neighbours < 1:
            raise ValueError("k_neighbours must be at least 1")
        if platform_prior_weight < 1.0 or task_prior_weight < 1.0:
            raise ValueError("prior weights must be >= 1.0")
        if isinstance(knowledge_base, str | Path):
            knowledge_base = _load_knowledge_base(Path(knowledge_base))
        self._kb = knowledge_base
        self._k = int(k_neighbours)
        self._platform_prior_weight = float(platform_prior_weight)
        self._task_prior_weight = float(task_prior_weight)
        raw_task = str(getattr(knowledge_base, "task", "spatial_domain"))
        # Accept legacy task names from older knowledge bases.
        if raw_task in {"domain_detection", "domain"}:
            raw_task = AnalysisTask.SPATIAL_DOMAIN.value
        self._task = raw_task

        source_order = list(self._kb.feature_order or RECOMMENDATION_FEATURE_ORDER)
        keep = [
            index
            for index, name in enumerate(source_order)
            if name
            not in {
                "n_domains",
                "domain_balance",
                "domain_spatial_coherence",
            }
        ]
        self._feature_order = [source_order[index] for index in keep]
        if not self._feature_order:
            raise ValueError("knowledge base contains no target-free features")

        self._dataset_meta: dict[str, dict[str, Any]] = {
            str(name): dict(meta) for name, meta in getattr(self._kb, "dataset_meta", {}).items()
        }

        self._ref_features: dict[str, np.ndarray] = {}
        vectors: list[np.ndarray] = []
        for name in self._kb.dataset_order():
            vector = self._kb.features.get(name)
            if vector is None:
                continue
            vector = np.asarray(vector, dtype=float)
            if vector.ndim != 1 or len(vector) != len(source_order):
                raise ValueError(
                    f"feature vector for {name!r} has length {vector.size}; "
                    f"expected {len(source_order)}"
                )
            projected = vector[keep]
            vectors.append(projected)
            self._ref_features[name] = projected
        self._ref_names = list(self._ref_features.keys())
        if len(vectors) < 2:
            raise ValueError("knowledge base requires at least two datasets with features")
        raw_reference = np.asarray(vectors, dtype=float)
        self._impute, self._mean, self._std = _fit_feature_space(raw_reference)
        self._ref_matrix = _transform_feature_space(
            raw_reference, self._impute, self._mean, self._std
        )
        self._global_best_method, self._global_best_score = _global_best_baseline(self._kb)

    # ------------------------------------------------------------------
    def recommend(
        self,
        data: SpatialTable,
        *,
        dataset_name: str = "user_dataset",
        task: str | AnalysisTask | None = None,
        platform: str | None = None,
        spatial_context_policy: str | None = None,
    ) -> Recommendation:
        """Produce a ranked configuration recommendation for *data*.

        Parameters
        ----------
        task
            Analysis target (``spatial_domain`` or ``cell_type``).  Defaults to
            the knowledge-base task.
        platform
            Assay platform prior (``visium``, ``xenium``, ``merfish``, …).
            Also read from ``data.uns['platform']`` / ``data.uns['assay']``.
        spatial_context_policy
            Preferred spatial-context policy label (``off`` / ``default`` /
            ``high``, or ``sw0.0``-style keys).  Used to re-rank matching
            ``method@policy`` configurations.
        """
        query_task = str(task) if task is not None else self._task
        if query_task in {"domain_detection", "domain"}:
            query_task = AnalysisTask.SPATIAL_DOMAIN.value
        query_platform = classify_platform(
            platform
            or data.uns.get("platform")
            or data.uns.get("assay")
            or data.uns.get("technology")
        )
        policy = spatial_context_policy or default_spatial_context_policy(query_task)

        # Query features permanently exclude domain labels (no target leakage).
        feats = extract_features(data, include_domain=False)
        vec = feature_vector(feats, order=self._feature_order)
        vec_2d = vec.reshape(1, -1)
        vec_std = _transform_feature_space(vec_2d, self._impute, self._mean, self._std)

        neighbours = self._find_neighbours(
            vec_std.ravel(),
            query_task=query_task,
            query_platform=query_platform,
        )
        ranked = self._rank_methods(
            neighbours,
            preferred_policy=policy,
            query_task=query_task,
        )
        warnings = self._applicability_warnings(
            neighbours, ranked, query_task=query_task, query_platform=query_platform
        )
        diagnostics = self._baseline_diagnostics(ranked, neighbours)
        if diagnostics.get("beats_global_best_baseline") is False:
            warnings.append(
                "Neighbour-weighted selection does not beat the global-best "
                f"baseline ({self._global_best_method}). Prefer the global "
                "default, or run a short multi-method ensemble and inspect "
                "disagreement regions."
            )
        ensemble = _ensemble_strategy(ranked, neighbours)

        recommendation = Recommendation(
            task=query_task,
            dataset_name=dataset_name,
            ranked_methods=ranked,
            neighbours=neighbours,
            feature_vector=vec.tolist(),
            ensemble_strategy=ensemble,
            feature_order=list(self._feature_order),
            warnings=warnings,
            platform=query_platform,
            spatial_context_policy=policy,
            global_best_method=self._global_best_method,
            global_best_score=self._global_best_score,
            selection_regret_vs_oracle_neighbours=diagnostics.get(
                "selection_regret_vs_oracle_neighbours"
            ),
            selection_regret_vs_global_best=diagnostics.get("selection_regret_vs_global_best"),
            beats_global_best_baseline=diagnostics.get("beats_global_best_baseline"),
        )
        # Active-learning calibration: when personalisation fails to beat the
        # global-best baseline, propose dataset×method pairs that maximise EIG.
        try:
            from .active_calibration import attach_calibration

            attach_calibration(self, recommendation, top_n=10, always=False)
        except Exception as exc:  # calibration is advisory — never fail recommend
            recommendation.warnings.append(
                f"Active calibration unavailable: {type(exc).__name__}: {exc}"
            )
        return recommendation

    # ------------------------------------------------------------------
    def _find_neighbours(
        self,
        query_vec: np.ndarray,
        *,
        query_task: str,
        query_platform: str | None,
    ) -> list[dict[str, Any]]:
        """Return the *k* nearest reference datasets with prior-adjusted weights."""
        if self._ref_matrix is None:
            return []

        cosine = _cosine_similarity(query_vec, self._ref_matrix)
        similarities = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

        adjusted = similarities.copy()
        meta_rows: list[dict[str, Any]] = []
        for i, name in enumerate(self._ref_names):
            meta = self._dataset_meta.get(name, {})
            ref_platform = classify_platform(meta.get("platform") or meta.get("assay"))
            ref_task = str(meta.get("task") or self._task)
            if ref_task in {"domain_detection", "domain"}:
                ref_task = AnalysisTask.SPATIAL_DOMAIN.value
            boost = 1.0
            if query_platform and ref_platform and query_platform == ref_platform:
                boost *= self._platform_prior_weight
            if query_task and ref_task and query_task == ref_task:
                boost *= self._task_prior_weight
            # Penalise cross-task neighbours hard: domain ↔ cell_type transfer is unsafe.
            if query_task and ref_task and query_task != ref_task:
                boost *= 0.25
            adjusted[i] = min(1.0, float(similarities[i]) * boost)
            meta_rows.append(
                {
                    "platform": ref_platform,
                    "task": ref_task,
                    "ground_truth_kind": meta.get("ground_truth_kind"),
                    "prior_boost": round(boost, 4),
                }
            )

        top_k = np.argsort(adjusted, kind="stable")[::-1][: min(self._k, len(self._ref_names))]

        neighbours: list[dict[str, Any]] = []
        for idx in top_k:
            sim = float(adjusted[idx])
            name = self._ref_names[idx]
            row = {
                "name": name,
                "similarity": round(sim, 4),
                "raw_similarity": round(float(similarities[idx]), 4),
                "distance": round(1.0 - sim, 4),
                "best_method": self._kb.best_method.get(name, "?"),
            }
            row.update(meta_rows[idx])
            neighbours.append(row)
        return neighbours

    # ------------------------------------------------------------------
    def _rank_methods(
        self,
        neighbours: list[dict[str, Any]],
        *,
        preferred_policy: str | None,
        query_task: str,
    ) -> list[MethodScore]:
        """Rank methods / method@policy configurations with policy preference."""
        if not neighbours:
            return []

        method_names = self._kb.method_order()
        wins: dict[str, int] = {m: 0 for m in method_names}
        for nbr in neighbours:
            best = self._kb.best_method.get(nbr["name"])
            if best in wins:
                wins[best] += 1

        total_weight = sum(float(nbr["similarity"]) for nbr in neighbours)
        preferred_tokens = _policy_tokens(preferred_policy)
        ranked: list[MethodScore] = []
        for method in method_names:
            values: list[float] = []
            weights: list[float] = []
            evidence: dict[str, float] = {}
            for nbr in neighbours:
                name = nbr["name"]
                value = self._kb.performance.get(name, {}).get(method, float("nan"))
                if value is None or not np.isfinite(value):
                    continue
                values.append(float(value))
                weights.append(float(nbr["similarity"]))
                evidence[name] = float(value)
            if not values:
                continue

            value_array: np.ndarray = np.asarray(values, dtype=float)
            weight_array: np.ndarray = np.asarray(weights, dtype=float)
            valid_weight = float(weight_array.sum())
            score = float(np.average(value_array, weights=weight_array))
            variance = float(np.average((value_array - score) ** 2, weights=weight_array))
            effective_n = valid_weight**2 / max(float(np.sum(weight_array**2)), 1e-12)
            uncertainty = float(np.sqrt(variance / max(effective_n, 1.0)))
            coverage = valid_weight / max(total_weight, 1e-12)
            confidence = float(np.mean(weight_array)) * coverage * min(1.0, effective_n / 2.0)
            base_method, policy = split_method_policy(method)
            prior_boost = 0.0
            if preferred_tokens and policy:
                if any(token in policy.lower() for token in preferred_tokens):
                    prior_boost = 0.02
                    score += prior_boost
            elif preferred_tokens and policy is None:
                # Bare method names inherit the task default policy softly.
                if query_task == AnalysisTask.SPATIAL_DOMAIN.value and "high" in preferred_tokens:
                    prior_boost = 0.005
                    score += prior_boost

            ranked.append(
                MethodScore(
                    method=method,
                    score=round(score, 4),
                    confidence=round(min(confidence, 1.0), 4),
                    wins=wins.get(method, 0),
                    neighbour_scores=evidence,
                    uncertainty=round(uncertainty, 4),
                    support=len(values),
                    coverage=round(coverage, 4),
                    base_method=base_method,
                    spatial_context_policy=policy,
                    prior_boost=round(prior_boost, 4),
                )
            )

        reverse = bool(getattr(self._kb, "higher_is_better", True))
        ranked.sort(key=lambda item: item.score, reverse=reverse)
        return ranked

    def _applicability_warnings(
        self,
        neighbours: list[dict[str, Any]],
        ranked: list[MethodScore],
        *,
        query_task: str,
        query_platform: str | None,
    ) -> list[str]:
        warnings: list[str] = []
        if len(self._ref_names) < 8:
            warnings.append(
                f"Knowledge base has only {len(self._ref_names)} reference datasets; "
                "recommendations are indicative, not generalisation claims."
            )
        if query_platform and not any(nbr.get("platform") == query_platform for nbr in neighbours):
            warnings.append(
                f"No reference neighbour shares platform={query_platform!r}; "
                "platform transfer is unreliable."
            )
        mismatched = [
            nbr["name"] for nbr in neighbours if nbr.get("task") and nbr.get("task") != query_task
        ]
        if mismatched:
            warnings.append(
                "Some neighbours come from a different analysis task "
                f"({mismatched}); their weight was down-scaled."
            )
        if not ranked:
            warnings.append("No method configurations had finite scores on neighbours.")
        # Detect possible self-supervised GT contamination in the KB.
        for nbr in neighbours:
            kind = str(nbr.get("ground_truth_kind") or "").lower()
            if kind in {"self_supervised", "leiden", "louvain"}:
                warnings.append(
                    f"Neighbour {nbr['name']!r} uses ground_truth_kind={kind!r}, "
                    "which is invalid for spatial_domain evaluation."
                )
        return warnings

    def _baseline_diagnostics(
        self,
        ranked: list[MethodScore],
        neighbours: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not ranked or not neighbours:
            return {}
        higher = bool(getattr(self._kb, "higher_is_better", True))
        chosen = ranked[0].method
        # Oracle on neighbours: mean of each neighbour's best method score.
        oracle_values = []
        chosen_values = []
        global_values = []
        for nbr in neighbours:
            row = self._kb.performance.get(nbr["name"], {})
            finite = {
                method: float(value)
                for method, value in row.items()
                if value is not None and np.isfinite(value)
            }
            if not finite:
                continue
            oracle = max(finite.values()) if higher else min(finite.values())
            oracle_values.append(oracle)
            if chosen in finite:
                chosen_values.append(finite[chosen])
            if self._global_best_method and self._global_best_method in finite:
                global_values.append(finite[self._global_best_method])
        if not oracle_values or not chosen_values:
            return {
                "beats_global_best_baseline": None,
                "selection_regret_vs_oracle_neighbours": None,
                "selection_regret_vs_global_best": None,
            }
        chosen_mean = float(np.mean(chosen_values))
        oracle_mean = float(np.mean(oracle_values))
        regret_oracle = oracle_mean - chosen_mean if higher else chosen_mean - oracle_mean
        if global_values:
            global_mean = float(np.mean(global_values))
            regret_global = global_mean - chosen_mean if higher else chosen_mean - global_mean
            beats = regret_global < -1e-12  # strictly better than global-best
        else:
            regret_global = None
            beats = None
        return {
            "selection_regret_vs_oracle_neighbours": round(regret_oracle, 4),
            "selection_regret_vs_global_best": (
                None if regret_global is None else round(regret_global, 4)
            ),
            "beats_global_best_baseline": beats,
        }

    # ------------------------------------------------------------------
    def save_knowledge_base(self, path: str | Path) -> Path:
        """Persist the current knowledge base as JSON for later reuse."""
        path = Path(path)
        payload = {
            "schema_version": 3,
            "task": self._task,
            "metric": str(getattr(self._kb, "metric", "score")),
            "higher_is_better": bool(getattr(self._kb, "higher_is_better", True)),
            "feature_schema": "histoweave.target_free.v1",
            "performance": _json_safe(self._kb.performance),
            "features": {
                name: _json_safe(vec.tolist()) for name, vec in self._ref_features.items()
            },
            "best_method": self._kb.best_method,
            "niches": _json_safe(self._kb.niches),
            "feature_order": self._feature_order,
            "method_count": len(self._kb.method_order()),
            "dataset_count": len(self._ref_names),
            "timings": _json_safe(self._kb.timings),
            "dataset_meta": _json_safe(self._dataset_meta),
            "global_best_method": self._global_best_method,
            "global_best_score": self._global_best_score,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        try:
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


# ---------------------------------------------------------------------------
# Knowledge base persistence
# ---------------------------------------------------------------------------
def _load_knowledge_base(path: Path) -> LandscapeResult:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"knowledge base does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"knowledge base is invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("knowledge base must contain a JSON object")
    if raw.get("schema_version") not in {1, 2, 3}:
        raise ValueError(f"unsupported knowledge-base schema {raw.get('schema_version')!r}")
    order = raw.get("feature_order", list(RECOMMENDATION_FEATURE_ORDER))
    if not isinstance(order, list) or not all(isinstance(name, str) for name in order):
        raise ValueError("knowledge base feature_order must be a list of names")

    features: dict[str, np.ndarray] = {
        str(name): np.asarray(
            [float("nan") if value is None else float(value) for value in vector],
            dtype=float,
        )
        for name, vector in raw.get("features", {}).items()
    }
    performance = {
        str(dataset): {
            str(method): float("nan") if value is None else float(value)
            for method, value in row.items()
        }
        for dataset, row in raw.get("performance", {}).items()
    }
    shared = sorted(set(features) & set(performance))
    if len(shared) < 2:
        raise ValueError(
            "knowledge base requires at least two datasets with features and performance"
        )
    for name in shared:
        vector = features[name]
        if vector.ndim != 1 or len(vector) != len(order):
            raise ValueError(
                f"feature vector for {name!r} has length {vector.size}; expected {len(order)}"
            )
    if not any(np.isfinite(value) for row in performance.values() for value in row.values()):
        raise ValueError("knowledge base contains no finite method performance score")

    dataset_meta_raw = raw.get("dataset_meta", {})
    dataset_meta = (
        {str(name): dict(meta) for name, meta in dataset_meta_raw.items()}
        if isinstance(dataset_meta_raw, dict)
        else {}
    )

    from .landscape import _embed_datasets

    embedding = _embed_datasets(features)
    task = str(raw.get("task", "spatial_domain"))
    if task in {"domain_detection", "domain"}:
        task = AnalysisTask.SPATIAL_DOMAIN.value
    return LandscapeResult(
        performance=performance,
        features=features,
        embedding=embedding,
        best_method=raw.get("best_method", {}),
        niches=raw.get("niches", {}),
        timings=raw.get("timings", {}),
        feature_order=list(order),
        method_count=raw.get("method_count", 0),
        dataset_count=raw.get("dataset_count", 0),
        task=task,
        metric=str(raw.get("metric", "score")),
        higher_is_better=bool(raw.get("higher_is_better", True)),
        dataset_meta=dataset_meta,
    )


def _global_best_baseline(kb: LandscapeResult) -> tuple[str | None, float | None]:
    """Return the method with the best mean score across all datasets."""
    method_names = kb.method_order()
    if not method_names:
        return None, None
    higher = bool(getattr(kb, "higher_is_better", True))
    best_name: str | None = None
    best_score = -np.inf if higher else np.inf
    for method in method_names:
        values = [
            float(row[method])
            for row in kb.performance.values()
            if method in row and row[method] is not None and np.isfinite(row[method])
        ]
        if not values:
            continue
        mean = float(np.mean(values))
        if (higher and mean > best_score) or (not higher and mean < best_score):
            best_score = mean
            best_name = method
    if best_name is None:
        return None, None
    return best_name, round(float(best_score), 4)


def _policy_tokens(policy: str | None) -> set[str]:
    if not policy:
        return set()
    text = str(policy).strip().lower()
    tokens = {text}
    aliases = {
        "off": {"off", "sw0.0", "0.0", "expr", "expression"},
        "default": {"default", "sw0.3", "0.3", "mid"},
        "high": {"high", "sw0.8", "0.8", "strong", "spatial"},
        "sw0.0": {"off", "sw0.0", "0.0"},
        "sw0.3": {"default", "sw0.3", "0.3"},
        "sw0.8": {"high", "sw0.8", "0.8"},
    }
    tokens |= aliases.get(text, set())
    return tokens


# ---------------------------------------------------------------------------
# Ensemble strategy builder
# ---------------------------------------------------------------------------
def _ensemble_strategy(
    ranked: list[MethodScore],
    neighbours: list[dict[str, Any]],
) -> str:
    """Build a human-readable ensemble strategy from top-2 methods.

    When the top-2 methods have close scores (confidence < 0.8), recommend
    consensus voting: assign each cell to the domain label both methods agree
    on, and flag cells where they disagree as "uncertain."
    """
    if not ranked:
        return "No ensemble can be suggested without benchmark evidence."
    if len(ranked) == 1:
        return f"Use {ranked[0].method} (only available method with evidence)."

    m1, m2 = ranked[0], ranked[1]
    gap = m1.score - m2.score

    if gap > 0.10:
        return f"Strong recommendation: {m1.method} (outperforms {m2.method} by {gap:.3f} ARI)."
    if gap > 0.03:
        return (
            f"Prefer {m1.method} over {m2.method} "
            f"(marginal lead, ΔARI={gap:.3f}). "
            f"Consider running both and inspecting disagreement regions."
        )
    return (
        f"Consensus ensemble: run both {m1.method} and {m2.method}, "
        f"assign cells where they agree (expected ≥{0.85:.0%} of cells), "
        f"flag disagreement regions as uncertain for downstream review."
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _cosine_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine similarity between a 1‑D query vector and rows of a 2‑D reference."""
    q_norm = np.linalg.norm(query)
    r_norms = np.linalg.norm(reference, axis=1)
    denom = q_norm * r_norms + 1e-12
    return query @ reference.T / denom


def _fit_feature_space(
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit imputation and scaling statistics on raw reference features."""
    reference = np.atleast_2d(reference).astype(float)
    impute = np.zeros(reference.shape[1], dtype=float)
    for index in range(reference.shape[1]):
        finite = reference[:, index][np.isfinite(reference[:, index])]
        impute[index] = float(np.median(finite)) if len(finite) else 0.0
    filled = np.where(np.isfinite(reference), reference, impute)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std[std < 1e-12] = 1.0
    return impute, mean, std


def _transform_feature_space(
    values: np.ndarray,
    impute: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply reference-fit imputation and scaling to reference or query rows."""
    values = np.atleast_2d(values).astype(float)
    if values.shape[1] != len(impute):
        raise ValueError(f"feature vector has {values.shape[1]} values; expected {len(impute)}")
    filled = np.where(np.isfinite(values), values, impute)
    return (filled - mean) / std


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
