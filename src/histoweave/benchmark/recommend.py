"""Method recommendation engine — from benchmark data to actionable guidance.

Given a new dataset (a :class:`~histoweave.data.SpatialTable` with unknown ground truth),
the recommender:

1. Extracts the 19‑dim feature vector (same space the landscape lives in).
2. Finds the *k* most similar reference datasets via cosine similarity.
3. Ranks methods by similarity-weighted mean performance on those neighbours.
4. Returns a :class:`Recommendation` with ranked methods, per-method confidence,
   an ensemble strategy, and diagnostic metadata.

The knowledge base is a persisted :class:`LandscapeResult` (or equivalent JSON) that
encodes  dataset‑features + performance‑matrix  for every task.  As new benchmark runs
complete, the recommender improves without code changes.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class MethodScore:
    """One ranked method with its recommendation metadata."""

    method: str
    score: float  # similarity-weighted mean ARI (higher = better)
    confidence: float  # 0‑1, gap to next-best method (1 = clear winner)
    wins: int  # how many neighbour datasets this method won
    neighbour_scores: dict[str, float]  # dataset_name → ARI on that neighbour

    uncertainty: float = 0.0
    support: int = 0
    coverage: float = 0.0

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

    def top(self, n: int = 3) -> list[MethodScore]:
        return self.ranked_methods[:n]

    def best(self) -> MethodScore | None:
        return self.ranked_methods[0] if self.ranked_methods else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": self.task,
            "dataset_name": self.dataset_name,
            "ranked_methods": [m.to_dict() for m in self.ranked_methods],
            "neighbours": _json_safe(self.neighbours),
            "feature_order": self.feature_order,
            "feature_vector": _json_safe(self.feature_vector),
            "warnings": self.warnings,
            "ensemble_suggestion": self.ensemble_strategy,
        }

    def summary(self) -> str:
        lines = [f"Recommendation for {self.dataset_name!r} [{self.task}]:"]
        for i, m in enumerate(self.top(3), 1):
            lines.append(
                f"  {i}. {m.method:<20} score={m.score:.3f}  "
                f"uncertainty={m.uncertainty:.3f}  "
                f"support={m.support}/{len(self.neighbours)}"
            )
        if self.ensemble_strategy:
            lines.append(f"  Ensemble suggestion: {self.ensemble_strategy}")
        if self.neighbours:
            nbr_names = [n["name"] for n in self.neighbours]
            lines.append(f"  Based on: {', '.join(nbr_names)}")
        return "\n".join(lines)


class MethodRecommender:
    """Recommend methods for a new dataset using a pre-computed landscape.

    Parameters
    ----------
    knowledge_base
        A :class:`LandscapeResult` from a prior ``run_landscape()`` call,
        or a path to a JSON knowledge-base file written by
        :meth:`save_knowledge_base`.
    k_neighbours
        Number of nearest reference datasets to consult (default 3).
    """

    def __init__(
        self,
        knowledge_base: LandscapeResult | str | Path,
        *,
        k_neighbours: int = 3,
    ) -> None:
        if k_neighbours < 1:
            raise ValueError("k_neighbours must be at least 1")
        if isinstance(knowledge_base, str | Path):
            knowledge_base = _load_knowledge_base(Path(knowledge_base))
        self._kb = knowledge_base
        self._k = int(k_neighbours)
        self._task = str(getattr(knowledge_base, "task", "domain_detection"))

        source_order = list(
            self._kb.feature_order or RECOMMENDATION_FEATURE_ORDER
        )
        keep = [
            index
            for index, name in enumerate(source_order)
            if name not in {
                "n_domains",
                "domain_balance",
                "domain_spatial_coherence",
            }
        ]
        self._feature_order = [source_order[index] for index in keep]
        if not self._feature_order:
            raise ValueError("knowledge base contains no target-free features")

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
            raise ValueError(
                "knowledge base requires at least two datasets with features"
            )
        raw_reference = np.asarray(vectors, dtype=float)
        self._impute, self._mean, self._std = _fit_feature_space(raw_reference)
        self._ref_matrix = _transform_feature_space(
            raw_reference, self._impute, self._mean, self._std
        )

    # ------------------------------------------------------------------
    def recommend(
        self,
        data: SpatialTable,
        *,
        dataset_name: str = "user_dataset",
    ) -> Recommendation:
        """Produce a ranked method recommendation for *data*."""
        # Query features are extracted before any method runs and permanently
        # exclude domain labels to prevent benchmark target leakage.
        feats = extract_features(data, include_domain=False)
        vec = feature_vector(feats, order=self._feature_order)
        vec_2d = vec.reshape(1, -1)
        vec_std = _transform_feature_space(
            vec_2d, self._impute, self._mean, self._std
        )

        # Cosine similarity to every reference dataset.
        neighbours = self._find_neighbours(vec_std.ravel())

        # Similarity-weighted method scores.
        ranked = self._rank_methods(neighbours)

        # Build ensemble strategy from top-2 methods.
        ensemble = _ensemble_strategy(ranked, neighbours)

        return Recommendation(
            task=self._task,
            dataset_name=dataset_name,
            ranked_methods=ranked,
            neighbours=neighbours,
            feature_vector=vec.tolist(),
            ensemble_strategy=ensemble,
            feature_order=list(self._feature_order),
        )

    # ------------------------------------------------------------------
    def _find_neighbours(self, query_vec: np.ndarray) -> list[dict[str, Any]]:
        """Return the *k* nearest reference datasets by cosine similarity."""
        if self._ref_matrix is None:
            return []

        cosine = _cosine_similarity(query_vec, self._ref_matrix)
        similarities = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
        top_k = np.argsort(similarities, kind="stable")[::-1][
            : min(self._k, len(self._ref_names))
        ]

        neighbours: list[dict[str, Any]] = []
        for idx in top_k:
            sim = float(similarities[idx])
            name = self._ref_names[idx]
            neighbours.append({
                "name": name,
                "similarity": round(sim, 4),
                "distance": round(1.0 - sim, 4),
                "best_method": self._kb.best_method.get(name, "?"),
            })
        return neighbours

    # ------------------------------------------------------------------
    def _rank_methods(self, neighbours: list[dict[str, Any]]) -> list[MethodScore]:
        """Rank methods by similarity-weighted mean ARI across neighbours."""
        if not neighbours:
            return []

        method_names = self._kb.method_order()
        wins: dict[str, int] = {m: 0 for m in method_names}
        for nbr in neighbours:
            best = self._kb.best_method.get(nbr["name"])
            if best in wins:
                wins[best] += 1

        total_weight = sum(float(nbr["similarity"]) for nbr in neighbours)
        ranked: list[MethodScore] = []
        for method in method_names:
            values: list[float] = []
            weights: list[float] = []
            evidence: dict[str, float] = {}
            for nbr in neighbours:
                name = nbr["name"]
                value = self._kb.performance.get(name, {}).get(
                    method, float("nan")
                )
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
            variance = float(
                np.average((value_array - score) ** 2, weights=weight_array)
            )
            effective_n = valid_weight**2 / max(
                float(np.sum(weight_array**2)), 1e-12
            )
            uncertainty = float(np.sqrt(variance / max(effective_n, 1.0)))
            coverage = valid_weight / max(total_weight, 1e-12)
            confidence = (
                float(np.mean(weight_array))
                * coverage
                * min(1.0, effective_n / 2.0)
            )
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
                )
            )

        reverse = bool(getattr(self._kb, "higher_is_better", True))
        ranked.sort(key=lambda item: item.score, reverse=reverse)
        return ranked

    # ------------------------------------------------------------------
    def save_knowledge_base(self, path: str | Path) -> Path:
        """Persist the current knowledge base as JSON for later reuse."""
        path = Path(path)
        payload = {
            "schema_version": 2,
            "task": self._task,
            "metric": str(getattr(self._kb, "metric", "score")),
            "higher_is_better": bool(
                getattr(self._kb, "higher_is_better", True)
            ),
            "feature_schema": "histoweave.target_free.v1",
            "performance": _json_safe(self._kb.performance),
            "features": {
                name: _json_safe(vec.tolist())
                for name, vec in self._ref_features.items()
            },
            "best_method": self._kb.best_method,
            "niches": _json_safe(self._kb.niches),
            "feature_order": self._feature_order,
            "method_count": len(self._kb.method_order()),
            "dataset_count": len(self._ref_names),
            "timings": _json_safe(self._kb.timings),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
            )
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
    if raw.get("schema_version") not in {1, 2}:
        raise ValueError(
            f"unsupported knowledge-base schema {raw.get('schema_version')!r}"
        )
    order = raw.get(
        "feature_order", list(RECOMMENDATION_FEATURE_ORDER)
    )
    if not isinstance(order, list) or not all(
        isinstance(name, str) for name in order
    ):
        raise ValueError("knowledge base feature_order must be a list of names")

    features: dict[str, np.ndarray] = {
        str(name): np.asarray(
            [
                float("nan") if value is None else float(value)
                for value in vector
            ],
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
            "knowledge base requires at least two datasets with features "
            "and performance"
        )
    for name in shared:
        vector = features[name]
        if vector.ndim != 1 or len(vector) != len(order):
            raise ValueError(
                f"feature vector for {name!r} has length {vector.size}; "
                f"expected {len(order)}"
            )
    if not any(np.isfinite(value) for row in performance.values() for value in row.values()):
        raise ValueError("knowledge base contains no finite method performance score")

    from .landscape import _embed_datasets

    embedding = _embed_datasets(features)
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
        task=str(raw.get("task", "domain_detection")),
        metric=str(raw.get("metric", "score")),
        higher_is_better=bool(raw.get("higher_is_better", True)),
    )


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
        return (
            f"Run {ranked[0].method}; only one method has usable "
            "neighbour evidence."
        )
    first, second = ranked[:2]
    return (
        f"Run {first.method} and {second.method}; align their labels, retain "
        "consensus regions, and flag disagreements for review. This is a "
        "suggestion, not an executed ensemble."
    )

    if len(ranked) < 2:
        if ranked:
            return f"Use {ranked[0].method} (only available method)."
        return "No method available."

    m1, m2 = ranked[0], ranked[1]
    gap = m1.score - m2.score

    if gap > 0.10:
        return (
            f"Strong recommendation: {m1.method} "
            f"(outperforms {m2.method} by {gap:.3f} ARI)."
        )
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
        raise ValueError(
            f"feature vector has {values.shape[1]} values; expected {len(impute)}"
        )
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
