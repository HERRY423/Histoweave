"""Independent statistical review layer for method rankings.

This module is intentionally separate from landscape *scoring* and
recommendation *heuristics*.  It answers three questions a Nature-Methods
reviewer will ask of any leaderboard:

1. **Score uncertainty** — cell-bootstrap CIs on ARI (or any paired labels).
2. **Rank uncertainty** — dataset-bootstrap stability of ranks + a simple
   Dirichlet–multinomial Bayesian posterior over ranks.
3. **Pairwise significance with FDR** — paired permutation tests across
   datasets, corrected by Benjamini–Hochberg / Holm / BY.

None of these paths mutate existing ``LandscapeResult.performance`` semantics;
callers attach the returned :class:`StatsReviewReport` as an optional artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .._math import adjusted_rand_index
from .multiple_testing import FdrMethod, pairwise_fdr_table

HigherIsBetter = bool


@dataclass
class BootstrapARIResult:
    mean: float
    ci_low: float
    ci_high: float
    n_boot: int
    fraction: float
    point: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MethodRankSummary:
    method: str
    mean_score: float
    mean_rank: float
    rank_ci_low: float
    rank_ci_high: float
    p_best: float  # Bayesian posterior P(rank == 1)
    rank_stability: float  # 1 - (rank IQR / (n_methods-1)), in [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StatsReviewReport:
    """Machine-readable statistical review of a performance matrix."""

    schema_version: int = 1
    protocol: str = "histoweave.stats_review.v1"
    n_datasets: int = 0
    n_methods: int = 0
    n_boot: int = 0
    n_perm: int = 0
    fdr_method: str = "bh"
    alpha: float = 0.05
    higher_is_better: bool = True
    methods: list[str] = field(default_factory=list)
    rank_summary: list[dict[str, Any]] = field(default_factory=list)
    pairwise: dict[str, Any] = field(default_factory=dict)
    bayesian_rank_posterior: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "n_datasets": self.n_datasets,
            "n_methods": self.n_methods,
            "n_boot": self.n_boot,
            "n_perm": self.n_perm,
            "fdr_method": self.fdr_method,
            "alpha": self.alpha,
            "higher_is_better": self.higher_is_better,
            "methods": list(self.methods),
            "rank_summary": list(self.rank_summary),
            "pairwise": dict(self.pairwise),
            "bayesian_rank_posterior": {
                m: list(v) for m, v in self.bayesian_rank_posterior.items()
            },
            "notes": list(self.notes),
        }


def bootstrap_ari(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    *,
    n_boot: int = 200,
    fraction: float = 0.8,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapARIResult:
    """Cell-level bootstrap CI for ARI (refit-free; resamples observations)."""
    truth = np.asarray(labels_true)
    pred = np.asarray(labels_pred)
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have equal length")
    n = int(truth.shape[0])
    if n < 4:
        point = float(adjusted_rand_index(truth, pred))
        return BootstrapARIResult(point, point, point, 0, fraction, point)

    point = float(adjusted_rand_index(truth, pred))
    rng = np.random.default_rng(seed)
    m = max(2, int(round(fraction * n)))
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.choice(n, size=m, replace=True)
        samples[b] = adjusted_rand_index(truth[idx], pred[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapARIResult(
        mean=float(np.mean(samples)),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
        fraction=float(fraction),
        point=point,
    )


def performance_to_matrix(
    performance: dict[str, dict[str, float]],
    *,
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert landscape performance dict → (n_datasets × n_methods) matrix."""
    ds = datasets or sorted(performance)
    meths = methods or sorted({m for row in performance.values() for m in row})
    mat = np.full((len(ds), len(meths)), np.nan, dtype=float)
    for i, d in enumerate(ds):
        row = performance.get(d, {})
        for j, m in enumerate(meths):
            val = row.get(m, np.nan)
            mat[i, j] = float(val) if val is not None else np.nan
    return mat, ds, meths


def ranks_from_scores(
    scores: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> np.ndarray:
    """Rank methods for one score vector (1 = best).  NaNs get worst rank."""
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    ranks = np.full(n, float(n), dtype=float)  # NaN / non-finite → worst
    orderable = np.where(np.isfinite(s))[0]
    if orderable.size == 0:
        return ranks
    vals = s[orderable]
    keyed = -vals if higher_is_better else vals
    order = np.argsort(keyed, kind="mergesort")
    sorted_vals = keyed[order]
    placed = np.empty(order.size, dtype=float)
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        placed[order[i : j + 1]] = avg_rank
        i = j + 1
    ranks[orderable] = placed
    return ranks


def bootstrap_rank_stability(
    performance: dict[str, dict[str, float]],
    *,
    n_boot: int = 500,
    seed: int = 0,
    higher_is_better: bool = True,
    ci: float = 0.95,
) -> tuple[list[MethodRankSummary], dict[str, list[float]], np.ndarray]:
    """Resample *datasets* with replacement; track rank distributions.

    Returns
    -------
    summaries
        Per-method mean rank + CI + P(best).
    posterior
        method → probability mass over ranks 1..M (Dirichlet-multinomial with
        uniform prior α=1, posterior mean of rank frequencies).
    rank_samples
        Array shape ``(n_boot, n_methods)`` of ranks.
    """
    mat, _datasets, methods = performance_to_matrix(performance)
    n_ds, n_m = mat.shape
    if n_ds == 0 or n_m == 0:
        return [], {}, np.zeros((0, 0))

    rng = np.random.default_rng(seed)
    rank_samples = np.empty((n_boot, n_m), dtype=float)
    win_counts = np.zeros(n_m, dtype=float)
    rank_hist = np.zeros((n_m, n_m), dtype=float)  # method × rank_index

    for b in range(n_boot):
        idx = rng.integers(0, n_ds, size=n_ds)
        boot = mat[idx]
        # Mean score per method over resampled datasets (nanmean).
        means = np.nanmean(boot, axis=0)
        ranks = ranks_from_scores(means, higher_is_better=higher_is_better)
        rank_samples[b] = ranks
        best = int(np.argmin(ranks))
        win_counts[best] += 1.0
        for j in range(n_m):
            # rank 1 → index 0
            r_idx = int(np.clip(np.rint(ranks[j]) - 1, 0, n_m - 1))
            rank_hist[j, r_idx] += 1.0

    alpha = (1.0 - ci) / 2.0
    # Dirichlet posterior mean: (counts + 1) / (n_boot + n_m)
    prior = 1.0
    posterior: dict[str, list[float]] = {}
    summaries: list[MethodRankSummary] = []
    for j, method in enumerate(methods):
        col = rank_samples[:, j]
        lo, hi = np.quantile(col, [alpha, 1.0 - alpha])
        iqr = float(np.subtract(*np.quantile(col, [0.75, 0.25])))
        stability = 1.0 - (iqr / max(n_m - 1, 1))
        post = (rank_hist[j] + prior) / (n_boot + prior * n_m)
        posterior[method] = [float(x) for x in post]
        # Point mean score over all datasets
        mean_score = float(np.nanmean(mat[:, j]))
        summaries.append(
            MethodRankSummary(
                method=method,
                mean_score=mean_score,
                mean_rank=float(np.mean(col)),
                rank_ci_low=float(lo),
                rank_ci_high=float(hi),
                p_best=float(post[0]),
                rank_stability=float(np.clip(stability, 0.0, 1.0)),
            )
        )
    summaries.sort(key=lambda s: (s.mean_rank, -s.mean_score))
    return summaries, posterior, rank_samples


def paired_permutation_pvalue(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    alternative: Literal["two-sided", "greater"] = "two-sided",
) -> float:
    """Paired permutation test on per-dataset score differences (A − B).

    Under the null, method labels are exchangeable within each dataset, so
    the sign of each paired difference is randomised.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    diff = a[mask] - b[mask]
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _i in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diff.size)
        stat = float(np.mean(diff * signs))
        if alternative == "greater":
            extreme += int(stat >= obs - 1e-15)
        else:
            extreme += int(abs(stat) >= abs(obs) - 1e-15)
    # Add-one smoothing so p never hits exactly 0.
    return float((extreme + 1) / (n_perm + 1))


def pairwise_permutation_matrix(
    performance: dict[str, dict[str, float]],
    *,
    n_perm: int = 1000,
    seed: int = 0,
    methods: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Symmetric matrix of paired permutation p-values (diagonal = nan)."""
    mat, _ds, meths = performance_to_matrix(performance, methods=methods)
    n_m = len(meths)
    pmat = np.full((n_m, n_m), np.nan, dtype=float)
    for i in range(n_m):
        for j in range(i + 1, n_m):
            p = paired_permutation_pvalue(
                mat[:, i],
                mat[:, j],
                n_perm=n_perm,
                seed=seed + 17 * i + 31 * j,
            )
            pmat[i, j] = p
            pmat[j, i] = p
    return pmat, meths


def review_landscape(
    performance: dict[str, dict[str, float]],
    *,
    n_boot: int = 500,
    n_perm: int = 1000,
    seed: int = 0,
    higher_is_better: bool = True,
    fdr_method: FdrMethod = "bh",
    alpha: float = 0.05,
) -> StatsReviewReport:
    """Full statistical review of a multi-dataset performance matrix.

    Combines bootstrap rank stability, Bayesian rank posterior, and
    FDR-corrected pairwise permutation tests.
    """
    summaries, posterior, _ranks = bootstrap_rank_stability(
        performance,
        n_boot=n_boot,
        seed=seed,
        higher_is_better=higher_is_better,
    )
    pmat, methods = pairwise_permutation_matrix(performance, n_perm=n_perm, seed=seed + 7)
    pairwise = pairwise_fdr_table(pmat, methods, method=fdr_method, alpha=alpha)
    notes = [
        "Ranks from dataset-bootstrap of mean scores (not cell-bootstrap).",
        "Bayesian rank posterior uses Dirichlet(1,…,1) prior over rank bins.",
        "Pairwise tests are paired sign-flip permutations across datasets.",
        f"Multiple comparisons controlled by {fdr_method} at alpha={alpha}.",
    ]
    if len(performance) < 3:
        notes.append(
            "WARNING: fewer than 3 datasets — permutation power and rank CIs "
            "are weak; treat significance claims cautiously."
        )

    return StatsReviewReport(
        n_datasets=len(performance),
        n_methods=len(methods),
        n_boot=n_boot,
        n_perm=n_perm,
        fdr_method=fdr_method,
        alpha=alpha,
        higher_is_better=higher_is_better,
        methods=methods,
        rank_summary=[s.to_dict() for s in summaries],
        pairwise=pairwise,
        bayesian_rank_posterior=posterior,
        notes=notes,
    )


def review_leaderboard_scores(
    scores: dict[str, float],
    *,
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    """Attach ranks to a single-dataset score dict (no resampling possible)."""
    methods = list(scores)
    arr = np.array([scores[m] for m in methods], dtype=float)
    ranks = ranks_from_scores(arr, higher_is_better=higher_is_better)
    rows = []
    for m, s, r in zip(methods, arr, ranks, strict=True):
        rows.append(
            {
                "method": m,
                "score": None if not np.isfinite(s) else float(s),
                "rank": float(r),
            }
        )
    rows.sort(key=lambda row: row["rank"] if isinstance(row["rank"], (int, float)) else 1e9)
    return rows
