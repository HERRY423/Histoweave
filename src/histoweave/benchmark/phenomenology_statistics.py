"""Coverage-aware summaries and paired inference for phenomenology benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCIENTIFIC_FAILURES = {
    "invalid_input",
    "method_error",
    "timeout",
    "oom",
    "budget_exceeded",
}
ENVIRONMENT_GAPS = {"backend_unavailable", "fixture_unavailable"}


def coverage_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Report applicability, execution and success coverage without conflation."""

    required = {"method", "category", "role", "track", "applicability", "status"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"runs table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    keys = ["category", "method", "version", "role", "track"]
    for values, group in runs.groupby(keys, dropna=False, sort=True):
        applicable = group["applicability"].astype(bool)
        applicable_count = int(applicable.sum())
        executable = applicable & ~group["status"].isin(ENVIRONMENT_GAPS)
        successful = applicable & (group["status"] == "ok")
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "design_units": int(len(group)),
                "applicable_units": applicable_count,
                "applicability_coverage": applicable_count / max(len(group), 1),
                "execution_coverage": int(executable.sum()) / max(applicable_count, 1),
                "success_coverage": int(successful.sum()) / max(applicable_count, 1),
                "environment_gap_units": int(
                    (applicable & group["status"].isin(ENVIRONMENT_GAPS)).sum()
                ),
                "scientific_failure_units": int(
                    (applicable & group["status"].isin(SCIENTIFIC_FAILURES)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def capability_index(
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute role-conditional capability components without producing ranks.

    Environment gaps and N/A cells are excluded from the scientific denominator;
    applicable runs that reached the backend but failed receive a primary score of zero.
    """

    weights = weights or {
        "recovery": 0.50,
        "robustness": 0.25,
        "reliability": 0.15,
        "efficiency": 0.10,
    }
    if set(weights) != {"recovery", "robustness", "reliability", "efficiency"}:
        raise ValueError("weights must define recovery, robustness, reliability and efficiency")
    if not np.isclose(sum(weights.values()), 1.0) or any(value < 0 for value in weights.values()):
        raise ValueError("capability weights must be non-negative and sum to one")

    primary = metrics.loc[metrics["primary"].astype(bool), ["run_id", "normalized_value"]]
    if primary["run_id"].duplicated().any():
        raise ValueError("each successful run must expose exactly one primary metric")
    frame = runs.merge(primary, on="run_id", how="left", validate="one_to_one")
    frame = frame.loc[frame["applicability"].astype(bool)].copy()
    frame = frame.loc[~frame["status"].isin(ENVIRONMENT_GAPS | {"not_tunable"})].copy()
    frame.loc[frame["status"].isin(SCIENTIFIC_FAILURES), "normalized_value"] = 0.0
    if frame.loc[frame["status"] == "ok", "normalized_value"].isna().any():
        raise ValueError("successful runs are missing a primary normalized metric")

    frame["efficiency_run"] = _efficiency_per_run(frame)
    keys = ["category", "method", "version", "role", "track", "phenomenon"]
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        clean = group[group["condition"] == "clean"]
        recovery = float(clean["normalized_value"].mean()) if len(clean) else np.nan
        robustness_values: list[float] = []
        for replicate, replicate_group in group.groupby("replicate"):
            del replicate
            clean_values = replicate_group.loc[
                replicate_group["condition"] == "clean", "normalized_value"
            ]
            if len(clean_values) != 1:
                continue
            clean_score = float(clean_values.iloc[0])
            perturbed = replicate_group.loc[
                replicate_group["condition"] != "clean", "normalized_value"
            ]
            robustness_values.extend(
                1.0 - max(0.0, clean_score - float(value)) for value in perturbed
            )
        robustness = float(np.mean(robustness_values)) if robustness_values else np.nan
        success_rate = float(np.mean(group["status"] == "ok"))
        condition_spreads = [
            float(values.std(ddof=1))
            for _, values in group.loc[group["status"] == "ok"].groupby("condition")[
                "normalized_value"
            ]
            if len(values) > 1
        ]
        repeatability = (
            1.0 - min(float(np.mean(condition_spreads)) * 2.0, 1.0)
            if condition_spreads
            else success_rate
        )
        reliability = 0.5 * success_rate + 0.5 * repeatability
        efficiency = float(group["efficiency_run"].mean())
        components = {
            "recovery": recovery,
            "robustness": robustness,
            "reliability": reliability,
            "efficiency": efficiency,
        }
        available_weight = sum(
            weights[name] for name, value in components.items() if np.isfinite(value)
        )
        index = (
            sum(weights[name] * value for name, value in components.items() if np.isfinite(value))
            / available_weight
            if available_weight
            else np.nan
        )
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "recovery_score": recovery,
                "robustness_score": robustness,
                "reliability_score": reliability,
                "efficiency_score": efficiency,
                "capability_index": index,
                "scientific_runs": int(len(group)),
                "successful_runs": int((group["status"] == "ok").sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    value_column: str = "normalized_value",
    group_columns: Sequence[str] = (
        "category",
        "method",
        "role",
        "track",
        "phenomenon",
        "condition",
    ),
    replicate_column: str = "replicate",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> pd.DataFrame:
    """Bootstrap replicate blocks; pairing across methods/conditions stays intact."""

    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    required = set(group_columns) | {replicate_column, value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame is missing columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        by_replicate = group.groupby(replicate_column)[value_column].mean().dropna()
        observed = by_replicate.to_numpy(dtype=float)
        if len(observed) == 0:
            continue
        samples = rng.choice(observed, size=(n_resamples, len(observed)), replace=True).mean(axis=1)
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                "mean": float(np.mean(observed)),
                "ci_low": float(np.quantile(samples, alpha)),
                "ci_high": float(np.quantile(samples, 1.0 - alpha)),
                "n_replicates": int(len(observed)),
                "n_resamples": int(n_resamples),
            }
        )
    return pd.DataFrame(rows)


def paired_method_comparisons(
    frame: pd.DataFrame,
    *,
    value_column: str = "normalized_value",
    family_columns: Sequence[str] = ("role", "track", "metric"),
    pairing_columns: Sequence[str] = ("phenomenon", "condition", "replicate"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Exploratory within-family paired tests with BH-FDR correction."""

    required = set(family_columns) | set(pairing_columns) | {"method", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"comparison frame is missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for family_values, family in frame.groupby(list(family_columns), dropna=False, sort=True):
        methods = sorted(family["method"].unique())
        family_records: list[dict[str, object]] = []
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1 :]:
                left = family.loc[family["method"] == left_method]
                right = family.loc[family["method"] == right_method]
                paired = left.merge(
                    right,
                    on=list(pairing_columns),
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                if len(paired) < 2:
                    continue
                differences = paired[f"{value_column}_left"] - paired[f"{value_column}_right"]
                try:
                    p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
                family_records.append(
                    {
                        **dict(zip(family_columns, family_values, strict=True)),
                        "method_left": left_method,
                        "method_right": right_method,
                        "n_pairs": int(len(paired)),
                        "mean_difference": float(differences.mean()),
                        "median_difference": float(differences.median()),
                        "p_value": p_value,
                    }
                )
        if family_records:
            adjusted = benjamini_hochberg(
                [cast(float, record["p_value"]) for record in family_records]
            )
            for record, q_value in zip(family_records, adjusted, strict=True):
                record["q_value"] = q_value
                record["reject_fdr_0_05"] = bool(q_value <= alpha)
            records.extend(family_records)
    return pd.DataFrame(records)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone BH-adjusted q-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional array in [0, 1]")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _efficiency_per_run(frame: pd.DataFrame) -> pd.Series:
    successful = frame["status"] == "ok"
    score = pd.Series(0.0, index=frame.index)
    for _, group in frame.loc[successful].groupby(["role", "track"], dropna=False):
        time_score = _inverse_log_scale(group["seconds"].to_numpy(dtype=float))
        memory_score = _inverse_log_scale(group["peak_rss_mb"].to_numpy(dtype=float))
        score.loc[group.index] = np.sqrt(time_score * memory_score)
    return score


def _inverse_log_scale(values: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(values, 0.0))
    low, high = float(np.min(logged)), float(np.max(logged))
    if np.isclose(low, high):
        return np.ones_like(logged)
    return 1.0 - (logged - low) / (high - low)
