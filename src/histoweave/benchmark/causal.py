"""Causal performance landscape — from correlation to intervention.

Histoweave's :class:`~histoweave.benchmark.recommend.MethodRecommender` reasons by
*analogy*: it embeds a new dataset in a 16-feature space, finds the *k* nearest
reference datasets, and recommends whatever method won on those neighbours. That is
similarity-based inference, and it can fail systematically when a user's dataset
differs from the atlas along one decisive axis that the neighbours happen to share.

This module answers the question the recommender cannot: **what *causes* one method
to beat another?**  Because Histoweave owns the synthetic data-generating process, we
can perform genuine *interventions* — ``do(marker_gene_lift = x)`` — regenerate data
with everything else held fixed, and measure how each domain-detection method's ARI
responds.  ``marker_gene_lift`` is the dominant driver of ``spatial_autocorrelation``
(raising lift from 2 → 12 moves spatial autocorrelation ~0.27 → 0.68), so this
recovers the ``spatial_autocorrelation → method_performance`` story grounded in the
true DGP rather than an observational correlation.

Honesty about confounding
--------------------------
One generator knob moves several features at once (raising ``marker_gene_lift`` also
lowers ``effective_rank_90``).  Rather than pretend a single feature was isolated, the
estimand is defined at the **knob level**::

    ACE(method) = E[ARI | do(lift = hi)] − E[ARI | do(lift = lo)]

and every result carries a **feature-displacement table** recording which of the 16
features co-moved across the intervention grid.  The confounding is surfaced, not
hidden.  "Causal" here means *interventional on the synthetic DGP* (a true ``do``),
not causal discovery from observational data.

Outputs
-------
* **ACE table** — per method: hi−lo ARI contrast with a bootstrap 95 % CI and a
  significance flag (CI excludes 0).
* **Dose-response grid** — mean ARI at every lift level (full curve retained so a
  slope can be derived later without recompute).
* **Feature displacement** — mean ± sd of all 16 features at every lift level.
* **Causal graph** — a bipartite ``lift → method`` SVG; edge width ∝ |ACE|, colour =
  sign (green helps / red hurts), solid if significant else faded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .._math import adjusted_rand_index
from ..plugins import MethodCategory, create_method, list_methods
from .features import RECOMMENDATION_FEATURE_ORDER, extract_features, feature_vector

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults for the pilot intervention (locked with the project owner).
# ---------------------------------------------------------------------------
DEFAULT_KNOB = "marker_gene_lift"
DEFAULT_GRID: tuple[float, ...] = (2.0, 4.5, 7.0, 9.5, 12.0)
DEFAULT_N_SEEDS = 10
DEFAULT_FIXED_PARAMS: dict[str, Any] = {
    "n_cells": 500,
    "n_domains": 4,
    "noise": 0.25,
    "layout": "blob",
    "n_genes": 50,
}
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 12345


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class CausalEffect:
    """Average causal effect of the intervention on one method's ARI."""

    method: str
    ace: float  # E[ARI | do(hi)] − E[ARI | do(lo)]
    ci_low: float  # bootstrap 95 % lower bound
    ci_high: float  # bootstrap 95 % upper bound
    significant: bool  # CI excludes zero
    ari_lo: float  # mean ARI at the low anchor
    ari_hi: float  # mean ARI at the high anchor
    support_lo: int  # finite seed replicates at low anchor
    support_hi: int  # finite seed replicates at high anchor

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalLandscapeResult:
    """Complete output of an interventional (``do``) performance-landscape run."""

    # method → CausalEffect (hi − lo contrast with bootstrap CI)
    effects: dict[str, CausalEffect]

    # method → {lift_level: mean ARI over seeds}  (full dose-response curve)
    grid_means: dict[str, dict[float, float]]

    # method → {lift_level: [ARI per seed]}  (raw replicates feeding the bootstrap)
    seed_replicates: dict[str, dict[float, list[float]]]

    # lift_level → feature → {"mean", "sd"}  (confounding disclosure; 16 features)
    feature_displacement: dict[float, dict[str, dict[str, float]]]

    # metadata
    knob: str = DEFAULT_KNOB
    grid: list[float] = field(default_factory=lambda: list(DEFAULT_GRID))
    lo: float = DEFAULT_GRID[0]
    hi: float = DEFAULT_GRID[-1]
    n_seeds: int = DEFAULT_N_SEEDS
    fixed_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FIXED_PARAMS))
    feature_order: list[str] = field(default_factory=lambda: list(RECOMMENDATION_FEATURE_ORDER))
    task: str = "domain_detection"
    metric: str = "ARI"
    higher_is_better: bool = True
    method_count: int = 0

    # ------------------------------------------------------------------
    def ranked_effects(self) -> list[CausalEffect]:
        """Effects sorted by ACE descending (biggest positive causal effect first)."""
        return sorted(
            self.effects.values(),
            key=lambda e: e.ace if np.isfinite(e.ace) else -np.inf,
            reverse=True,
        )

    def significant_effects(self) -> list[CausalEffect]:
        return [e for e in self.ranked_effects() if e.significant]

    def feature_shift(self, feature: str) -> tuple[float, float]:
        """Mean value of *feature* at the (lo, hi) anchors — the confounding view."""
        lo = self.feature_displacement.get(self.lo, {}).get(feature, {}).get("mean", float("nan"))
        hi = self.feature_displacement.get(self.hi, {}).get(feature, {}).get("mean", float("nan"))
        return float(lo), float(hi)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable ACE table, styled after ``LandscapeResult.summary()``."""
        lines: list[str] = []
        lines.append(
            f"Causal landscape: do({self.knob}) "
            f"{self.lo:g} → {self.hi:g}  "
            f"[{len(self.grid)} levels × {self.n_seeds} seeds × {self.method_count} methods]"
        )
        # Show the primary knob → feature confounding up front.
        sa_lo, sa_hi = self.feature_shift("spatial_autocorrelation")
        er_lo, er_hi = self.feature_shift("effective_rank_90")
        lines.append(
            f"  Knob moved: spatial_autocorrelation {sa_lo:.2f}→{sa_hi:.2f}, "
            f"effective_rank_90 {er_lo:.1f}→{er_hi:.1f} (co-moved / confounded)"
        )
        lines.append(
            f"  ACE = E[ARI|do({self.knob}={self.hi:g})] − E[ARI|do({self.knob}={self.lo:g})]"
        )
        lines.append("  " + "-" * 68)
        lines.append(f"  {'method':<28}{'ACE':>8}{'95% CI':>20}{'sig':>6}")
        for e in self.ranked_effects():
            ci = f"[{e.ci_low:+.3f}, {e.ci_high:+.3f}]" if np.isfinite(e.ci_low) else "[   n/a    ]"
            ace = f"{e.ace:+.3f}" if np.isfinite(e.ace) else "  n/a"
            flag = "*" if e.significant else ""
            lines.append(f"  {e.method:<28}{ace:>8}{ci:>20}{flag:>6}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "knob": self.knob,
            "grid": _json_safe(self.grid),
            "lo": self.lo,
            "hi": self.hi,
            "n_seeds": self.n_seeds,
            "fixed_params": _json_safe(self.fixed_params),
            "task": self.task,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "method_count": self.method_count,
            "feature_order": self.feature_order,
            "effects": {m: _json_safe(e.to_dict()) for m, e in self.effects.items()},
            "grid_means": _json_safe(self.grid_means),
            "seed_replicates": _json_safe(self.seed_replicates),
            "feature_displacement": _json_safe(self.feature_displacement),
        }


# ---------------------------------------------------------------------------
# Interventional benchmark loop
# ---------------------------------------------------------------------------
def run_causal_landscape(
    *,
    knob: str = DEFAULT_KNOB,
    grid: tuple[float, ...] | list[float] = DEFAULT_GRID,
    n_seeds: int = DEFAULT_N_SEEDS,
    fixed_params: dict[str, Any] | None = None,
    methods: list[str] | None = None,
    base_seed: int = 0,
    progress: bool = False,
) -> CausalLandscapeResult:
    """Estimate the causal effect of an intervention on method performance.

    For every level of *knob* and every seed, a synthetic dataset is regenerated with
    all other generator parameters held at *fixed_params*.  Each registered
    domain-detection method is run (with the true cluster count passed in) and scored
    by ARI against ground truth.  The average causal effect (ACE) for each method is
    the hi−lo contrast of mean ARI, with a nonparametric bootstrap 95 % CI over the
    per-seed replicates.

    Parameters
    ----------
    knob
        Generator parameter to intervene on.  Only ``"marker_gene_lift"`` is validated
        for the pilot; other continuous knobs of :func:`make_synthetic` are accepted.
    grid
        Ordered intervention levels.  ``grid[0]`` is the low anchor, ``grid[-1]`` the
        high anchor for the ACE contrast.
    n_seeds
        Independent synthetic replicates per level (drives CI width).
    fixed_params
        Generator parameters held constant (defaults isolate *knob*).
    methods
        Domain-detection method names.  ``None`` → every registered method.
    base_seed
        Seeds used are ``base_seed + level_index * 1000 + seed_index``.
    progress
        When ``True``, print per-level throughput to stdout.
    """
    if knob == "layout":
        raise ValueError("layout is categorical; use a continuous knob for ACE contrasts")
    grid = [float(x) for x in grid]
    if len(grid) < 2:
        raise ValueError("grid needs at least a low and a high anchor")
    fixed = dict(DEFAULT_FIXED_PARAMS)
    if fixed_params:
        fixed.update(fixed_params)
    if knob in fixed:
        # The knob must vary, not be pinned by fixed_params.
        fixed.pop(knob)

    from ..datasets.synthetic import make_synthetic

    if methods is None:
        methods = [m["name"] for m in list_methods(MethodCategory.DOMAIN_DETECTION)]

    # Sniff which extra params each method declares (mirror run_task_landscape).
    method_param_names: dict[str, set[str]] = {}
    for name in methods:
        try:
            method_param_names[name] = set(
                create_method(MethodCategory.DOMAIN_DETECTION, name).params.keys()
            )
        except Exception:
            method_param_names[name] = set()

    true_k = int(fixed.get("n_domains", DEFAULT_FIXED_PARAMS["n_domains"]))

    # method → level → [ARI per seed]
    replicates: dict[str, dict[float, list[float]]] = {m: {lv: [] for lv in grid} for m in methods}
    # level → feature → [value per seed]
    feat_acc: dict[float, dict[str, list[float]]] = {lv: {} for lv in grid}

    for li, level in enumerate(grid):
        t0 = time.perf_counter()
        for si in range(n_seeds):
            seed = base_seed + li * 1000 + si
            gen_kwargs = dict(fixed)
            gen_kwargs[knob] = level
            data = make_synthetic(seed=seed, **gen_kwargs)

            # Feature vector (16 target-free features) → displacement accumulator.
            feats = extract_features(data, include_domain=False)
            fv = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
            for fname, fval in zip(RECOMMENDATION_FEATURE_ORDER, fv, strict=True):
                feat_acc[level].setdefault(fname, []).append(float(fval))

            # Normalize once so every method sees identical input.
            normalized = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(data.copy())

            data.obs["domain_truth"].to_numpy()
            for name in methods:
                try:
                    params: dict[str, Any] = {}
                    if "n_domains" in method_param_names.get(name, set()):
                        params["n_domains"] = true_k
                    result = create_method(MethodCategory.DOMAIN_DETECTION, name, **params).run(
                        normalized.copy()
                    )
                    pred = result.obs["domain"].to_numpy()
                    truth_aligned = data.obs.loc[result.obs_names, "domain_truth"].to_numpy()
                    ari = float(adjusted_rand_index(truth_aligned, pred))
                except Exception:
                    ari = float("nan")
                replicates[name][level].append(ari)
        if progress:
            dt = time.perf_counter() - t0
            _LOGGER.info(
                "  [level %s/%s] %s=%g: %s seeds × %s methods in %.1fs",
                li + 1,
                len(grid),
                knob,
                level,
                n_seeds,
                len(methods),
                dt,
            )

    # Feature displacement table: mean ± sd per level (nan-robust).
    displacement: dict[float, dict[str, dict[str, float]]] = {}
    for level in grid:
        displacement[level] = {}
        for fname in RECOMMENDATION_FEATURE_ORDER:
            vals = np.asarray(feat_acc[level].get(fname, []), dtype=float)
            finite = vals[np.isfinite(vals)]
            displacement[level][fname] = {
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "sd": float(np.std(finite)) if finite.size else float("nan"),
            }

    # Grid means + ACE with bootstrap CI.
    lo, hi = grid[0], grid[-1]
    grid_means: dict[str, dict[float, float]] = {}
    effects: dict[str, CausalEffect] = {}
    for name in methods:
        gm: dict[float, float] = {}
        for level in grid:
            vals = np.asarray(replicates[name][level], dtype=float)
            finite = vals[np.isfinite(vals)]
            gm[level] = float(np.mean(finite)) if finite.size else float("nan")
        grid_means[name] = gm

        lo_vals = np.asarray(replicates[name][lo], dtype=float)
        hi_vals = np.asarray(replicates[name][hi], dtype=float)
        effects[name] = _bootstrap_effect(name, lo_vals, hi_vals)

    return CausalLandscapeResult(
        effects=effects,
        grid_means=grid_means,
        seed_replicates={m: {lv: list(v) for lv, v in d.items()} for m, d in replicates.items()},
        feature_displacement=displacement,
        knob=knob,
        grid=list(grid),
        lo=lo,
        hi=hi,
        n_seeds=n_seeds,
        fixed_params=fixed,
        method_count=len(methods),
    )


# ---------------------------------------------------------------------------
# Bootstrap ACE
# ---------------------------------------------------------------------------
def _bootstrap_effect(
    method: str,
    lo_vals: np.ndarray,
    hi_vals: np.ndarray,
    *,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> CausalEffect:
    """Hi−lo ACE with a nonparametric bootstrap 95 % CI over seed replicates."""
    lo_f = lo_vals[np.isfinite(lo_vals)]
    hi_f = hi_vals[np.isfinite(hi_vals)]
    ari_lo = float(np.mean(lo_f)) if lo_f.size else float("nan")
    ari_hi = float(np.mean(hi_f)) if hi_f.size else float("nan")

    if lo_f.size < 2 or hi_f.size < 2:
        return CausalEffect(
            method=method,
            ace=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            significant=False,
            ari_lo=ari_lo,
            ari_hi=ari_hi,
            support_lo=int(lo_f.size),
            support_hi=int(hi_f.size),
        )

    ace = ari_hi - ari_lo
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    boot = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        lo_s = rng.choice(lo_f, size=lo_f.size, replace=True)
        hi_s = rng.choice(hi_f, size=hi_f.size, replace=True)
        boot[b] = hi_s.mean() - lo_s.mean()
    ci_low = float(np.percentile(boot, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    significant = bool(ci_low > 0.0 or ci_high < 0.0)

    return CausalEffect(
        method=method,
        ace=float(ace),
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        ari_lo=ari_lo,
        ari_hi=ari_hi,
        support_lo=int(lo_f.size),
        support_hi=int(hi_f.size),
    )


# ---------------------------------------------------------------------------
# Causal graph SVG
# ---------------------------------------------------------------------------
def causal_graph_svg(result: CausalLandscapeResult, width: int = 860, height: int = 520) -> str:
    """Render the bipartite ``knob → method`` causal graph as an SVG.

    A single left node is the intervention (``marker_gene_lift``); the right column
    lists methods sorted by ACE.  Each edge encodes the average causal effect:
    width ∝ |ACE|, colour = sign (green helps, red hurts), solid when the 95 % CI
    excludes zero, faded and thin otherwise.  Text stays selectable (not outlined).
    """
    import html

    ranked = result.ranked_effects()
    if not ranked or all(not np.isfinite(e.ace) for e in ranked):
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="40">'
            '<text x="10" y="25" font-size="12">(no causal effects estimated)</text></svg>'
        )

    pad = 40
    left_x = pad + 70
    # Leave room on the right for the longest method label + its ACE/CI caption.
    right_x = width - pad - 320
    node_top = pad + 40
    node_gap = (height - node_top - pad) / max(len(ranked), 1)
    knob_y = height / 2

    finite_abs = [abs(e.ace) for e in ranked if np.isfinite(e.ace)]
    max_abs = max(finite_abs) if finite_abs else 1.0
    max_abs = max(max_abs, 1e-6)

    pos = "#2CA02C"  # green — helps
    neg = "#D62728"  # red — hurts

    edges: list[str] = []
    nodes: list[str] = []
    for i, e in enumerate(ranked):
        y = node_top + i * node_gap + node_gap / 2
        if np.isfinite(e.ace):
            w = 1.0 + 7.0 * (abs(e.ace) / max_abs)
            color = pos if e.ace >= 0 else neg
            # Keep non-significant edges faintly visible without implying significance.
            opacity = 0.95 if e.significant else 0.4
            dash = "" if e.significant else ' stroke-dasharray="4 3"'
            edges.append(
                f'<path d="M {left_x + 12} {knob_y:.1f} '
                f"C {(left_x + right_x) / 2:.1f} {knob_y:.1f}, "
                f'{(left_x + right_x) / 2:.1f} {y:.1f}, {right_x - 6} {y:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="{w:.2f}" '
                f'stroke-opacity="{opacity}"{dash}/>'
            )
            label = f"ACE={e.ace:+.3f} [{e.ci_low:+.3f},{e.ci_high:+.3f}]" + (
                "" if e.significant else " (ns)"
            )
        else:
            edges.append(
                f'<path d="M {left_x + 12} {knob_y:.1f} '
                f"C {(left_x + right_x) / 2:.1f} {knob_y:.1f}, "
                f'{(left_x + right_x) / 2:.1f} {y:.1f}, {right_x - 6} {y:.1f}" '
                f'fill="none" stroke="#999" stroke-width="1" '
                f'stroke-opacity="0.2" stroke-dasharray="2 3"/>'
            )
            label = "insufficient support"

        weight = "600" if getattr(e, "significant", False) else "400"
        nodes.append(
            f'<circle cx="{right_x:.1f}" cy="{y:.1f}" r="5" fill="#4C78A8" '
            f'stroke="#333" stroke-width="0.5"/>'
            f'<text x="{right_x + 10:.1f}" y="{y - 2:.1f}" font-size="11" '
            f'font-weight="{weight}" fill="currentColor">{html.escape(e.method)}</text>'
            f'<text x="{right_x + 10:.1f}" y="{y + 10:.1f}" font-size="8.5" '
            f'fill="#666">{html.escape(label)}</text>'
        )

    # Intervention node + its measured feature displacement caption.
    sa_lo, sa_hi = result.feature_shift("spatial_autocorrelation")
    knob_node = (
        f'<circle cx="{left_x:.1f}" cy="{knob_y:.1f}" r="12" fill="#EECA3B" '
        f'stroke="#333" stroke-width="1"/>'
        f'<text x="{left_x:.1f}" y="{knob_y - 20:.1f}" text-anchor="middle" '
        f'font-size="11" font-weight="600" fill="currentColor">'
        f"do({html.escape(result.knob)})</text>"
        f'<text x="{left_x:.1f}" y="{knob_y + 28:.1f}" text-anchor="middle" '
        f'font-size="9" fill="#666">{result.lo:g} → {result.hi:g}</text>'
        f'<text x="{left_x:.1f}" y="{knob_y + 40:.1f}" text-anchor="middle" '
        f'font-size="8" fill="#666">spatial_ac {sa_lo:.2f}→{sa_hi:.2f}</text>'
    )

    title = "Causal Performance Landscape: intervention → method ARI"
    subtitle = (
        "Edge width ∝ |ACE|.  Green = helps, red = hurts.  "
        "Solid = 95% CI excludes 0; faded = not significant."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui, sans-serif">'
        f'<text x="{pad}" y="24" font-size="13" font-weight="600" fill="currentColor">'
        f"{html.escape(title)}</text>"
        f'<text x="{pad}" y="40" font-size="10" fill="#666">{html.escape(subtitle)}</text>'
        f"{''.join(edges)}{knob_node}{''.join(nodes)}</svg>"
    )


# ---------------------------------------------------------------------------
# JSON helper (shared convention with recommend.py)
# ---------------------------------------------------------------------------
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
