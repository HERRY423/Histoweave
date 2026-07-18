"""Method failure fingerprint atlas — *how* methods fail, not only *where*.

Failure-boundary mapping (:mod:`histoweave.benchmark.failure_boundary`) locates
the parameter value at which a method drops below an acceptability threshold.
This module classifies the **failure mode** using contingency structure between
planted truth and predicted labels (label-permutation invariant):

* **Fragmentation** — one true domain is split into ≥5 predicted fragments.
* **Merge** — one predicted cluster absorbs ≥3 true domains.
* **Noise** — predicted clusters that each cover <5 % of observations.
* **Structural** — recovers domains on easy data but collapses on multi-domain
  or high-noise regimes.

Each method receives a **failure fingerprint**: a 4-vector
``[fragmentation, merge, noise, structural] ∈ [0, 1]⁴`` describing how
performance degrades when approaching its failure boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .._math import adjusted_rand_index
from ..data import SpatialTable
from ..plugins import MethodCategory, create_method

FINGERPRINT_SCHEMA_VERSION = 1
FINGERPRINT_ORDER: tuple[str, ...] = (
    "fragmentation",
    "merge",
    "noise",
    "structural",
)

# Classification thresholds (documented defaults; overridable).
DEFAULT_FRAG_MIN_CLUSTERS = 5
DEFAULT_MERGE_MIN_DOMAINS = 3
DEFAULT_NOISE_FRAC = 0.05
DEFAULT_MIN_SHARE = 0.05  # ignore tiny contingency cells when counting splits/merges
DEFAULT_TAU = 0.7


@dataclass
class FailureModeProfile:
    """Contingency-based failure diagnosis for one (truth, prediction) pair."""

    fragmentation: float  # [0, 1] severity
    merge: float
    noise: float
    ari: float
    fragmentation_flag: bool
    merge_flag: bool
    noise_flag: bool
    n_true: int
    n_pred: int
    max_true_fragments: int
    max_pred_true_domains: int
    n_micro_clusters: int
    micro_cluster_obs_fraction: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def mode_vector(self) -> dict[str, float]:
        return {
            "fragmentation": float(self.fragmentation),
            "merge": float(self.merge),
            "noise": float(self.noise),
        }


@dataclass
class FailureFingerprint:
    """Four-dimensional failure fingerprint for one method."""

    method: str
    vector: dict[str, float]  # keys in FINGERPRINT_ORDER
    dominant_mode: str
    n_evaluations: int
    n_failure_evaluations: int
    mean_ari_easy: float | None = None
    mean_ari_hard: float | None = None
    axis_contributions: dict[str, dict[str, float]] = field(default_factory=dict)
    schema_version: int = FINGERPRINT_SCHEMA_VERSION

    def as_array(self, order: Sequence[str] = FINGERPRINT_ORDER) -> np.ndarray:
        return np.array([float(self.vector.get(k, 0.0)) for k in order], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "vector": {k: _json_float(self.vector.get(k, 0.0)) for k in FINGERPRINT_ORDER},
            "dominant_mode": self.dominant_mode,
            "n_evaluations": self.n_evaluations,
            "n_failure_evaluations": self.n_failure_evaluations,
            "mean_ari_easy": _json_float(self.mean_ari_easy),
            "mean_ari_hard": _json_float(self.mean_ari_hard),
            "axis_contributions": {
                axis: {k: _json_float(v) for k, v in modes.items()}
                for axis, modes in self.axis_contributions.items()
            },
        }

    def summary(self) -> str:
        parts = [f"{k}={self.vector.get(k, 0.0):.2f}" for k in FINGERPRINT_ORDER]
        return (
            f"{self.method}: [{', '.join(parts)}]  "
            f"dominant={self.dominant_mode}  "
            f"(n_fail={self.n_failure_evaluations}/{self.n_evaluations})"
        )


@dataclass
class FailureFingerprintAtlas:
    """Collection of per-method fingerprints from a probe or boundary study."""

    fingerprints: list[FailureFingerprint]
    tau: float = DEFAULT_TAU
    protocol: str = "histoweave.failure_fingerprint.v1"
    conditions: dict[str, Any] = field(default_factory=dict)
    schema_version: int = FINGERPRINT_SCHEMA_VERSION

    def by_method(self) -> dict[str, FailureFingerprint]:
        return {fp.method: fp for fp in self.fingerprints}

    def matrix(self) -> dict[str, list[float]]:
        """method → [frag, merge, noise, structural]."""
        return {fp.method: fp.as_array().tolist() for fp in self.fingerprints}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "tau": self.tau,
            "conditions": self.conditions,
            "fingerprint_order": list(FINGERPRINT_ORDER),
            "fingerprints": [fp.to_dict() for fp in self.fingerprints],
            "matrix": self.matrix(),
        }

    def summary(self) -> str:
        lines = [
            f"Failure fingerprint atlas  (n_methods={len(self.fingerprints)}, tau={self.tau})",
            f"  order: {list(FINGERPRINT_ORDER)}",
        ]
        for fp in sorted(self.fingerprints, key=lambda f: f.method):
            lines.append(f"  {fp.summary()}")
        return "\n".join(lines)


# ======================================================================
# Contingency-based mode classification
# ======================================================================
def classify_domain_failure(
    labels_true: np.ndarray | Sequence,
    labels_pred: np.ndarray | Sequence,
    *,
    frag_min_clusters: int = DEFAULT_FRAG_MIN_CLUSTERS,
    merge_min_domains: int = DEFAULT_MERGE_MIN_DOMAINS,
    noise_frac: float = DEFAULT_NOISE_FRAC,
    min_share: float = DEFAULT_MIN_SHARE,
) -> FailureModeProfile:
    """Classify fragmentation / merge / noise failures from a label pair.

    Uses the contingency table between truth and prediction.  Counts are
    **label-permutation invariant** (only co-occurrence structure matters).
    """
    truth = np.asarray(labels_true)
    pred = np.asarray(labels_pred)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("labels_true and labels_pred must be 1-D arrays of equal length")
    n = int(truth.shape[0])
    if n == 0:
        raise ValueError("empty label arrays")

    true_ids, true_inv = np.unique(truth, return_inverse=True)
    pred_ids, pred_inv = np.unique(pred, return_inverse=True)
    n_true = int(true_ids.size)
    n_pred = int(pred_ids.size)
    contingency = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(contingency, (true_inv, pred_inv), 1)

    true_sizes = contingency.sum(axis=1).astype(float)
    pred_sizes = contingency.sum(axis=0).astype(float)

    # --- Fragmentation: true domain split across many predicted clusters ---
    max_true_fragments = 0
    frag_severities: list[float] = []
    for i in range(n_true):
        if true_sizes[i] <= 0:
            continue
        # Count predicted clusters that claim ≥ min_share of this true domain.
        shares = contingency[i, :] / true_sizes[i]
        n_frags = int(np.sum(shares >= min_share))
        # Also count any non-empty assignment if min_share is strict.
        if n_frags == 0:
            n_frags = int(np.sum(contingency[i, :] > 0))
        max_true_fragments = max(max_true_fragments, n_frags)
        # Severity: excess fragments beyond (frag_min_clusters - 1), scaled.
        excess = max(0, n_frags - 1) / max(frag_min_clusters - 1, 1)
        frag_severities.append(float(np.clip(excess, 0.0, 1.0)))
    fragmentation = float(np.max(frag_severities)) if frag_severities else 0.0
    # Boost to 1.0 when hard threshold is met.
    fragmentation_flag = max_true_fragments >= frag_min_clusters
    if fragmentation_flag:
        fragmentation = max(fragmentation, 1.0)

    # --- Merge: predicted cluster absorbs many true domains ---
    max_pred_true_domains = 0
    merge_severities: list[float] = []
    for j in range(n_pred):
        if pred_sizes[j] <= 0:
            continue
        shares = contingency[:, j] / pred_sizes[j]
        n_doms = int(np.sum(shares >= min_share))
        if n_doms == 0:
            n_doms = int(np.sum(contingency[:, j] > 0))
        max_pred_true_domains = max(max_pred_true_domains, n_doms)
        excess = max(0, n_doms - 1) / max(merge_min_domains - 1, 1)
        merge_severities.append(float(np.clip(excess, 0.0, 1.0)))
    merge = float(np.max(merge_severities)) if merge_severities else 0.0
    merge_flag = max_pred_true_domains >= merge_min_domains
    if merge_flag:
        merge = max(merge, 1.0)

    # --- Noise: micro-clusters < noise_frac of observations ---
    micro_mask = pred_sizes < (noise_frac * n)
    n_micro = int(np.sum(micro_mask))
    micro_obs_frac = float(pred_sizes[micro_mask].sum() / n) if n_micro else 0.0
    # Severity: fraction of clusters that are micro, and mass they hold.
    micro_cluster_frac = float(n_micro / max(n_pred, 1))
    noise = float(np.clip(0.5 * micro_cluster_frac + 0.5 * micro_obs_frac / noise_frac, 0.0, 1.0))
    noise_flag = n_micro > 0 and micro_obs_frac > 0.0
    if noise_flag and n_micro >= 2:
        noise = max(noise, min(1.0, micro_cluster_frac))

    try:
        ari = float(adjusted_rand_index(truth, pred))
    except Exception:
        ari = float("nan")

    return FailureModeProfile(
        fragmentation=fragmentation,
        merge=merge,
        noise=noise,
        ari=ari,
        fragmentation_flag=fragmentation_flag,
        merge_flag=merge_flag,
        noise_flag=noise_flag,
        n_true=n_true,
        n_pred=n_pred,
        max_true_fragments=max_true_fragments,
        max_pred_true_domains=max_pred_true_domains,
        n_micro_clusters=n_micro,
        micro_cluster_obs_fraction=micro_obs_frac,
        details={
            "frag_min_clusters": frag_min_clusters,
            "merge_min_domains": merge_min_domains,
            "noise_frac": noise_frac,
            "min_share": min_share,
        },
    )


def structural_severity(
    easy_ari: float,
    hard_ari: float,
    *,
    tau: float = DEFAULT_TAU,
    collapse_threshold: float = 0.2,
) -> float:
    """Structural failure: strong on easy data, near-collapse on hard data.

    Returns a severity in [0, 1].  1.0 when easy ≥ tau and hard ≤ collapse.

    Parameters
    ----------
    collapse_threshold
        ARI below which hard-condition performance is deemed a structural
        collapse.  The default (0.2) encodes the heuristic that a method
        achieving ARI ≤ 0.2 on hard data is performing near the level of
        randomly shuffling one true domain label — no meaningful structure
        remains.  This value is calibrated against the synthetic generators
        in ``histoweave.datasets`` where domains are Voronoi-tessellated
        blobs with planted marker genes; it may not transfer to real-tissue
        data without recalibration.
    """
    if not (np.isfinite(easy_ari) and np.isfinite(hard_ari)):
        return 0.0
    if easy_ari < tau:
        # Never worked on easy data — not a "structural collapse" pattern.
        return 0.0
    # How far hard score fell relative to the easy→collapse gap.
    gap = easy_ari - hard_ari
    if gap <= 0:
        return 0.0
    # Full severity when hard ≤ collapse_threshold and easy ≥ tau.
    if hard_ari <= collapse_threshold:
        return 1.0
    # Partial: scale by how close hard is to collapse.
    span = max(easy_ari - collapse_threshold, 1e-8)
    return float(np.clip((easy_ari - hard_ari) / span, 0.0, 1.0))


# ======================================================================
# Fingerprint aggregation
# ======================================================================
def aggregate_fingerprint(
    method: str,
    profiles: Sequence[FailureModeProfile],
    *,
    structural: float = 0.0,
    mean_ari_easy: float | None = None,
    mean_ari_hard: float | None = None,
    axis_contributions: dict[str, dict[str, float]] | None = None,
    tau: float = DEFAULT_TAU,
) -> FailureFingerprint:
    """Aggregate per-run profiles into a 4-vector fingerprint."""
    if not profiles:
        vector = {k: 0.0 for k in FINGERPRINT_ORDER}
        return FailureFingerprint(
            method=method,
            vector=vector,
            dominant_mode="none",
            n_evaluations=0,
            n_failure_evaluations=0,
            mean_ari_easy=mean_ari_easy,
            mean_ari_hard=mean_ari_hard,
            axis_contributions=dict(axis_contributions or {}),
        )

    frag = float(np.mean([p.fragmentation for p in profiles]))
    merge = float(np.mean([p.merge for p in profiles]))
    noise = float(np.mean([p.noise for p in profiles]))
    struct = float(np.clip(structural, 0.0, 1.0))
    vector = {
        "fragmentation": frag,
        "merge": merge,
        "noise": noise,
        "structural": struct,
    }
    # Failure evaluations: ARI below tau or any hard flag.
    n_fail = sum(
        1
        for p in profiles
        if (np.isfinite(p.ari) and p.ari < tau)
        or p.fragmentation_flag
        or p.merge_flag
        or p.noise_flag
    )
    dominant = max(vector, key=lambda k: vector[k])
    if vector[dominant] < 1e-9:
        dominant = "none"
    return FailureFingerprint(
        method=method,
        vector=vector,
        dominant_mode=dominant,
        n_evaluations=len(profiles),
        n_failure_evaluations=n_fail,
        mean_ari_easy=mean_ari_easy,
        mean_ari_hard=mean_ari_hard,
        axis_contributions=dict(axis_contributions or {}),
    )


# ======================================================================
# Probe: easy vs hard synthetic conditions → atlas
# ======================================================================
# Compact grids for CI-friendly fingerprint probes.
_EASY = dict(n_cells=200, n_genes=40, n_domains=3, noise=0.15, marker_gene_lift=8.0, layout="blob")
_HARD_NOISE = dict(
    n_cells=200, n_genes=40, n_domains=3, noise=0.85, marker_gene_lift=4.0, layout="blob"
)
_HARD_DOMAINS = dict(
    n_cells=240, n_genes=40, n_domains=8, noise=0.30, marker_gene_lift=5.0, layout="blob"
)


def run_failure_fingerprint_probe(
    methods: Sequence[str] | None = None,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    tau: float = DEFAULT_TAU,
    progress: bool = False,
) -> FailureFingerprintAtlas:
    """Run methods on easy / high-noise / multi-domain data and build fingerprints.

    Conditions
    ----------
    * **easy** — low noise, few domains (baseline recovery check).
    * **hard_noise** — high multiplicative noise.
    * **hard_domains** — many spatial domains.

    Fragmentation / merge / noise severities are averaged over hard conditions.
    Structural severity compares mean ARI(easy) vs mean ARI(hard).
    """
    from ..datasets import make_synthetic
    from ..logging import get_logger
    from .failure_boundary import probe_runnable

    logger = get_logger("histoweave.benchmark.failure_fingerprint")
    import histoweave.plugins.builtin  # noqa: F401

    if methods is None:
        runnable, _excluded = probe_runnable("domain_detection")
        methods = runnable
    methods = list(methods)
    if not methods:
        return FailureFingerprintAtlas(fingerprints=[], tau=tau)

    conditions: list[tuple[str, dict[str, Any]]] = [
        ("easy", dict(_EASY)),
        ("hard_noise", dict(_HARD_NOISE)),
        ("hard_domains", dict(_HARD_DOMAINS)),
    ]
    profiles: dict[str, list[FailureModeProfile]] = {m: [] for m in methods}
    hard_profiles: dict[str, list[FailureModeProfile]] = {m: [] for m in methods}
    ari_easy: dict[str, list[float]] = {m: [] for m in methods}
    ari_hard: dict[str, list[float]] = {m: [] for m in methods}
    by_condition: dict[str, dict[str, list[FailureModeProfile]]] = {
        m: {"hard_noise": [], "hard_domains": []} for m in methods
    }

    for cond_name, cfg in conditions:
        for seed in seeds:
            if progress:
                logger.info("fingerprint probe condition=%s seed=%s", cond_name, seed)
            ds = make_synthetic(seed=int(seed), **cfg)
            for method in methods:
                profile, ari = _run_and_classify(
                    ds, method, seed=int(seed), n_domains=int(cfg["n_domains"])
                )
                if profile is None:
                    continue
                profiles[method].append(profile)
                if cond_name == "easy":
                    ari_easy[method].append(ari)
                else:
                    ari_hard[method].append(ari)
                    hard_profiles[method].append(profile)
                    by_condition[method][cond_name].append(profile)

    fingerprints: list[FailureFingerprint] = []
    for method in methods:
        easy_mean = float(np.mean(ari_easy[method])) if ari_easy[method] else float("nan")
        hard_mean = float(np.mean(ari_hard[method])) if ari_hard[method] else float("nan")
        struct = structural_severity(easy_mean, hard_mean, tau=tau)
        use_profiles = hard_profiles[method] or profiles[method]
        axis_contributions = {
            name: {
                "fragmentation": _mean_attr(plist, "fragmentation"),
                "merge": _mean_attr(plist, "merge"),
                "noise": _mean_attr(plist, "noise"),
            }
            for name, plist in by_condition[method].items()
        }
        fp = aggregate_fingerprint(
            method,
            use_profiles,
            structural=struct,
            mean_ari_easy=easy_mean if np.isfinite(easy_mean) else None,
            mean_ari_hard=hard_mean if np.isfinite(hard_mean) else None,
            axis_contributions=axis_contributions,
            tau=tau,
        )
        fp.n_evaluations = len(profiles[method])
        fingerprints.append(fp)

    return FailureFingerprintAtlas(
        fingerprints=fingerprints,
        tau=tau,
        conditions={
            "easy": dict(_EASY),
            "hard_noise": dict(_HARD_NOISE),
            "hard_domains": dict(_HARD_DOMAINS),
            "seeds": list(seeds),
            "methods": list(methods),
        },
    )


def _run_and_classify(
    dataset: SpatialTable,
    method: str,
    *,
    seed: int,
    n_domains: int,
) -> tuple[FailureModeProfile | None, float]:
    """Normalize, run one domain method, classify failure modes."""
    try:
        prep = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k")
        data = prep.run(dataset.copy())
        probe = create_method(MethodCategory.DOMAIN_DETECTION, method)
        params: dict[str, Any] = {}
        if "n_domains" in probe.params:
            params["n_domains"] = int(n_domains)
        if "random_state" in probe.params:
            params["random_state"] = int(seed)
        result = create_method(MethodCategory.DOMAIN_DETECTION, method, **params).run(data)
        if "domain" not in result.obs.columns:
            return None, float("nan")
        # Align to surviving obs after QC-like drops (normalization usually keeps all).
        truth = dataset.obs.loc[result.obs_names, "domain_truth"].to_numpy()
        pred = result.obs["domain"].to_numpy()
        profile = classify_domain_failure(truth, pred)
        return profile, profile.ari
    except Exception:
        return None, float("nan")


def _mean_attr(profiles: Sequence[FailureModeProfile], attr: str) -> float:
    if not profiles:
        return 0.0
    vals = [float(getattr(p, attr)) for p in profiles]
    return float(np.mean(vals)) if vals else 0.0


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def write_fingerprint_atlas(
    atlas: FailureFingerprintAtlas,
    out_dir: str | Any,
) -> dict[str, Any]:
    """Persist atlas JSON + markdown summary. Returns written paths as strings."""
    import json
    from pathlib import Path
    from uuid import uuid4

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    json_path = root / "failure_fingerprints.json"
    temporary = json_path.with_name(f".{json_path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(atlas.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(json_path)
    finally:
        temporary.unlink(missing_ok=True)
    paths["json"] = json_path

    md_path = root / "failure_fingerprints.md"
    lines = [
        "# Method failure fingerprint atlas",
        "",
        f"Protocol: `{atlas.protocol}` · tau={atlas.tau}",
        "",
        "Each method has a 4-vector "
        "`[fragmentation, merge, noise, structural] ∈ [0,1]⁴` describing "
        "how performance degrades near its failure boundary.",
        "",
        "| Method | Frag | Merge | Noise | Structural | Dominant | n_fail |",
        "|--------|-----:|------:|------:|-----------:|----------|-------:|",
    ]
    for fp in sorted(atlas.fingerprints, key=lambda f: f.method):
        v = fp.vector
        lines.append(
            f"| `{fp.method}` | {v.get('fragmentation', 0):.2f} | "
            f"{v.get('merge', 0):.2f} | {v.get('noise', 0):.2f} | "
            f"{v.get('structural', 0):.2f} | {fp.dominant_mode} | "
            f"{fp.n_failure_evaluations}/{fp.n_evaluations} |"
        )
    lines.extend(
        [
            "",
            "## Mode definitions",
            "",
            "- **Fragmentation**: one true domain split into ≥5 predicted fragments.",
            "- **Merge**: one predicted cluster absorbs ≥3 true domains.",
            "- **Noise**: micro-clusters each covering <5% of observations.",
            "- **Structural**: recovers on easy data but collapses under high noise "
            "or many domains.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    paths["markdown"] = md_path
    return paths

