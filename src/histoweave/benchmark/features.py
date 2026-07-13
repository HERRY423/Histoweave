"""Quantitative spatial-transcriptomics dataset characterisation.

Every dataset is reduced to a ~20‑dimensional feature vector so that datasets
live in a common embedding space where *method performance niches* can be
discovered.  Features span four dimensions:

* **Expression** — sparsity, library-size distribution, expression entropy.
* **Spatial** — autocorrelation (Moran's I), cluster tendency (Hopkins),
  nearest-neighbour spacing, local density variation.
* **Geometry** — effective rank, singular-value entropy, aspect ratio.
* **Domain** — number of domains, label balance, spatial coherence (when
  ground-truth labels are available).

All extractors are pure NumPy / SciPy so the module stays dependency-light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from ..data import SpatialTable

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Friendly names shown in legends / axis labels.
FEATURE_LABELS: dict[str, str] = {
    "n_obs": "N observations",
    "n_vars": "N variables",
    "aspect_ratio": "n_vars / n_obs",
    "sparsity": "Sparsity (zero fraction)",
    "mean_nonzero": "Mean non-zero expr",
    "library_mean": "Library size (mean)",
    "library_cv": "Library size (CV)",
    "expression_entropy": "Expression entropy",
    "spatial_autocorrelation": "Spatial autocorrelation (Moran's I)",
    "cluster_tendency": "Cluster tendency (Hopkins)",
    "mean_nn_distance": "Mean NN distance",
    "spatial_density_cv": "Spatial density (CV)",
    "spatial_entropy": "Spatial entropy",
    "effective_rank_90": "Effective rank (90 %)",
    "effective_rank_95": "Effective rank (95 %)",
    "sv_entropy": "Singular-value entropy",
    "n_domains": "N domains",
    "domain_balance": "Domain balance (entropy)",
    "domain_spatial_coherence": "Domain spatial coherence",
}


def extract_features(
    data: SpatialTable,
    *,
    include_domain: bool = True,
) -> dict[str, float]:
    """Extract a standardised feature vector from a spatial dataset.

    Parameters
    ----------
    data
        A processed :class:`SpatialTable`.  Expression features are computed on
        ``data.X``; spatial features require ``obsm['spatial']``.
    include_domain
        When ``True`` and ground-truth labels are present in ``obs['domain_truth']``,
        domain-level features are appended.  Omit when running on user data that
        has no ground truth.
    """
    features: dict[str, float] = {}

    # -- expression ---------------------------------------------------------
    X = np.asarray(data.X, dtype=float)
    features["n_obs"] = float(data.n_obs)
    features["n_vars"] = float(data.n_vars)
    features["aspect_ratio"] = data.n_vars / max(data.n_obs, 1)
    features.update(_expression_features(X))

    # -- spatial ------------------------------------------------------------
    coords = data.spatial
    if coords is not None:
        features.update(_spatial_features(X, coords))
    else:
        for key in _SPATIAL_KEYS:
            features[key] = float("nan")

    # -- geometry -----------------------------------------------------------
    features.update(_geometry_features(X))

    # -- domain (ground truth) -----------------------------------------------
    if include_domain:
        features.update(_domain_features(data))
    else:
        for key in _DOMAIN_KEYS:
            features[key] = float("nan")

    return features


def feature_vector(features: dict[str, float], order: list[str] | None = None) -> np.ndarray:
    """Return a 1‑D array of feature values in a consistent order."""
    if order is None:
        order = DEFAULT_FEATURE_ORDER
    return np.array([features.get(k, float("nan")) for k in order], dtype=float)


def feature_dataframe(datasets: dict[str, SpatialTable]) -> pd.DataFrame:
    """:func:`extract_features` for every named dataset, returned as a DataFrame."""
    import pandas as pd

    rows: dict[str, dict[str, float]] = {}
    for name, table in datasets.items():
        rows[name] = extract_features(table)
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Feature order (stable across runs)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_ORDER = [
    "n_obs",
    "n_vars",
    "aspect_ratio",
    "sparsity",
    "mean_nonzero",
    "library_mean",
    "library_cv",
    "expression_entropy",
    "spatial_autocorrelation",
    "cluster_tendency",
    "mean_nn_distance",
    "spatial_density_cv",
    "spatial_entropy",
    "effective_rank_90",
    "effective_rank_95",
    "sv_entropy",
    "n_domains",
    "domain_balance",
    "domain_spatial_coherence",
]

# Features available before an analysis method has been selected or any ground
# truth is known. Recommendation must use this schema: domain-level features
# are useful for retrospective landscape analysis, but including them in
# nearest-neighbour retrieval would leak benchmark labels.
RECOMMENDATION_FEATURE_ORDER = [
    key
    for key in DEFAULT_FEATURE_ORDER
    if key not in {"n_domains", "domain_balance", "domain_spatial_coherence"}
]

_SPATIAL_KEYS = {
    "spatial_autocorrelation",
    "cluster_tendency",
    "mean_nn_distance",
    "spatial_density_cv",
    "spatial_entropy",
}
_DOMAIN_KEYS = {"n_domains", "domain_balance", "domain_spatial_coherence"}


# ====================================================================
# Expression features
# ====================================================================
def _expression_features(X: np.ndarray) -> dict[str, float]:
    nz = X[X > 0] if X.size else np.array([0.0])
    lib_sizes = X.sum(axis=1)
    mean_expr = np.clip(X.mean(axis=0), 0.0, None)
    # Shannon entropy of the normalised mean-expression distribution.
    total_mean = float(mean_expr.sum())
    if total_mean > 0:
        probabilities = mean_expr / total_mean
        positive = probabilities > 0
        entropy = -float(np.sum(probabilities[positive] * np.log2(probabilities[positive])))
    else:
        entropy = 0.0

    return {
        "sparsity": float(1.0 - (X > 0).sum() / max(X.size, 1)),
        "mean_nonzero": float(nz.mean()) if len(nz) else 0.0,
        "library_mean": float(lib_sizes.mean()),
        "library_cv": float(lib_sizes.std() / (lib_sizes.mean() + 1e-8)),
        "expression_entropy": entropy,
    }


# ====================================================================
# Spatial features
# ====================================================================
def _spatial_features(X: np.ndarray, coords: np.ndarray) -> dict[str, float]:
    n = coords.shape[0]

    # Moran's I on the top-10 highest-variance genes
    try:
        from ..plugins.builtin.spatial_svg import _build_spatial_weight_matrix

        var_genes = np.argsort(X.var(axis=0))[::-1][: min(10, X.shape[1])]
        moran_vals: list[float] = []
        W, W_sum = _build_spatial_weight_matrix(coords, 6)
        Xc = X - X.mean(axis=0, keepdims=True)
        for g in var_genes:
            num = float(Xc[:, g].T @ W @ Xc[:, g])
            denom = float((Xc[:, g] ** 2).sum()) + 1e-12
            moran_vals.append((n / W_sum) * num / denom)
        spatial_ac = float(np.mean(moran_vals)) if moran_vals else float("nan")
    except Exception:
        spatial_ac = float("nan")

    # Hopkins statistic
    hopkins = _hopkins_statistic(coords, sample_size=min(50, n))

    # Mean nearest-neighbour distance (normalised by sqrt(area))
    nn_dist = _mean_nn_distance(coords)

    # CV of local density (10 × 10 grid)
    density_cv = _spatial_density_cv(coords)

    # 2D histogram entropy
    spatial_ent = _spatial_entropy(coords)

    return {
        "spatial_autocorrelation": spatial_ac,
        "cluster_tendency": hopkins,
        "mean_nn_distance": nn_dist,
        "spatial_density_cv": density_cv,
        "spatial_entropy": spatial_ent,
    }


def _hopkins_statistic(coords: np.ndarray, sample_size: int = 50) -> float:
    """Hopkins statistic: 1.0 = highly clustered, 0.5 = uniform random, <0.5 = regular."""
    n = coords.shape[0]
    k = min(sample_size, n)
    rng = np.random.default_rng(42)

    # k random points from the bounding box
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    random_pts = rng.uniform(mins, maxs, size=(k, coords.shape[1]))

    # k real points sampled without replacement
    real_idx = rng.choice(n, size=k, replace=False)
    real_pts = coords[real_idx]

    # Nearest-neighbour distances for both sets to the full coordinate set.
    # Real sampled points must exclude their own row; otherwise every distance
    # is zero and Hopkins degenerates to ~1 for every dataset.
    def _nn_sum(pts: np.ndarray, *, self_indices: np.ndarray | None = None) -> float:
        dists = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=2)
        if self_indices is not None:
            dists[np.arange(len(pts)), self_indices] = np.inf
        return float(dists.min(axis=1).sum())

    w = _nn_sum(random_pts)
    u = _nn_sum(real_pts, self_indices=real_idx)
    return w / (u + w + 1e-12)


def _mean_nn_distance(coords: np.ndarray) -> float:
    """Mean nearest-neighbour distance, normalised by sqrt(convex-hull area)."""
    n = coords.shape[0]
    if n < 2:
        return 0.0
    k = min(6, n - 1)
    try:
        from scipy.spatial import KDTree

        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=k + 1)
        # skip self (col 0)
        raw = float(dists[:, 1:].mean())
    except Exception:
        return float("nan")
    # Normalise by sqrt(bounding-box area)
    span = coords.max(axis=0) - coords.min(axis=0)
    area = np.prod(span[span > 0]) if span.max() > 0 else 1.0
    return raw / (np.sqrt(area) + 1e-12)


def _spatial_density_cv(coords: np.ndarray, grid_size: int = 10) -> float:
    """CV of cell density across a grid_size × grid_size grid."""
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    span = maxs - mins
    if np.any(span < 1e-12):
        return 0.0
    bins = [
        np.linspace(mins[d] - 1e-9, maxs[d] + 1e-9, grid_size + 1) for d in range(coords.shape[1])
    ]
    hist, _ = np.histogramdd(coords, bins=bins)  # type: ignore[call-overload]
    counts = hist.ravel()
    counts = counts[counts > 0]
    if len(counts) < 2:
        return 0.0
    return float(counts.std() / (counts.mean() + 1e-12))


def _spatial_entropy(coords: np.ndarray, grid_size: int = 10) -> float:
    """Shannon entropy of the 2D spatial histogram (higher = more uniform)."""
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    span = maxs - mins
    if np.any(span < 1e-12):
        return 0.0
    bins = [
        np.linspace(mins[d] - 1e-9, maxs[d] + 1e-9, grid_size + 1) for d in range(coords.shape[1])
    ]
    hist, _ = np.histogramdd(coords, bins=bins)  # type: ignore[call-overload]
    probs = hist.ravel() / hist.sum()
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


# ====================================================================
# Geometry features
# ====================================================================
def _geometry_features(X: np.ndarray) -> dict[str, float]:
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        # Economy SVD — use the smaller dimension
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        total_var = float((S**2).sum())
        if total_var > 0:
            cumsum = np.cumsum(S**2) / total_var
            eff_rank_90 = float(np.searchsorted(cumsum, 0.90) + 1)
            eff_rank_95 = float(np.searchsorted(cumsum, 0.95) + 1)
            probs = (S**2) / total_var
            sv_entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            sv_entropy = sv_entropy / max(np.log(len(S)), 1e-12)  # normalise
        else:
            eff_rank_90 = eff_rank_95 = sv_entropy = 0.0
    except np.linalg.LinAlgError:
        eff_rank_90 = eff_rank_95 = sv_entropy = float("nan")

    return {
        "effective_rank_90": eff_rank_90,
        "effective_rank_95": eff_rank_95,
        "sv_entropy": sv_entropy,
    }


# ====================================================================
# Domain features (ground truth)
# ====================================================================
def _domain_features(data: SpatialTable) -> dict[str, float]:
    truth_col = None
    for candidate in ("domain_truth", "domain", "cell_type"):
        if candidate in data.obs:
            truth_col = candidate
            break

    if truth_col is None:
        return {
            "n_domains": float("nan"),
            "domain_balance": float("nan"),
            "domain_spatial_coherence": float("nan"),
        }

    labels = data.obs[truth_col].to_numpy()
    uniq, counts = np.unique(labels, return_counts=True)
    n_domains = float(len(uniq))
    probs = counts / counts.sum()
    balance = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    balance = balance / max(np.log(len(uniq)), 1e-12)  # normalise

    # Spatial coherence: mean Moran's I of one-hot domain indicators
    coords = data.spatial
    if coords is not None and n_domains > 1:
        try:
            from ..plugins.builtin.spatial_svg import _build_spatial_weight_matrix

            W, W_sum = _build_spatial_weight_matrix(coords, 6)
            moran_per_domain = []
            for label in uniq:
                indicator = (labels == label).astype(float)
                indicator -= indicator.mean()
                denom = float((indicator**2).sum()) + 1e-12
                num = float(indicator.T @ W @ indicator)
                moran_per_domain.append((data.n_obs / W_sum) * num / denom)
            coherence = float(np.mean(moran_per_domain))
        except Exception:
            coherence = float("nan")
    else:
        coherence = float("nan")

    return {
        "n_domains": n_domains,
        "domain_balance": balance,
        "domain_spatial_coherence": coherence,
    }
