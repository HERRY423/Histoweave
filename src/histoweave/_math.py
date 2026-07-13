"""Small, dependency-free numerical helpers.

These are deliberately minimal NumPy implementations so the scaffold runs with zero
heavy dependencies. In a real deployment these steps are delegated to the wrapped
methods (scikit-learn, scanpy, RAPIDS, ...); they are isolated here behind plain
functions so that substitution is trivial.
"""

from __future__ import annotations

import numpy as np


def zscore(X: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Standardize to zero mean / unit variance along ``axis``."""
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + eps)


def pca(X: np.ndarray, n_components: int, random_state: int = 0) -> np.ndarray:
    """Truncated PCA on mean-centred data. Returns the scores matrix.

    Two equivalent routes, chosen by shape:

    * When features outnumber observations (the spatial-omics norm — thousands of genes,
      fewer spots), eigendecompose the small ``n x n`` Gram matrix. This yields the same
      scores as the SVD (up to each component's arbitrary sign) without ever forming the
      ``n x d`` right singular vectors that the plain SVD computes and we immediately
      discard — the difference between seconds and a minute on a real Visium slide.
    * Otherwise fall back to the economy SVD.

    Both routes are deterministic for a given input.
    """
    n_components = int(min(n_components, min(X.shape)))
    Xc = X - X.mean(axis=0, keepdims=True)
    n_obs, n_features = Xc.shape

    if n_features > n_obs:
        gram = Xc @ Xc.T  # (n_obs, n_obs) — small when genes >> spots
        eigvals, eigvecs = np.linalg.eigh(gram)  # ascending order
        top = np.argsort(eigvals)[::-1][:n_components]
        singular_values = np.sqrt(np.clip(eigvals[top], 0.0, None))
        return eigvecs[:, top] * singular_values

    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


def kmeans(
    X: np.ndarray,
    k: int,
    n_iter: int = 100,
    n_init: int = 4,
    random_state: int = 0,
) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding. Returns integer cluster labels.

    Runs ``n_init`` restarts and keeps the lowest-inertia solution, so results are
    deterministic for a fixed ``random_state``.
    """
    k = int(min(k, X.shape[0]))
    best_labels: np.ndarray | None = None
    best_inertia = np.inf
    for init in range(n_init):
        rng = np.random.default_rng(random_state + init)
        centers = _kmeanspp_init(X, k, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(n_iter):
            dists = _sqdist(X, centers)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                members = X[labels == c]
                if len(members):
                    centers[c] = members.mean(axis=0)
        inertia = _sqdist(X, centers)[np.arange(X.shape[0]), labels].sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    assert best_labels is not None
    return best_labels


def _sqdist(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[rng.integers(n)]
    closest = ((X - centers[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        probs = closest / closest.sum() if closest.sum() > 0 else np.full(n, 1 / n)
        centers[c] = X[rng.choice(n, p=probs)]
        closest = np.minimum(closest, ((X - centers[c]) ** 2).sum(axis=1))
    return centers


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` nearest neighbours (by Euclidean distance) per point.

    Brute force — fine for the small canonical datasets this scaffold targets; a real
    deployment uses squidpy's spatial graph on chunked data.
    """
    k = int(min(k, coords.shape[0]))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return np.argsort(d, axis=1)[:, :k]


def neighborhood_mean(features: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    """Average each point's features over its ``k`` spatial neighbours.

    This is the ingredient that makes clustering *spatially aware* (the idea behind
    BANKSY / neighbourhood-augmented domain detection).
    """
    idx = knn_indices(coords, k)
    return features[idx].mean(axis=1)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index between two labelings (clustering-agreement metric)."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(contingency, (class_idx, cluster_idx), 1)

    def comb2(x: np.ndarray | np.int64) -> np.ndarray | np.int64:
        return x * (x - 1) // 2

    sum_comb_c = comb2(contingency.sum(axis=1)).sum()
    sum_comb_k = comb2(contingency.sum(axis=0)).sum()
    sum_comb = comb2(contingency).sum()
    n = labels_true.shape[0]
    total = comb2(np.int64(n))
    if total == 0:
        return 1.0
    expected = sum_comb_c * sum_comb_k / total
    max_index = (sum_comb_c + sum_comb_k) / 2
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


def proportions_rmsd(true: np.ndarray, pred: np.ndarray) -> float:
    """Root-mean-square deviation between two proportion matrices (rows sum to 1).

    Lower is better; 0 = identical, max = √2 (completely disjoint supports).
    """
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"shape mismatch: true {true_arr.shape} vs pred {pred_arr.shape}"
        )
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))
