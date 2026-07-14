"""Batch-integration / harmonisation methods.

Multi-sample spatial experiments need to remove technical batch effects before
joint analysis. This module provides simple, dependency-light integration methods
that run without the heavy scvi-tools / Harmony stacks:

* ``combat`` — Empirical Bayes batch-effect correction (ComBat).
  Implemented in pure NumPy/scipy following the original Johnson, Li & Rabinovic
  (2007) parametric ComBat model. Suitable for merging spatial samples from
  different slides / runs / donors when a per-observation batch label is available.

* ``harmony`` — Harmony embedding-space batch integration (Korsunsky et al., 2019),
  wrapping the pure-Python ``harmonypy`` reference implementation. Operates on a
  low-dimensional embedding (typically PCA) held in ``obsm`` and writes a
  batch-corrected embedding back to a new ``obsm`` key, which downstream neighbour
  graphs / clustering / domain detection can consume. This is the scanpy-standard
  integration path and does not touch ``X``.

Method-to-wrap candidates for later phases (when scvi-tools is available):
  * scVI / scANVI (Lopez et al., 2018; Xu et al., 2020)
  * BBKNN (Polanski et al., 2020)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodMaturity, MethodSpec, ParamSpec
from ..registry import register

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


@register
class ComBatIntegration(Method):
    """Parametric ComBat batch correction for spatial transcriptomics.

    Adjusts each gene's expression so that the mean and variance per batch match
    the global (pooled) mean and variance.  This is the standard first-pass
    integration method — fast, interpretable, and dependency-free.

    The corrected matrix replaces ``data.X``; the original is stashed in a new
    layer ``layers['pre_combat']`` so the transformation is reversible.

    References
    ----------
    Johnson, Li & Rabinovic (2007). Adjusting batch effects in microarray
    expression data using empirical Bayes methods. *Biostatistics* 8(1).
    """

    spec = MethodSpec(
        name="combat",
        category=MethodCategory.INTEGRATION,
        version="0.1.0",
        summary="Parametric ComBat batch-effect correction (NumPy implementation).",
        params=(
            ParamSpec("batch_key", "str", "batch", "obs column with batch labels."),
            ParamSpec("parametric", "bool", True, "Use parametric (True) or non-parametric."),
            ParamSpec("store_pre", "str", "pre_combat", "Layer name for pre-correction matrix."),
        ),
        assumptions=(
            "obs[batch_key] holds batch labels (integers or strings).",
            "Normalised, log-transformed expression recommended.",
        ),
        wraps="ComBat (Johnson et al. 2007)",
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        batch_key = self.params["batch_key"]
        if batch_key not in data.obs:
            columns = list(data.obs.columns)
            raise ValueError(
                f"Batch key '{batch_key}' not found in obs. Available columns: {columns}"
            )

        batches = data.obs[batch_key].to_numpy()
        unique_batches = np.unique(batches)
        if len(unique_batches) < 2:
            # Single batch: nothing to correct.
            return self.finalize(data, step="integration")
        too_small = [str(batch) for batch in unique_batches if np.sum(batches == batch) < 2]
        if too_small:
            raise ValueError(
                "ComBat requires at least two observations per batch; "
                f"undersized batches: {too_small}"
            )

        X = np.asarray(data.X, dtype=float)
        data.layers.setdefault(self.params["store_pre"], X.copy())

        if self.params["parametric"]:
            X_corrected = _combat_parametric(X, batches, unique_batches)
        else:
            X_corrected = _combat_nonparametric(X, batches, unique_batches)

        data.X = X_corrected
        data.uns["integration"] = {
            "method": "combat",
            "batch_key": batch_key,
            "n_batches": int(len(unique_batches)),
            "batches": [str(b) for b in unique_batches],
            "parametric": self.params["parametric"],
        }
        return self.finalize(data, step="integration")


# ---------------------------------------------------------------------------
# Parametric ComBat (Johnson et al. 2007)
# ---------------------------------------------------------------------------
def _combat_parametric(
    X: np.ndarray,
    batches: np.ndarray,
    unique_batches: np.ndarray,
) -> np.ndarray:
    """Parametric ComBat: pool means and variances via empirical Bayes.

    Algorithm (per gene):
    1. Standardise within each batch: Z_bg = (Y_bg - μ_bg) / σ_bg
    2. Pool parameters across genes via empirical Bayes shrinkage toward the
       grand mean/variance.
    3. Rescale to the global mean and variance.
    """
    _, n_genes = X.shape
    # Design matrix: grand mean (intercept) + batch indicators (one-hot)
    # We use the simpler "standardise then pool" form from the original paper.

    # Per-batch per-gene mean and variance
    batch_means = {}
    batch_vars = {}
    for b in unique_batches:
        mask = batches == b
        Xb = X[mask]
        batch_means[b] = Xb.mean(axis=0)
        batch_vars[b] = Xb.var(axis=0, ddof=1) + 1e-8

    # Grand (pooled) mean and variance per gene
    grand_mean = X.mean(axis=0)
    grand_var = X.var(axis=0, ddof=1) + 1e-8

    # Step 1: standardise within each batch
    Z = np.zeros_like(X)
    for b in unique_batches:
        mask = batches == b
        Z[mask] = (X[mask] - batch_means[b]) / np.sqrt(batch_vars[b])

    # Step 2: empirical Bayes shrinkage of batch parameters toward the grand
    # mean/variance (per gene).  γ*_g and δ*_g are the pooled estimates.
    # We use a simple prior: shrink toward 0 mean, 1 variance.
    gamma_star = {}
    delta_star = {}
    for b in unique_batches:
        mask = batches == b
        nb = mask.sum()
        # Prior: gamma ~ N(0, τ²), delta ~ IG(λ, θ)
        # Empirical estimates from the standardised data:
        gamma_hat = Z[mask].mean(axis=0)  # should be ~0 after standardisation
        delta_hat = Z[mask].var(axis=0, ddof=1) + 1e-8  # should be ~1

        # Shrinkage: weighted average of batch-specific and pooled estimates
        # τ² estimated from variance of gamma_hat across genes
        tau2 = max(np.var(gamma_hat) - np.mean(delta_hat) / nb, 1e-6)
        gamma_star[b] = (nb * tau2) / (nb * tau2 + delta_hat) * gamma_hat

        # λ, θ for the inverse-gamma prior on δ
        # Simple shrinkage: pull toward 1.0
        lambda_prior = 2.0
        theta_prior = 1.0
        n_star = nb + 2 * lambda_prior
        delta_star[b] = (nb * delta_hat + 2 * lambda_prior * theta_prior) / n_star
        delta_star[b] = np.maximum(delta_star[b], 1e-8)

    # Step 3: adjust
    X_adj = np.zeros_like(X)
    for b in unique_batches:
        mask = batches == b
        X_adj[mask] = (Z[mask] - gamma_star[b]) / np.sqrt(delta_star[b]) * np.sqrt(
            grand_var
        ) + grand_mean

    return X_adj


def _combat_nonparametric(
    X: np.ndarray,
    batches: np.ndarray,
    unique_batches: np.ndarray,
) -> np.ndarray:
    """Non-parametric ComBat: rank-based quantile normalisation across batches.

    Simpler than the parametric version: for each gene independently, the values
    in each batch are replaced by their ranks, which are then mapped to the
    global quantile function for that gene.
    """
    n_obs, n_genes = X.shape
    X_adj = np.zeros_like(X)

    for g in range(n_genes):
        xg = X[:, g]
        from scipy.stats import rankdata

        # Quantile-normalise each batch independently to the pooled distribution.
        # The previous implementation ranked the pooled vector once and never used
        # ``batches`` at all, making the advertised batch correction a silent no-op.
        for batch in unique_batches:
            mask = batches == batch
            within_batch = xg[mask]
            ranks = rankdata(within_batch, method="average")
            quantiles = (ranks - 0.5) / len(within_batch)
            X_adj[mask, g] = np.quantile(xg, quantiles, method="linear")

    return X_adj


@register
class HarmonyIntegration(Method):
    """Harmony batch integration in embedding space (Korsunsky et al., 2019).

    Harmony iteratively clusters cells in a low-dimensional embedding and applies
    a soft, cluster-aware linear correction that removes batch offsets while
    preserving biological structure. Unlike :class:`ComBatIntegration`, it does
    **not** modify the expression matrix ``X``; it corrects an embedding (usually
    PCA) and writes the result to a new ``obsm`` key. Downstream neighbour graphs,
    clustering, and spatial-domain detection then run on the corrected embedding.

    Behaviour
    ---------
    * If ``use_rep`` is present in ``obsm`` it is used directly as the input
      embedding. Otherwise a PCA of ``X`` (``n_pcs`` components) is computed on the
      fly (requires scikit-learn) and stored in ``obsm['X_pca']``.
    * The corrected embedding is written to ``obsm[key_added]`` (default
      ``'X_pca_harmony'``), matching the scanpy convention so existing
      embedding-consuming methods pick it up with no change.
    * With a single batch, the input embedding is copied through unchanged.

    References
    ----------
    Korsunsky et al. (2019). Fast, sensitive and accurate integration of
    single-cell data with Harmony. *Nature Methods* 16, 1289–1296.
    """

    spec = MethodSpec(
        name="harmony",
        category=MethodCategory.INTEGRATION,
        version="0.1.0",
        summary="Harmony embedding-space batch integration (harmonypy).",
        params=(
            ParamSpec("batch_key", "str", "batch", "obs column with batch labels."),
            ParamSpec(
                "use_rep",
                "str",
                "X_pca",
                "obsm key of the input embedding; computed via PCA if absent.",
            ),
            ParamSpec(
                "key_added",
                "str",
                "X_pca_harmony",
                "obsm key for the batch-corrected embedding.",
            ),
            ParamSpec("n_pcs", "int", 30, "PCA components if the embedding is built.", minimum=2),
            ParamSpec("theta", "float", 2.0, "Diversity-clustering penalty.", minimum=0.0),
            ParamSpec("max_iter_harmony", "int", 10, "Max Harmony iterations.", minimum=1),
            ParamSpec("seed", "int", 0, "Random seed for PCA / Harmony.", minimum=0),
        ),
        assumptions=(
            "obs[batch_key] holds batch labels (integers or strings).",
            "A meaningful low-dimensional embedding (PCA) captures the shared signal.",
            "Normalised, log-transformed expression recommended before PCA.",
        ),
        assays=("visium", "xenium", "cosmx", "merscope"),
        maturity=MethodMaturity.BETA,
        wraps="harmonypy.run_harmony (Korsunsky et al. 2019)",
        language="python",
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        batch_key = self.params["batch_key"]
        if batch_key not in data.obs:
            columns = list(data.obs.columns)
            raise ValueError(
                f"Batch key '{batch_key}' not found in obs. Available columns: {columns}"
            )

        embedding = self._get_or_build_embedding(data)
        batches = data.obs[batch_key].astype(str).to_numpy()
        unique_batches = np.unique(batches)
        key_added = self.params["key_added"]

        if len(unique_batches) < 2:
            # Single batch: nothing to correct, pass the embedding through.
            data.obsm[key_added] = np.asarray(embedding, dtype=float).copy()
            data.uns["integration"] = {
                "method": "harmony",
                "batch_key": batch_key,
                "n_batches": int(len(unique_batches)),
                "batches": [str(b) for b in unique_batches],
                "use_rep": self.params["use_rep"],
                "key_added": key_added,
                "note": "single batch — embedding copied unchanged",
            }
            return self.finalize(data, step="integration")

        corrected = self._run_harmony(embedding, data.obs, batch_key)
        data.obsm[key_added] = corrected
        data.uns["integration"] = {
            "method": "harmony",
            "batch_key": batch_key,
            "n_batches": int(len(unique_batches)),
            "batches": [str(b) for b in unique_batches],
            "use_rep": self.params["use_rep"],
            "key_added": key_added,
            "theta": float(self.params["theta"]),
            "max_iter_harmony": int(self.params["max_iter_harmony"]),
        }
        return self.finalize(data, step="integration")

    # -- helpers ---------------------------------------------------------------
    def _get_or_build_embedding(self, data: SpatialTable) -> np.ndarray:
        use_rep = self.params["use_rep"]
        if use_rep in data.obsm:
            return np.asarray(data.obsm[use_rep], dtype=float)

        try:
            from sklearn.decomposition import PCA
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                f"obsm['{use_rep}'] is absent and scikit-learn is required to build a "
                "PCA embedding. Install with: pip install 'histoweave-spatial[harmony]'"
            ) from exc

        X = np.asarray(data.X, dtype=float)
        n_comp = int(min(self.params["n_pcs"], min(X.shape) - 1))
        n_comp = max(n_comp, 2)
        pca = PCA(n_components=n_comp, random_state=int(self.params["seed"]))
        emb = pca.fit_transform(X)
        # Persist so callers / provenance can see the embedding that Harmony used.
        data.obsm.setdefault("X_pca", emb)
        return emb

    def _run_harmony(
        self, embedding: np.ndarray, obs: pd.DataFrame, batch_key: str
    ) -> np.ndarray:
        try:
            import harmonypy
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                "harmonypy is required for Harmony integration. "
                "Install with: pip install 'histoweave-spatial[harmony]'"
            ) from exc

        try:  # harmonypy exposes a seed knob in recent releases only
            import random

            random.seed(int(self.params["seed"]))
            np.random.seed(int(self.params["seed"]))
        except Exception:  # pragma: no cover - defensive
            pass

        embedding = np.asarray(embedding, dtype=float)
        n_cells, n_features = embedding.shape
        ho = harmonypy.run_harmony(
            embedding,
            obs,
            [batch_key],
            theta=float(self.params["theta"]),
            max_iter_harmony=int(self.params["max_iter_harmony"]),
        )
        # harmonypy's ``Z_corr`` orientation changed across releases: older
        # versions return (n_features, n_cells), newer ones (n_cells, n_features).
        # Orient robustly by matching the observation axis rather than assuming.
        z = np.asarray(ho.Z_corr, dtype=float)
        if z.shape == (n_cells, n_features):
            return z
        if z.shape == (n_features, n_cells):
            return z.T
        # Fallback: put the axis whose length equals n_cells first.
        if z.shape[0] == n_cells:
            return z
        if z.shape[1] == n_cells:
            return z.T
        raise ValueError(  # pragma: no cover - defensive
            f"unexpected harmonypy Z_corr shape {z.shape} for embedding {embedding.shape}"
        )
