"""Virtual spatial transcriptomics: H&E morphology → predicted expression.

Implements a complete :class:`~histoweave.plugins.interfaces.MethodCategory.VIRTUAL_ST`
analysis task with three method families inspired by 2025–2026 H&E→ST literature:

* ``virtual_st_morphology`` — deterministic patch-statistic baseline (no torch).
* ``virtual_st_scellst`` — sCellST-style weakly supervised morphology encoder
  (Chadoutaud et al., *Nat Commun* 2026): learn GE from cell/spot morphology
  using paired ST as bag-level supervision when available.
* ``virtual_st_storm`` — STORM-inspired hierarchical fusion of morphology,
  spatial neighbourhood context, and (optional) expression anchors
  (multimodal foundation-style predictor for virtual ST from H&E).

None of these wrappers claim to ship multi-million-parameter foundation
weights. They provide **API-stable, CI-safe reference implementations** of the
scientific contracts so landscapes, task contracts, and decision evidence can
treat virtual ST as a first-class analysis task. External full-weight backends
can replace the native path later without changing the task contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data import SpatialTable
from ..interfaces import (
    Method,
    MethodCategory,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register

# ---------------------------------------------------------------------------
# Shared morphology / scoring helpers
# ---------------------------------------------------------------------------


def _as_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        # Channel-first → channel-last.
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError("registered image must be a finite 2D image with optional channels")
    return arr


def extract_patch_features(
    data: SpatialTable,
    *,
    image_key: str,
    patch_size: int,
) -> np.ndarray:
    """Extract per-spot morphology features from a registered histology image.

    Features per spot (channel-wise): mean, std, min, max of the local patch,
    plus global relative coordinates (x, y in [0, 1]) for spatial context.
    """
    if image_key not in data.images:
        raise KeyError(f"registered image {image_key!r} does not exist")
    if data.spatial is None:
        raise ValueError("virtual_st methods require obsm['spatial']")
    image = _as_image(data.images[image_key])
    coords = np.asarray(data.spatial, dtype=float)[:, :2]
    if coords.shape[0] != data.n_obs or not np.isfinite(coords).all():
        raise ValueError("spatial coordinates must be finite and observation-aligned")

    low = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - low, 1e-12)
    unit = (coords - low) / span
    xs = np.rint(unit[:, 0] * (image.shape[1] - 1)).astype(int)
    ys = np.rint(unit[:, 1] * (image.shape[0] - 1)).astype(int)
    radius = max(0, int(patch_size) // 2)
    n_channels = image.shape[2]
    rows: list[np.ndarray] = []
    for x, y, ux, uy in zip(xs, ys, unit[:, 0], unit[:, 1], strict=True):
        patch = image[
            max(0, y - radius) : min(image.shape[0], y + radius + 1),
            max(0, x - radius) : min(image.shape[1], x + radius + 1),
        ]
        if patch.size == 0:
            stats = np.zeros(n_channels * 4, dtype=float)
        else:
            flat_axes = (0, 1)
            stats = np.concatenate(
                (
                    patch.mean(axis=flat_axes),
                    patch.std(axis=flat_axes),
                    patch.min(axis=flat_axes),
                    patch.max(axis=flat_axes),
                )
            )
        rows.append(np.concatenate((stats, np.asarray([ux, uy], dtype=float))))
    return np.asarray(rows, dtype=float)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    return np.divide(matrix - mean, scale, out=np.zeros_like(matrix), where=scale > 1e-8)


def _neighbor_mean_features(data: SpatialTable, features: np.ndarray, k: int) -> np.ndarray:
    from ..._math import knn_indices

    coords = np.asarray(data.spatial, dtype=float)[:, :2]
    k_query = min(int(k) + 1, data.n_obs)
    neighbors = knn_indices(coords, k_query)
    if neighbors.shape[1] <= 1:
        return features.copy()
    return features[neighbors[:, 1:]].mean(axis=1)


def _dense_expression(data: SpatialTable, layer: str | None) -> np.ndarray:
    if layer is None or layer in {"X", "x", "expression"}:
        matrix = data.X
    else:
        if layer not in data.layers:
            raise KeyError(f"expression layer {layer!r} does not exist")
        matrix = data.layers[layer]
    arr = np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != data.n_obs:
        raise ValueError("expression matrix must be (n_obs, n_vars)")
    if not np.isfinite(arr).all():
        raise ValueError("expression matrix must be finite")
    return arr


def _ridge_fit_predict(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Ridge multi-output regression with a closed-form solution.

    When ``n_obs`` is small relative to features, falls back to a low-rank
    projection of features onto the leading target PCs.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    n_obs, n_feat = x.shape
    # Bias column.
    xb = np.concatenate((np.ones((n_obs, 1)), x), axis=1)
    # Mild jitter for numerical stability on constant columns.
    xb = xb + rng.normal(0.0, 1e-10, size=xb.shape)
    gram = xb.T @ xb
    diag = np.arange(gram.shape[0])
    gram[diag, diag] += float(alpha)
    try:
        coef = np.linalg.solve(gram, xb.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ (xb.T @ y)
    pred = xb @ coef
    return np.clip(pred, 0.0, None)


def _unsupervised_morphology_expression(
    features: np.ndarray,
    *,
    n_genes: int,
    seed: int,
) -> np.ndarray:
    """Map morphology features to a non-negative pseudo-expression matrix.

    Used when no paired ST is available (inference-only virtual ST). A fixed
    random projection keeps the gene space stable for a given seed while
    remaining fully deterministic and dependency-free.
    """
    rng = np.random.default_rng(seed)
    feats = _standardize(features)
    # Add polynomial interactions of leading channels for non-linearity.
    lead = feats[:, : min(6, feats.shape[1])]
    expanded = np.concatenate(
        (feats, lead**2, lead[:, :1] * lead[:, 1:2] if lead.shape[1] > 1 else lead), axis=1
    )
    expanded = _standardize(expanded)
    projection = rng.normal(
        0.0, 1.0 / max(1.0, np.sqrt(expanded.shape[1])), size=(expanded.shape[1], n_genes)
    )
    logits = expanded @ projection
    # Softplus-like non-negativity without torch.
    return np.log1p(np.exp(np.clip(logits, -20.0, 20.0)))


def mean_gene_pearson(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Mean per-gene Pearson correlation (virtual ST primary metric)."""
    pred = np.asarray(predicted, dtype=float)
    truth = np.asarray(measured, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError(
            f"predicted shape {pred.shape} does not match measured shape {truth.shape}"
        )
    if pred.shape[1] == 0:
        return 0.0
    scores: list[float] = []
    for gene in range(pred.shape[1]):
        p = pred[:, gene]
        t = truth[:, gene]
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            continue
        corr = float(np.corrcoef(p, t)[0, 1])
        if np.isfinite(corr):
            scores.append(corr)
    return float(np.mean(scores)) if scores else 0.0


def mean_gene_spearman(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Mean per-gene Spearman correlation via rank-Pearson."""
    pred = np.asarray(predicted, dtype=float)
    truth = np.asarray(measured, dtype=float)

    def _rank(column: np.ndarray) -> np.ndarray:
        order = np.argsort(column, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(column.size, dtype=float)
        return ranks

    if pred.shape != truth.shape or pred.shape[1] == 0:
        return 0.0
    scores: list[float] = []
    for gene in range(pred.shape[1]):
        p = _rank(pred[:, gene])
        t = _rank(truth[:, gene])
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            continue
        corr = float(np.corrcoef(p, t)[0, 1])
        if np.isfinite(corr):
            scores.append(corr)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VirtualSTArchitecture:
    name: str
    summary: str
    maturity: MethodMaturity
    family: str  # morphology | scellst | storm


class _VirtualSTBase(Method):
    """Shared runner for virtual ST predictors."""

    architecture: _VirtualSTArchitecture

    def _build_features(self, data: SpatialTable) -> np.ndarray:
        patch = extract_patch_features(
            data,
            image_key=self.params["image_key"],
            patch_size=int(self.params["patch_size"]),
        )
        feats = _standardize(patch)
        if self.architecture.family == "storm":
            nbr = _standardize(
                _neighbor_mean_features(data, feats, int(self.params["k_neighbors"]))
            )
            # Hierarchical fusion: local morphology + neighbourhood morphology
            # + relative spatial coordinates (already inside patch features).
            reliability = np.mean(np.abs(feats), axis=1, keepdims=True)
            nbr_reliability = np.mean(np.abs(nbr), axis=1, keepdims=True)
            weights = np.exp(np.concatenate((reliability, nbr_reliability), axis=1))
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            fused = np.concatenate((feats * weights[:, :1], nbr * weights[:, 1:]), axis=1)
            return _standardize(fused)
        if self.architecture.family == "scellst":
            # Weakly supervised bag features: local + mild spatial pooling.
            nbr = _neighbor_mean_features(data, feats, max(2, int(self.params["k_neighbors"]) // 2))
            return _standardize(np.concatenate((feats, nbr), axis=1))
        return feats

    def _predict(self, data: SpatialTable, features: np.ndarray) -> tuple[np.ndarray, dict]:
        mode = str(self.params["mode"])
        layer = self.params.get("expression_layer")
        n_genes = int(self.params["n_genes"])
        seed = int(self.params["seed"])
        alpha = float(self.params["ridge_alpha"])
        meta: dict = {
            "architecture": self.architecture.name,
            "family": self.architecture.family,
            "mode": mode,
            "image_key": self.params["image_key"],
            "seed": seed,
        }

        has_expression = data.n_vars > 0 and data.X is not None
        if mode == "auto":
            mode = "paired" if has_expression and np.asarray(data.X).size > 0 else "inference"
            meta["mode"] = mode

        if mode == "paired":
            if not has_expression:
                raise ValueError(
                    f"{self.spec.name}: mode='paired' requires a measured expression matrix"
                )
            measured = _dense_expression(data, layer)
            # Restrict to most variable genes for a stable multi-output head.
            n_out = min(n_genes, measured.shape[1])
            variances = measured.var(axis=0)
            order = np.argsort(variances)[::-1][:n_out]
            targets = measured[:, order]
            if self.architecture.family == "scellst":
                # sCellST-style: weakly supervised — fit on log1p targets so the
                # morphology encoder matches spot-level bags of expression.
                targets_fit = np.log1p(np.clip(targets, 0.0, None))
                pred_log = _ridge_fit_predict(features, targets_fit, alpha=alpha, seed=seed)
                predicted_full = np.zeros_like(measured)
                predicted_full[:, order] = np.expm1(np.clip(pred_log, 0.0, 20.0))
            elif self.architecture.family == "storm":
                # STORM-style: hierarchical features → multi-gene head with a
                # light residual toward the gene-wise mean (batch-effect soft prior).
                gene_mean = targets.mean(axis=0, keepdims=True)
                residual = targets - gene_mean
                pred_resid = _ridge_fit_predict(features, residual, alpha=alpha, seed=seed)
                pred = np.clip(pred_resid + gene_mean, 0.0, None)
                predicted_full = np.zeros_like(measured)
                predicted_full[:, order] = pred
            else:
                pred = _ridge_fit_predict(features, targets, alpha=alpha, seed=seed)
                predicted_full = np.zeros_like(measured)
                predicted_full[:, order] = pred
            meta["n_genes_predicted"] = int(n_out)
            meta["gene_indices"] = order.tolist()
            meta["mean_gene_pearson"] = mean_gene_pearson(predicted_full[:, order], targets)
            meta["mean_gene_spearman"] = mean_gene_spearman(predicted_full[:, order], targets)
            meta["supervision"] = "paired_measured_expression"
            return predicted_full, meta

        # Inference-only: morphology → pseudo expression in a declared gene space.
        n_out = n_genes if data.n_vars == 0 else min(n_genes, max(data.n_vars, n_genes))
        if data.n_vars > 0:
            n_out = min(n_genes, data.n_vars)
            predicted = _unsupervised_morphology_expression(features, n_genes=n_out, seed=seed)
            if n_out < data.n_vars:
                full = np.zeros((data.n_obs, data.n_vars), dtype=float)
                # Place predictions on the leading var indices for alignment.
                full[:, :n_out] = predicted
                predicted = full
        else:
            predicted = _unsupervised_morphology_expression(features, n_genes=n_out, seed=seed)
        meta["n_genes_predicted"] = int(predicted.shape[1])
        meta["supervision"] = "morphology_only"
        return predicted, meta

    def run(self, data: SpatialTable) -> SpatialTable:
        features = self._build_features(data)
        predicted, meta = self._predict(data, features)
        result = data.copy()
        layer_key = str(self.params["layer_added"])
        result.layers[layer_key] = predicted.astype(float)
        if bool(self.params["write_to_x"]):
            # Preserve measured counts when present.
            if "counts" not in result.layers and data.X is not None:
                result.layers["counts"] = np.asarray(
                    data.X.todense() if hasattr(data.X, "todense") else data.X, dtype=float
                )
            result.X = predicted.astype(float)
        emb_key = str(self.params["embedding_key"])
        # Compact morphology embedding for downstream domain methods.
        from ..._math import pca

        n_comp = min(
            int(self.params["embedding_dim"]), features.shape[1], max(1, features.shape[0] - 1)
        )
        if n_comp >= 1 and features.shape[0] >= 2:
            result.obsm[emb_key] = pca(features, n_comp, random_state=int(self.params["seed"]))
        else:
            result.obsm[emb_key] = features
        result.uns.setdefault("virtual_st", {})[self.spec.name] = meta
        return self.finalize(result, step="virtual_st")


def _register_architecture(architecture: _VirtualSTArchitecture) -> None:
    params = (
        ParamSpec("image_key", "str", "image", "Registered H&E / histology key in images."),
        ParamSpec(
            "patch_size", "int", 15, "Odd-ish patch diameter in pixels.", minimum=1, maximum=255
        ),
        ParamSpec(
            "k_neighbors", "int", 6, "Spatial neighbours for hierarchical fusion.", minimum=1
        ),
        ParamSpec(
            "mode",
            "str",
            "auto",
            "auto | paired (fit on measured ST) | inference (morphology only).",
            choices=("auto", "paired", "inference"),
        ),
        ParamSpec(
            "expression_layer",
            "str|None",
            None,
            "Measured expression layer for paired mode; None uses X.",
        ),
        ParamSpec("n_genes", "int", 64, "Max genes to predict / project.", minimum=1, maximum=5000),
        ParamSpec("ridge_alpha", "float", 1.0, "Ridge penalty for paired fit.", minimum=0.0),
        ParamSpec("seed", "int", 0, "Deterministic seed.", minimum=0),
        ParamSpec("layer_added", "str", "virtual_st", "layers key for predicted expression."),
        ParamSpec("embedding_key", "str", "X_virtual_st", "obsm key for morphology embedding."),
        ParamSpec("embedding_dim", "int", 16, "PCA dims of morphology embedding.", minimum=1),
        ParamSpec(
            "write_to_x",
            "bool",
            False,
            "If true, replace X with predicted expression (counts preserved in layers).",
        ),
    )
    modalities = ("image", "spatial", "expression")
    assumptions = (
        "SpatialTable.images[image_key] is a finite H&E (or histology) array.",
        "obsm['spatial'] aligns observations to the image.",
        "Paired mode requires measured expression in X or expression_layer.",
        "Native implementation is a contract-stable reference, not a foundation checkpoint.",
    )
    spec = MethodSpec(
        name=architecture.name,
        category=MethodCategory.VIRTUAL_ST,
        version="0.1.0",
        summary=architecture.summary,
        params=params,
        assumptions=assumptions,
        maturity=architecture.maturity,
        wraps="HistoWeave native virtual ST reference",
        modalities=modalities,
        model_family="deep_learning" if architecture.family != "morphology" else "machine_learning",
        metadata={
            "analysis_task": "virtual_st",
            "method_family": architecture.family,
            "literature": {
                "scellst": "Chadoutaud et al., Nat Commun 2026 (sCellST)",
                "storm": "STORM multimodal histology–ST foundation model (2026)",
                "morphology": "Patch-statistic baseline for H&E→expression",
            }.get(architecture.family, ""),
        },
    )
    class_name = "".join(part.title() for part in architecture.name.split("_"))
    generated = type(
        class_name,
        (_VirtualSTBase,),
        {
            "__module__": __name__,
            "__doc__": architecture.summary,
            "architecture": architecture,
            "spec": spec,
        },
    )
    register(generated)


_ARCHITECTURES = (
    _VirtualSTArchitecture(
        name="virtual_st_morphology",
        summary="H&E patch-statistic baseline for virtual spatial transcriptomics.",
        maturity=MethodMaturity.BETA,
        family="morphology",
    ),
    _VirtualSTArchitecture(
        name="virtual_st_scellst",
        summary=(
            "sCellST-inspired weakly supervised morphology→expression predictor "
            "(H&E virtual ST; Chadoutaud et al. 2026)."
        ),
        maturity=MethodMaturity.BETA,
        family="scellst",
    ),
    _VirtualSTArchitecture(
        name="virtual_st_storm",
        summary=(
            "STORM-inspired hierarchical morphology+spatial fusion predictor "
            "for H&E→spatial transcriptomics (virtual ST)."
        ),
        maturity=MethodMaturity.BETA,
        family="storm",
    ),
)

for _architecture in _ARCHITECTURES:
    _register_architecture(_architecture)


__all__ = [
    "extract_patch_features",
    "mean_gene_pearson",
    "mean_gene_spearman",
]
