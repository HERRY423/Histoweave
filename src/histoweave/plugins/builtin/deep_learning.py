"""Compact PyTorch representation models for expression and registered images.

All models share validation, deterministic training, resource bounds, and provenance.
Architecture names select materially different corruption, variational, graph, or
multimodal feature paths while keeping a stable plugin contract.
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


@dataclass(frozen=True)
class _Architecture:
    name: str
    summary: str
    maturity: MethodMaturity
    multimodal: bool = False
    graph: bool = False
    denoising: bool = False
    variational: bool = False
    fusion: str = "concatenate"


_ARCHITECTURES = (
    _Architecture(
        "spatial_autoencoder",
        "Bounded autoencoder for spatial expression embeddings.",
        MethodMaturity.PRODUCTION,
    ),
    _Architecture(
        "denoising_spatial_autoencoder",
        "Denoising autoencoder robust to expression dropout.",
        MethodMaturity.PRODUCTION,
        denoising=True,
    ),
    _Architecture(
        "variational_spatial_autoencoder",
        "Variational autoencoder with a regularized latent space.",
        MethodMaturity.PRODUCTION,
        variational=True,
    ),
    _Architecture(
        "graph_expression_autoencoder",
        "Expression autoencoder augmented with spatial-neighbor features.",
        MethodMaturity.PRODUCTION,
        graph=True,
    ),
    _Architecture(
        "image_expression_autoencoder",
        "Joint autoencoder over expression and registered image patches.",
        MethodMaturity.BETA,
        multimodal=True,
    ),
    _Architecture(
        "image_expression_contrastive",
        "Scale-balanced dual-view image and expression representation.",
        MethodMaturity.BETA,
        multimodal=True,
        fusion="contrastive",
    ),
    _Architecture(
        "image_expression_attention",
        "Reliability-gated fusion of image and expression features.",
        MethodMaturity.BETA,
        multimodal=True,
        fusion="attention",
    ),
    _Architecture(
        "multimodal_graph_fusion",
        "Image-expression fusion with spatial-neighborhood context.",
        MethodMaturity.BETA,
        multimodal=True,
        graph=True,
        fusion="graph",
    ),
)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    return np.divide(matrix - mean, scale, out=np.zeros_like(matrix), where=scale > 1e-8)


def _neighbor_mean(data: SpatialTable, matrix: np.ndarray, k: int) -> np.ndarray:
    from ..._math import knn_indices

    coords = data.spatial
    if coords is None:
        raise ValueError("graph deep-learning models require obsm['spatial']")
    coords = np.asarray(coords, dtype=float)
    if data.n_obs < 2 or coords.shape[0] != data.n_obs or not np.isfinite(coords).all():
        raise ValueError("spatial coordinates must be finite and observation-aligned")
    # knn_indices includes self; drop it for a pure neighbourhood mean.
    k_query = min(int(k) + 1, data.n_obs)
    neighbors = knn_indices(coords[:, :2], k_query)
    if neighbors.shape[1] <= 1:
        return matrix.copy()
    return matrix[neighbors[:, 1:]].mean(axis=1)


def _image_features(data: SpatialTable, image_key: str, patch_size: int) -> np.ndarray:
    if image_key not in data.images:
        raise KeyError(f"registered image {image_key!r} does not exist")
    if data.spatial is None:
        raise ValueError("image-expression models require obsm['spatial']")
    image = np.asarray(data.images[image_key], dtype=float)
    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim == 3 and image.shape[0] <= 4 and image.shape[-1] > 4:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("registered image must be a finite 2D image with optional channels")

    coords = np.asarray(data.spatial, dtype=float)[:, :2]
    low = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - low, 1e-12)
    unit = (coords - low) / span
    xs = np.rint(unit[:, 0] * (image.shape[1] - 1)).astype(int)
    ys = np.rint(unit[:, 1] * (image.shape[0] - 1)).astype(int)
    radius = patch_size // 2
    features = []
    for x, y in zip(xs, ys, strict=True):
        patch = image[
            max(0, y - radius) : min(image.shape[0], y + radius + 1),
            max(0, x - radius) : min(image.shape[1], x + radius + 1),
        ]
        features.append(np.concatenate((patch.mean(axis=(0, 1)), patch.std(axis=(0, 1)))))
    return np.asarray(features, dtype=float)


class DeepRepresentationMethod(Method):
    """Shared bounded trainer for the registered architecture variants."""

    architecture: _Architecture

    def _features(self, data: SpatialTable) -> tuple[np.ndarray, list[str]]:
        expression = np.asarray(data.X, dtype=float)
        if expression.ndim != 2 or not np.isfinite(expression).all():
            raise ValueError(f"{self.spec.name}: X must be a finite two-dimensional matrix")
        if (expression < 0).any():
            raise ValueError(f"{self.spec.name}: X must contain non-negative expression")

        max_features = min(self.params["max_features"], data.n_vars)
        variance_order = np.argsort(expression.var(axis=0))[::-1][:max_features]
        expression = _standardize(np.log1p(expression[:, variance_order]))
        blocks = [expression]
        modalities = ["expression"]

        if self.architecture.graph:
            blocks.append(_standardize(_neighbor_mean(data, expression, self.params["k"])))
            modalities.append("spatial")

        if self.architecture.multimodal:
            image = _standardize(
                _image_features(data, self.params["image_key"], self.params["patch_size"])
            )
            if self.architecture.fusion == "contrastive":
                expression_norm = np.linalg.norm(blocks[0], axis=1, keepdims=True)
                image_norm = np.linalg.norm(image, axis=1, keepdims=True)
                blocks[0] = np.divide(
                    blocks[0],
                    expression_norm,
                    out=np.zeros_like(blocks[0]),
                    where=expression_norm > 0,
                )
                image = np.divide(image, image_norm, out=np.zeros_like(image), where=image_norm > 0)
            elif self.architecture.fusion == "attention":
                expr_reliability = np.mean(np.abs(blocks[0]), axis=1, keepdims=True)
                image_reliability = np.mean(np.abs(image), axis=1, keepdims=True)
                weights = np.exp(np.concatenate((expr_reliability, image_reliability), axis=1))
                weights /= weights.sum(axis=1, keepdims=True)
                blocks[0] *= weights[:, :1]
                image *= weights[:, 1:]
            blocks.append(image)
            modalities.append("image")
        return np.concatenate(blocks, axis=1).astype(np.float32), modalities

    def run(self, data: SpatialTable) -> SpatialTable:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                f"{self.spec.name} requires PyTorch; install histoweave-spatial[deep-learning]"
            ) from exc

        features, used_modalities = self._features(data)
        torch.manual_seed(self.params["seed"])
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        tensor = torch.as_tensor(features, dtype=torch.float32)
        input_dim = features.shape[1]
        latent_dim = min(self.params["latent_dim"], input_dim)
        hidden_dim = min(128, max(16, latent_dim * 2))

        class Autoencoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, values):
                latent = self.encoder(values)
                return self.decoder(latent), latent, None

        class VariationalAutoencoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
                self.mu = nn.Linear(hidden_dim, latent_dim)
                self.log_variance = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, values):
                hidden = self.hidden(values)
                mean = self.mu(hidden)
                log_variance = self.log_variance(hidden).clamp(-8.0, 8.0)
                if self.training:
                    latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
                else:
                    latent = mean
                return self.decoder(latent), latent, log_variance

        model = VariationalAutoencoder() if self.architecture.variational else Autoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params["learning_rate"])
        losses: list[float] = []
        model.train()
        for _ in range(self.params["epochs"]):
            optimizer.zero_grad()
            training_input = tensor
            if self.architecture.denoising:
                training_input = tensor + torch.randn_like(tensor) * self.params["noise"]
            reconstruction, latent, log_variance = model(training_input)
            loss = nn.functional.mse_loss(reconstruction, tensor)
            if log_variance is not None:
                loss = loss + self.params["kl_weight"] * (
                    -0.5 * torch.mean(1.0 + log_variance - latent.square() - log_variance.exp())
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            _, latent, _ = model(tensor)
        result = data.copy()
        result.obsm[self.params["key_added"]] = latent.cpu().numpy()
        result.uns.setdefault("deep_learning", {})[self.spec.name] = {
            "architecture": self.architecture.name,
            "modalities": used_modalities,
            "latent_dim": latent_dim,
            "epochs": self.params["epochs"],
            "final_loss": losses[-1],
            "seed": self.params["seed"],
        }
        return self.finalize(result)


def _register_architecture(architecture: _Architecture) -> None:
    modalities = (
        ("expression", "image", "spatial")
        if architecture.multimodal and architecture.graph
        else ("expression", "image")
        if architecture.multimodal
        else ("expression", "spatial")
        if architecture.graph
        else ("expression",)
    )
    params = (
        ParamSpec("latent_dim", "int", 16, minimum=2, maximum=256),
        ParamSpec("epochs", "int", 25, minimum=1, maximum=1000),
        ParamSpec("learning_rate", "float", 0.001, minimum=1e-8, maximum=1.0),
        ParamSpec("max_features", "int", 256, minimum=2, maximum=4096),
        ParamSpec("k", "int", 6, minimum=1, maximum=128),
        ParamSpec("image_key", "str", "image"),
        ParamSpec("patch_size", "int", 9, minimum=1, maximum=255),
        ParamSpec("noise", "float", 0.10, minimum=0.0, maximum=2.0),
        ParamSpec("kl_weight", "float", 0.001, minimum=0.0, maximum=1.0),
        ParamSpec("seed", "int", 0, minimum=0),
        ParamSpec("key_added", "str", f"X_{architecture.name}"),
    )
    spec = MethodSpec(
        name=architecture.name,
        category=MethodCategory.INTEGRATION,
        version="1.0.0",
        summary=architecture.summary,
        params=params,
        assumptions=("PyTorch 2.x is installed.", "X contains non-negative expression."),
        maturity=architecture.maturity,
        wraps="PyTorch 2.x",
        modalities=modalities,
        model_family="deep_learning",
    )
    class_name = "".join(part.title() for part in architecture.name.split("_"))
    generated = type(
        class_name,
        (DeepRepresentationMethod,),
        {
            "__module__": __name__,
            "__doc__": architecture.summary,
            "architecture": architecture,
            "spec": spec,
        },
    )
    register(generated)


for _architecture in _ARCHITECTURES:
    _register_architecture(_architecture)
