"""Built-in spatial-domain detection (neighbourhood-augmented k-means).

A deliberately small stand-in for wrapped methods like BANKSY / STAGATE / GraphST /
SpaGCN. It demonstrates the interface *and* the key idea those methods share: mixing a
cell's own expression with its spatial neighbourhood so that detected domains are
spatially coherent rather than purely transcriptomic clusters.
"""

from __future__ import annotations

import pandas as pd

from ..._math import kmeans, neighborhood_mean, pca, zscore
from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register


@register
class KMeansDomains(Method):
    spec = MethodSpec(
        name="kmeans",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="PCA + neighbourhood-augmented k-means (BANKSY-style).",
        params=(
            ParamSpec(
                "n_domains",
                "int|None",
                None,
                "Clusters; falls back to uns['n_domains'].",
                minimum=2,
            ),
            ParamSpec("n_pcs", "int", 15, "PCs used for clustering.", minimum=1),
            ParamSpec(
                "n_neighbors", "int", 12, "Spatial neighbours for smoothing.", minimum=1
            ),
            ParamSpec(
                "spatial_weight",
                "float",
                0.3,
                "0=expression only, 1=neighbourhood only.",
                minimum=0,
                maximum=1,
            ),
            ParamSpec("key_added", "str", "domain", "obs column for the result."),
            ParamSpec("random_state", "int", 0, "Seed for reproducibility."),
        ),
        assumptions=("obsm['spatial'] present for the neighbourhood term.",),
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        data = data.copy()
        k = self.params["n_domains"] or data.uns.get("n_domains")
        if not k:
            raise ValueError("n_domains not given and uns['n_domains'] is absent")

        feats = zscore(data.X)
        scores = pca(feats, self.params["n_pcs"], self.params["random_state"])

        w = float(self.params["spatial_weight"])
        coords = data.spatial
        if coords is not None and w > 0:
            nbr = neighborhood_mean(scores, coords, self.params["n_neighbors"])
            embedding = (1 - w) * zscore(scores) + w * zscore(nbr)
        else:
            embedding = zscore(scores)

        labels = kmeans(embedding, int(k), random_state=self.params["random_state"])

        key = self.params["key_added"]
        data.obs[key] = pd.Categorical([f"domain_{lab}" for lab in labels])
        data.obsm["X_pca"] = scores
        return self.finalize(data, step="domain_detection")
