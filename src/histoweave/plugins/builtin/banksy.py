"""BANKSY spatial-domain detection through the shared R container bridge."""

from __future__ import annotations

from ...data import SpatialTable
from ..interfaces import MethodCategory, MethodSpec, ParamSpec
from ..registry import register
from ._r_base import RContainerMethod


@register
class BANKSYDomains(RContainerMethod):
    """Run Bioconductor BANKSY without duplicating the h5ad/R bridge."""

    spec = MethodSpec(
        name="banksy",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="BANKSY neighbourhood-augmented spatial-domain detection.",
        params=(
            ParamSpec(
                "lambda_param", "float", 0.8,
                "Weight assigned to spatial-neighbourhood features.",
                minimum=0.0, maximum=1.0,
            ),
            ParamSpec("k_geom", "int", 15, "Spatial neighbours used by BANKSY.", minimum=1),
            ParamSpec("npcs", "int", 20, "BANKSY principal components.", minimum=2),
            ParamSpec(
                "algorithm", "str", "leiden", "Clustering algorithm.",
                choices=("leiden", "louvain", "kmeans", "mclust"),
            ),
            ParamSpec("resolution", "float", 0.8, "Graph-clustering resolution.", minimum=0.0),
            ParamSpec("n_domains", "int", 5, "Clusters for k-means or mclust.", minimum=2),
            ParamSpec("seed", "int", 0, "Random seed used for clustering.", minimum=0),
        ),
        assumptions=(
            "obsm['spatial'] contains two-dimensional coordinates.",
            "X contains non-negative expression values; raw counts are recommended.",
            "The histoweave-r image contains Bioconductor Banksy.",
        ),
        assays=("visium", "xenium", "cosmx"),
        wraps="Bioconductor::Banksy",
        language="container",
    )
    r_script = "/usr/local/bin/histoweave-banksy.R"

    def _validate_input(self, data: SpatialTable) -> None:
        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for BANKSY domain detection")
        if data.spatial.shape[1] != 2:
            raise ValueError("BANKSY currently requires two-dimensional spatial coordinates")

    def _build_r_args(self, data: SpatialTable) -> list[str]:
        return [
            f"lambda={self.params['lambda_param']}",
            f"k_geom={self.params['k_geom']}",
            f"npcs={self.params['npcs']}",
            f"algorithm={self.params['algorithm']}",
            f"resolution={self.params['resolution']}",
            f"n_domains={self.params['n_domains']}",
            f"seed={self.params['seed']}",
        ]

    def _validate_r_output(self, data: SpatialTable) -> None:
        if "domain" not in data.obs:
            raise RuntimeError("BANKSY output is missing obs['domain']")
        if data.obs["domain"].isna().any():
            raise RuntimeError("BANKSY output contains missing domain labels")
