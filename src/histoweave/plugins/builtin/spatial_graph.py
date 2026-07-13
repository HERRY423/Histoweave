"""Spatial neighbourhood analysis via networkx graph construction.

Spatial transcriptomics is fundamentally graph-structured: cells/spots are nodes
and their spatial proximity defines edges. This plugin constructs a spatial graph
from coordinates (k-NN or radius-based) and computes graph-theoretic properties —
degree, clustering coefficient, centrality — that characterise tissue architecture.

These metrics are useful both as QC diagnostics (e.g. "does this region have
unexpectedly sparse sampling?") and as features for downstream analysis (e.g.
graph centrality as a proxy for niche hub-ness).
"""

from __future__ import annotations

import numpy as np

from ...data import SpatialTable
from ..interfaces import Method, MethodCategory, MethodSpec, ParamSpec
from ..registry import register


@register
class SpatialGraphMetrics(Method):
    """Construct a spatial proximity graph and compute per-node graph metrics.

    Writes the following into ``obs``:

    * ``spatial_degree`` — number of neighbours within the graph.
    * ``spatial_clustering`` — local clustering coefficient (Watts–Strogatz).
    * ``spatial_centrality`` — eigenvector centrality (PageRank-like hub score).

    The graph itself is stored in ``uns['spatial_graph']`` as a serialisable
    edge list so downstream methods (e.g. GraphST-style GNNs) can reuse it.
    """

    spec = MethodSpec(
        name="spatial_graph",
        category=MethodCategory.NEIGHBORHOOD,
        version="0.1.0",
        summary="Spatial k-NN graph construction + per-node graph metrics.",
        params=(
            ParamSpec("k", "int", 8, "Number of spatial neighbours per node.", minimum=1),
            ParamSpec(
                "mode",
                "str",
                "knn",
                "'knn' or 'radius' (radius in coordinate units).",
                choices=("knn", "radius"),
            ),
            ParamSpec(
                "radius", "float", 15.0, "Radius for 'radius' mode.", minimum=1e-12
            ),
        ),
        assumptions=("obsm['spatial'] present.", "networkx installed."),
        wraps="networkx + scipy.spatial",
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        import networkx as nx

        data = data.copy()
        coords = data.spatial
        if coords is None:
            raise ValueError("obsm['spatial'] is required for spatial graph construction")

        n = data.n_obs
        mode = self.params["mode"]
        if mode == "knn":
            k = int(min(self.params["k"], n - 1))
            edges = _knn_edges(coords, k)
        elif mode == "radius":
            edges = _radius_edges(coords, self.params["radius"])
        else:
            raise ValueError(f"Unknown mode '{mode}'; use 'knn' or 'radius'")

        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        # Per-node metrics
        data.obs["spatial_degree"] = [G.degree(i) for i in range(n)]
        data.obs["spatial_clustering"] = [
            nx.clustering(G, i) if G.degree(i) > 1 else 0.0 for i in range(n)
        ]

        # Eigenvector centrality (for the largest connected component)
        centrality = np.zeros(n, dtype=float)
        if G.number_of_edges() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            sub = G.subgraph(largest_cc)
            ecc = nx.eigenvector_centrality_numpy(sub)
            for node, val in ecc.items():
                centrality[node] = float(val)
        data.obs["spatial_centrality"] = centrality

        # Store the graph as a serialisable edge list.
        data.uns["spatial_graph"] = {
            "n_nodes": n,
            "n_edges": G.number_of_edges(),
            "mode": mode,
            "params": {"k": self.params["k"], "radius": self.params["radius"]},
            "edges": list(G.edges()),
        }
        return self.finalize(data, step="neighborhood")


def _knn_edges(coords: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Symmetric k-NN edges (undirected)."""
    from scipy.spatial import KDTree

    n = coords.shape[0]
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # k+1 includes self
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in idx[i, 1:]:
            u, v = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((u, v))
    return list(edges)


def _radius_edges(coords: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Edges between all point pairs within ``radius``."""
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    return [(int(u), int(v)) for u, v in pairs]
