"""Tutorial 2 companion — define and exercise a custom method plugin.

    python examples/tutorial_custom_plugin.py

Builds a spatial neighbourhood-smoothing plugin, runs it on synthetic data, and
benchmarks it alongside the built-in methods. See docs/tutorials/02_custom_plugin_development.md.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import histoweave as ts
from histoweave.plugins import (
    Method,
    MethodCategory,
    MethodSpec,
    ParamSpec,
    create_method,
    register,
)


@register
class SpatialSmooth(Method):
    """Denoise expression by mixing each spot with its spatial neighbours."""

    spec = MethodSpec(
        name="spatial_smooth",
        category=MethodCategory.NORMALIZATION,
        version="0.1.0",
        summary="k-NN spatial-neighbourhood expression smoothing.",
        params=(
            ParamSpec("k", "int", 6, "Spatial neighbours per spot.", minimum=1),
            ParamSpec(
                "alpha", "float", 0.5,
                "Self weight; 1.0 keeps the spot unchanged.",
                minimum=0.0, maximum=1.0,
            ),
        ),
        assumptions=("obsm['spatial'] present.",),
        wraps="scipy.spatial.cKDTree",
        language="python",
    )

    def run(self, data):
        data = data.copy()
        coords = data.spatial
        if coords is None:
            raise ValueError("obsm['spatial'] is required for spatial smoothing")

        k = int(min(self.params["k"], data.n_obs - 1))
        alpha = float(self.params["alpha"])

        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1)
        neighbour_mean = np.asarray(data.X)[idx[:, 1:]].mean(axis=1)

        data.layers["pre_smooth"] = np.asarray(data.X).copy()
        data.X = alpha * np.asarray(data.X) + (1.0 - alpha) * neighbour_mean
        data.uns["spatial_smooth"] = {"k": k, "alpha": alpha}
        return self.finalize(data, step="normalization")


def main() -> None:
    print(f"HistoWeave v{ts.__version__}\n")

    data = ts.datasets.make_synthetic(n_cells=400, n_genes=40, n_domains=3, seed=0)
    print("Input:", repr(data))

    smoothed = create_method("normalization", "spatial_smooth", k=8, alpha=0.6).run(data)
    print("\nAfter spatial_smooth:")
    print("  provenance:", smoothed.provenance[-1])
    print("  pre_smooth layer stored:", "pre_smooth" in smoothed.layers)
    delta = np.abs(np.asarray(smoothed.X) - np.asarray(data.X)).mean()
    print(f"  mean |Delta X| = {delta:.4f}")

    print("\nBenchmarking domain detection (custom plugin lives alongside built-ins):")
    from histoweave.benchmark import domain_detection_task, run_benchmark

    bench = run_benchmark(domain_detection_task(data))
    for row in bench.leaderboard[:5]:
        print(f"  {row['rank']}. {row['method']:<16} score={row['score']}")


if __name__ == "__main__":
    main()
