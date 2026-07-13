"""Tiny, deterministic synthetic spatial datasets.

The engineering philosophy in the plan is "test & benchmark continuously on tiny
canonical datasets." This module fabricates a small tissue-like sample with known
spatial domains and domain-specific marker genes, so every layer of the platform can
be exercised in CI in milliseconds and with a *known ground truth* for benchmarking.

For the performance-landscape pilot, :func:`make_benchmark_suite` produces a family of
datasets that systematically vary along axes the field cares about: sparsity, spatial
autocorrelation, domain count, domain separability, and sample size.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data import Provenance, SpatialTable


# ====================================================================
# Benchmark suite — systematically varied datasets
# ====================================================================
@dataclass
class BenchmarkSuite:
    """A named, parameterised collection of synthetic spatial datasets.

    Each preset varies one or two data-generation knobs while holding others
    constant so that the resulting feature embedding exposes clean axes of
    variation for the landscape analysis.
    """

    datasets: dict[str, SpatialTable] = field(default_factory=dict)
    presets: dict[str, dict] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.datasets)

    def __iter__(self):
        return iter(self.datasets.items())


def make_benchmark_suite(seed: int = 42) -> BenchmarkSuite:
    """Generate 10 synthetic datasets spanning diverse tissue architectures.

    Preset design rationale
    -----------------------
    ===================== ===== ===== ===== ===== ====== ==============
    Preset                 n     doms  noise lift  layout  signature
    ===================== ===== ===== ===== ===== ====== ==============
    ``clean_easy``         600   3     0.15  8×    blob    Well-separated
    ``noisy_hard``         600   3     0.55  3×    blob    Weak signal
    ``many_small_domains`` 600   6     0.25  6×    blob    Many domains
    ``few_large_domains``  300   2     0.20  10×   blob    Two halves
    ``dense_regular``      900   4     0.20  7×    grid    Visium-like
    ``sparse_scattered``   200   3     0.30  9×    blob    Sparse sampling
    ``tumor_mimic``        800   5     0.25  —      zones   TIME architecture
    ``devel_gradient``     600   5     0.20  —      grad    Pseudotime axis
    ``tumor_noisy``        800   5     0.50  —      zones   TIME hard mode
    ``devel_branching``    700   6     0.22  —      grad    Branching pseudotime
    ===================== ===== ===== ===== ===== ====== ==============

    All are deterministic for a fixed *seed*.
    """
    presets: dict[str, dict[str, Any]] = {
        "clean_easy": dict(
            n_cells=600, n_domains=3, noise=0.15, lift=8.0, layout="blob", label="Clean & easy"
        ),
        "noisy_hard": dict(
            n_cells=600, n_domains=3, noise=0.55, lift=3.0, layout="blob", label="Noisy & hard"
        ),
        "many_small_domains": dict(
            n_cells=600,
            n_domains=6,
            noise=0.25,
            lift=6.0,
            layout="blob",
            label="Many small domains",
        ),
        "few_large_domains": dict(
            n_cells=300,
            n_domains=2,
            noise=0.20,
            lift=10.0,
            layout="blob",
            label="Few large domains",
        ),
        "dense_regular": dict(
            n_cells=900, n_domains=4, noise=0.20, lift=7.0, layout="grid", label="Dense & regular"
        ),
        "sparse_scattered": dict(
            n_cells=200,
            n_domains=3,
            noise=0.30,
            lift=9.0,
            layout="blob",
            label="Sparse & scattered",
        ),
    }

    datasets: dict[str, SpatialTable] = {}
    for name, cfg in presets.items():
        datasets[name] = make_synthetic(
            n_cells=int(cfg["n_cells"]),
            n_domains=int(cfg["n_domains"]),
            noise=float(cfg["noise"]),
            marker_gene_lift=float(cfg["lift"]),
            layout=str(cfg["layout"]),
            seed=seed + zlib.crc32(name.encode("utf-8")) % 10000,
        )

    # Tissue-specific architectures — fundamentally different spatial layouts
    # that cannot be represented by the blob/grid parameterisation.
    datasets["tumor_mimic"] = make_tumor_microenvironment(
        n_cells=800,
        n_genes=200,
        noise=0.25,
        seed=seed + zlib.crc32(b"tumor_mimic") % 10000,
    )
    datasets["devel_gradient"] = make_developmental_gradient(
        n_cells=600,
        n_genes=150,
        n_states=5,
        expression_noise=0.20,
        seed=seed + zlib.crc32(b"devel_gradient") % 10000,
    )
    datasets["tumor_noisy"] = make_tumor_microenvironment(
        n_cells=800,
        n_genes=200,
        noise=0.50,
        seed=seed + zlib.crc32(b"tumor_noisy") % 10000,
    )
    datasets["devel_branching"] = make_developmental_gradient(
        n_cells=700,
        n_genes=150,
        n_states=6,
        expression_noise=0.22,
        seed=seed + zlib.crc32(b"devel_branching") % 10000,
    )

    return BenchmarkSuite(datasets=datasets, presets=presets)


def make_synthetic(
    n_cells: int = 600,
    n_genes: int = 50,
    n_domains: int = 3,
    grid: tuple[float, float] = (100.0, 100.0),
    marker_genes_per_domain: int = 5,
    noise: float = 0.3,
    marker_gene_lift: float | None = None,
    layout: str = "blob",
    add_mito: bool = False,
    seed: int | None = 0,
) -> SpatialTable:
    """Generate a synthetic spatial sample with ground-truth domains.

    Cells are scattered in 2D space and assigned to one of ``n_domains`` spatial
    domains. Each domain over-expresses a dedicated block of marker genes, so both
    spatial-domain detection and marker-based annotation have signal to recover.

    Parameters
    ----------
    marker_gene_lift
        Mean Poisson rate for marker genes inside their domain. When *None*
        (default) a uniform draw in [6, 10) is used, matching the original
        behaviour.  Lower values (2-4) make domains harder to separate.
    layout
        ``"blob"`` — domain centroids are uniformly scattered; cells assigned
        to the nearest centroid (Voronoi tessellation).  ``"grid"`` — cells
        are placed on a regular grid with gentle jitter and domain centroids
        are arranged in a regular pattern.

    Returns
    -------
    SpatialTable
        ``obs["domain_truth"]`` holds the ground-truth domain label; ``obsm["spatial"]``
        holds coordinates; ``uns["marker_genes"]`` maps each domain to its markers.
    """
    rng = np.random.default_rng(seed)

    # 1. Spatial layout -------------------------------------------------------
    if layout == "grid":
        side = int(np.ceil(np.sqrt(n_cells)))
        xs, ys = np.meshgrid(np.linspace(0, grid[0], side), np.linspace(0, grid[1], side))
        coords = np.column_stack([xs.ravel()[:n_cells], ys.ravel()[:n_cells]])
        coords += rng.normal(scale=2.0, size=coords.shape)  # jitter
        # Place domain centroids on a regular sub-grid
        dom_side = int(np.ceil(np.sqrt(n_domains)))
        cx = np.linspace(grid[0] * 0.2, grid[0] * 0.8, dom_side)
        cy = np.linspace(grid[1] * 0.2, grid[1] * 0.8, dom_side)
        cxs, cys = np.meshgrid(cx, cy)
        centroids = np.column_stack([cxs.ravel()[:n_domains], cys.ravel()[:n_domains]])
    else:  # blob — random centroids
        centroids = rng.uniform([0, 0], grid, size=(n_domains, 2))
        coords = rng.uniform([0, 0], grid, size=(n_cells, 2))

    dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
    domain = dists.argmin(axis=1)

    # 2. Assign disjoint marker-gene blocks to each domain.
    n_markers = n_domains * marker_genes_per_domain
    n_markers = min(n_markers, n_genes)
    marker_genes: dict[str, list[str]] = {}
    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]

    # 3. Base expression = low background; markers get a domain-specific lift.
    base_rate = rng.uniform(0.5, 1.5, size=n_genes)
    X = np.asarray(rng.poisson(lam=np.broadcast_to(base_rate, (n_cells, n_genes))), dtype=float)

    for d in range(n_domains):
        start = d * marker_genes_per_domain
        stop = min(start + marker_genes_per_domain, n_markers)
        block = list(range(start, stop))
        if not block:
            break
        marker_genes[f"domain_{d}"] = [gene_names[i] for i in block]
        in_domain = domain == d
        if marker_gene_lift is not None:
            lift = np.full(len(block), marker_gene_lift)
        else:
            lift = rng.uniform(6.0, 10.0, size=len(block))
        X[np.ix_(in_domain, block)] += rng.poisson(
            lam=np.broadcast_to(lift, (in_domain.sum(), len(block)))
        )

    # 4. Multiplicative noise to blur the boundaries.
    X *= rng.lognormal(mean=0.0, sigma=noise, size=X.shape)
    X = np.rint(X).astype(float)

    # 5. Optional mitochondrial genes (off by default for benchmark suite).
    if add_mito and n_genes >= 3:
        n_mito = max(1, n_genes // 20)
        mito_idx = np.arange(n_genes - n_mito, n_genes)
        for i in mito_idx:
            gene_names[i] = f"MT-{i:03d}"
        n_bad = max(1, n_cells // 25)
        bad = rng.choice(n_cells, size=n_bad, replace=False)
        X[bad] *= 0.2
        X[np.ix_(bad, mito_idx)] += rng.poisson(lam=40.0, size=(n_bad, n_mito))
        X = np.rint(X).astype(float)

    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical([f"domain_{d}" for d in domain])},
        index=[f"cell_{i:05d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"mito": [name.startswith("MT-") for name in gene_names]},
        index=gene_names,
    )

    table = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": coords},
        uns={
            "marker_genes": marker_genes,
            "n_domains": n_domains,
            "assay": "synthetic",
        },
    )
    table.record(
        Provenance(
            step="ingestion",
            method="make_synthetic",
            method_version="0.0.1",
            params={
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_domains": n_domains,
                "seed": seed,
            },
        )
    )
    return table


def make_mixture_synthetic(
    n_spots: int = 300,
    n_genes: int = 36,
    n_cell_types: int = 3,
    n_markers_per_type: int = 5,
    noise: float = 0.15,
    seed: int | None = 0,
) -> SpatialTable:
    """Generate synthetic spots that are *mixtures* of cell-type expression programs.

    Each spot draws a random composition vector from a Dirichlet distribution; its
    expression is the weighted sum of the cell-type profiles plus noise. Marker genes
    (disjoint blocks) give each cell type a unique signature, so deconvolution methods
    have detectable signal.

    Returns a ``SpatialTable`` whose ``obsm['proportions_truth']`` holds the
    ground-truth (n_spots × n_cell_types) composition matrix and ``uns['marker_genes']``
    maps cell type → list of marker gene names.
    """
    rng = np.random.default_rng(seed)

    # -- cell-type expression profiles (one per cell type) --------------------
    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]
    n_markers_tot = min(n_cell_types * n_markers_per_type, n_genes)
    profiles = rng.uniform(0.5, 1.5, size=(n_cell_types, n_genes))

    marker_genes: dict[str, list[str]] = {}
    for ct in range(n_cell_types):
        start = ct * n_markers_per_type
        stop = min(start + n_markers_per_type, n_markers_tot)
        block = list(range(start, stop))
        if not block:
            break
        marker_genes[f"cell_type_{ct}"] = [gene_names[i] for i in block]
        profiles[ct, block] = rng.uniform(8.0, 14.0, size=len(block))

    # -- per-spot composition (Dirichlet) ------------------------------------
    alpha = rng.uniform(0.6, 1.8, size=n_cell_types)
    proportions = rng.dirichlet(alpha, size=n_spots)  # (n_spots, n_cell_types)

    # -- mixture expression --------------------------------------------------
    X = proportions @ profiles  # (n_spots, n_genes)
    X += rng.lognormal(mean=0.0, sigma=noise, size=X.shape) * X.mean()
    X = np.rint(np.clip(X, 0, None)).astype(float)

    # -- spatial coordinates (grid-like for visual interest) ------------------
    side = int(np.ceil(np.sqrt(n_spots)))
    xs, ys = np.meshgrid(np.linspace(0, 100, side), np.linspace(0, 100, side))
    coords = np.column_stack([xs.ravel()[:n_spots], ys.ravel()[:n_spots]])
    coords += rng.normal(scale=1.0, size=coords.shape)  # jitter

    obs = pd.DataFrame(index=[f"spot_{i:05d}" for i in range(n_spots)])
    var = pd.DataFrame(index=gene_names)

    table = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": coords, "proportions_truth": proportions},
        uns={
            "marker_genes": marker_genes,
            "n_cell_types": n_cell_types,
            "assay": "mixture_synthetic",
        },
    )
    table.record(
        Provenance(
            step="ingestion",
            method="make_mixture_synthetic",
            method_version="0.0.1",
            params={
                "n_spots": n_spots,
                "n_genes": n_genes,
                "seed": seed,
            },
        )
    )
    return table


def make_tumor_microenvironment(
    n_cells: int = 800,
    n_genes: int = 200,
    *,
    necrotic_fraction: float = 0.15,
    immune_fraction: float = 0.20,
    margin_width: float = 0.12,
    noise: float = 0.25,
    seed: int | None = 0,
) -> SpatialTable:
    """Generate a synthetic tumor microenvironment (TIME) sample.

    Five concentric zones radiating from a necrotic core: necrotic,
    peri_necrotic, invasive_margin, immune_hotspot, stroma.
    Immune hotspots are scattered TLS-like patches with high immune markers.
    """
    rng = np.random.default_rng(seed)

    center = np.array([50.0, 50.0])
    coords = rng.uniform(0, 100, size=(n_cells, 2))
    dists = np.linalg.norm(coords - center, axis=1)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    max_dist: float = float(np.max(dists))
    necrotic_r = necrotic_fraction * max_dist
    margin_inner = necrotic_r + 0.03 * max_dist
    margin_outer = margin_inner + margin_width * max_dist

    n_hotspots = rng.integers(3, 7)
    hotspot_angles = rng.uniform(0, 2 * np.pi, size=n_hotspots)
    hotspot_radii = rng.uniform(0.35 * max_dist, 0.85 * max_dist, size=n_hotspots)
    hotspot_widths = rng.uniform(0.06, 0.14, size=n_hotspots)

    domain: np.ndarray = np.full(n_cells, -1, dtype=int)
    for i in range(n_cells):
        d, a = dists[i], angles[i]
        in_hotspot = False
        for ha, hr, hw in zip(hotspot_angles, hotspot_radii, hotspot_widths, strict=True):
            adiff = min(abs(a - ha), abs(a - ha - 2 * np.pi), abs(a - ha + 2 * np.pi))
            if adiff < hw and abs(d - hr) < hw * max_dist:
                in_hotspot = True
                break
        if in_hotspot:
            domain[i] = 3
        elif d <= necrotic_r:
            domain[i] = 0
        elif d <= margin_inner:
            domain[i] = 1
        elif d <= margin_outer:
            domain[i] = 2
        else:
            domain[i] = 4

    zone_names = np.array(
        [
            "necrotic",
            "peri_necrotic",
            "invasive_margin",
            "immune_hotspot",
            "stroma",
        ]
    )

    gene_names = [f"gene_{i:04d}" for i in range(n_genes)]
    n_per_zone = n_genes // 5
    marker_genes: dict[str, list[str]] = {}
    for zidx, zname in enumerate(zone_names):
        start = zidx * n_per_zone
        marker_genes[str(zname)] = [gene_names[i] for i in range(start, min(start + 5, n_genes))]

    base_rate = rng.uniform(0.3, 1.0, size=n_genes)
    X = np.asarray(rng.poisson(lam=np.broadcast_to(base_rate, (n_cells, n_genes))), dtype=float)
    lifts = {0: 1.5, 1: 4.0, 2: 8.0, 3: 6.0, 4: 3.0}

    for d_idx in range(5):
        in_zone = domain == d_idx
        n_in = int(in_zone.sum())
        if n_in == 0:
            continue
        zname = str(zone_names[d_idx])
        for g_name in marker_genes.get(zname, []):
            try:
                g_idx = gene_names.index(g_name)
            except ValueError:
                continue
            X[in_zone, g_idx] += rng.poisson(lam=lifts[d_idx], size=n_in)

    gradient = np.clip(1.0 - dists / max_dist, 0.2, 1.0)
    X = X * gradient[:, None]
    X *= rng.lognormal(mean=0.0, sigma=noise, size=X.shape)
    X = np.rint(np.clip(X, 0, None)).astype(float)

    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical(zone_names[domain])},
        index=[f"cell_{i:05d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    table = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": coords},
        uns={
            "marker_genes": marker_genes,
            "n_domains": 5,
            "assay": "synthetic_tumor",
            "architecture": "concentric_zones",
        },
    )
    table.record(
        Provenance(
            step="ingestion",
            method="make_tumor_microenvironment",
            method_version="0.1.0",
            params={"n_cells": n_cells, "n_genes": n_genes, "seed": seed},
        )
    )
    return table


def make_developmental_gradient(
    n_cells: int = 600,
    n_genes: int = 150,
    *,
    n_states: int = 5,
    pseudotime_noise: float = 0.12,
    expression_noise: float = 0.20,
    seed: int | None = 0,
) -> SpatialTable:
    """Generate a synthetic developmental-gradient tissue.

    Cells follow a pseudotime axis (left → right). Adjacent states share
    ~30% of marker genes, creating a smooth developmental continuum.
    """
    rng = np.random.default_rng(seed)

    side = int(np.ceil(np.sqrt(n_cells)))
    xs, ys = np.meshgrid(np.linspace(0, 100, side), np.linspace(0, 100, side))
    coords = np.column_stack([xs.ravel()[:n_cells], ys.ravel()[:n_cells]])
    coords += rng.normal(scale=1.5, size=coords.shape)

    pseudotime = coords[:, 0] / 100.0
    pseudotime += rng.normal(scale=pseudotime_noise, size=n_cells)
    pseudotime = np.clip(pseudotime, 0.0, 1.0)

    boundaries = np.linspace(0, 1, n_states + 1)
    domain = np.clip(np.digitize(pseudotime, boundaries[1:]), 0, n_states - 1)
    state_names = np.array([f"state_{i}" for i in range(n_states)])

    gene_names = [f"gene_{i:04d}" for i in range(n_genes)]
    genes_per_state = n_genes // n_states
    marker_sets: dict[str, set[str]] = {}
    for s in range(n_states):
        start = s * genes_per_state
        end = min(start + genes_per_state, n_genes)
        core = {gene_names[i] for i in range(start, end)}
        if s > 0:
            prev = marker_sets[f"state_{s - 1}"]
            n_bridge = max(1, len(prev) // 3)
            bridge = set(
                rng.choice(
                    sorted(prev),
                    size=min(n_bridge, len(prev)),
                    replace=False,
                )
            )
            core |= bridge
        marker_sets[f"state_{s}"] = core

    marker_genes = {s: sorted(ms) for s, ms in marker_sets.items()}

    base_rate = rng.uniform(0.3, 0.9, size=n_genes)
    X = np.asarray(rng.poisson(lam=np.broadcast_to(base_rate, (n_cells, n_genes))), dtype=float)

    for s in range(n_states):
        in_state = domain == s
        n_in = int(in_state.sum())
        if n_in == 0:
            continue
        for g_name in marker_genes[f"state_{s}"]:
            try:
                g_idx = gene_names.index(g_name)
            except ValueError:
                continue
            activation = np.interp(
                pseudotime[in_state],
                [s / n_states, (s + 1) / n_states],
                [0.5, 1.0],
            )
            X[in_state, g_idx] += rng.poisson(lam=activation * 6.0)

    for _i in range(min(10, n_genes)):
        driver_idx = rng.integers(0, n_genes)
        X[:, driver_idx] += rng.poisson(lam=np.sin(pseudotime * np.pi) * 4.0 + 1.0)

    X *= rng.lognormal(mean=0.0, sigma=expression_noise, size=X.shape)
    X = np.rint(np.clip(X, 0, None)).astype(float)

    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical(state_names[domain]), "pseudotime": pseudotime},
        index=[f"cell_{i:05d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    table = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": coords},
        uns={
            "marker_genes": {str(k): v for k, v in marker_genes.items()},
            "n_domains": n_states,
            "assay": "synthetic_developmental",
            "architecture": "pseudotime_gradient",
        },
    )
    table.record(
        Provenance(
            step="ingestion",
            method="make_developmental_gradient",
            method_version="0.1.0",
            params={
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_states": n_states,
                "seed": seed,
            },
        )
    )
    return table
