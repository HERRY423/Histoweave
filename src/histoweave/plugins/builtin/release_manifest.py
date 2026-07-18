"""Audited release manifest for built-in method maturity and capabilities.

Maturity policy (enforced by :mod:`histoweave.plugins.interfaces`)
------------------------------------------------------------------
* **experimental** — stable contract only; may be baseline / research / unvalidated.
* **beta** — wraps the real upstream library; mock/integration tests present.
* **production** — pinned runtime path, real-data integration tests, ops diagnostics.
* **contract_validated** — multi-dataset *interface / mock / structural* gates
  (CI-safe; no claim of scientific concordance on real ground truth).
* **validated** — multi-dataset *scientific* concordance (e.g. ARI vs manual layers).

Unified counting (release freeze)
---------------------------------
* **10 scientifically validated** methods → :data:`SCIENTIFIC_VALIDATED_METHODS`
* **3 contract-validated** methods → :data:`CONTRACT_VALIDATED_METHODS`
* **13 multi-dataset evidence packages** total → :data:`MULTI_DATASET_EVIDENCE_METHODS`

This manifest deliberately de-inflates maturity: teaching baselines and in-house
autoencoders are *not* production, and no method is scientifically validated
without documented multi-dataset concordance evidence.
"""

from __future__ import annotations

from dataclasses import replace

from ..interfaces import MethodMaturity
from ..registry import _REGISTRY

# ---------------------------------------------------------------------------
# Evidence packages — kind: scientific | contract
# ---------------------------------------------------------------------------
# Each entry documents why a multi-dataset gate was claimed.  Keep this list
# short and honest.  ``kind`` must be "scientific" or "contract".

VALIDATION_EVIDENCE: dict[str, dict[str, str]] = {
    # --- Scientific validation (real multi-dataset concordance) -------------
    "banksy_py": {
        "kind": "scientific",
        "protocol": "histoweave.landscape.dlpfc_real.v1 + variance_decomposition",
        "datasets": "DLPFC Visium 151507/669/670/673/674 (Maynard 2021)",
        "metric": "ARI vs manual cortical layers; factorial variance decomposition",
        "notes": "Native BANKSY scaffold; official R BANKSY remains the canonical wrap.",
        "report": "docs/methods/validation/banksy_py.md",
    },
    "spectral": {
        "kind": "scientific",
        "protocol": "histoweave.landscape.dlpfc_real.v1 + dlpfc_spatial_aware.v1",
        "datasets": "DLPFC Visium 5-slice difficulty gradient",
        "metric": "ARI vs manual layers across seeds and spatial_weight policies",
        "notes": "sklearn SpectralClustering with spatial-neighbourhood embedding.",
        "report": "docs/methods/validation/spectral.md",
    },
    "gaussian_mixture": {
        "kind": "scientific",
        "protocol": "histoweave.landscape.dlpfc_real.v1 + dlpfc_spatial_aware.v1",
        "datasets": "DLPFC Visium 5-slice difficulty gradient",
        "metric": "ARI vs manual layers; frequently top spatial_weight configuration",
        "notes": "sklearn GaussianMixture with spatial-context policy.",
        "report": "docs/methods/validation/gaussian_mixture.md",
    },
    "agglomerative": {
        "kind": "scientific",
        "protocol": "histoweave.method_validation.multidataset.v1 (figure3 + dlpfc 5x10 + 5x15)",
        "datasets": "Figure3 synthetic ×3; DLPFC Visium 5 slices (Maynard 2021)",
        "metric": "ARI vs planted domains / manual layers; best spatial_weight per slice",
        "notes": "sklearn AgglomerativeClustering (Ward) on spatial-PCA embedding.",
        "report": "docs/methods/validation/agglomerative.md",
    },
    "birch": {
        "kind": "scientific",
        "protocol": "histoweave.method_validation.multidataset.v1 (figure3 + dlpfc 5x10 + 5x15)",
        "datasets": "Figure3 synthetic ×3; DLPFC Visium 5 slices (Maynard 2021)",
        "metric": "ARI vs planted domains / manual layers; best spatial_weight per slice",
        "notes": "sklearn Birch on spatial-PCA embedding; hierarchical baseline.",
        "report": "docs/methods/validation/birch.md",
    },
    "minibatch_kmeans": {
        "kind": "scientific",
        "protocol": "histoweave.method_validation.multidataset.v1 (figure3 + dlpfc 5x10)",
        "datasets": "Figure3 synthetic ×3; DLPFC Visium 5 slices (Maynard 2021)",
        "metric": "ARI vs planted domains / manual layers across seeds",
        "notes": "sklearn MiniBatchKMeans; scalable k-means baseline for large n.",
        "report": "docs/methods/validation/minibatch_kmeans.md",
    },
    "banksy": {
        "kind": "scientific",
        "protocol": "histoweave.method_validation.multidataset.v1 + sota_domains.v1",
        "datasets": "DLPFC Visium 5 slices via banksy_py multi-seed ARI; R wrap contract tests",
        "metric": "ARI vs manual layers (native banksy_py proxy) + R container contract",
        "notes": (
            "Production wrap is Bioconductor::Banksy; multi-slice numeric ARI from "
            "native banksy_py scaffold (same family). See docs/methods/validation/banksy.md."
        ),
        "report": "docs/methods/validation/banksy.md",
    },
    "spagcn": {
        "kind": "scientific",
        "protocol": "histoweave.method_validation.sota_batch.v1 + histoweave.sota_dlpfc.v1",
        "datasets": "DLPFC Visium 151507/669/670/673/674 × 3 seeds (official SpaGCN)",
        "metric": "ARI vs manual layers; mean≈0.32 across 15 successful cells",
        "notes": "Official SpaGCN==1.2.7; no silent substitute when backend missing.",
        "report": "docs/methods/validation/spagcn.md",
    },
    "graphst": {
        "kind": "scientific",
        "protocol": "histoweave.sota_dlpfc.v1 (official GraphST real ARI)",
        "datasets": "DLPFC Visium 151507/669/670/673/674 × 3 seeds (max_obs=1000, epochs=120)",
        "metric": "ARI vs manual layers; mean≈0.121 across 15/15 successful cells",
        "notes": (
            "Official JinmiaoChenLab/GraphST via GraphST.GraphST.GraphST; "
            "adapter fixed for package layout. Full-slice / 600-epoch re-runs optional."
        ),
        "report": "docs/methods/validation/graphst.md",
    },
    "stagate": {
        "kind": "scientific",
        "protocol": "histoweave.sota_dlpfc.v1 (official STAGATE_pyG real ARI)",
        "datasets": "DLPFC Visium 151507/669/670/673/674 × 3 seeds (max_obs=1000, epochs=150)",
        "metric": "ARI vs manual layers; mean≈0.285 across 15/15 successful cells",
        "notes": (
            "Official QIFEIDKN/STAGATE_pyG; Windows torch-sparse soft-shim when sparse "
            "extension fails to load; edge_index path used. Full-slice re-runs optional."
        ),
        "report": "docs/methods/validation/stagate.md",
    },
    # --- Contract validation (mock / interface / structural multi-dataset) --
    "cell2location": {
        "kind": "contract",
        "protocol": "histoweave.method_validation.cell2location_structural.v1",
        "datasets": "3 synthetic marker mixtures + DLPFC 151507/669/673 (subsampled)",
        "metric": "Contract success, shared-gene coverage, abundance/proportion simplex",
        "notes": (
            "Structural multi-dataset validation with mock Cell2location backend for CI; "
            "no marker-score fallback. Full Pyro posterior quality is out of CI scope."
        ),
        "report": "docs/methods/validation/cell2location.md",
    },
    "rctd": {
        "kind": "contract",
        "protocol": "histoweave.method_validation.sota_batch.v1",
        "datasets": "3 synthetic + 3 DLPFC structural (reference + counts + driver gate)",
        "metric": "Fail-closed without R driver 6/6; no marker-score fallback",
        "notes": (
            "spacexr::RCTD production path requires Rscript + driver script; "
            "multi-dataset gate validates hard failure and reference/count contracts."
        ),
        "report": "docs/methods/validation/rctd.md",
    },
    "spatialde": {
        "kind": "contract",
        "protocol": "histoweave.method_validation.sota_batch.v1",
        "datasets": "3 synthetic + DLPFC 151507/669/673 multi-dataset SVG ranking",
        "metric": "SVG contract 6/6; top genes + significance flags exported",
        "notes": (
            "CI uses mock SpatialDE/NaiveDE for multi-dataset I/O; install "
            "histoweave-spatial[spatialde] for real GP p-values."
        ),
        "report": "docs/methods/validation/spatialde.md",
    },
}

# Canonical partitions — single source of truth for 10 / 3 / 13 counts.
SCIENTIFIC_VALIDATED_METHODS: frozenset[str] = frozenset(
    name for name, meta in VALIDATION_EVIDENCE.items() if meta.get("kind") == "scientific"
)
CONTRACT_VALIDATED_METHODS: frozenset[str] = frozenset(
    name for name, meta in VALIDATION_EVIDENCE.items() if meta.get("kind") == "contract"
)
MULTI_DATASET_EVIDENCE_METHODS: frozenset[str] = frozenset(VALIDATION_EVIDENCE)

# Backward-compatible alias: "validated" maturity == scientific only.
VALIDATED_METHODS: frozenset[str] = SCIENTIFIC_VALIDATED_METHODS

# Freeze checks — fail loud if the ledger drifts.
assert len(SCIENTIFIC_VALIDATED_METHODS) == 10, (
    f"expected 10 scientific validated methods, got {len(SCIENTIFIC_VALIDATED_METHODS)}: "
    f"{sorted(SCIENTIFIC_VALIDATED_METHODS)}"
)
assert len(CONTRACT_VALIDATED_METHODS) == 3, (
    f"expected 3 contract-validated methods, got {len(CONTRACT_VALIDATED_METHODS)}: "
    f"{sorted(CONTRACT_VALIDATED_METHODS)}"
)
assert len(MULTI_DATASET_EVIDENCE_METHODS) == 13, (
    f"expected 13 multi-dataset evidence packages, got {len(MULTI_DATASET_EVIDENCE_METHODS)}"
)
assert SCIENTIFIC_VALIDATED_METHODS.isdisjoint(CONTRACT_VALIDATED_METHODS)
assert SCIENTIFIC_VALIDATED_METHODS | CONTRACT_VALIDATED_METHODS == MULTI_DATASET_EVIDENCE_METHODS

# ---------------------------------------------------------------------------
# Production: operationally reliable, not necessarily state-of-the-art science
# ---------------------------------------------------------------------------
PRODUCTION_METHODS = {
    # Ingestion readers
    "cosmx_reader",
    "merfish_reader",
    "merscope_reader",
    "slideseq_reader",
    "stereoseq_reader",
    "visium_reader",
    "xenium_reader",
    # QC / normalization (deterministic, well-defined transforms)
    "basic_qc",
    "gene_complexity_qc",
    "library_size_qc",
    "mitochondrial_qc",
    "arcsinh_transform",
    "clr_per_cell",
    "library_size_scale",
    "log1p_cp10k",
    "sqrt_transform",
    "tfidf_l2",
    # Domain baselines (sklearn family) — elevated when evidence present
    "agglomerative",
    "birch",
    "bisecting_kmeans",
    "dbscan",
    "kmeans",
    "mean_shift",
    "minibatch_kmeans",
    "optics",
    # Integration / neighborhood / SVG statistics
    "combat",
    "harmony",
    "spatial_graph",
    "gearys_c",
    "morans_i",
    "spatial_variance_ratio",
    # Container / R bridges with declared runtimes
    "banksy",
    "nnsvg",
    "r_lognorm",
}
# Elevated methods leave the pure-production set.
PRODUCTION_METHODS -= set(MULTI_DATASET_EVIDENCE_METHODS)

# ---------------------------------------------------------------------------
# Beta: real upstream wraps without multi-dataset validation gates
# ---------------------------------------------------------------------------
BETA_METHODS = {
    "banksy_py",
    "spectral",
    "gaussian_mixture",
    "cell2location",
    "cellpose2",
    "celltypist",
    "liana_plus",
    "scanvi",
    "sctransform",
    "spatialde",
    # First-class SOTA plugins
    "spagcn",
    "graphst",
    "stagate",
    "bayesspace",
    "rctd",
    # Multimodal representation experiments that wrap torch
    "image_expression_attention",
    "image_expression_autoencoder",
    "image_expression_contrastive",
    "multimodal_graph_fusion",
}
BETA_METHODS -= set(MULTI_DATASET_EVIDENCE_METHODS)

# ---------------------------------------------------------------------------
# Experimental: teaching baselines, research incubator, unvalidated DL toys
# ---------------------------------------------------------------------------
RESEARCH_METHODS = {
    "weave_adaptive_radius_graph",
    "weave_adaptive_saturation_qc",
    "weave_anchor_residual_integrate",
    "weave_anisotropy_svg",
    "weave_bootstrap_robust_svg",
    "weave_boundary_aware_domains",
    "weave_boundary_svg",
    "weave_expression_spatial_graph",
    "weave_graph_diffusion_normalize",
    "weave_hotspot_svg",
    "weave_multiscale_consensus_domains",
    "weave_multiscale_svg",
    "weave_mutual_knn_graph",
    "weave_neighbor_discordance_qc",
    "weave_neighbor_marker_annotate",
    "weave_rank_stabilize",
    "weave_robust_pearson_residual",
    "weave_spatial_entropy_qc",
    "weave_spatial_median_normalize",
    "weave_spatial_quantile_integrate",
    "weave_spatial_simplex_deconv",
    "weave_topology_regularized_domains",
    "weave_uncertainty_domains",
}

EXPERIMENTAL_BASELINES = {
    "marker_deconv",
    "marker_score",
    "denoising_spatial_autoencoder",
    "graph_expression_autoencoder",
    "spatial_autoencoder",
    "variational_spatial_autoencoder",
}

EXPERIMENTAL_METHODS = RESEARCH_METHODS | EXPERIMENTAL_BASELINES

DEEP_LEARNING_METHODS = {
    "cell2location",
    "cellpose2",
    "denoising_spatial_autoencoder",
    "graph_expression_autoencoder",
    "graphst",
    "image_expression_attention",
    "image_expression_autoencoder",
    "image_expression_contrastive",
    "multimodal_graph_fusion",
    "scanvi",
    "spagcn",
    "spatial_autoencoder",
    "stagate",
    "variational_spatial_autoencoder",
}

MACHINE_LEARNING_METHODS = {
    "agglomerative",
    "banksy_py",
    "birch",
    "bisecting_kmeans",
    "celltypist",
    "dbscan",
    "gaussian_mixture",
    "harmony",
    "kmeans",
    "mean_shift",
    "minibatch_kmeans",
    "optics",
    "spectral",
}

MODALITY_OVERRIDES = {
    "cellpose2": ("image",),
    "scanvi": ("expression", "labels"),
}


def apply_builtin_release_manifest() -> None:
    """Apply and validate the evidence-reviewed built-in release manifest."""

    classes = {
        cls.spec.name: cls
        for cls in _REGISTRY.values()
        if cls.__module__.startswith("histoweave.plugins.builtin.")
    }
    declared = (
        PRODUCTION_METHODS
        | BETA_METHODS
        | set(SCIENTIFIC_VALIDATED_METHODS)
        | set(CONTRACT_VALIDATED_METHODS)
        | EXPERIMENTAL_METHODS
    )
    if set(classes) != declared:
        missing = sorted(set(classes) - declared)
        stale = sorted(declared - set(classes))
        raise RuntimeError(
            f"builtin maturity manifest drift: unclassified={missing}, missing={stale}"
        )

    for name, cls in classes.items():
        if name in SCIENTIFIC_VALIDATED_METHODS:
            maturity = MethodMaturity.VALIDATED
        elif name in CONTRACT_VALIDATED_METHODS:
            maturity = MethodMaturity.CONTRACT_VALIDATED
        elif name in PRODUCTION_METHODS:
            maturity = MethodMaturity.PRODUCTION
        elif name in BETA_METHODS:
            maturity = MethodMaturity.BETA
        else:
            maturity = MethodMaturity.EXPERIMENTAL
        model_family = (
            "deep_learning"
            if name in DEEP_LEARNING_METHODS
            else "machine_learning"
            if name in MACHINE_LEARNING_METHODS
            else cls.spec.model_family
        )
        metadata = dict(cls.spec.metadata)
        if name in VALIDATION_EVIDENCE:
            metadata["validation_evidence"] = VALIDATION_EVIDENCE[name]
            metadata["validation_kind"] = VALIDATION_EVIDENCE[name]["kind"]
        if name in RESEARCH_METHODS:
            metadata.setdefault("track", "research")
            metadata.setdefault("novelty", "unvalidated")
        if name in EXPERIMENTAL_BASELINES:
            metadata.setdefault("track", "baseline")
        cls.spec = replace(
            cls.spec,
            maturity=maturity,
            model_family=model_family,
            modalities=MODALITY_OVERRIDES.get(name, cls.spec.modalities),
            metadata=metadata,
        )
