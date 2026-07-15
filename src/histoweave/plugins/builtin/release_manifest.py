"""Audited release manifest for built-in method maturity and capabilities."""

from __future__ import annotations

from dataclasses import replace

from ..interfaces import MethodMaturity
from ..registry import _REGISTRY

PRODUCTION_METHODS = {
    "agglomerative",
    "arcsinh_transform",
    "banksy",
    "banksy_py",
    "basic_qc",
    "birch",
    "bisecting_kmeans",
    "clr_per_cell",
    "combat",
    "cosmx_reader",
    "dbscan",
    "denoising_spatial_autoencoder",
    "gene_complexity_qc",
    "gearys_c",
    "gaussian_mixture",
    "graph_expression_autoencoder",
    "kmeans",
    "library_size_qc",
    "library_size_scale",
    "log1p_cp10k",
    "marker_deconv",
    "marker_score",
    "mean_shift",
    "merfish_reader",
    "merscope_reader",
    "harmony",
    "minibatch_kmeans",
    "mitochondrial_qc",
    "morans_i",
    "nnsvg",
    "optics",
    "r_lognorm",
    "slideseq_reader",
    "spatial_autoencoder",
    "spatial_graph",
    "spatial_variance_ratio",
    "spatialde",
    "spectral",
    "sqrt_transform",
    "stereoseq_reader",
    "tfidf_l2",
    "variational_spatial_autoencoder",
    "visium_reader",
    "xenium_reader",
}

BETA_METHODS = {
    "cell2location",
    "cellpose2",
    "celltypist",
    "image_expression_attention",
    "image_expression_autoencoder",
    "image_expression_contrastive",
    "liana_plus",
    "multimodal_graph_fusion",
    "scanvi",
    "sctransform",
}

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

DEEP_LEARNING_METHODS = {
    "cell2location",
    "cellpose2",
    "denoising_spatial_autoencoder",
    "graph_expression_autoencoder",
    "image_expression_attention",
    "image_expression_autoencoder",
    "image_expression_contrastive",
    "multimodal_graph_fusion",
    "scanvi",
    "spatial_autoencoder",
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
    declared = PRODUCTION_METHODS | BETA_METHODS | RESEARCH_METHODS
    if set(classes) != declared:
        missing = sorted(set(classes) - declared)
        stale = sorted(declared - set(classes))
        raise RuntimeError(
            f"builtin maturity manifest drift: unclassified={missing}, missing={stale}"
        )

    for name, cls in classes.items():
        maturity = (
            MethodMaturity.PRODUCTION
            if name in PRODUCTION_METHODS
            else MethodMaturity.BETA
            if name in BETA_METHODS
            else MethodMaturity.EXPERIMENTAL
        )
        model_family = (
            "deep_learning"
            if name in DEEP_LEARNING_METHODS
            else "machine_learning"
            if name in MACHINE_LEARNING_METHODS
            else cls.spec.model_family
        )
        cls.spec = replace(
            cls.spec,
            maturity=maturity,
            model_family=model_family,
            modalities=MODALITY_OVERRIDES.get(name, cls.spec.modalities),
        )
