# HistoWeave research method incubator

HistoWeave includes a deliberately separate research track for native algorithm
candidates. These methods are executable and tested, but they are **not claimed to be
scientifically validated, publication-novel, or superior to established tools**. Their
names use the `weave_` prefix, their maturity is `experimental`, and their registry
metadata records `metadata.track = research` and `novelty = unvalidated`.

The incubator targets gaps highlighted by recent spatial-omics work: interpretable
multi-scale representations, spatially robust variable-gene scores, explicit boundary
and uncertainty modeling, cross-section integration, and quantitative niche context.
Relevant reference directions include
[STAMP](https://doi.org/10.1038/s41592-024-02463-8),
[BANKSY](https://doi.org/10.1038/s41588-024-01664-3),
[multiscale topology](https://doi.org/10.1038/s41586-024-07563-1),
[HEARTSVG](https://doi.org/10.1038/s41467-024-49846-1),
[SANTO](https://doi.org/10.1038/s41467-024-50308-x), and
[NiCo](https://doi.org/10.1038/s41467-024-54973-w). The implementations below are
independent, dependency-light combinations designed for ablation and benchmarking;
they do not reproduce or rename those published methods.

## Candidate inventory

| Category | Method | Research hypothesis |
|---|---|---|
| QC | `weave_spatial_entropy_qc` | Local expression entropy exposes spatially isolated low-information observations. |
| QC | `weave_neighbor_discordance_qc` | Expression disagreement with nearby observations is an interpretable spatial QC signal. |
| QC | `weave_adaptive_saturation_qc` | Detection saturation should be judged against a dataset-adaptive depth envelope. |
| Normalization | `weave_spatial_median_normalize` | Blending cell and neighborhood size factors reduces local depth artifacts. |
| Normalization | `weave_graph_diffusion_normalize` | A bounded diffusion step can stabilize sparse normalized expression. |
| Normalization | `weave_rank_stabilize` | Per-cell fractional ranks provide a depth-robust nonparametric representation. |
| Normalization | `weave_robust_pearson_residual` | Prior-shrunk gene frequencies and clipping can bound Pearson-residual influence. |
| Domains | `weave_multiscale_consensus_domains` | Consensus across neighborhood scales is more stable than one fixed graph scale. |
| Domains | `weave_boundary_aware_domains` | Local expression gradients should reduce smoothing across tissue boundaries. |
| Domains | `weave_topology_regularized_domains` | Iterative spatial-neighbor Potts regularization can improve coherence without a GNN. |
| Domains | `weave_uncertainty_domains` | Ensemble disagreement should be exposed as per-observation domain uncertainty. |
| SVG | `weave_multiscale_svg` | A gene should rank highly only when spatial structure persists across graph scales. |
| SVG | `weave_boundary_svg` | Edge-wise gradients identify genes concentrated at spatial transitions. |
| SVG | `weave_hotspot_svg` | Concentrated local maxima complement global autocorrelation statistics. |
| SVG | `weave_anisotropy_svg` | Direction-dependent covariance reveals oriented tissue programs. |
| SVG | `weave_bootstrap_robust_svg` | Bootstrap mean, spread, and sign stability can trade sensitivity for rank stability. |
| Neighborhood | `weave_adaptive_radius_graph` | Local-density radii avoid overconnecting dense areas and isolating sparse areas. |
| Neighborhood | `weave_mutual_knn_graph` | Mutual neighbors provide a conservative, density-aware tissue graph. |
| Neighborhood | `weave_expression_spatial_graph` | Joint spatial and expression distances distinguish contact from state similarity. |
| Integration | `weave_spatial_quantile_integrate` | Batch quantile alignment followed by bounded spatial smoothing preserves anatomy. |
| Integration | `weave_anchor_residual_integrate` | Mutual spatial-expression anchors and median residuals provide a transparent correction baseline. |
| Annotation | `weave_neighbor_marker_annotate` | Marker evidence and neighbor consensus should be reported together. |
| Deconvolution | `weave_spatial_simplex_deconv` | Marker projections constrained to the simplex can be stabilized by graph-Laplacian regularization. |

## Graduation criteria

A research candidate can move to beta only after all of the following are available:

1. multi-dataset benchmarks against named baselines with fixed train/test protocols;
2. ablations isolating every new component;
3. noise, sparsity, density, and platform-shift sensitivity analyses;
4. calibrated uncertainty or error control where the output implies confidence;
5. independent biological review and reference-concordance tests;
6. documented runtime and memory scaling.

Until those gates pass, release coverage reports keep research candidates separate from
the production-quality denominator.
