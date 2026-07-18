# Cryptic component (rank-2) — `dlpfc_151508`

**Size:** 138 spots · **rank:** 1 (0 = largest; spatially contiguous cryptic niche).

## Manual-layer composition (domain_truth inside component)

| Layer | n spots | fraction |
|-------|--------:|---------:|
| Layer 3 | 138 | 1.000 |

**Dominant truth label:** `Layer 3` (100% of component). If external contacts are also almost exclusively this layer, the niche is **intra-layer substructure** (methods disagree *inside* a manual domain), not an inter-layer boundary ribbon.

## Adjacency to WM / L1–L6

- Internal kNN edges (within component): **568** (68.6% of all component-incident edges)
- External edges (to outside): **260**
- Top abutting layers by contact count: **Layer 3**
- Primary abutment distribution: `{'Layer 3': 115, 'internal_only': 23}`

| Layer | External contacts | Fraction | Background frac | Enrichment | Spots with this as primary abut |
|-------|------------------:|---------:|----------------:|-----------:|--------------------------------:|
| Layer 1 | 0 | 0.000 | 0.204 | 0.0 | 0 |
| Layer 2 | 0 | 0.000 | 0.070 | 0.0 | 0 |
| Layer 3 | 260 | 1.000 | 0.294 | 3.403 | 115 |
| Layer 4 | 0 | 0.000 | 0.088 | 0.0 | 0 |
| Layer 5 | 0 | 0.000 | 0.174 | 0.0 | 0 |
| Layer 6 | 0 | 0.000 | 0.124 | 0.0 | 0 |
| WM | 0 | 0.000 | 0.047 | 0.0 | 0 |

### Geometric interpretation

- **Intra-`Layer 3` compact niche:** every component spot and every external neighbour contact is `Layer 3`. Multi-method uncertainty is flagging a **subregion inside Layer 3**, not a WM↔cortex or L5↔L6 border.
- Internal edge fraction 68.6% indicates a compact blob (more self-contained niche).

## Marker genes (component vs all other spots)

Wilcoxon rank-sum on library-size log1p expression; BH-FDR. MT/Ribo/IG/HB genes excluded from ranking.

### Up in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `SAA1` | 0.906 | 0.450 | 1.011 | 2.33e-07 |
| `MGP` | 1.492 | 0.878 | 0.766 | 2.33e-07 |
| `HOPX` | 1.431 | 0.979 | 0.547 | 2.22e-04 |
| `GAP43` | 1.850 | 1.363 | 0.440 | 3.07e-04 |
| `CALM2` | 3.030 | 2.655 | 0.191 | 3.07e-04 |
| `ENC1` | 2.379 | 1.828 | 0.380 | 7.02e-04 |
| `OSBPL1A` | 0.538 | 0.278 | 0.952 | 8.51e-04 |
| `SCGB2A2` | 3.008 | 2.261 | 0.412 | 8.51e-04 |
| `GRIA2` | 0.962 | 0.609 | 0.660 | 9.96e-04 |
| `TPM1` | 0.593 | 0.315 | 0.911 | 1.03e-03 |
| `RGS4` | 1.492 | 1.085 | 0.460 | 2.19e-03 |
| `FABP4` | 0.630 | 0.374 | 0.753 | 3.25e-03 |
| `PHYHIP` | 1.413 | 1.044 | 0.437 | 3.25e-03 |
| `NEFL` | 1.959 | 1.559 | 0.330 | 4.78e-03 |
| `CTSD` | 1.640 | 1.240 | 0.403 | 5.56e-03 |

### Down in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `MBP` | 1.498 | 2.100 | -0.487 | 2.33e-07 |
| `GFAP` | 1.018 | 1.680 | -0.723 | 1.35e-05 |
| `PTGDS` | 1.688 | 2.007 | -0.249 | 5.56e-03 |
| `S100B` | 0.996 | 1.384 | -0.475 | 5.56e-03 |
| `PLP1` | 1.044 | 1.439 | -0.464 | 1.35e-02 |

## Markers vs major abutting layers

### vs `Layer 3`

| Gene | log2FC | padj | mean_in | mean_layer |
|------|-------:|-----:|--------:|-----------:|
| `SCGB2A2` | 0.854 | 5.23e-20 | 3.008 | 1.664 |
| `SCGB1D2` | 1.036 | 4.18e-15 | 1.700 | 0.829 |
| `MGP` | 0.924 | 4.62e-10 | 1.492 | 0.786 |
| `SAA1` | 1.202 | 2.03e-08 | 0.906 | 0.394 |
| `MUC1` | 0.889 | 7.02e-07 | 1.175 | 0.634 |
| `COL1A2` | 1.318 | 7.40e-06 | 0.494 | 0.198 |
| `FABP4` | 1.110 | 6.95e-05 | 0.630 | 0.292 |
| `TFF3` | 0.867 | 7.88e-03 | 0.662 | 0.363 |
| `CTSD` | 0.348 | 7.88e-03 | 1.640 | 1.288 |
| `TFF1` | 0.775 | 8.13e-03 | 0.728 | 0.426 |

## Claim bounds

1. This is a **single-slice** deep-dive of a geometric cryptic component.
2. Marker lists are differential expression, not causal cell-state proof.
3. Upgrade requires protein/IF validation and multi-slice replication of the same abutting-layer pattern + marker panel.

Artifacts: `C:/Spatial Transcriptomics/histoweave/research/discovery_uncertainty_niches/results/dlpfc_151508/component_rank1_n138`
