# Cryptic component (rank-2) — `dlpfc_151673`

**Size:** 24 spots · **rank:** 1 (0 = largest; spatially contiguous cryptic niche).

## Manual-layer composition (domain_truth inside component)

| Layer | n spots | fraction |
|-------|--------:|---------:|
| Layer 6 | 24 | 1.000 |

**Dominant truth label:** `Layer 6` (100% of component). If external contacts are also almost exclusively this layer, the niche is **intra-layer substructure** (methods disagree *inside* a manual domain), not an inter-layer boundary ribbon.

## Adjacency to WM / L1–L6

- Internal kNN edges (within component): **60** (41.7% of all component-incident edges)
- External edges (to outside): **84**
- Top abutting layers by contact count: **Layer 6**
- Primary abutment distribution: `{'Layer 6': 24}`

| Layer | External contacts | Fraction | Background frac | Enrichment | Spots with this as primary abut |
|-------|------------------:|---------:|----------------:|-----------:|--------------------------------:|
| Layer 1 | 0 | 0.000 | 0.076 | 0.0 | 0 |
| Layer 2 | 0 | 0.000 | 0.070 | 0.0 | 0 |
| Layer 3 | 0 | 0.000 | 0.276 | 0.0 | 0 |
| Layer 4 | 0 | 0.000 | 0.061 | 0.0 | 0 |
| Layer 5 | 0 | 0.000 | 0.188 | 0.0 | 0 |
| Layer 6 | 84 | 1.000 | 0.186 | 5.37 | 24 |
| WM | 0 | 0.000 | 0.143 | 0.0 | 0 |

### Geometric interpretation

- **Intra-`Layer 6` compact niche:** every component spot and every external neighbour contact is `Layer 6`. Multi-method uncertainty is flagging a **subregion inside Layer 6**, not a WM↔cortex or L5↔L6 border.
- Internal edge fraction 41.7% indicates a compact blob (more self-contained niche).

## Marker genes (component vs all other spots)

Wilcoxon rank-sum on library-size log1p expression; BH-FDR. MT/Ribo/IG/HB genes excluded from ranking.

### Up in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `SCGB2A2` | 3.704 | 1.727 | 1.101 | 1.00e-09 |
| `SCGB1D2` | 2.436 | 0.911 | 1.419 | 2.04e-08 |
| `TFF3` | 1.037 | 0.484 | 1.100 | 4.81e-02 |

## Markers vs major abutting layers

### vs `Layer 6`

| Gene | log2FC | padj | mean_in | mean_layer |
|------|-------:|-----:|--------:|-----------:|
| `SCGB2A2` | 0.740 | 9.83e-07 | 3.704 | 2.218 |
| `SCGB1D2` | 1.000 | 1.93e-04 | 2.436 | 1.218 |

## Claim bounds

1. This is a **single-slice** deep-dive of a geometric cryptic component.
2. Marker lists are differential expression, not causal cell-state proof.
3. Upgrade requires protein/IF validation and multi-slice replication of the same abutting-layer pattern + marker panel.

Artifacts: `C:/Spatial Transcriptomics/histoweave/research/discovery_uncertainty_niches/results/dlpfc_151673/component_rank1_n24`
