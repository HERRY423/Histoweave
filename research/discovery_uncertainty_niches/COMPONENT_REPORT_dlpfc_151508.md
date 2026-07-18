# Largest cryptic component — `dlpfc_151508`

**Size:** 154 spots (spatially contiguous cryptic niche).

## Manual-layer composition (domain_truth inside component)

| Layer | n spots | fraction |
|-------|--------:|---------:|
| Layer 6 | 154 | 1.000 |

**Dominant truth label:** `Layer 6` (100% of component). If external contacts are also almost exclusively this layer, the niche is **intra-layer substructure** (methods disagree *inside* a manual domain), not an inter-layer boundary ribbon.

## Adjacency to WM / L1–L6

- Internal kNN edges (within component): **627** (67.9% of all component-incident edges)
- External edges (to outside): **297**
- Top abutting layers by contact count: **Layer 6**
- Primary abutment distribution: `{'Layer 6': 128, 'internal_only': 26}`

| Layer | External contacts | Fraction | Background frac | Enrichment | Spots with this as primary abut |
|-------|------------------:|---------:|----------------:|-----------:|--------------------------------:|
| Layer 1 | 0 | 0.000 | 0.205 | 0.0 | 0 |
| Layer 2 | 0 | 0.000 | 0.070 | 0.0 | 0 |
| Layer 3 | 0 | 0.000 | 0.328 | 0.0 | 0 |
| Layer 4 | 0 | 0.000 | 0.088 | 0.0 | 0 |
| Layer 5 | 0 | 0.000 | 0.174 | 0.0 | 0 |
| Layer 6 | 297 | 1.000 | 0.088 | 11.394 | 128 |
| WM | 0 | 0.000 | 0.047 | 0.0 | 0 |

### Geometric interpretation

- **Intra-`Layer 6` compact niche:** every component spot and every external neighbour contact is `Layer 6`. Multi-method uncertainty is flagging a **subregion inside Layer 6**, not a WM↔cortex or L5↔L6 border.
- Internal edge fraction 67.9% indicates a compact blob (more self-contained niche).

## Marker genes (component vs all other spots)

Wilcoxon rank-sum on library-size log1p expression; BH-FDR. MT/Ribo/IG/HB genes excluded from ranking.

### Up in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `MBP` | 2.808 | 2.055 | 0.450 | 8.50e-17 |
| `SCGB2A2` | 3.437 | 2.243 | 0.616 | 3.06e-15 |
| `PLP1` | 2.062 | 1.404 | 0.555 | 2.06e-09 |
| `KRT8` | 1.080 | 0.561 | 0.946 | 5.31e-07 |
| `S100A11` | 1.132 | 0.600 | 0.915 | 1.14e-06 |
| `SCGB1D2` | 1.881 | 1.255 | 0.584 | 3.64e-06 |
| `AGR2` | 0.580 | 0.265 | 1.130 | 2.57e-04 |
| `TMSB10` | 3.407 | 3.166 | 0.106 | 2.57e-04 |
| `SPP1` | 0.510 | 0.242 | 1.077 | 5.05e-04 |
| `COL1A2` | 0.644 | 0.316 | 1.028 | 1.67e-03 |
| `MOBP` | 0.620 | 0.325 | 0.933 | 1.67e-03 |
| `SLC17A7` | 1.796 | 1.509 | 0.251 | 9.17e-03 |
| `CCK` | 2.735 | 2.582 | 0.083 | 1.32e-02 |
| `TUBA1A` | 1.860 | 1.669 | 0.156 | 3.93e-02 |

### Down in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `MTRNR2L12` | 2.946 | 3.266 | -0.149 | 1.64e-06 |
| `PDZD4` | 0.186 | 0.533 | -1.519 | 1.65e-04 |
| `VDAC2` | 0.223 | 0.543 | -1.283 | 5.17e-04 |
| `ATP5PD` | 0.528 | 0.906 | -0.779 | 1.67e-03 |
| `COX6C` | 2.224 | 2.643 | -0.249 | 2.07e-03 |
| `SST` | 0.185 | 0.476 | -1.365 | 3.33e-03 |
| `MTRNR2L1` | 2.094 | 2.471 | -0.238 | 4.28e-03 |
| `FABP7` | 0.096 | 0.333 | -1.793 | 4.93e-03 |
| `NHP2` | 0.399 | 0.714 | -0.842 | 5.29e-03 |
| `HOPX` | 0.639 | 1.007 | -0.655 | 6.00e-03 |

## Markers vs major abutting layers

### vs `Layer 6`

_No up-genes at padj ≤ 0.05._

## Claim bounds

1. This is a **single-slice** deep-dive of a geometric cryptic component.
2. Marker lists are differential expression, not causal cell-state proof.
3. Upgrade requires protein/IF validation and multi-slice replication of the same abutting-layer pattern + marker panel.

Artifacts: `C:/Spatial Transcriptomics/histoweave/research/discovery_uncertainty_niches/results/dlpfc_151508/largest_component`
