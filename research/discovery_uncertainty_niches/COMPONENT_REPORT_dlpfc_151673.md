# Cryptic component (largest) — `dlpfc_151673`

**Size:** 47 spots · **rank:** 0 (0 = largest; spatially contiguous cryptic niche).

## Manual-layer composition (domain_truth inside component)

| Layer | n spots | fraction |
|-------|--------:|---------:|
| Layer 3 | 47 | 1.000 |

**Dominant truth label:** `Layer 3` (100% of component). If external contacts are also almost exclusively this layer, the niche is **intra-layer substructure** (methods disagree *inside* a manual domain), not an inter-layer boundary ribbon.

## Adjacency to WM / L1–L6

- Internal kNN edges (within component): **176** (62.4% of all component-incident edges)
- External edges (to outside): **106**
- Top abutting layers by contact count: **Layer 3**
- Primary abutment distribution: `{'Layer 3': 45, 'internal_only': 2}`

| Layer | External contacts | Fraction | Background frac | Enrichment | Spots with this as primary abut |
|-------|------------------:|---------:|----------------:|-----------:|--------------------------------:|
| Layer 1 | 0 | 0.000 | 0.077 | 0.0 | 0 |
| Layer 2 | 0 | 0.000 | 0.071 | 0.0 | 0 |
| Layer 3 | 106 | 1.000 | 0.264 | 3.783 | 45 |
| Layer 4 | 0 | 0.000 | 0.061 | 0.0 | 0 |
| Layer 5 | 0 | 0.000 | 0.189 | 0.0 | 0 |
| Layer 6 | 0 | 0.000 | 0.194 | 0.0 | 0 |
| WM | 0 | 0.000 | 0.144 | 0.0 | 0 |

### Geometric interpretation

- **Intra-`Layer 3` compact niche:** every component spot and every external neighbour contact is `Layer 3`. Multi-method uncertainty is flagging a **subregion inside Layer 3**, not a WM↔cortex or L5↔L6 border.
- Internal edge fraction 62.4% indicates a compact blob (more self-contained niche).

## Marker genes (component vs all other spots)

Wilcoxon rank-sum on library-size log1p expression; BH-FDR. MT/Ribo/IG/HB genes excluded from ranking.

### Up in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `GAP43` | 2.162 | 1.575 | 0.457 | 3.80e-04 |
| `HAPLN4` | 0.796 | 0.398 | 0.998 | 3.25e-03 |
| `VSNL1` | 3.038 | 2.585 | 0.233 | 3.25e-03 |
| `SAA1` | 0.647 | 0.293 | 1.143 | 7.08e-03 |
| `ENC1` | 2.814 | 2.242 | 0.328 | 7.16e-03 |
| `NRGN` | 3.502 | 3.147 | 0.154 | 1.07e-02 |
| `HS6ST1` | 0.736 | 0.401 | 0.878 | 1.16e-02 |
| `NRXN1` | 1.252 | 0.808 | 0.631 | 1.89e-02 |
| `C14orf132` | 0.799 | 0.458 | 0.805 | 2.43e-02 |
| `CCK` | 2.896 | 2.464 | 0.233 | 2.73e-02 |
| `CAMKK2` | 1.157 | 0.736 | 0.651 | 2.92e-02 |
| `TPRG1L` | 1.094 | 0.688 | 0.668 | 3.14e-02 |
| `VTI1B` | 0.883 | 0.546 | 0.693 | 3.74e-02 |
| `FAM213A` | 1.086 | 0.705 | 0.622 | 3.74e-02 |
| `HOPX` | 1.439 | 0.980 | 0.554 | 4.70e-02 |

### Down in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `MBP` | 2.238 | 3.142 | -0.489 | 4.73e-06 |
| `KRT19` | 0.261 | 0.867 | -1.732 | 2.39e-03 |
| `KRT8` | 0.040 | 0.555 | -3.789 | 2.39e-03 |
| `PLP1` | 1.609 | 2.379 | -0.564 | 7.25e-03 |
| `KRT18` | 0.042 | 0.452 | -3.413 | 1.15e-02 |
| `GFAP` | 1.053 | 1.783 | -0.760 | 1.16e-02 |
| `CRYAB` | 1.014 | 1.619 | -0.674 | 2.43e-02 |
| `IGFBP5` | 0.144 | 0.521 | -1.858 | 3.22e-02 |
| `S100B` | 1.006 | 1.552 | -0.625 | 3.34e-02 |
| `CNP` | 0.917 | 1.526 | -0.734 | 3.74e-02 |

## Markers vs major abutting layers

### vs `Layer 3`

_No up-genes at padj ≤ 0.05._

## Claim bounds

1. This is a **single-slice** deep-dive of a geometric cryptic component.
2. Marker lists are differential expression, not causal cell-state proof.
3. Upgrade requires protein/IF validation and multi-slice replication of the same abutting-layer pattern + marker panel.

Artifacts: `C:/Spatial Transcriptomics/histoweave/research/discovery_uncertainty_niches/results/dlpfc_151673/largest_component`
