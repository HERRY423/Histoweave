# Cryptic component (largest) — `dlpfc_151669`

**Size:** 137 spots · **rank:** 0 (0 = largest; spatially contiguous cryptic niche).

## Manual-layer composition (domain_truth inside component)

| Layer | n spots | fraction |
|-------|--------:|---------:|
| Layer 3 | 137 | 1.000 |

**Dominant truth label:** `Layer 3` (100% of component). If external contacts are also almost exclusively this layer, the niche is **intra-layer substructure** (methods disagree *inside* a manual domain), not an inter-layer boundary ribbon.

## Adjacency to WM / L1–L6

- Internal kNN edges (within component): **534** (65.0% of all component-incident edges)
- External edges (to outside): **288**
- Top abutting layers by contact count: **Layer 3**
- Primary abutment distribution: `{'Layer 3': 116, 'internal_only': 21}`

| Layer | External contacts | Fraction | Background frac | Enrichment | Spots with this as primary abut |
|-------|------------------:|---------:|----------------:|-----------:|--------------------------------:|
| Layer 1 | 0 | 0.000 | 0.196 | 0.0 | 0 |
| Layer 2 | 0 | 0.000 | 0.067 | 0.0 | 0 |
| Layer 3 | 288 | 1.000 | 0.259 | 3.859 | 116 |
| Layer 4 | 0 | 0.000 | 0.106 | 0.0 | 0 |
| Layer 5 | 0 | 0.000 | 0.173 | 0.0 | 0 |
| Layer 6 | 0 | 0.000 | 0.102 | 0.0 | 0 |
| WM | 0 | 0.000 | 0.068 | 0.0 | 0 |

### Geometric interpretation

- **Intra-`Layer 3` compact niche:** every component spot and every external neighbour contact is `Layer 3`. Multi-method uncertainty is flagging a **subregion inside Layer 3**, not a WM↔cortex or L5↔L6 border.
- Internal edge fraction 65.0% indicates a compact blob (more self-contained niche).

## Marker genes (component vs all other spots)

Wilcoxon rank-sum on library-size log1p expression; BH-FDR. MT/Ribo/IG/HB genes excluded from ranking.

### Up in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `CARTPT` | 0.746 | 0.406 | 0.876 | 3.26e-04 |
| `ENC1` | 2.492 | 2.185 | 0.189 | 3.97e-03 |
| `CHN1` | 2.667 | 2.372 | 0.170 | 3.97e-03 |
| `CKB` | 3.022 | 2.836 | 0.092 | 5.58e-03 |
| `NSG2` | 1.129 | 0.828 | 0.446 | 7.27e-03 |
| `HOPX` | 1.598 | 1.284 | 0.316 | 1.11e-02 |
| `TESPA1` | 0.798 | 0.533 | 0.581 | 1.14e-02 |
| `ALDOA` | 0.973 | 0.694 | 0.486 | 1.15e-02 |
| `RRAGA` | 0.832 | 0.581 | 0.520 | 1.42e-02 |
| `YWHAH` | 2.724 | 2.503 | 0.122 | 1.58e-02 |
| `CAPNS1` | 1.413 | 1.130 | 0.323 | 2.88e-02 |
| `NRGN` | 3.155 | 2.961 | 0.091 | 3.21e-02 |
| `SYNE1` | 0.571 | 0.374 | 0.611 | 3.25e-02 |

### Down in component

| Gene | mean_in | mean_out | log2FC | padj |
|------|--------:|---------:|-------:|-----:|
| `SCGB2A2` | 1.266 | 1.928 | -0.608 | 8.62e-05 |
| `SCGB1D2` | 0.557 | 1.031 | -0.888 | 3.26e-04 |
| `MBP` | 1.916 | 2.224 | -0.215 | 7.00e-04 |
| `PLP1` | 1.331 | 1.628 | -0.291 | 1.56e-02 |
| `DBI` | 0.704 | 1.008 | -0.519 | 1.62e-02 |
| `EFHD2` | 0.217 | 0.412 | -0.929 | 4.12e-02 |

## Markers vs major abutting layers

### vs `Layer 3`

_No up-genes at padj ≤ 0.05._

## Claim bounds

1. This is a **single-slice** deep-dive of a geometric cryptic component.
2. Marker lists are differential expression, not causal cell-state proof.
3. Upgrade requires protein/IF validation and multi-slice replication of the same abutting-layer pattern + marker panel.

Artifacts: `C:/Spatial Transcriptomics/histoweave/research/discovery_uncertainty_niches/results/dlpfc_151669/largest_component`
