# Xenium lymph node cryptic-niche discovery

**Dataset:** `xenium_human_lymph_node` · assay=xenium · tissue=lymph_node
**Expression source:** `official_10x_cell_feature_matrix`

## Pipeline (same architecture as DLPFC)

1. Non-oracle *K* (silhouette) + fine *K* ensemble
2. Domain methods: kmeans / spectral / banksy_py / gaussian_mixture
3. Target-free `boundary_uncertainty`
4. Cryptic = high-U ∧ ¬ pathology boundary
5. Contiguous components + **lymphoid** panels (B / T / GC)

## Geometry

| Metric | Value |
|--------|------:|
| n cells | 15000 |
| pathology domains | 3 |
| estimated K | 4 |
| ensemble K | 4 |
| high-U cells | 3121 |
| cryptic cells | 3109 (99.6% of high-U) |
| AUROC(U → pathology boundary) | 0.439 |
| components ≥30 | 5 |

## Components

| Rank | n | Dominant pathology | Purity | Class | dir_ok | GC Δrest | B Δrest | T Δrest | Abut |
|-----:|--:|--------------------|-------:|-------|:------:|---------:|--------:|--------:|------|
| 0 | 38 | Lymph node | 1.00 | LN_parenchyma | N | -0.033 | -0.212 | -0.082 | — |
| 1 | 34 | Adipose tissue | 1.00 | Adipose_like | N | -0.085 | 0.178 | -0.180 | — |
| 2 | 33 | Lymph node | 1.00 | LN_parenchyma | Y | -0.057 | -0.109 | 0.063 | — |
| 3 | 31 | Lymph node | 1.00 | LN_parenchyma | Y | -0.047 | 0.069 | -0.053 | — |
| 4 | 31 | Lymph node | 1.00 | LN_parenchyma | Y | 0.000 | 0.075 | 0.116 | — |

**Direction-ok components:** 3/5

## Cross-tissue takeaway vs DLPFC

| | DLPFC (Visium) | Lymph node (Xenium) |
|--|----------------|---------------------|
| Domain GT | cortical layers | pathology polygons (LN / GC aggregate / adipose) |
| Molecular panels | ENC1/HOPX vs MBP | B-follicle / T-zone / GC programs |
| Pipeline | identical architecture | identical architecture |

## Provenance note

**This run uses official 10x counts** (`official_10x_cell_feature_matrix`).
GeoJSON pathology polygons provide domain truth (auto-calibrated micron→pixel
transform; rare GC domain floor-sampled). Synthetic control arm metrics and the
full before/after table live in `OFFICIAL_SWAP_COMPARISON.md`.

Artifacts: `C:/Spatial Transcriptomics/histoweave/research/discovery_xenium_lymph/results`
