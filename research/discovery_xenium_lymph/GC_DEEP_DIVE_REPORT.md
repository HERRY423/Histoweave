# GC-enriched cryptic components — Xenium lymph node deep-dive

**Dataset:** `xenium_human_lymph_node` · **expression_source:** `official_10x_cell_feature_matrix`
**AUROC(U → pathology boundary)** from discovery run: 0.4393127680984638

## Selected components

| Rank | n | Dominant pathology | Reason | GC Δrest | B Δrest | T Δrest | Sig↑ genes | GC genes↑ | Abut | Internal edge frac |
|-----:|--:|--------------------|--------|---------:|--------:|--------:|-----------:|-----------|------|-------------------:|
| 4 | 31 | Lymph node | top_GC_panel_delta_rest | 0.003 | 0.073 | 0.109 | 1 | — | Lymph node | 0.46 |
| 0 | 38 | Lymph node | top_GC_panel_delta_rest | -0.033 | -0.211 | -0.083 | 0 | — | Lymph node | 0.41 |
| 3 | 31 | Lymph node | top_GC_panel_delta_rest | -0.045 | 0.062 | -0.055 | 4 | — | Lymph node | 0.35 |

## Cross-tissue对照 (DLPFC largest-component protocol)

| Aspect | DLPFC Visium | Xenium LN (this run) |
|--------|--------------|----------------------|
| Contiguous cryptic components | yes | yes |
| Adjacency table | WM / L1–L6 | pathology domains (LN / GC / adipose) |
| DE vs rest + abutting | yes | yes |
| Hard same-domain contrast | Layer 3 / Layer 6 | dominant pathology label |
| Pre-registered panels | ENC1/HOPX vs MBP | B / T / GC lymphoid |
| Largest-component parity | rank-0 always reported | rank-0 always included |

## Interpretation guide

- **Pathology GC label + GC panel↑**: strongest architectural hit for GC-like niche.
- **LN parenchyma + GC panel↑**: candidate cryptic GC-like subregion inside bulk LN (methods disagree; polygon GT may be too coarse).
- **High internal edge fraction**: compact niche (blob), not ribbon boundary.
- **same-domain hard DE empty**: mirrors DLPFC L3 pattern — direction vs rest works, intra-domain hard gate often fails.

Per-component reports live under `C:/Spatial Transcriptomics/histoweave/research/discovery_xenium_lymph/results/gc_deep_dive/`.
