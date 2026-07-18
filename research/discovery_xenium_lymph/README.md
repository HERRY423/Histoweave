# Xenium lymph node discovery (second tissue context)

Applies the **same cryptic-niche architecture** used on DLPFC Visium to a
**non-brain, Xenium** tissue: human reactive lymph node.

## Why this tissue

| | DLPFC | Lymph node |
|--|-------|------------|
| Platform | Visium | **Xenium** |
| Organ | Cortex | **Lymph node** |
| Domain GT | Manual layers | **Pathology polygons** |
| Molecular panels | ENC1/HOPX vs MBP | **B-follicle / T-zone / GC** |

## Run

```bash
# Build bundle — prefers official matrix (CDN download if needed)
python research/discovery_xenium_lymph/prepare_bundle.py

# Discovery + lymphoid panels
python research/discovery_xenium_lymph/run_discovery_ln.py

# GC-enriched component deep-dive (DLPFC largest-component protocol)
python research/discovery_xenium_lymph/analyze_gc_components.py

# One-shot: swap official → rediscovery → panel/AUROC compare → GC dive
python research/discovery_xenium_lymph/swap_and_rerun.py

# CLI
histoweave discovery xenium-lymph
histoweave discovery xenium-lymph --swap-official
histoweave discovery xenium-lymph --gc-deep-dive
```

Force synthetic counts (architecture-only control):

```bash
python research/discovery_xenium_lymph/prepare_bundle.py --force-synthetic
```

## Provenance

| Source | Files | `expression_source` |
|--------|-------|---------------------|
| **Official (default when available)** | `cell_feature_matrix.h5` + `cells.csv.gz` + `annotation.geojson` | `official_10x_cell_feature_matrix` |
| Synthetic fallback | GeoJSON polygons + domain-conditioned counts | `domain_conditioned_synthetic_pending_official_matrix` |

Official CDN (v3.0.0):

* `…/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_cell_feature_matrix.h5`
* `…/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_cells.csv.gz`

GeoJSON micron→pixel transform is **auto-calibrated** when the identity map
labels ~0 cells. Rare pathology domains (GC aggregates) get a **stratified floor**
so they are not wiped at `max_cells=15k`.

## Outputs

| Path | Content |
|------|---------|
| `results/LYMPH_DISCOVERY_REPORT.md` | Discovery summary, AUROC, panel table |
| `results/slice_summary.json` | Machine-readable metrics |
| `results/components_panel.csv` | Per-component B/T/GC Δ |
| `results/gc_deep_dive/` | DE + adjacency deep-dives |
| `OFFICIAL_SWAP_COMPARISON.md` | Synthetic vs official reassessment |
| `GC_DEEP_DIVE_REPORT.md` | GC selection synthesis |

## Current official-run headline (2026-07-17)

* n=15 000 · genes=4624 · AUROC≈**0.44**
* components≥30: **5** · direction-ok **3/5**
* GC panel Δ near null (synthetic over-claim corrected)
* Best GC-ranked cryptic niche: intra-LN `CLEC10A` up, **not** a pathology-GC blob

See `OFFICIAL_SWAP_COMPARISON.md` for the full synthetic→official table.
