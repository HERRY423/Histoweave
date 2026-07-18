# Official matrix swap — panel Δ & AUROC reassessment

**Date:** 2026-07-17
**Dataset:** Xenium Prime Human Lymph Node Reactive FFPE (10x public preview v3.0.0)

## What was swapped

| | Synthetic (pre-swap) | Official (post-swap) |
|--|---------------------:|---------------------:|
| expression_source | `domain_conditioned_synthetic_pending_official_matrix` | **`official_10x_cell_feature_matrix`** |
| n cells | 12 000 | **15 000** |
| n genes | 266 (panel-ish) | **4 624** (Prime 5K) |
| pathology domains | LN / GC aggregate / Adipose | same (GeoJSON auto-calibrated) |
| matrix files | — | `cell_feature_matrix.h5` + `cells.csv.gz` (CDN 3.0.0) |
| geojson transform | N/A (points sampled in polygons) | scale≈**3.724**, offset=(−500, 1500); sample label frac **0.93** |

### Engineering fixes required for official path

1. Boolean mask dtype crash in `prepare_human_lymph_node.py` (StringDtype × `&=`).
2. Prefer GeoJSON `name` over short `classification.name` (was collapsing GC → `"Lymphoid"`).
3. Rare-domain floor in stratified subsample (GC was reduced to **1 cell** under pure proportional quotas).
4. Auto-calibrate micron→pixel GeoJSON transform (identity labelled ~47 cells).
5. Expand lymphoid panels for genes present on Prime 5K (`IL7`, `LMO2`, `CXCL13`, …).

## Geometry / uncertainty

| Metric | Synthetic | Official | Δ |
|--------|----------:|---------:|--:|
| AUROC(U → pathology boundary) | 0.431 | **0.439** | +0.008 |
| high-U cells | 2 419 | 3 121 | +702 |
| cryptic cells | 2 342 | 3 109 | +767 |
| cryptic / high-U | 96.8% | **99.6%** | +2.8 pp |
| estimated K | 2 | 4 | +2 |
| components ≥30 | 9 | **5** | −4 |
| direction-ok | 9/9 | **3/5** | — |

## Panel Δ (component-level)

Synthetic components had **large positive** B/T/GC Δrest by construction (domain-conditioned counts).
Official counts **collapse those inflated panel shifts**:

| Metric | Synthetic (typical) | Official (this run) |
|--------|--------------------:|--------------------:|
| GC Δrest mean | ~+0.27 | **≈ −0.04** |
| GC Δrest max | ~+0.46 | **≈ +0.00** |
| B Δrest mean | ~+0.49 | **≈ +0.00** |
| T Δrest mean | ~+0.51 | **≈ −0.03** |
| direction-ok rate | 100% | **60%** (3/5) |

Hard same-domain (same pathology label) panel shifts remain near zero / non-significant — same pattern as DLPFC L3 “direction vs rest works / hard gate fails”.

## GC deep-dive (official)

See `results/gc_deep_dive/GC_DEEP_DIVE_REPORT.md`.

| Rank | n | Dom | Reason | GC Δrest | Notable DE |
|-----:|--:|-----|--------|----------:|------------|
| 4 | 31 | LN | top GC Δ | +0.003 | `CLEC10A` up (padj≈0.01) vs rest & same-domain |
| 0 | 38 | LN | largest | −0.033 | no padj≤0.05 up genes |
| 3 | 31 | LN | next GC | −0.045 | few DE hits |

- **No cryptic component** is dominated by pathology label `Lymphoid aggregate + germinal center` (only 50 GC-labelled cells after floor sampling; none form a high-U contiguous blob ≥30).
- Best “GC-enriched” component is still **intra-LN parenchyma** with **null GC panel** — honest negative relative to synthetic over-claim.
- Compact geometry (internal edge frac 0.35–0.46) matches DLPFC “blob not ribbon” niche shape.

## Interpretation (claim ladder)

| Claim | Synthetic | Official |
|-------|-----------|----------|
| Pipeline runs cross-tissue | Level 0–1 ✅ | Level 0–1 ✅ |
| U tracks pathology boundaries | weak AUROC 0.43 | weak AUROC **0.44** (unchanged) |
| Cryptic components = GC biology | **over-stated** (synthetic panels) | **not supported** by GC panel / DE |
| Intra-domain niches exist geometrically | yes | yes (LN / adipose pure components) |
| Wet-lab IF priority | not indicated | **not indicated** for GC panel on these components |

**Bottom line:** Official matrix swap **worked end-to-end**. AUROC is essentially unchanged; **panel Δ and direction-ok collapse from synthetic inflated positives to near-null** — the scientifically correct outcome. GC deep-dive does **not** promote a Level-2 GC niche claim on this section without finer GC polygon coverage or protein validation.

## Reproduce

```bash
# Official rebuild (downloads matrix if missing) + discovery + GC deep-dive
python research/discovery_xenium_lymph/swap_and_rerun.py

# Or stepwise
python research/discovery_xenium_lymph/prepare_bundle.py
python research/discovery_xenium_lymph/run_discovery_ln.py
python research/discovery_xenium_lymph/analyze_gc_components.py

# CLI
histoweave discovery xenium-lymph --swap-official
histoweave discovery xenium-lymph --gc-deep-dive
```

Artifacts: `research/discovery_xenium_lymph/results/`
Comparison JSON can be regenerated via `swap_and_rerun.py` (writes `swap_comparison.json`).
