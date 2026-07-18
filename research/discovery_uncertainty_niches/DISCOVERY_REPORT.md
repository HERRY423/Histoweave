# Uncertainty-niche discovery (DLPFC multi-method)

**Tools:** HistoWeave non-oracle *K*, multi-method domain ensemble (kmeans / spectral /
banksy_py / GMM × coarse+fine *K*), boundary-uncertainty maps, Moran's I + BH-FDR,
spatial-shift nulls, cryptic-component geometry.

**Question:** Do high multi-method uncertainty zones contain cryptic spatial expression
programs or contiguous tissue niches that are *not* explained only by known cortical
layer boundaries?

**Global decision:** `NO_GO_OR_WEAK` for a **gene-program discovery claim**.
**Per-slice geometric status:** all three donor slices → `GEOMETRIC_CANDIDATE`.

## What this run actually found (honest)

| Finding | Support | Claim level |
|---------|---------|-------------|
| Non-oracle silhouette *K* collapses to **K=2** (grey/white-matter axis) on all 3 donors, while anatomy has 7–8 layers | 3/3 slices | **Methodological**: default *K* estimation does not recover laminar count; fine *K* must be a separate sensitivity track |
| High multi-method uncertainty is **mostly cryptic** (61–76% of high-U spots are *not* known layer boundaries) | 3/3 | **Geometric**: disagreement is not just “layer edges” |
| Known-boundary AUROC of uncertainty is only **0.45–0.60** | 3/3 | Uncertainty map is a weak predictor of manual layer edges alone |
| Contiguous cryptic niches exist (5–7 components/slice; largest 47–154 spots) | 3/3 | **Geometric candidate regions** for follow-up — *not* yet a named cell state |
| SVG genes with Moran FDR are abundant, but **none** pass spatial-shift FDR for cryptic enrichment | 3/3 | **No gene-level program claim** after proper nulls |

**Bottom line for biology:** HistoWeave can *map* multi-method disagreement niches that
are spatially coherent and largely off known layer boundaries. Turning those niches into
a new **cell state / mechanism** requires a gene (or protein) program that survives
spatial nulls and replicates — which this pre-registered gene gate **rejects** on DLPFC.
That rejection is itself a result: public DLPFC is a hard place to claim “new regions”
without orthogonal assays.

## Pre-registered gates

```json
{
  "min_slices_replicating": 2,
  "max_fdr_q": 0.05,
  "min_shift_null_p": 0.05,
  "min_cryptic_fraction": 0.05,
  "max_known_boundary_fraction_of_high_u": 0.85
}
```

## Per-slice summary

| Slice | n | oracle K | est. K | ens. K | high-U | cryptic | cryptic/high-U | AUROC(known) | #comp | largest | SVG FDR | cryptic SVG | status |
|-------|--:|---------:|-------:|-------:|-------:|--------:|---------------:|-------------:|------:|--------:|--------:|------------:|--------|
| dlpfc_151508 | 4381 | 7 | 2 | 7 | 928 | 648 | 0.698 | 0.581 | 5 | 154 | 1183 | 0 | `GEOMETRIC_CANDIDATE` |
| dlpfc_151669 | 3645 | 8 | 2 | 7 | 771 | 586 | 0.760 | 0.452 | 7 | 137 | 1269 | 0 | `GEOMETRIC_CANDIDATE` |
| dlpfc_151673 | 3611 | 7 | 2 | 7 | 730 | 446 | 0.611 | 0.603 | 6 | 47 | 1377 | 0 | `GEOMETRIC_CANDIDATE` |

### Notes

- **dlpfc_151508:** no_fdr_svg_in_cryptic;geometric_cryptic_niches;component_sizes=[154, 138, 23, 18, 15]; top genes: —
- **dlpfc_151669:** no_fdr_svg_in_cryptic;geometric_cryptic_niches;component_sizes=[137, 20, 17, 17, 16]; top genes: —
- **dlpfc_151673:** no_fdr_svg_in_cryptic;geometric_cryptic_niches;component_sizes=[47, 24, 18, 17, 17]; top genes: —

## Cross-slice replicated cryptic genes

_No gene passed multi-slice replication gates._

## Interpretation bounds (honest)

1. DLPFC layers are a **saturated public benchmark**. Recovering layers is not a biological discovery; cryptic niches are *candidates* only.
2. Non-oracle *K* and multi-method uncertainty reduce method-choice artifacts, but residual technical effects (depth, batch) remain.
3. Upgrade path to a claim: independent imaging/protein validation, perturbation or orthogonal platform, and pre-registered effect sizes.
4. If `high_u` mostly equals known boundaries (AUROC high + cryptic fraction low), the result is a **method-consistency diagnostic**, not a new tissue region.

**Rationale:** Requires ≥2 slices with CANDIDATE status and ≥3 genes replicated across ≥2 slices after SVG FDR + spatial-shift FDR.

## Artifacts

- `results/<slice>/spot_uncertainty_map.csv`
- `results/<slice>/svg_morans_fdr.csv`
- `results/<slice>/cryptic_gene_enrichment.csv`
- `results/cross_slice_replication.json`
- `results/slice_summaries.csv`
