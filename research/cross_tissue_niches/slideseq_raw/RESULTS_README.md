# Slide-seqV2 raw-count recovery: results and claim boundary

## Outcome

The original SCP815 `Puck_200115_08` count matrix was recovered and converted
to a count-valid H5AD. The cached Squidpy expression matrix was never used for
SCT, scVI, or module scoring; it contributed labels, coordinates, and gene names
only.

- Raw matrix: 53,208 beads × 23,264 genes; 22,396,657 nonzeros; 32,311,360 UMI.
- Stored counts: int32, finite, non-negative, integer; range 1–1,917.
- Written H5AD: `X` and `layers['counts']` are CSR int32 with exactly identical
  `data`, `indices`, and `indptr` arrays (read-back audit).
- Annotation match: 41,786/41,786 cached barcodes and coordinates matched
  exactly; this annotates 78.5333% of raw beads. Maximum coordinate error is 0.
- Gene match: 4,000/4,000 cached genes are present in the raw matrix.

## SCT technical validation

A deterministic cluster-stratified subset of 4,000 annotated beads was used.
The gene set was 2,000 raw-count HVGs plus predefined markers (2,024 genes in
the union). Genuine `sctransform::vst` v2 Pearson residuals modeled all 2,024
genes and were finite (range -1.0211 to 63.2456).

Median per-gene absolute correlation with log library depth was 0.1260 for raw
log1p counts, 0.0940 for library-normalized log1p, and 0.01333 for SCT
residuals. This is a technical depth-dependence diagnostic, not a biological
endpoint.

## scVI technical validation

scVI 1.3.3 trained for exactly 40 CPU epochs on the same 4,000 × 2,024 raw-count
pilot with 20 latent dimensions. No `batch_key` was supplied and no synthetic
batch was created. Raw-count input, finite latent output, and finite non-negative
normalized expression all passed QC; normalized cell sums were approximately
10,000.

Median absolute correlation with log library depth was 0.12649 for scVI
normalized expression and 0.12060 for the latent coordinates, versus 0.09396
for ordinary library-normalized log1p. Thus this single-puck smoke test does not
show improved depth removal and cannot be cited as batch correction or
generalization.

## Pre-specified vascular-barrier directional check

The weak DLPFC scVI candidate was tested using the unchanged nine-gene
`vascular_barrier` definition from `dlpfc_sct_scvi/results/module_spec.tsv`.
Only five genes were available in the fixed 2,024-gene pilot (`CLDN5`, `PECAM1`,
`KDR`, `SLC2A1`, `MFSD2A`); `VWF`, `EMCN`, `RAMP2`, and `ABCB1` were missing and
were not substituted.

The exposure was six-neighbor Shannon entropy of cached hippocampal cluster
labels. Regressions controlled log library depth, local spacing, focal cluster,
and six-neighbor vascular abundance. P-values below are from 999 deterministic
two-dimensional toroidal spatial shifts.

| Branch | beta | Analytic p | Spatial-shift p | DLPFC direction match |
|---|---:|---:|---:|---|
| lognorm | +0.02819 | 0.0881 | 0.055 | No (DLPFC was negative) |
| SCT | +0.02975 | 0.0749 | 0.051 | No (DLPFC was negative) |
| scVI | +0.04325 | 0.00344 | 0.090 | Yes |

All Slide-seq branches were positive and lognorm/SCT scores were highly
correlated (r=0.956), but the pre-specified scVI spatial p<=0.05 criterion failed
and the DLPFC lognorm/SCT directions did not transfer. This is technical
directional evidence only; `biological_validation=false`.

## Claim boundary

This public dataset contains one puck (`biological_n=1`) and the labels come
from the same puck. It cannot establish independent replication, a conserved
cross-tissue program, or a Nature Methods-level biological discovery.

Key machine-readable files are `results/count_contract.json`,
`results/annotation_match.json`, `results/sct_technical_metrics.json`,
`results/scvi_technical_metrics.json`, and
`results/vascular_external_hypothesis.json`. Exact shift nulls are in
`results/vascular_external_hypothesis_shift_nulls.npz`.
