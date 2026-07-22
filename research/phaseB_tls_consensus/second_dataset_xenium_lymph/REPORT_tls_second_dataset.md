# TLS discovery transport: second independent dataset

## Result

The original breast-cancer Visium observation is now paired with the official
10x Xenium Prime reactive lymph-node dataset. The direct, locked definition
(`B_score > 90th percentile AND T_score > 90th percentile`) **did not
replicate**:

- Breast Visium: Moran's I **0.665**,
  99 foci, contiguity
  **0.727**.
- Lymph-node Xenium: Moran's I **0.190**,
  29 co-high cells, contiguity **0.000**.
- Direct foci versus the 50 pathology
  germinal-center cells: F1 **0.000**
  (0 intersecting cells).
- Fixed k=20 B/T neighbourhood co-localisation AUROC for
  pathology GC: **0.364**.

## Interpretation and claim boundary

This is an informative negative transport test. A Visium spot can contain both
B and T cells, whereas a Xenium observation is an individual cell; identical
per-unit co-expression is therefore not measurement-invariant. The fixed
neighbourhood sensitivity did not rescue the result, but the GC reference is
only 50 retained cells after the documented
stratified subsample. The result does not disprove the breast-cancer niche and
does not establish TLS generalisation. It establishes that HistoWeave must make
TLS endpoints assay-aware and retain the global/abstention default when the
endpoint does not transport.

## Reproduction

```bash
python research/phaseB_tls_consensus/analyze_tls_second_dataset.py
```
