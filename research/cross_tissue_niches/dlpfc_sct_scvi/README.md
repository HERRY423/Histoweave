# DLPFC SCT/scVI spatial-neighborhood robustness pilot

This analysis audits all 12 human DLPFC Visium sections and maps them explicitly
to three donors. The default pilot selects one section per donor by median spot
count without consulting any expression outcome. It runs three independent
branches from the same integer UMI matrix:

1. library-size log normalization;
2. per-section SCTransform v2 Pearson residuals through the existing HistoWeave
   R bridge;
3. scVI 1.3.3 on raw integer counts, with section as the batch key.

SCTransform residuals are never supplied to scVI. The response is a predeclared
astrocyte ion-homeostasis, oligodendrocyte/myelin, vascular/barrier, or combined
GEI module score. The exposure is six-neighbor layer entropy. Within-section
effects adjust for layer, library depth, and local spot spacing. Sections are
aggregated within donor before the three-donor summary; spatial circular shifts
provide a topology-aware null. Leave-one-donor-out prediction is descriptive
and is not treated as independent cell-level inference.

Run with the provided scVI environment:

```powershell
& 'C:\Users\13264\anaconda3\envs\scvi-env\python.exe' `
  'C:\Spatial Transcriptomics\histoweave\research\cross_tissue_niches\dlpfc_sct_scvi\run_dlpfc_sct_scvi.py' `
  --scope pilot --n-hvg 1900 --scvi-epochs 80 --permutations 199
```

Use `--scope all` only after the pilot passes the data and runtime gates. Even a
robust DLPFC result is a candidate for cross-region validation, not cross-tissue
validation on its own.
