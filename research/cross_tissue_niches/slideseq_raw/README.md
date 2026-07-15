# Raw Slide-seqV2 recovery and technical validation

This directory recovers `Puck_200115_08` from the original SCP815 RData and
keeps every expression-dependent operation on finite, non-negative integer UMI
counts. The cached Squidpy H5AD contributes only barcode-matched annotations,
coordinates, and gene names; its normalized `X` and misleading `counts` layer
are never used.

## Reproduce

```powershell
& 'C:\Users\13264\anaconda3\python.exe' `
  'C:\Spatial Transcriptomics\histoweave\research\cross_tissue_niches\slideseq_raw\run_raw_recovery.py'
```

The raw workflow calls `export_raw_rdata.R`, writes the complete annotated
raw-count H5AD, checks the count contract, chooses a deterministic stratified
4,000-bead pilot, and calls `run_sct_pilot.R` for genuine
`sctransform::vst(vst.flavor='v2', residual_type='pearson')` residuals.

The optional scVI smoke test uses the same raw-count pilot and an existing
isolated environment:

```powershell
& 'C:\Users\13264\anaconda3\envs\scvi-env\python.exe' `
  'C:\Spatial Transcriptomics\histoweave\research\cross_tissue_niches\slideseq_raw\run_scvi_smoke.py'
```

## Claim boundary

This is one public hippocampal puck (`biological_n=1`). Results establish input
legality and generate candidates only. They do not establish independent
replication, cross-tissue generalization, or a Nature Methods-level biological
claim. Labels copied from the same puck are descriptive strata, not external
validation.
