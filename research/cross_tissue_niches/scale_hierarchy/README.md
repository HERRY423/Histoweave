# Conserved spatial-scale hierarchy screen

This isolated analysis tests a pre-specified topology hypothesis: broad
neuronal homotypy should persist over a longer native-nearest-neighbour scale
than astrocyte–vascular cross-association in DLPFC Visium, hypothalamic MERFISH,
and hippocampal Slide-seqV2.

The screen is deliberately conservative:

- DLPFC uses fixed marker-score fields from audited integer raw counts.
- MERFISH and Slide-seqV2 use labels only; their cached normalized matrices are
  never passed to SCT or scVI.
- Oligodendrocytes are excluded, so the known white/grey-matter oligodendrocyte
  axis cannot create the result.
- Radius is reported in platform-native NN units.
- Two-dimensional toroidal spatial shifts provide the null, and max-T controls
  selection across the eight tested scales.
- A positive result is only a candidate warranting independent-animal
  replication, not a discovery claim.

Run from the repository with the existing Anaconda Python:

```powershell
& 'C:\Users\13264\anaconda3\python.exe' `
  'C:\Spatial Transcriptomics\histoweave\research\cross_tissue_niches\scale_hierarchy\run_scale_hierarchy.py'
```

All reproducible tables, the JSON go/no-go decision, and PNG/PDF/SVG figures are
written to `scale_hierarchy/results/`.
