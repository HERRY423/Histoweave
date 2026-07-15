# DLPFC SCT/scVI neighborhood pilot: final QA and interpretation

## Outcome first

The predeclared glial-enriched-interface (GEI) hypothesis did **not** survive
the three normalization/modeling branches. No primary module passed the
predeclared robustness gate. The strongest positive scVI-only result,
`vascular_barrier`, was directionally reversed by both log normalization and
SCTransform and must therefore be rejected as a normalization-robust candidate.

This is a useful negative result: lognorm and SCT agree closely across the 12
donor-by-module effect pairs (Pearson `r=0.908`, Spearman `rho=0.888`), whereas
scVI agrees poorly with lognorm (`r=0.200`) and SCT (`r=0.124`). The discrepancy
is biological-model dependence, not evidence of a conserved cross-tissue niche.

## Data and model contract

- All 12 DLPFC files passed an integer, finite, non-negative raw-count audit;
  `X` was exactly identical to `layers['counts']` in every file.
- The audit covered 47,338 spots. The experiment used one outcome-blind,
  median-size section per donor: `151508`, `151669`, and `151673`.
- scVI used all 11,637 selected-section spots and 1,927 genes. Spatial effect
  tests used the 11,532 spots with valid layer labels.
- Biological `n=3` donors. Sections are never treated as independent biological
  replicates.
- SCTransform 0.4.3 v2 ran on exact raw UMI matrices and modeled 1,876, 1,856,
  and 1,876 genes in the three sections. `glmGamPoi` was unavailable, so the
  package's valid but slower native fallback was used.
- scVI 1.3.3 ran for 80 CPU epochs with section as batch and decoded each marker
  after averaging over the three counterfactual section batches.
- SCTransform residuals were never supplied to scVI.
- Each spatial test used 199 within-section circular shifts along the first
  tissue axis. Effects adjust for layer, log library depth, and local spot
  spacing, then aggregate section effects within donor before the three-donor
  summary.

## Primary module results

`beta` is the partial standardized association with six-neighbor layer entropy.
The interval is a 95% t interval over the three biological donors. `q` is BH
adjusted over the unified module-by-branch spatial-shift tests. LODO reports the
number of held-out donors with positive incremental R-squared from adding the
neighborhood term.

| Module | Branch | beta | 95% donor CI | spatial q | donor direction | LODO positive |
|---|---:|---:|---:|---:|---:|---:|
| astro_ion | lognorm | -0.0154 | [-0.0442, 0.0134] | 0.104 | 3/3 negative | 0/3 |
| astro_ion | SCT | -0.0200 | [-0.0699, 0.0300] | 0.068 | 2/3 negative | 0/3 |
| astro_ion | scVI | +0.0163 | [-0.0247, 0.0573] | 0.104 | 2/3 positive | 2/3 |
| oligo_myelin | lognorm | -0.0160 | [-0.0439, 0.0119] | 0.070 | 3/3 negative | 2/3 |
| oligo_myelin | SCT | -0.0148 | [-0.0450, 0.0153] | 0.104 | 3/3 negative | 3/3 |
| oligo_myelin | scVI | -0.0172 | [-0.0366, 0.0022] | 0.148 | 3/3 negative | 2/3 |
| vascular_barrier | lognorm | -0.0115 | [-0.0564, 0.0334] | 0.276 | 2/3 negative | 0/3 |
| vascular_barrier | SCT | -0.0157 | [-0.0708, 0.0394] | 0.150 | 2/3 negative | 0/3 |
| vascular_barrier | scVI | +0.0268 | [-0.0309, 0.0845] | 0.030 | 3/3 positive | 3/3 |
| GEI | lognorm | -0.0241 | [-0.0469, -0.0014] | 0.030 | 3/3 negative | 2/3 |
| GEI | SCT | -0.0334 | [-0.0629, -0.0039] | 0.030 | 3/3 negative | 2/3 |
| GEI | scVI | +0.0106 | [-0.0390, 0.0601] | 0.360 | 2/3 positive | 2/3 |

The housekeeping control was positive in all three branches (scVI
`beta=0.0360`, 95% CI `[0.0157, 0.0563]`, `q=0.030`). The neuronal-synaptic
control was positive in lognorm/SCT but near zero in scVI. These controls show
that layer entropy retains broad spatial-expression structure and further argue
against a glia-specific claim.

## Predeclared decision gate

A primary module had to satisfy all of the following:

1. the same effect sign in lognorm, SCT, and scVI;
2. minimum absolute branch effect at least 0.05;
3. at least two of three donors matching the branch direction in every branch;
4. spatial-shift `q <= 0.10` in every branch;
5. positive LODO incremental R-squared in at least two of three donors in every
   branch.

No module passed. `oligo_myelin` was the only primary module with the same sign
in all three branches, but its largest absolute effect was only 0.0172 and not
all branch q-values passed. `vascular_barrier` failed the first, second, fourth,
and fifth criteria. The scVI vascular result is therefore a branch-specific
candidate that was explicitly falsified by the two count-normalization views.

## scVI model QC

- Full-data ELBO: `-459.513`.
- Full-data reconstruction loss: `436.701`.
- Section ELBOs: `151508=-363.057`, `151669=-465.079`,
  `151673=-570.897`.
- The model reached the configured 80 epochs and the saved weights were used for
  post-hoc ELBO, latent, and normalized-expression calculations.
- Training history was not retained because the Lightning logger was disabled;
  this is a reproducibility limitation, although the model weights and post-hoc
  QC are present.

## QA results and artifacts

- Every module score is finite in the final 11,532-row spot table.
- All requested marker genes were used in every branch: astro 9/9, oligo 10/10,
  vascular 9/9, neuronal control 7/7, housekeeping control 6/6.
- `results.json` parses, all CSV/NPZ files are non-empty, and PNG/PDF/SVG exports
  exist.
- `figure1_dlpfc_robustness.png`: 6,600 x 5,345 pixels.
- `figure2_representative_spatial.png`: 4,874 x 2,225 pixels.

Core outputs:

- `overall_effects.csv`: module-by-branch donor summaries and spatial q-values.
- `donor_effects.csv`: the actual biological-replicate effects.
- `leave_one_donor_out_prediction.csv`: held-out-donor incremental R-squared.
- `module_branch_concordance.csv`: predeclared gate results.
- `branch_pair_correlations.csv`: cross-branch agreement.
- `spot_module_scores.csv.gz`: auditable per-spot inputs to effect estimation.
- `spatial_shift_nulls.npz`: exact null distributions.
- `results.json`: machine-readable manifest and interpretation.

## Reproducibility status

`run_sct_scores.R` is the actual dependency-light Matrix Market SCTransform
runner used for these results. The older Python driver still points to the
project R bridge, which requires an unavailable R `anndata` package; a fresh
one-command run will stop at that bridge until the Python driver is patched to
call `run_sct_scores.R`. The completed output is valid, but this one-command
orchestration issue must be fixed before external handoff.

This pilot is DLPFC-only. It cannot establish cross-tissue, cross-species, or
cross-platform conservation. Its defensible conclusion is that the proposed
GEI/vascular neighborhood signal is not normalization robust in the current
human DLPFC pilot; independent raw-count tissue cohorts are required before any
Nature Methods-level biological claim.
