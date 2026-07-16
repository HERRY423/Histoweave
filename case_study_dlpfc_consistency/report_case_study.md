# Case study: cross-method consistency on DLPFC 6-layer cortex

**Dataset:** human dorsolateral prefrontal cortex (DLPFC) slice **151673**, 10x Visium,
spatialLIBD manual layer annotation (Maynard et al., *Nature Neuroscience* 2021).
**Task:** recover the 6 cortical layers + white matter (7 domains) and quantify **which
layer boundaries a single method misses** that the others resolve.
**Platform:** HistoWeave plugin interface (`banksy_py`, `gaussian_mixture`, `scanvi`).

- 3,611 annotated spots · 2,000 HVGs · 7 ground-truth domains
- **1,023 boundary spots (28.3%)** — a spot is a boundary spot if any of its 6 nearest
  spatial neighbours belongs to a different manual layer. These transition zones are the
  hard, biologically meaningful part of the problem.

## Methods compared

| Method | Family | Uses spatial context | Uses labels |
|---|---|---|---|
| **BANKSY** (`banksy_py`) | neighbourhood-augmented clustering | ✅ own + neighbour-mean + azimuthal-gradient features (λ=0.8) | ❌ unsupervised |
| **GMM** (`gaussian_mixture`) | expression-only soft clustering | ❌ | ❌ unsupervised |
| **scANVI** (`scanvi`) | semi-supervised deep generative | ❌ (expression latent) | ✅ 20% of manual layers seeded |

scANVI is run in its intended regime: a small (20%) random seed of manual layer labels is
revealed, the rest hidden as `Unknown`, and the model propagates labels through the scVI
latent space. This mirrors the realistic setting where a pathologist annotates a fraction
of spots and the model completes the rest.

## Results

| Method | ARI | NMI | Accuracy | Interior acc. | **Boundary acc.** | Runtime (s) |
|---|---|---|---|---|---|---|
| BANKSY | 0.284 | 0.428 | 0.543 | 0.640 | 0.298 | 12.6 |
| GMM | 0.208 | 0.286 | 0.403 | 0.459 | 0.262 | 7.6 |
| **scANVI** | **0.488** | **0.516** | **0.727** | **0.801** | **0.541** | 84.3 |

**Key observations**

1. **Every method degrades sharply at boundaries.** Interior accuracy exceeds boundary
   accuracy by 24–34 points for all three methods — the layer transitions are where
   method choice matters most.
2. **Spatial context beats pure expression, but supervision beats both.** BANKSY's
   neighbourhood term lifts ARI from 0.208 (GMM) to 0.284; a 20% label seed (scANVI)
   nearly doubles it again to 0.488 and roughly doubles boundary accuracy (0.541 vs
   0.262–0.298).
3. **No method is uniformly best at boundaries.** The unique-miss analysis shows each
   method fails on boundary spots the other two get right.

### Boundaries missed *uniquely* by one method

For each method we count boundary spots it labels incorrectly **while the other two are
both correct** — the layer transitions that method alone cannot resolve.

| Method | Boundary spots missed | **Uniquely missed** (others correct) | % of all boundary spots |
|---|---|---|---|
| BANKSY | 718 | 29 | 2.83% |
| GMM | 755 | **68** | **6.65%** |
| scANVI | 470 | 32 | 3.13% |

- **GMM uniquely misses the most boundaries (68 spots, 6.7%)** — without a spatial term it
  scatters transition spots that both the spatially-aware (BANKSY) and label-aware
  (scANVI) methods recover.
- Even the strongest method (scANVI) has **32 boundary spots it alone gets wrong** that
  BANKSY + GMM both resolve — a concrete argument for the multi-method consensus that
  HistoWeave is built to orchestrate. Relying on any single method silently loses these
  transition zones.

### Target-free uncertainty map recovers single-method blind spots

The unique-miss count above uses the manual layers to define an error after the fact. To
make the signal usable when manual layers are unavailable, we additionally computed a
**target-free boundary-uncertainty map** from the three prediction maps alone. For every
method, a 6-nearest-neighbour edge receives a boundary vote when the method assigns its two
endpoints to different domains. The spot score aggregates the Bernoulli entropy of these
cross-method edge votes. It is invariant to each method's arbitrary cluster-label names;
manual layers and the `is_boundary` field are used only for post-hoc validation.

The resulting map identifies areas in which a single method's smooth boundary conflicts
with a boundary supported by the other methods:

| Validation target | Positives | AUROC | Enrichment at uncertainty >= P80 | Recall at >= P80 |
|---|---:|---:|---:|---:|
| manual boundary spot | 1,023 | 0.633 | 1.29x | 44.7% |
| any uniquely missed boundary | 129 | **0.702** | **1.83x** | **63.6%** |
| uniquely missed by BANKSY | 29 | 0.660 | 1.79x | 62.1% |
| uniquely missed by GMM | 68 | 0.702 | 1.82x | 63.2% |
| uniquely missed by scANVI | 32 | **0.720** | **1.89x** | **65.6%** |

At the inclusive 80th-percentile threshold, the map flags 1,253 of 3,611 spots and
contains 82 of the 129 single-method blind spots. The flagged set is larger than exactly
20% because many spots tie at the P80 score; the enrichment calculation uses the full
inclusive set. These results support the intended use of uncertainty as a review-priority
map, not as a replacement for manual annotation or a calibrated probability of error.

## Figures

- `figures/fig1_layer_comparison.svg` — spatial maps: manual layers vs BANKSY / GMM / scANVI.
- `figures/fig2_boundary_misses.svg` — interior vs boundary accuracy, and uniquely-missed
  boundary spots per method.
- `figures/fig3_agreement_map.svg` — per-spot cross-method agreement (0–3 methods correct);
  low-agreement regions trace the layer boundaries.
- `figures/fig4_uncertainty_boundary_map.svg` — manual boundary reference, target-free
  uncertainty, uniquely missed boundaries, and enrichment at uncertainty >= P80.

Machine-readable uncertainty artifacts are in `results/per_spot_uncertainty.csv`,
`results/uncertainty_hotspots.csv`, and `results/uncertainty_validation.json`; the
checksummed build record is `results/uncertainty_manifest.json`.

## Interactive comparison (Vitessce)

- `vitessce_config.json` — self-contained Vitessce v1 config exposing manual layers, all
  three method predictions, per-spot agreement (`n/3`), and boundary mask as linked
  `obsSets`. Selecting a cluster in the scatterplot drives the marker-gene heatmap via the
  enhanced bidirectional linking (task 3).
- `case_study_report.html` — standalone interactive HTML report (Vitessce loaded from CDN,
  data inlined; opens with no server).

## Data / method notes and limitations

- **Single slice, within-study.** Numbers are illustrative of relative method behaviour on
  one well-characterised section, not a generalization claim; a study-grouped multi-slice
  protocol is required for that (see `5x10_dlpfc_benchmark/` and `study_grouped_*`).
- **`banksy_py` is a native scaffold** reimplementation of the BANKSY feature construction
  (own + neighbour-mean + azimuthal-gradient, λ weighting) used because the canonical
  Bioconductor `banksy` wrapper needs the `histoweave-r` container. It reproduces BANKSY's
  qualitative behaviour (λ↑ ⇒ more spatial coherence ⇒ higher ARI) but is not the exact
  upstream implementation.
- **scANVI ran CPU-only, capped epochs** (scVI 50 / scANVI 25). With GPU and longer
  training the absolute numbers would improve; the relative ordering is the message.

## Reproduce

```bash
python case_study_dlpfc_consistency/prepare.py        # fetch + build 151673.h5ad
python case_study_dlpfc_consistency/run_case_study.py # methods, metrics, figures, config
python case_study_dlpfc_consistency/build_uncertainty_case.py # target-free map + validation
```

The last command can also be rerun independently from the archived
`case_study_dlpfc_consistency/results/per_spot.csv`; it does not require refetching the raw
Visium data.
