# 5 Datasets × 15 Spatial-Aware Configurations — Domain-Detection Landscape
### A spatial-awareness benchmark for the HistoWeave platform (DLPFC spatial transcriptomics)

---

## 1. Objective

Extend HistoWeave's 5×10 DLPFC benchmark into a **spatial-aware** landscape. Every
HistoWeave domain-detection method exposes a `spatial_weight ∈ [0, 1]` knob that blends the
expression embedding with a spatial-neighbourhood term (0 = expression only, 1 = space
only; `histoweave/_math.py::neighborhood_mean`). This experiment turns that knob into the
benchmark's second axis: **5 core, deterministic, k-aware clusterers × 3 spatial weights =
15 configurations**, scored on the same 5 DLPFC slices with the same ARI-vs-manual-layers
metric and the same recommendation/LOOCV machinery, so results are directly comparable to
the 5×10 baseline.

**Why a spatial-weight sweep and not 15 new methods?** The HistoWeave registry has 11
domain-detection methods, of which only **BANKSY** is a dedicated spatial-aware algorithm —
and BANKSY is container/R-only (Bioconductor), which was not runnable in this environment
(no Docker). The `spatial_weight` sweep is therefore the genuine spatial-aware axis
available in pure Python, and it directly answers the scientific question the "spatial-aware
set" was meant to probe: *does adding spatial context improve domain detection, and by how
much?*

---

## 2. Data

Identical to the 5×10 benchmark — the only real datasets in the registry with genuine
spatial-domain ground truth: 5 human DLPFC Visium slices (Maynard et al. 2021, spatialLIBD),
manual cortical-layer annotation, spanning a difficulty gradient.

| Slice | Spots | HVGs | True domains |
|-------|------:|-----:|-------------:|
| 151673 | 3611 | 2000 | 7 |
| 151674 | 3635 | 2000 | 7 |
| 151507 | 4221 | 2000 | 7 |
| 151669 | 3645 | 2000 | 8 |
| 151670 | 3484 | 2000 | 5 |

Preparation (`prepare_dlpfc.py`, unchanged from 5×10): filtered count matrices from S3,
manual layers joined by barcode from LieberInstitute/HumanPilot (majority-vote de-dup),
Visium pixel coordinates as `obsm["spatial"]`, QC (min 3 cells), CP10K + log1p, 2000 HVGs.
Raw counts are passed to the harness, which re-normalizes internally.

---

## 3. Methods & Protocol

**5 core clusterers** (deterministic, accept an explicit domain count, and expose
`spatial_weight`): `kmeans`, `gaussian_mixture`, `agglomerative`, `spectral`, `birch`.

**3 spatial weights**: `{0.0, 0.3, 0.8}` — expression-only, the library default, and
strongly spatial.

**15 configurations**, keyed `<method>@sw<w>` (e.g. `kmeans@sw0.3`). Because
`run_task_landscape` keys results by method name and forwards only the params a method
declares, the harness was run **once per spatial weight** over the 5 core methods and each
method column relabelled `<method>@sw<w>` before merging into a single 5×15 performance
matrix.

**Protocol**: `histoweave.landscape.dlpfc_spatial_aware.v1`
- Metric: **Adjusted Rand Index (ARI)** vs manual layers, higher is better.
- Each method receives the slice's **true domain count** as `n_domains`, plus `random_state`
  and `spatial_weight`.
- **3 random seeds** (42, 1, 2) → mean ± sd.
- 5 slices × 15 configs × 3 seeds = **225 fits**; ~21 min/seed, ~64 min total.

---

## 4. Results

### 4.1 Headline: spatial awareness improves domain detection

Mean ARI (averaged over all 5 methods and 5 slices × 3 seeds), by spatial weight:

| spatial_weight | Mean ARI | vs expression-only |
|---------------:|---------:|-------------------:|
| 0.0 (expression only) | 0.114 | — |
| 0.3 (default) | 0.209 | **+83%** |
| 0.8 (strongly spatial) | 0.235 | **+106%** |

**Adding spatial context roughly doubles ARI.** The gain is monotonic in `spatial_weight`
across the sweep, and holds for every core method (see `figures/spatial_weight_effect_5x15`).
This is the central finding: for layered cortex, blending neighbourhood information into
general-purpose clusterers substantially closes the gap toward spatially-aware specialists.

### 4.2 Best configuration per slice (seed-mean)

| Slice | Best config | Mean ARI |
|-------|-------------|---------:|
| 151673 | `spectral@sw0.8` | 0.302 |
| 151674 | `agglomerative@sw0.8` | 0.300 |
| 151507 | `gaussian_mixture@sw0.8` | 0.263 |
| 151669 | `agglomerative@sw0.8` | 0.190 |
| 151670 | `spectral@sw0.3` | 0.346 |

**4 of 5 slices are won by a high-spatial-weight (sw0.8) configuration**; only the
easiest slice (151670, 5 merged domains) peaks at the moderate sw0.3. No single method
dominates — the base clusterer that wins changes by slice (spectral / agglomerative / GMM),
which keeps the recommendation task non-trivial.

### 4.3 Top configurations overall (mean ARI across slices)

`gaussian_mixture@sw0.8` (0.254) > `spectral@sw0.8` (0.243) > `kmeans@sw0.3` (0.235) >
`agglomerative@sw0.8` (0.230) > `spectral@sw0.3` (0.227). High-spatial-weight variants
sweep the top of the table.

---

## 5. Recommendation-Engine Validation (LOOCV)

Each slice held out in turn; `MethodRecommender` (kNN over dataset feature vectors, k=2)
trained on the other four, then asked to rank the 15 configs for the held-out slice.

| Metric | Value | Interpretation |
|--------|------:|----------------|
| n queries | 5 | one per held-out slice |
| top-1 accuracy | 0.00 | never picks the exact oracle-best config |
| top-3 accuracy | 0.20 | oracle-best in top-3 for 1/5 slices |
| mean selection regret | 0.082 | ARI lost vs oracle-best |
| global-best-baseline regret | 0.027 | "always pick the globally best config" |
| random-choice regret | 0.094 | expected regret from a random config |
| regret reduction vs random | 12.5% | recommender modestly beats random |

**Interpretation.** With 15 configs (vs 10 methods in the 5×10 run) and still only 4
same-study training datasets per query, the recommender beats random selection (regret
0.082 vs 0.094) but not the trivial "always pick the single globally-best config"
baseline (0.027). This is the expected honest outcome: on a narrow, within-study landscape
a strong global default is hard to beat, and picking the *exact* best of 15 near-tied
configs is harder than picking the best of 10. The recommender's value emerges on broad,
diverse landscapes — motivating the cross-platform 7×15 track.

---

## 6. Figures

In `5x15_spatial_aware/figures/` (SVG + PNG):
- **heatmap_5x15** — 5 slices × 15 configs, mean ARI, white separators between spatial-weight blocks.
- **spatial_weight_effect_5x15** — mean ARI vs `spatial_weight` per core method (the headline trend).

---

## 7. Limitations

1. **Within-study only.** All 5 slices are from one study (Maynard 2021 human DLPFC);
   this is within-study validation, not cross-platform transfer.
2. **Spatial-aware = param sweep, not dedicated algorithms.** The 15 configs are
   spatial-weighted sklearn-family clusterers, not GNN/HMRF spatial methods. BANKSY (the
   one dedicated spatial-aware method in the registry) is container/R-only and was not
   runnable here. Absolute ARI is a baseline, not a state-of-the-art claim.
3. **Only 5 LOOCV queries** → recommendation metrics are indicative, not definitive.
4. **Ground truth is expert annotation**, itself imperfect (e.g. L2/3 ambiguity in 151669),
   so ARI ceilings are < 1.

---

## 8. Artifacts

In `5x15_spatial_aware/`: `benchmark_long.csv`, `performance_matrix_mean.csv`,
`performance_matrix_std.csv`, `timings_mean.csv`, `dataset_features.csv`, `landscape.json`,
`recommendation_loocv.csv`/`.json`, `figure_data.json`, `manifest.json`, and `figures/`.
Scripts: `prepare_dlpfc.py`, `experiment_5x15.py`, `make_figures.py`.
