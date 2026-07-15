# Cross-Platform Spatial-Aware Landscape — 8 Datasets × 15 Configurations
### Domain detection across Visium, MERFISH, Slide-seqV2, and Xenium (HistoWeave)

---

## 1. Objective

Test whether HistoWeave's spatial-aware domain-detection sweep generalises **across
sequencing platforms**, and whether the recommendation engine transfers knowledge between
technologies with very different resolution, gene panels, and ground-truth types. This
track reuses the 5×15 design — 5 core clusterers × 3 spatial weights = 15 configurations —
but scores it on **8 real datasets spanning 4 platforms**.

> **Naming note.** The track was scoped as "7×15". In practice we secured **8 genuine
> real-data datasets** (5 Visium DLPFC + MERFISH + Slide-seqV2 + a full-resolution Xenium
> breast-cancer sample downloaded and processed from the 10x bundle), so the delivered
> landscape is **8×15**. More real data is strictly better for a cross-platform benchmark;
> nothing was simulated.

---

## 2. Data

| Dataset | Platform | Cells | Genes (HVG) | Domains | Ground-truth kind |
|---------|----------|------:|------------:|--------:|-------------------|
| 151673 | Visium | 3611 | 2000 | 7 | expert manual cortical layers |
| 151674 | Visium | 3635 | 2000 | 7 | expert manual cortical layers |
| 151507 | Visium | 4221 | 2000 | 7 | expert manual cortical layers |
| 151669 | Visium | 3645 | 2000 | 8 | expert manual cortical layers |
| 151670 | Visium | 3484 | 2000 | 5 | expert manual cortical layers |
| merfish | MERFISH | 6000 | 159 | 15 | proxy: `Cell_class` |
| slideseqv2 | Slide-seqV2 | 6000 | 2000 | 14 | proxy: `cluster` |
| xenium | Xenium | 5942 | 312 | 14 | proxy: Leiden cluster |

**Ground-truth caveat (central to interpretation).** Only the Visium DLPFC slices have
*true spatial-domain* annotation (expert cortical layers). The three imaging/bead platforms
have no gold-standard spatial domains, so we use **proxy labels**: published cell-type
annotations (MERFISH `Cell_class`, Slide-seqV2 `cluster`) or, for Xenium, a Leiden
transcriptomic clustering. ARI on those platforms therefore measures **recovery of
cell-type / transcriptomic structure**, not recovery of spatial tissue domains. This
distinction drives the headline result below.

**Preparation.** Cross-platform datasets were subsampled to ≤6,000 cells (fixed seed 0).
This was required because HistoWeave's spatial neighbourhood (`_math.knn_indices`) is a
brute-force O(n²) distance computation — at 50k cells it OOM'd (~20 GB); 6,000 cells matches
the ~3.6k Visium scale and keeps an identical code path. Counts were reconstructed per
platform (integer detection vs `expm1` of log-normalised values), QC-filtered, and reduced
to ≤2000 HVGs. Scripts: `prep_merfish.py`, `prep_slideseqv2.py`, `prep_xenium.py`,
`_prep_common.py`.

---

## 3. Methods & Protocol

Identical machinery to 5×15. **15 configs** = {kmeans, gaussian_mixture, agglomerative,
spectral, birch} × spatial_weight {0.0, 0.3, 0.8}, keyed `<method>@sw<w>`. Each method
receives the dataset's domain count as `n_domains`, plus `random_state` and `spatial_weight`.
Harness run once per weight and columns relabelled/merged into an 8×15 matrix.

**Protocol**: `histoweave.landscape.cross_platform_spatial_aware.v1`. Metric: **ARI** vs the
(true or proxy) labels. **3 seeds** (42, 1, 2). 8 × 15 × 3 = **360 fits**.

---

## 4. Results

### 4.1 Headline: the *optimal* spatial weight depends on what ground truth represents

Mean ARI by platform × spatial weight (averaged over methods × seeds):

| Platform | sw0.0 (expr only) | sw0.3 | sw0.8 (strongly spatial) | Best |
|----------|------------------:|------:|-------------------------:|------|
| **Visium** (true spatial domains) | 0.114 | 0.209 | **0.235** | sw0.8 |
| **MERFISH** (proxy cell-type) | **0.378** | 0.307 | 0.042 | sw0.0 |
| **Slide-seqV2** (proxy cluster) | 0.050 | **0.075** | 0.067 | sw0.3 |
| **Xenium** (proxy Leiden) | **0.494** | 0.458 | 0.242 | sw0.0 |

**The key cross-platform finding: spatial weighting helps only when the ground truth is
itself spatial.** On Visium, where labels are contiguous cortical layers, more spatial
weight monotonically *improves* ARI (0.114 → 0.235). On MERFISH and Xenium, where labels are
non-spatial cell types, expression-only wins and heavy spatial smoothing is actively harmful
(MERFISH 0.378 → 0.042; Xenium 0.494 → 0.242) — smoothing over neighbours blurs the
transcriptomic identity that defines those proxy labels. Slide-seqV2 is low and flat (bead
sparsity + weak transcriptomic proxy).

This is a *scientifically correct* outcome, not a bug: it shows the benchmark is sensitive to
the nature of the ground truth, and it warns that a spatial prior is not universally
beneficial — it must match the biological question.

### 4.2 Best configuration per dataset (seed-mean)

| Dataset | Best config | ARI |
|---------|-------------|----:|
| 151673 (Visium) | `spectral@sw0.8` | 0.302 |
| 151674 (Visium) | `agglomerative@sw0.8` | 0.300 |
| 151507 (Visium) | `gaussian_mixture@sw0.8` | 0.263 |
| 151669 (Visium) | `agglomerative@sw0.8` | 0.190 |
| 151670 (Visium) | `spectral@sw0.3` | 0.346 |
| merfish (MERFISH) | `spectral@sw0.0` | 0.553 |
| slideseqv2 (Slide-seqV2) | `agglomerative@sw0.3` | 0.094 |
| xenium (Xenium) | `agglomerative@sw0.0` | 0.521 |

The winning config's spatial weight cleanly splits by ground-truth type: **sw0.8/0.3 for
Visium, sw0.0 for the imaging platforms.** No single base method dominates across platforms.

---

## 5. Recommendation-Engine Validation (LOOCV)

Each dataset held out; `MethodRecommender` (kNN over feature vectors, k=min(2, n−1)) trained
on the other 7, then ranked the 15 configs for the held-out dataset.

| Metric | Value |
|--------|------:|
| n queries | 8 |
| top-1 accuracy | 0.00 |
| top-3 accuracy | 0.125 |
| mean selection regret | 0.113 |
| global-best-baseline regret | 0.078 |
| random-choice regret | 0.117 |
| regret reduction vs random | 3.1% |

**Interpretation.** Cross-platform transfer is *hard*, and the metrics say so honestly. With
platforms this heterogeneous (and opposite optimal spatial weights across ground-truth
types), a kNN recommender trained on 7 neighbours barely beats random (0.113 vs 0.117) and
does not beat the global-best default (0.078). The reason is structural: the best config for
a held-out Visium slice is a high-spatial-weight variant, but its nearest feature-space
neighbours may be imaging datasets whose best config is expression-only — so kNN transfers
the wrong prior. **This is the most important lesson of the track**: naive feature-similarity
recommendation fails when the *right* method depends on ground-truth semantics that aren't
captured in the dataset feature vector (cell count, gene count, sparsity, etc.). A
platform-/label-type-aware feature would likely be needed to recover transfer performance.

---

## 6. Figures

In `7x15_cross_platform/figures/` (SVG + PNG):
- **heatmap_7x15** — 8 datasets × 15 configs, mean ARI, platform in row labels.
- **spatial_weight_by_platform_7x15** — mean ARI vs spatial_weight, one line per platform
  (the headline divergence: Visium rises, MERFISH/Xenium fall).

---

## 7. Limitations

1. **Proxy ground truth on 3/4 platforms.** MERFISH/Slide-seqV2/Xenium ARI measures
   cell-type/transcriptomic recovery, not spatial-domain recovery. Cross-platform ARI values
   are **not** directly comparable in absolute terms across ground-truth types.
2. **Subsampled to 6,000 cells** (brute-force KNN memory limit) — not full-resolution;
   imaging platforms in particular have far more cells natively.
3. **Spatial-aware = parameter sweep**, not dedicated spatial algorithms. BANKSY (the only
   dedicated spatial-aware method in the registry) is container/R-only and was not runnable
   here (no Docker).
4. **8 LOOCV queries across 4 platforms** → recommendation metrics are indicative; the poor
   transfer is itself a finding, not merely noise.
5. Xenium proxy labels are a Leiden clustering we computed, so its "ground truth" is
   circular w.r.t. transcriptomic clustering methods — treat its high ARI with caution.

---

## 8. Artifacts

In `7x15_cross_platform/`: `benchmark_long.csv`, `performance_matrix_mean.csv`,
`performance_matrix_std.csv`, `timings_mean.csv`, `dataset_features.csv`, `landscape.json`,
`recommendation_loocv.csv`/`.json`, `figure_data.json`, `manifest.json`, and `figures/`.
Scripts: `_prep_common.py`, `prep_merfish.py`, `prep_slideseqv2.py`, `prep_xenium.py`,
`experiment_7x15.py`, `make_figures_7x15.py`.
