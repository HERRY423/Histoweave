# 5 Datasets × 10 Methods Spatial-Domain Performance Landscape
### A real-data benchmark for the HistoWeave platform (DLPFC spatial transcriptomics)

---

## 1. Objective

Design and run a **5-dataset × 10-method performance-landscape experiment** for spatial
domain detection, as a *real-data upgrade* of HistoWeave's existing `figure3` protocol
(10 methods × 3 **synthetic** datasets, ARI metric). The experiment reuses the repo's
own benchmark harness (`run_task_landscape`), recommendation engine (`MethodRecommender`,
LOOCV), and ARI scoring, so results are directly comparable to the platform's design.

Per the agreed design choices, the experiment uses **real data**, arranges datasets as a
**difficulty gradient**, and evaluates each dataset at **its own true domain count**.

---

## 2. Data

The only real datasets in the HistoWeave registry that carry genuine **spatial-domain**
ground truth (`domain_truth`) are the human DLPFC Visium slices from Maynard et al. 2021
(spatialLIBD), annotated with manual cortical layers (Layer 1–6 + white matter). Five
slices were selected to span a difficulty gradient in true domain count and annotation
structure.

| Slice | Spots | HVGs | True domains | Annotated | Layers |
|-------|------:|-----:|-------------:|----------:|--------|
| 151673 | 3611 | 2000 | 7 | 99.2% | L1–L6, WM |
| 151674 | 3635 | 2000 | 7 | 99.0% | L1–L6, WM |
| 151507 | 4221 | 2000 | 7 | 99.9% | L1–L6, WM |
| 151669 | 3645 | 2000 | 8 | 99.6% | L1, L2, **L2/3**, L3, L4, L5, L6, WM |
| 151670 | 3484 | 2000 | 5 | 99.6% | **L2/3**, L4, L5, L6, WM (upper layers merged/missing) |

**Preparation** (`prepare_dlpfc.py`): filtered count matrices were downloaded directly
(the registry `.h5` files carry counts only, with placeholder checksums that always fail,
and no layer labels or coordinates). Manual layer annotations were joined by barcode from
the LieberInstitute/HumanPilot `Layer_Guesses` CSVs (majority-vote de-duplication where a
barcode appeared more than once), and Visium pixel coordinates from
`tissue_positions_list.txt` were attached as `obsm["spatial"]`. Spots without a layer label
were dropped. Standard QC followed: gene filtering (min 3 cells), CP10K normalization,
log1p, and 2000 highly-variable genes. Raw counts are passed to the harness, which
re-normalizes internally (avoids double normalization).

---

## 3. Methods & Protocol

**10 clustering methods** (identical to HistoWeave `figure3`, all scikit-learn):
`agglomerative, birch, bisecting_kmeans, dbscan, gaussian_mixture, kmeans, mean_shift,
minibatch_kmeans, optics, spectral`.

- 7 partitional/hierarchical methods receive the slice's **true domain count** as `n_domains`.
- 3 density/mode-seeking methods (`dbscan`, `optics`, `mean_shift`) auto-determine cluster count.
- All methods receive `random_state`.

**Protocol**: `histoweave.landscape.dlpfc_real.v1`
- Metric: **Adjusted Rand Index (ARI)** vs manual layers, higher is better.
- **3 random seeds** (42, 1, 2) — a robustness improvement over `figure3`'s single seed.
- Harness: `run_task_landscape(..., category=DOMAIN_DETECTION, extra_params_factory=...)`.
- Runtime: ~247 s/seed, ~12–13 min total.

---

## 4. Results

### 4.1 Performance matrix (mean ARI over 3 seeds)

| Slice | agglo | birch | bisect_km | dbscan | gmm | **kmeans** | mean_shift | minibatch_km | optics | spectral |
|-------|------:|------:|----------:|-------:|----:|-------:|-----------:|-------------:|-------:|---------:|
| 151673 | 0.148 | 0.170 | 0.216 | 0.000 | 0.235 | **0.250** | 0.013 | 0.229 | −0.001 | 0.242 |
| 151674 | 0.221 | 0.253 | 0.296 | 0.000 | 0.269 | **0.298** | 0.006 | 0.265 | 0.000 | 0.176 |
| 151507 | 0.211 | 0.132 | 0.250 | 0.000 | 0.223 | 0.240 | 0.012 | 0.237 | 0.000 | 0.200 |
| 151669 | 0.145 | 0.130 | **0.183** | 0.000 | 0.140 | 0.166 | 0.010 | 0.172 | 0.000 | 0.168 |
| 151670 | 0.280 | 0.188 | 0.193 | 0.000 | 0.178 | 0.221 | −0.021 | 0.211 | −0.005 | **0.346** |

(Bold = best method per slice by seed-mean.)

**Best method per slice (seed-mean):** 151673 → kmeans (0.250), 151674 → kmeans (0.298),
151507 → bisecting_kmeans (0.250), 151669 → bisecting_kmeans (0.183), 151670 → spectral (0.346).

### 4.2 Key observations

1. **Partitional/hierarchical methods land at ARI ≈ 0.15–0.35.** This is realistic for
   *general-purpose* clusterers scored against expert cortical layers: they capture
   substantial layer structure but none is designed to exploit spatial adjacency, so they
   plateau well below what spatially-aware domain methods achieve on DLPFC.
2. **Density/mode-seeking methods collapse to ARI ≈ 0** (dbscan exactly 0, optics and
   mean_shift near/below 0). Expected — cortical layers are elongated, contiguous bands,
   not density-separated blobs, so DBSCAN/OPTICS assign nearly everything to one cluster or
   noise, and mean_shift finds too few modes.
3. **The best method changes by slice** (kmeans / bisecting_kmeans / spectral). This
   heterogeneity is exactly what makes the *recommendation* task meaningful rather than
   trivial — there is no single method that dominates all five slices.
4. **Difficulty gradient is visible.** 151669 (8 domains, includes the ambiguous L2/3 band)
   is the hardest (best ARI only 0.183); 151670 (5 merged domains) is the easiest for the
   top method (spectral 0.346).

### 4.3 Runtime

All partitional methods run in ~3–4 s/slice. `mean_shift` is the slowest (~10–13 s),
`optics` intermediate (~5–5.6 s). Runtime is not a discriminator here — accuracy is.

---

## 5. Recommendation-Engine Validation (LOOCV)

Using HistoWeave's `MethodRecommender` (k-nearest-neighbour over dataset feature vectors),
each slice was held out in turn and the recommender trained on the other four.

| Metric | Value | Interpretation |
|--------|------:|----------------|
| n queries | 5 | one per held-out slice |
| **top-1 accuracy** | **0.00** | never picks the single oracle-best method |
| **top-3 accuracy** | **0.40** | oracle-best in top-3 for 2/5 slices (151673, 151674) |
| mean selection regret | 0.075 | ARI lost vs the oracle-best method |
| median selection regret | 0.050 | — |
| global-best-baseline regret | 0.055 | "always pick kmeans" baseline |
| random-choice regret | 0.116 | expected regret from picking a method at random |

**Interpretation.** The recommender **beats random selection** (regret 0.075 vs 0.116, a
~35% reduction) but does **not** beat the trivial "always pick kmeans" global baseline
(0.055). It cannot reliably identify the single best method (top-1 = 0), which is
unsurprising given only **4 training datasets per query**, all from the *same study* with
very similar feature profiles. This honest result matches HistoWeave's own caveats: the
recommendation engine needs a broad, diverse landscape to add value over a strong global
default.

---

## 6. Figures

All figures are in `5x10_dlpfc_benchmark/figures/` (SVG + PNG):

- **fig1_performance_heatmap** — 5 slices × 10 methods, mean ARI, annotated cells.
- **fig2_method_boxplot** — ARI distribution per method (across slices × seeds), sorted by mean.
- **fig3_landscape_embedding** — 2D dataset-feature landscape, points coloured by best method.
- **fig4_runtime** — mean runtime per method.

---

## 7. Limitations (read before reusing)

1. **Within-study only.** All 5 slices come from one study (spatialLIBD/Maynard 2021) with
   near-identical protocols. The landscape and recommender therefore reflect *within-study*
   generalization, not cross-platform/cross-tissue transfer.
2. **General clusterers, not spatial-domain specialists.** The `figure3` method set is
   scikit-learn baselines that ignore spatial coordinates. Absolute ARI values are lower
   than what spatially-aware methods (BayesSpace, SpaGCN, GraphST, etc.) reach on DLPFC;
   these numbers are a *baseline landscape*, not a claim about state of the art.
3. **Only 5 LOOCV queries.** Recommendation metrics (top-1/top-3, regret) are estimated
   from 5 points — high variance, indicative not definitive.
4. **Density methods are effectively floor references** (ARI ≈ 0); they are included for
   completeness and to anchor the low end of the landscape, not as viable choices for
   layered cortex.
5. **Ground truth is expert annotation**, itself imperfect (e.g. the L2/3 ambiguity in
   151669), so ARI ceilings are inherently < 1.

---

## 8. Artifacts

In `5x10_dlpfc_benchmark/`:
`benchmark_long.csv`, `performance_matrix_mean.csv`, `performance_matrix_std.csv`,
`timings_mean.csv`, `dataset_features.csv`, `landscape.json`,
`recommendation_loocv.csv` / `.json`, `figure_data.json`, `manifest.json`, and
`figures/` (4 figures × SVG+PNG). Prepared data + scripts: `prepare_dlpfc.py`,
`experiment_5x10_dlpfc.py`, `make_figures.py`.
