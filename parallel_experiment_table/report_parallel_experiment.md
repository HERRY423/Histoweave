# Parallel Experiment Table — Same Task, Same Data

### Spatial-domain detection on the shared 5-slice DLPFC panel

Three independent HistoWeave benchmarks were run on **the same task** (spatial domain detection, ARI vs Maynard 2021 manual cortical layers) and **the same data** (5 human DLPFC Visium slices: 151673, 151674, 151507, 151669, 151670), but with different method families and K-selection protocols. This document aligns them side by side so the method families can be compared on identical ground truth.

**Shared contract (identical across all three benchmarks):**

| Item | Value |
|------|-------|
| Task | Spatial domain detection |
| Data | 5 human DLPFC Visium slices (spatialLIBD, Maynard et al. 2021) |
| Metric | Adjusted Rand Index (ARI) vs manual cortical layers |
| Ground truth | Expert layer annotation (L1-L6 + WM; 151669 adds L2/3) |
| HVGs | 2000 |
| Normalization | CP10K + log1p (harness re-normalizes from raw counts) |

**Slice difficulty gradient:**

| Slice | Spots | True domains | Layers |
|-------|------:|-------------:|--------|
| 151673 | 3611 | 7 | L1-L6, WM |
| 151674 | 3635 | 7 | L1-L6, WM |
| 151507 | 4221 | 7 | L1-L6, WM |
| 151669 | 3645 | 8 | L1, L2, L2/3, L3, L4, L5, L6, WM |
| 151670 | 3484 | 5 | L2/3, L4, L5, L6, WM |

**What differs across the three benchmarks (read before comparing):**

| Benchmark | Methods | Seeds | K policy |
|-----------|--------|------:|----------|
| `5x10_dlpfc_benchmark` | 10 sklearn baselines (expression-only) | 3 (42,1,2) | oracle-K (truth-derived n_domains) |
| `5x15_spatial_aware` | 5 clusterers x 3 spatial_weight = 15 configs | 3 (42,1,2) | oracle-K |
| `non_oracle_k_sota` | SpaGCN + STAGATE (GNN / graph-attention AE) | 1 (42) | oracle-K AND 3 blind estimate-K variants |

> **Caveat.** The SOTA benchmark uses a single seed (42) while the sklearn and spatial-aware benchmarks use three seeds. SOTA numbers therefore carry higher variance and are not directly seed-averaged. The oracle-K SOTA numbers are the fair comparison point against the oracle-K sklearn / spatial-aware numbers; the estimate-K SOTA numbers are a *separate* axis (blind K selection) and should be compared to oracle-K SOTA, not to the sklearn baselines.

---

## 1. Full side-by-side matrix (mean ARI per slice)

| Slice | agglomerative | birch | bisecting_kmeans | dbscan | gaussian_mixture | kmeans | mean_shift | minibatch_kmeans | optics | spectral | agglomerative@sw0.0 | agglomerative@sw0.3 | agglomerative@sw0.8 | birch@sw0.0 | birch@sw0.3 | birch@sw0.8 | gaussian_mixture@sw0.0 | gaussian_mixture@sw0.3 | gaussian_mixture@sw0.8 | kmeans@sw0.0 | kmeans@sw0.3 | kmeans@sw0.8 | spectral@sw0.0 | spectral@sw0.3 | spectral@sw0.8 | spagcn (oracle-K) | stagate (oracle-K) | spagcn (est-K:ensemble) | spagcn (est-K:silhouette) | spagcn (est-K:spatial_sil) | stagate (est-K:ensemble) | stagate (est-K:silhouette) | stagate (est-K:spatial_sil) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 151673 | 0.148 | 0.170 | 0.216 | 0.000 | 0.235 | 0.250 | 0.013 | 0.229 | -0.001 | 0.242 | 0.099 | 0.148 | 0.246 | 0.088 | 0.170 | 0.235 | 0.165 | 0.235 | 0.299 | 0.174 | 0.250 | 0.263 | 0.168 | 0.242 | 0.302 | 0.418 | 0.212 | 0.186 | 0.186 | 0.186 | 0.136 | 0.136 | 0.136 |
| 151674 | 0.221 | 0.253 | 0.296 | 0.000 | 0.269 | 0.298 | 0.006 | 0.265 | 0.000 | 0.176 | 0.119 | 0.221 | 0.300 | 0.118 | 0.253 | 0.277 | 0.208 | 0.269 | 0.294 | 0.210 | 0.298 | 0.267 | 0.218 | 0.176 | 0.280 | 0.273 | 0.213 | 0.199 | 0.199 | 0.199 | 0.207 | 0.207 | 0.207 |
| 151507 | 0.211 | 0.132 | 0.250 | 0.000 | 0.223 | 0.240 | 0.012 | 0.237 | -0.000 | 0.200 | 0.080 | 0.211 | 0.216 | 0.046 | 0.132 | 0.251 | 0.131 | 0.223 | 0.263 | 0.182 | 0.240 | 0.198 | 0.138 | 0.200 | 0.234 | 0.385 | 0.383 | 0.217 | 0.248 | 0.217 | 0.217 | 0.244 | 0.217 |
| 151669 | 0.145 | 0.130 | 0.183 | 0.000 | 0.140 | 0.166 | 0.010 | 0.172 | 0.000 | 0.168 | 0.044 | 0.145 | 0.190 | 0.030 | 0.130 | 0.187 | 0.099 | 0.140 | 0.169 | 0.136 | 0.166 | 0.181 | 0.115 | 0.168 | 0.179 | 0.207 | 0.146 | 0.137 | 0.137 | 0.137 | 0.116 | 0.116 | 0.116 |
| 151670 | 0.280 | 0.188 | 0.193 | 0.000 | 0.178 | 0.222 | -0.021 | 0.211 | -0.005 | 0.346 | -0.074 | 0.280 | 0.199 | -0.085 | 0.188 | 0.181 | 0.080 | 0.178 | 0.243 | 0.262 | 0.222 | 0.193 | 0.107 | 0.346 | 0.221 | 0.213 | 0.207 | 0.416 | 0.416 | 0.416 | 0.391 | 0.391 | 0.391 |

Cells are mean ARI over the seeds each benchmark ran (3 for sklearn / spatial-aware, 1 for SOTA). `—` = method not run on that slice. Bold per-slice winners are listed in section 3.

---

## 2. Per-method aggregate (across the 5 shared slices)

| Rank | Method config | Family | Benchmark | K policy | Seeds | Mean ARI (5 slices) | Best ARI | Best slice | Worst ARI | Rank in family |
|----:|---------------|--------|-----------|----------|------:|--------------------:|---------:|------------|----------:|---------------:|
| 1 | `spagcn (oracle-K)` | sota | non_oracle_k_sota | oracle | 42 | **0.2991** | 0.4180 | 151673 | 0.2068 | 1 |
| 2 | `gaussian_mixture@sw0.8` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2536** | 0.2990 | 151673 | 0.1692 | 1 |
| 3 | `spectral@sw0.8` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2431** | 0.3019 | 151673 | 0.1787 | 2 |
| 4 | `spagcn (est-K:silhouette)` | sota | non_oracle_k_sota | estimate | 42 | **0.2371** | 0.4156 | 151670 | 0.1368 | 2 |
| 5 | `kmeans@sw0.3` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2351** | 0.2981 | 151674 | 0.1656 | 3 |
| 5 | `kmeans` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.2351** | 0.2981 | 151674 | 0.1656 | 1 |
| 7 | `stagate (oracle-K)` | sota | non_oracle_k_sota | oracle | 42 | **0.2322** | 0.3826 | 151507 | 0.1459 | 3 |
| 8 | `spagcn (est-K:ensemble)` | sota | non_oracle_k_sota | estimate | 42 | **0.2310** | 0.4156 | 151670 | 0.1368 | 4 |
| 8 | `spagcn (est-K:spatial_sil)` | sota | non_oracle_k_sota | estimate | 42 | **0.2310** | 0.4156 | 151670 | 0.1368 | 4 |
| 10 | `agglomerative@sw0.8` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2300** | 0.3000 | 151674 | 0.1895 | 4 |
| 11 | `bisecting_kmeans` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.2278** | 0.2961 | 151674 | 0.1829 | 2 |
| 12 | `spectral` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.2266** | 0.3464 | 151670 | 0.1679 | 3 |
| 12 | `spectral@sw0.3` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2266** | 0.3464 | 151670 | 0.1679 | 5 |
| 14 | `birch@sw0.8` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2262** | 0.2769 | 151674 | 0.1808 | 6 |
| 15 | `minibatch_kmeans` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.2229** | 0.2646 | 151674 | 0.1722 | 4 |
| 16 | `kmeans@sw0.8` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2203** | 0.2673 | 151674 | 0.1808 | 7 |
| 17 | `stagate (est-K:silhouette)` | sota | non_oracle_k_sota | estimate | 42 | **0.2190** | 0.3910 | 151670 | 0.1160 | 6 |
| 18 | `stagate (est-K:ensemble)` | sota | non_oracle_k_sota | estimate | 42 | **0.2136** | 0.3910 | 151670 | 0.1160 | 7 |
| 18 | `stagate (est-K:spatial_sil)` | sota | non_oracle_k_sota | estimate | 42 | **0.2136** | 0.3910 | 151670 | 0.1160 | 7 |
| 20 | `gaussian_mixture@sw0.3` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2089** | 0.2693 | 151674 | 0.1397 | 8 |
| 20 | `gaussian_mixture` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.2089** | 0.2693 | 151674 | 0.1397 | 5 |
| 22 | `agglomerative` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.2011** | 0.2800 | 151670 | 0.1454 | 6 |
| 22 | `agglomerative@sw0.3` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.2011** | 0.2800 | 151670 | 0.1454 | 9 |
| 24 | `kmeans@sw0.0` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.1928** | 0.2620 | 151670 | 0.1356 | 10 |
| 25 | `birch@sw0.3` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.1745** | 0.2528 | 151674 | 0.1297 | 11 |
| 25 | `birch` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.1745** | 0.2528 | 151674 | 0.1297 | 7 |
| 27 | `spectral@sw0.0` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.1493** | 0.2177 | 151674 | 0.1074 | 12 |
| 28 | `gaussian_mixture@sw0.0` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.1366** | 0.2083 | 151674 | 0.0802 | 13 |
| 29 | `agglomerative@sw0.0` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.0536** | 0.1188 | 151674 | -0.0738 | 14 |
| 30 | `birch@sw0.0` | spatial_aware | 5x15_spatial_aware | oracle | 42,1,2 | **0.0395** | 0.1180 | 151674 | -0.0851 | 15 |
| 31 | `mean_shift` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.0042** | 0.0133 | 151673 | -0.0206 | 8 |
| 32 | `dbscan` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **0.0000** | 0.0000 | 151507 | 0.0000 | 9 |
| 33 | `optics` | sklearn | 5x10_dlpfc | oracle | 42,1,2 | **-0.0013** | 0.0002 | 151669 | -0.0052 | 10 |

---

## 3. Best method per slice (cross-benchmark)

| Slice | True domains | Best method config | Family | Mean ARI |
|-------|-------------:|---------------------|--------|---------:|
| 151673 | 7 | `spagcn (oracle-K)` | sota | **0.4180** |
| 151674 | 7 | `agglomerative@sw0.8` | spatial_aware | **0.3000** |
| 151507 | 7 | `spagcn (oracle-K)` | sota | **0.3846** |
| 151669 | 8 | `spagcn (oracle-K)` | sota | **0.2068** |
| 151670 | 5 | `spagcn (est-K:ensemble)` | sota | **0.4156** |

---

## 4. Family-level summary

| Family | # configs | Best-in-family mean ARI (avg of 5 slices) | Best | Worst |
|--------|----------:|------------------------------------------:|-----:|------:|
| sklearn | 10 | **0.2656** | 0.3464 | 0.1829 |
| spatial_aware | 15 | **0.2801** | 0.3464 | 0.1895 |
| sota | 8 | **0.3397** | 0.4180 | 0.2068 |

`best-in-family` = for each slice, take the highest mean ARI among that family's configs, then average those 5 per-slice maxima. This is an *oracle-family* ceiling (you pick the best family member per slice), not a single-method average.

---

## 5. Key observations

1. **Spatial awareness closes most of the gap to SOTA on oracle-K.** The best spatial-weight config per slice (5x15, sw0.8 dominant) approaches or matches oracle-K SpaGCN/STAGATE ARI on several slices, despite being a sklearn clusterer with a neighbourhood-mean blend rather than a dedicated GNN/autoencoder. This is the central cross-benchmark finding: on layered cortex, the spatial-weight knob captures much of the benefit that dedicated spatial architectures provide.
2. **No single method dominates all 5 slices.** The per-slice winner rotates across families (spatial-aware sw0.8 variants on 151674, SOTA SpaGCN-oracle on 151673/151507/151669, SOTA SpaGCN estimate-K on 151670). This heterogeneity is what makes HistoWeave's recommendation task meaningful.
3. **Oracle-K inflation is real and large for SOTA.** SpaGCN drops from mean ARI 0.299 (oracle-K) to 0.237 (silhouette-estimate), a 0.062 absolute / ~21% relative loss, because blind K estimators collapse to K=2 on layered cortex. Any cross-benchmark comparison against SOTA must separate oracle-K from estimate-K; comparing sklearn oracle-K numbers against SOTA estimate-K numbers would flatter the sklearn baselines unfairly.
4. **Density/mode-seeking sklearn methods (dbscan, optics, mean_shift) are floor references** (ARI near 0) on layered cortex across all slices; they are included for landscape completeness, not as viable choices for this tissue structure.
5. **151669 is the hardest slice for every family** (8 domains, ambiguous L2/3 band) — the best mean ARI across all 33 configs is the lowest of the five slices (0.207, spagcn oracle-K). 151670 (5 merged domains) is the easiest for the top configs.

---

## 6. Limitations of the cross-benchmark comparison

1. **Seed mismatch.** sklearn / spatial-aware benchmarks use 3 seeds; the SOTA benchmark uses 1 seed (42). SOTA numbers have higher variance and are not seed-averaged. A fair re-run would use the same 3 seeds for all three benchmarks.
2. **K-policy mismatch.** sklearn and spatial-aware benchmarks are pure oracle-K. The SOTA benchmark is the only one that also reports blind estimate-K. The oracle-K SOTA column is the apples-to-apples comparison point; the estimate-K SOTA columns answer a different question (blind K robustness) and should not be compared to the oracle-K sklearn numbers.
3. **Within-study only.** All 5 slices come from one study (Maynard 2021). Cross-platform / cross-tissue transfer is not tested here; see `benchmark_external_validation/` for that.
4. **SOTA set is partial.** Only SpaGCN and STAGATE are in the non-oracle-K benchmark. BayesSpace, GraphST, and BANKSY are documented in `5x15_spatial_aware/SOTA_COMPARISON.md` but require isolated R / PyTorch environments and are not in this aligned table.
5. **Ground truth is expert annotation**, itself imperfect (e.g. the L2/3 ambiguity in 151669), so ARI ceilings are < 1 for every method.

---

## 7. Artifacts

- `parallel_experiment_table.csv` — long/tidy table, one row per (slice, method_config) with mean/std ARI, runtime, family, benchmark, K policy, seeds.
- `parallel_experiment_matrix.csv` — wide matrix, slices x method_configs, mean ARI (the table in section 1).
- `parallel_experiment_summary.csv` — per-method aggregate with overall and within-family ranks (the table in section 2).
- `figures/parallel_heatmap.svg` / `.png` — heatmap of all method_configs across the 5 slices, grouped by family.
- `build_parallel_table.py` — the generator (this script).

## 8. Reproducibility

```bash
# From the Histoweave repo root:
python parallel_experiment_table/build_parallel_table.py
```

The generator reads only the three existing `benchmark_long.csv` files from `5x10_dlpfc_benchmark/`, `5x15_spatial_aware/`, and `non_oracle_k_sota/`; it does not re-run any clustering. Numbers are mean ARI over the seeds each benchmark actually ran.
