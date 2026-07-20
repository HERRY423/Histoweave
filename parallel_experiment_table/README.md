# Parallel experiment table — same task, same data

**Protocol:** `histoweave.parallel_experiment_table.v1`  
**Role:** frozen **in-repo** side-by-side matrix aligning three DLPFC domain
benchmarks on identical task, data, and ground truth (manuscript / review Table).

This folder adds a **side-by-side (parallel) experiment table** that aligns the
three HistoWeave benchmarks which were run on the **same spatial-domain-detection
task** and the **same 5-slice DLPFC panel**, so their method families can be
compared on identical ground truth.

## Why this table exists

HistoWeave has three independent benchmarks that all evaluate spatial domain
detection by ARI vs the Maynard 2021 manual cortical layers on the same five
human DLPFC Visium slices (`151673, 151674, 151507, 151669, 151670`):

| Benchmark | Methods | Seeds | K policy |
|-----------|--------|------:|----------|
| `5x10_dlpfc_benchmark` | 10 sklearn baselines (expression-only) | 3 (42,1,2) | oracle-K |
| `5x15_spatial_aware` | 5 clusterers × 3 `spatial_weight` = 15 configs | 3 (42,1,2) | oracle-K |
| `non_oracle_k_sota` | SpaGCN + STAGATE (GNN / graph-attention AE) | 1 (42) | oracle-K + 3 blind estimate-K |

Each benchmark was reported in isolation. This table puts them next to each
other so a reader can see, on identical data, how the sklearn baselines, the
spatial-weight sweep, and the SOTA backends compare — and where the oracle-K
vs estimate-K axis changes the picture.

## Files

| File | What it is |
|------|------------|
| `build_parallel_table.py` | Generator. Reads the three existing `benchmark_long.csv` files from the repo, aligns them, writes the CSVs + report. Does **not** re-run any clustering. |
| `parallel_experiment_table.csv` | Long/tidy table: one row per (slice, method_config) with mean/std ARI, runtime, family, benchmark, K policy, seeds. |
| `parallel_experiment_matrix.csv` | Wide matrix: 5 slices × 33 method configs, mean ARI. |
| `parallel_experiment_summary.csv` | Per-method aggregate: mean ARI over 5 slices, best/worst slice, overall rank, within-family rank. |
| `report_parallel_experiment.md` | Human-readable side-by-side report (the main deliverable). |
| `figures/parallel_heatmap.svg` / `.png` | Heatmap of all 33 configs across the 5 slices, grouped by family. |

## Reproduce

```bash
# From the Histoweave repo root:
python parallel_experiment_table/build_parallel_table.py
python parallel_experiment_table/make_heatmap.py   # optional, regenerates the figure
```

Requires `pandas`, `numpy`, and `matplotlib` (all already in the project's
`[dev]` / `[all]` extras).

## Headline numbers

33 method configs (10 sklearn + 15 spatial-aware + 8 SOTA) on 5 slices.

| Rank | Method config | Family | Mean ARI (5 slices) |
|----:|----------------|--------|--------------------:|
| 1 | `spagcn (oracle-K)` | sota | **0.2991** |
| 2 | `gaussian_mixture@sw0.8` | spatial_aware | **0.2536** |
| 3 | `spectral@sw0.8` | spatial_aware | **0.2431** |
| 4 | `spagcn (est-K:silhouette)` | sota | **0.2371** |
| 5 | `kmeans` / `kmeans@sw0.3` | sklearn / spatial_aware | **0.2351** |

Best-in-family ceiling (avg of per-slice best): sota 0.340 > spatial_aware 0.280 > sklearn 0.266.

## Read before comparing

The three benchmarks differ in **seeds** (3 vs 1) and **K policy** (oracle-K
everywhere, plus blind estimate-K only in the SOTA benchmark). The oracle-K
SOTA column is the fair comparison point against the oracle-K sklearn /
spatial-aware numbers; the estimate-K SOTA columns answer a different question
(blind K robustness) and should not be compared to the oracle-K sklearn
numbers. See section 6 of `report_parallel_experiment.md` for the full caveat
list.
