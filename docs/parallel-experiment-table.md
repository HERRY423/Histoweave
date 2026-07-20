# Parallel experiment table — same task, same data

**Protocol:** `histoweave.parallel_experiment_table.v1`

Side-by-side alignment of three HistoWeave spatial-domain benchmarks on the
**same 5-slice DLPFC panel** and the **same ARI-vs-manual-layers task**:

| Benchmark | Family | K policy | Seeds |
|-----------|--------|----------|------:|
| `5x10_dlpfc_benchmark` | 10 sklearn baselines | oracle-K | 3 |
| `5x15_spatial_aware` | 15 spatial-weight configs | oracle-K | 3 |
| `non_oracle_k_sota` | SpaGCN + STAGATE | oracle-K + estimate-K | 1 |

## In-repo artefacts

| File | Description |
|------|-------------|
| [`parallel_experiment_table/report_parallel_experiment.md`](../parallel_experiment_table/report_parallel_experiment.md) | Full narrative report (primary) |
| [`parallel_experiment_table/parallel_experiment_summary.csv`](../parallel_experiment_table/parallel_experiment_summary.csv) | Ranked mean ARI (33 configs) |
| [`parallel_experiment_table/parallel_experiment_table.csv`](../parallel_experiment_table/parallel_experiment_table.csv) | Long tidy table |
| [`parallel_experiment_table/parallel_experiment_matrix.csv`](../parallel_experiment_table/parallel_experiment_matrix.csv) | Wide matrix |
| [`parallel_experiment_table/figures/parallel_heatmap.svg`](../parallel_experiment_table/figures/parallel_heatmap.svg) | Heatmap |

## Headline (mean ARI over 5 slices)

| Rank | Method config | Family | Mean ARI |
|----:|---------------|--------|---------:|
| 1 | `spagcn (oracle-K)` | sota | 0.2991 |
| 2 | `gaussian_mixture@sw0.8` | spatial_aware | 0.2536 |
| 3 | `spectral@sw0.8` | spatial_aware | 0.2431 |
| 4 | `spagcn (est-K:silhouette)` | sota | 0.2371 |
| 5 | `kmeans` / `kmeans@sw0.3` | sklearn / spatial_aware | 0.2351 |

## Caveats (read before comparing)

1. **Seed mismatch** — sklearn / spatial-aware use 3 seeds; SOTA uses seed 42 only.  
2. **K-policy mismatch** — fair cross-family comparison is **oracle-K only**; estimate-K is a separate axis (blind K robustness).  
3. **Within-study only** — all five slices from Maynard 2021; see `benchmark_external_validation/` for external panels.

## Reproduce

```bash
python parallel_experiment_table/build_parallel_table.py
python parallel_experiment_table/make_heatmap.py   # optional
python scripts/build_reference_artefact_manifest.py
```

See also [Reference artefacts](reference-artefacts.md) and
[validation index](methods/validation/index.md).
