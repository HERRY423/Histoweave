# Reference artefacts — availability policy

**Problem this page solves.** Evaluation numbers used in the decision protocol,
validation ledger, and manuscript drafts must be **reproducibly obtainable**.
Large raw matrices must **not** bloat git. This page is the single policy for
what is in-repo, what is ignored, and how to regenerate either class.

**Inventory:** [`reference_artefacts/MANIFEST.json`](../reference_artefacts/MANIFEST.json)  
**Builder / CI gate:** `python scripts/build_reference_artefact_manifest.py [--check]`

---

## Policy in one table

| Artefact class | Path pattern | Tracked? | Typical size | How to obtain |
|----------------|--------------|:--------:|--------------|---------------|
| Independent personalisation **summaries** | `independent_personalisation_results/*.{json,md}` | **Yes** | &lt; 50 KB each | clone repo **or** `scripts/run_independent_personalisation.py` |
| Protocol endpoint **summaries** | `protocol_endpoints_results/*.{json,md}` | **Yes** | &lt; 60 KB each | clone repo **or** `scripts/run_protocol_endpoints.py` |
| Non-oracle K dual-track tables | `non_oracle_k_sota/{summary,benchmark_long,dual_track_k}.*` | **Yes** | &lt; 100 KB | clone **or** `non_oracle_k_sota/run_non_oracle_k_sota.py` |
| Pareto / ISUS frozen report | `pareto_isus_results/*` | **Yes** | figures + JSON | clone **or** `histoweave pareto` / `histoweave isus --calibrate` |
| External negative holdout | `benchmark_external_validation/decision_validation.json` | **Yes** | &lt; 2 KB | clone (canonical negative control) |
| Parallel same-task table | `parallel_experiment_table/*` | **Yes** | summaries + heatmap | clone **or** `python parallel_experiment_table/build_parallel_table.py` |
| Per-seed cell dumps | `non_oracle_k_sota/cells/` | No | variable | regenerate with dual-track runner |
| Run logs | `**/*.log` | No | variable | local only |
| Raw expression / images | `*.h5ad`, `/data/`, `datasets_cache/` | No | 10 MB–GB | public portals + preparer scripts (below) |

**Rule of thumb:** if a file is needed to **quote a protocol number** in a paper
or README, it must be a **summary** under 5 MiB and listed in `MANIFEST.json`.
If it is needed only to **recompute** that number from counts, it may stay local.

---

## Primary in-repo bundles

### 1. Independent personalisation

| File | Role |
|------|------|
| [`independent_personalisation_summary.json`](../independent_personalisation_results/independent_personalisation_summary.json) | Primary endpoints (n units, gated regret, NI flags) |
| [`independent_personalisation_report.md`](../independent_personalisation_results/independent_personalisation_report.md) | Human table |
| [`cross_lab_reproducibility.json`](../independent_personalisation_results/cross_lab_reproducibility.json) | Δ regret CI, Kendall *W* |

Protocol: `histoweave.independent_personalisation.v1`  
README: [`independent_personalisation_results/README.md`](../independent_personalisation_results/README.md)

### 2. Protocol endpoints 1–5

| File | Role |
|------|------|
| [`protocol_endpoints_summary.json`](../protocol_endpoints_results/protocol_endpoints_summary.json) | Bundle index |
| [`oracle_k_leakage.json`](../protocol_endpoints_results/oracle_k_leakage.json) | Oracle − estimate ARI |
| [`selective_regret_coverage.json`](../protocol_endpoints_results/selective_regret_coverage.json) | Policy = often `always_global_default` |
| [`study_grouped_20_recommendation.json`](../protocol_endpoints_results/study_grouped_20_recommendation.json) | n=20 holdout |

README: [`protocol_endpoints_results/README.md`](../protocol_endpoints_results/README.md)

### 3. Non-oracle K / SOTA dual track

See [`non_oracle_k_sota/README.md`](../non_oracle_k_sota/README.md) and
[`docs/methods/validation/index.md`](methods/validation/index.md).

### 4. Pareto / ISUS

See [`pareto_isus_results/README.md`](../pareto_isus_results/README.md).

### 5. External validation negative control

[`benchmark_external_validation/decision_validation.json`](../benchmark_external_validation/decision_validation.json)
is intentionally **negative** (`beats_global_best: false`) and is used by the
[intercept case study](case-study-intercepted-recommendation.md).

### 6. Parallel experiment table (same task, same data)

| File | Role |
|------|------|
| [`parallel_experiment_summary.csv`](../parallel_experiment_table/parallel_experiment_summary.csv) | Ranked mean ARI over 5 DLPFC slices (33 configs) |
| [`parallel_experiment_table.csv`](../parallel_experiment_table/parallel_experiment_table.csv) | Long table: slice × method_config |
| [`parallel_experiment_matrix.csv`](../parallel_experiment_table/parallel_experiment_matrix.csv) | Wide matrix for heatmaps |
| [`report_parallel_experiment.md`](../parallel_experiment_table/report_parallel_experiment.md) | Narrative + caveats (seed / K-policy) |
| [`figures/parallel_heatmap.svg`](../parallel_experiment_table/figures/parallel_heatmap.svg) | Family-grouped heatmap |

Protocol: `histoweave.parallel_experiment_table.v1`  
Aligns `5x10_dlpfc_benchmark`, `5x15_spatial_aware`, and `non_oracle_k_sota` on
the shared 5-slice DLPFC spatial-domain panel. Rebuild:

```bash
python parallel_experiment_table/build_parallel_table.py
python parallel_experiment_table/make_heatmap.py   # optional
```

---

## Large inputs (not in git)

| Need | Obtain via |
|------|------------|
| DLPFC Visium H5AD for SOTA / dual-track | SpatialLIBD / 10x-compatible preparers under `benchmark_external_validation/prepare_*.py` and research preparers |
| Squidpy demo datasets | `squidpy.datasets.*` (runtime download) |
| Xenium / Visium HD panels | vendor portals + preparer scripts in `benchmark_external_validation/` |
| Local cache directory | `/data/anndata/` (gitignored) |

Never commit `*.h5ad` or `/data/`. The gitignore enforces this.

---

## Regeneration map

```bash
# Personalisation panel (writes independent_personalisation_results/)
python scripts/run_independent_personalisation.py

# Protocol endpoints bundle (writes protocol_endpoints_results/)
python scripts/run_protocol_endpoints.py

# Dual-track non-oracle K (writes non_oracle_k_sota/ summaries)
python non_oracle_k_sota/run_non_oracle_k_sota.py

# Same-task same-data parallel table (aligns three DLPFC benchmarks)
python parallel_experiment_table/build_parallel_table.py

# Refresh MANIFEST hashes after intentional re-runs
python scripts/build_reference_artefact_manifest.py
python scripts/build_reference_artefact_manifest.py --check
```

If regeneration **changes scientific numbers**, bump the relevant **protocol
version string** in the summary JSON and document the change in `CHANGELOG.md`.

---

## CI / developer gate

`tests/test_reference_artefacts_present.py` asserts:

1. Every **required** path exists.  
2. No required file exceeds the 5 MiB summary budget.  
3. `MANIFEST.json` hashes match the working tree (`--check` semantics).

---

## What changed vs earlier layout

| Before | After |
|--------|--------|
| Entire `independent_personalisation_results/` gitignored | Summaries **tracked**; only logs/h5ad/checkpoints ignored |
| Entire `protocol_endpoints_results/` gitignored | Summaries **tracked**; same ignore pattern |
| No central inventory | `reference_artefacts/MANIFEST.json` + this page |

---

## Related

- [Decision protocol](decision-protocol.md)
- [Validation index](methods/validation/index.md)
- [Intercept case study](case-study-intercepted-recommendation.md)
- [Roadmap](../ROADMAP.md)
