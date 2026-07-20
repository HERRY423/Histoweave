# Method validation reports (multi-dataset)

**Protocol family:** `histoweave.method_validation.*`

HistoWeave separates two multi-dataset evidence kinds (do **not** conflate them):

| Kind | Maturity | Count | Meaning |
|------|----------|------:|---------|
| **Scientific** | `validated` | **10** | Concordance vs independent ground truth on real multi-dataset grids (e.g. ARI) |
| **Contract** | `contract_validated` | **3** | Interface / mock / fail-closed structural gates across datasets (CI-safe) |
| **Total evidence packages** | — | **13** | `10 + 3` — single ledger in `release_manifest.VALIDATION_EVIDENCE` |

Canonical sets: `SCIENTIFIC_VALIDATED_METHODS`, `CONTRACT_VALIDATED_METHODS`,
`MULTI_DATASET_EVIDENCE_METHODS` in
[`release_manifest.py`](https://github.com/histoweave-spatial/histoweave/blob/main/src/histoweave/plugins/builtin/release_manifest.py).

## Scientific validated (10)

| Method | Category | Decision | Report |
|--------|----------|----------|--------|
| `agglomerative` | domain_detection | **validated** | [report](agglomerative.md) |
| `banksy` | domain_detection | **validated** | [report](banksy.md) |
| `banksy_py` | domain_detection | **validated** | [report](banksy_py.md) |
| `birch` | domain_detection | **validated** | [report](birch.md) |
| `gaussian_mixture` | domain_detection | **validated** | [report](gaussian_mixture.md) |
| `graphst` | domain_detection | **validated** | [report](graphst.md) |
| `minibatch_kmeans` | domain_detection | **validated** | [report](minibatch_kmeans.md) |
| `spagcn` | domain_detection | **validated** | [report](spagcn.md) |
| `spectral` | domain_detection | **validated** | [report](spectral.md) |
| `stagate` | domain_detection | **validated** | [report](stagate.md) |

## Contract validated (3)

| Method | Category | Decision | Report |
|--------|----------|----------|--------|
| `cell2location` | deconvolution | **contract_validated** | [report](cell2location.md) |
| `rctd` | deconvolution | **contract_validated** | [report](rctd.md) |
| `spatialde` | svg | **contract_validated** | [report](spatialde.md) |

Contract-validated methods wrap real upstream libraries and pass multi-dataset
**I/O / fail-closed / structural** gates (often with mock backends in CI). They are
**not** claimed as scientifically concordant until real multi-dataset label metrics
pass the scientific gate.

## SOTA ARI tracks (do not mix)

Published wrapper maturity reports historically used **oracle-K**
(`n_domains = domain_truth.nunique()`). Blind analyses must use
`k_policy=estimate`. Numbers from different tracks are **not** interchangeable.

| Method | Track | Mean ARI | Source |
|--------|-------|---------:|--------|
| SpaGCN | oracle-K | ≈0.317 | [spagcn.md](spagcn.md) / `5x15_spatial_aware/sota_benchmark_long.csv` |
| STAGATE | oracle-K (max_obs=1000) | ≈0.285 | [stagate.md](stagate.md) |
| GraphST | oracle-K | ≈0.12 | [graphst.md](graphst.md) |
| SpaGCN | estimate·silhouette | ≈0.237 | [non_oracle_k_sota](../../../non_oracle_k_sota/) |
| STAGATE | estimate·silhouette | ≈0.219 | same |
| SpaGCN | oracle−estimate drop (mean) | ≈0.062 | protocol endpoint `oracle_k_leakage` |
| SpaGCN | oracle−estimate drop (151673) | ≈0.232 | same |

Re-run endpoint: `python scripts/run_protocol_endpoints.py` (loads
`non_oracle_k_sota/benchmark_long.csv` when present).

## Reference artefacts (frozen result archives)

These directories are **reference artefacts** for the decision protocol and
SOTA audits. They are not method wrappers; cite them with their protocol
string and do not silently mix tracks.

**Availability policy:** summary JSON/MD/CSV are **tracked in git**. Raw H5AD
and logs are not. Canonical inventory:
[`reference_artefacts/MANIFEST.json`](../../../reference_artefacts/MANIFEST.json)
and [Reference artefacts](../../reference-artefacts.md).

| Directory | Protocol / role | In git? | Primary files |
|-----------|-----------------|:-------:|---------------|
| [`non_oracle_k_sota/`](../../../non_oracle_k_sota/) | `histoweave.non_oracle_k_sota.v1` + endpoint `histoweave.oracle_k_leakage.v1` | **yes** (summaries) | `benchmark_long.csv`, `dual_track_k.json`, `summary.json`, figures, report |
| [`pareto_isus_results/`](../../../pareto_isus_results/) | Pareto multi-objective report + post-hoc ISUS calibration (not a predictor) | **yes** | `pareto_report.json`, `isus_calibration.json`, figures |
| [`protocol_endpoints_results/`](../../../protocol_endpoints_results/) | Falsifiable endpoints 1–5 bundle | **yes** | `protocol_endpoints_summary.json`, study/selective/pareto/sota/oracle_k JSON |
| [`independent_personalisation_results/`](../../../independent_personalisation_results/) | Study-level gated personalisation | **yes** | `independent_personalisation_summary.json`, report, cross-lab JSON |
| [`benchmark_external_validation/decision_validation.json`](../../../benchmark_external_validation/decision_validation.json) | Negative external holdout control | **yes** | `beats_global_best: false` |
| [`parallel_experiment_table/`](../../../parallel_experiment_table/) | Same-task same-data side-by-side (sklearn / spatial-aware / SOTA) | **yes** | `parallel_experiment_summary.csv`, report, heatmap |

### How to cite a number from a reference artefact

1. Name the **protocol string** and **track** (`oracle` vs `estimate:…`).
2. Prefer machine-readable JSON/CSV over prose README approximations.
3. If regenerating, bump the protocol version when semantics change.
4. Confirm the file is listed in `reference_artefacts/MANIFEST.json` (run
   `python scripts/build_reference_artefact_manifest.py --check`).

## Batch narrative

See `research/method_validation/results/VALIDATION_BATCH_REPORT.md` and the
[validation protocol](https://github.com/histoweave-spatial/histoweave/blob/main/research/method_validation/PROTOCOL.md).

## Related

- [Method guide index](../index.md)
- [Method lifecycle](../../method-lifecycle.md)
- [Decision protocol](../../decision-protocol.md)
- [Roadmap](../../../ROADMAP.md)
- [Release manifest](https://github.com/histoweave-spatial/histoweave/blob/main/src/histoweave/plugins/builtin/release_manifest.py)
