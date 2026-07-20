# Reference artefacts (in-repo vs local-only)

HistoWeave freezes **small, citable evaluation summaries** in git and keeps
**large expression matrices / scratch dumps** out of the repository.

| Class | Location | In git? |
|-------|----------|:-------:|
| Decision / personalisation summaries | `independent_personalisation_results/` | **yes** |
| Protocol endpoints 1–5 | `protocol_endpoints_results/` | **yes** |
| Non-oracle K dual-track | `non_oracle_k_sota/` (summaries + long tables) | **yes** |
| Pareto / ISUS calibration | `pareto_isus_results/` | **yes** |
| External holdout negative control | `benchmark_external_validation/decision_validation.json` | **yes** |
| Raw H5AD / local caches | `/data/`, `*.h5ad`, `datasets_cache/` | **no** |
| Run logs / checkpoints | `**/*.log`, `**/checkpoints/` | **no** |

Machine-readable inventory: [`MANIFEST.json`](MANIFEST.json)  
(`schema`: `histoweave.reference_artefacts.manifest.v1`).

## Verify

```bash
python scripts/build_reference_artefact_manifest.py          # regenerate MANIFEST
python scripts/build_reference_artefact_manifest.py --check  # CI gate
```

## Full policy

See [docs/reference-artefacts.md](../docs/reference-artefacts.md).
