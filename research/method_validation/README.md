# Multi-dataset method validation

Promote methods to **validated** maturity with reproducible multi-dataset
evidence packages.

## Quick start

```bash
# 1) Structural multi-dataset suite for cell2location (mock backend)
python research/method_validation/run_cell2location_multidataset.py

# 2) SOTA batch: spagcn / graphst / stagate / rctd / spatialde
python research/method_validation/run_sota_batch_multidataset.py

# 3) Official GraphST / STAGATE multi-slice ARI (requires backends)
python research/method_validation/run_real_graphst_stagate_ari.py --methods graphst --max-obs 1000
# STAGATE needs a torch-sparse-compatible interpreter, e.g.:
#   $env:HISTOWEAVE_STAGATE_PYTHON = "C:\path\to\py312\python.exe"
python research/method_validation/run_real_graphst_stagate_ari.py --methods stagate --max-obs 1000

# 4) Compile formal reports from benchmark tables + JSON suites
python research/method_validation/compile_validation_evidence.py
```

Outputs:

| Path | Content |
|------|---------|
| `results/validation_summary.json` | Machine-readable gates |
| `results/VALIDATION_BATCH_REPORT.md` | Batch narrative |
| `docs/methods/validation/*.md` | Formal per-method reports |
| `docs/methods/validation/index.md` | Index |

## Expansion batches

**A — baselines:** `agglomerative` · `birch` · `minibatch_kmeans` · `banksy` · `cell2location`
**B — SOTA priority:** `spagcn` · `graphst` · `stagate` · `rctd` · `spatialde`

Evidence sources:

- Figure 3 synthetic landscape (`figure3_results/`)
- DLPFC 5×10 real (`5x10_dlpfc_benchmark/`)
- DLPFC 5×15 spatial-aware (`5x15_spatial_aware/`)
- SOTA multi-slice (`sota_benchmark_long.csv` — SpaGCN, banksy_py)
- cell2location + SOTA structural multi-dataset JSON suites

See [PROTOCOL.md](PROTOCOL.md).
