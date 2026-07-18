# SOTA DLPFC reproduction (P2)

This page is the operator contract for reproducing **official** spatial-domain
methods (SpaGCN, GraphST, STAGATE, BayesSpace) plus the native `banksy_py`
control on the Maynard 2021 DLPFC Visium slices.

## One-command probe

```bash
python scripts/run_sota_dlpfc.py --dry-run
# -> 5x15_spatial_aware/sota_probe.json
# -> 5x15_spatial_aware/sota_benchmark_long.csv  (status=skipped_*)
# -> 5x15_spatial_aware/sota_throughput.json
```

## Run what is installed

```bash
# Native control (always available with histoweave)
python scripts/run_sota_dlpfc.py --methods banksy_py

# Optional heavy backends (isolated interpreters via env vars)
export HISTOWEAVE_SPAGCN_PYTHON=/path/to/spagcn/python
export HISTOWEAVE_GRAPHST_PYTHON=/path/to/graphst/python
export HISTOWEAVE_STAGATE_PYTHON=/path/to/stagate/python
export HISTOWEAVE_R_LIB=/path/to/R/library
export HISTOWEAVE_SOTA_DEVICE=cpu   # or cuda

python scripts/run_sota_dlpfc.py --methods banksy_py,spagcn,graphst,stagate,bayesspace
```

## Merge into landscape + leaderboard

```bash
python scripts/build_merged_landscape.py \
  --sota-csv 5x15_spatial_aware/sota_benchmark_long.csv \
  --out figure3_results/landscape_dlpfc_merged.json

python leaderboard/generate.py
```

## Nextflow (optional HPC)

```bash
nextflow run workflows/nextflow/sota.nf \
  --repo $PWD \
  --methods banksy_py,spagcn \
  --outdir results/sota
```

Each `(method, slice, seed)` is an independent process with `errorStrategy
'ignore'`, so one missing backend does not kill the grid.

## Hard rules

| Rule | Why |
|------|-----|
| Fail closed on missing backend | No toy substitute under a SOTA name |
| `status` column required | Leaderboard / landscape can filter |
| Oracle `n_domains` documented | Comparability with published DLPFC setups |
| Task = `spatial_domain` only | Expert cortical layers, never Leiden |

Environment contract (YAML twin): [`workflows/sota/env_contract.yaml`](https://github.com/HERRY423/Histoweave/blob/main/workflows/sota/env_contract.yaml).

## Throughput report

`sota_throughput.json` records per-method success counts, mean ARI, and mean
seconds for cost planning on HPC or laptops.
