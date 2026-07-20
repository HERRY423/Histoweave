# Validation report — `graphst`

**Protocol:** `histoweave.sota_dlpfc.v1`
**Category:** `domain_detection`
**Decision:** **VALIDATED**
**Datasets (n=5):** 151507, 151669, 151670, 151673, 151674

## Gates

| Gate | Pass |
|------|:----:|
| `multi_slice_ari_ge_0.12` | ✅ |
| `multi_seed_success_ge_9` | ✅ |
| `official_backend` | ✅ |
| `limitations_documented` | ✅ |

## Metrics

### real_ari

```json
{
  "n_cells": 15,
  "n_success": 15,
  "mean_ari": 0.12063699622085217,
  "std_ari": 0.03818115794736281,
  "per_dataset_mean_ari": {
    "151507": 0.1271637835331029,
    "151669": 0.06494869837834567,
    "151670": 0.09745638604812684,
    "151673": 0.16134978430100386,
    "151674": 0.1522663288436816
  },
  "mean_seconds": 9.095066666666668,
  "statuses": [
    "success"
  ],
  "errors": []
}
```

### run_meta

```json
{
  "max_obs": 1000,
  "slices": [
    "151673",
    "151674",
    "151507",
    "151669",
    "151670"
  ],
  "seeds": [
    42,
    1,
    2
  ],
  "backend_mode": "official_real",
  "protocol": "histoweave.sota_dlpfc.v1"
}
```


## Sources

- `research/method_validation/results/graphst_stagate_real_ari.json`
- `5x15_spatial_aware/sota_benchmark_long.csv`

## Notes

- Official graphst multi-slice ARI mean=0.121 (15 cells, 5 slices).
- Spots subsampled to max_obs=1000 for CPU runtime.

## Limitations (independent review)

- ARI vs manual DLPFC layers with **oracle-K** domain count (track must be labeled).
- Epochs may be reduced vs paper defaults for wall-clock feasibility.
- Subsampling (if any) is disclosed in run_meta.max_obs.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
