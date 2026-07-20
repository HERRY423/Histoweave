# Validation report — `stagate`

**Protocol:** `histoweave.sota_dlpfc.v1`
**Category:** `domain_detection`
**Decision:** **VALIDATED**
**Datasets (n=5):** 151507, 151669, 151670, 151673, 151674
**Primary track for maturity gate:** **oracle-K** (with disclosed subsample)

> **Track warning.** Maturity ARI used oracle domain count and often
> `max_obs=1000`. Blind estimates live under `non_oracle_k_sota/` (full
> n_obs, seed 42).

## Gates

| Gate | Pass |
|------|:----:|
| `multi_slice_ari_ge_0.12` | ✅ |
| `multi_seed_success_ge_9` | ✅ |
| `official_backend` | ✅ |
| `limitations_documented` | ✅ |

## Metrics

### Track comparison (do not mix)

| Track | Mean ARI | Notes | Source |
|-------|---------:|-------|--------|
| **oracle-K** (3 seeds, max_obs=1000) | **0.285** | maturity gate | `real_ari` below |
| **estimate · silhouette** (seed 42, full n_obs) | **0.219** | dual-track | `non_oracle_k_sota/` |
| Oracle − estimate mean drop | **≈0.013** | full-n_obs oracle in non_oracle archive differs from max_obs=1000 track | endpoint `oracle_k_leakage` |

### real_ari (oracle-K track, max_obs=1000)

```json
{
  "n_cells": 15,
  "n_success": 15,
  "mean_ari": 0.28500538554104377,
  "std_ari": 0.09406168334640957,
  "k_policy": "oracle",
  "per_dataset_mean_ari": {
    "151507": 0.4323920250546108,
    "151669": 0.16389344040421294,
    "151670": 0.2532449445085327,
    "151673": 0.2939586536224667,
    "151674": 0.28153786411539544
  },
  "mean_seconds": 12.872533333333331,
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
  "protocol": "histoweave.sota_dlpfc.v1",
  "k_policy": "oracle"
}
```


## Sources

- `research/method_validation/results/graphst_stagate_real_ari.json`
- `5x15_spatial_aware/sota_benchmark_long.csv`
- `non_oracle_k_sota/benchmark_long.csv`

## Notes

- Official stagate multi-slice ARI mean=0.285 (15 cells, 5 slices, max_obs=1000).
- Estimate-track full-slice mean ARI≈0.219 under silhouette K (seed 42).

## Limitations (independent review)

- ARI vs manual DLPFC layers with oracle domain count for maturity.
- Epochs may be reduced vs paper defaults for wall-clock feasibility.
- Subsampling (if any) is disclosed in run_meta.max_obs.
- Non-oracle K estimators do not match true layer counts on these slices.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (`k_policy`, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
4. Do not quote max_obs=1000 oracle-K ARI as full-slide blind performance.
