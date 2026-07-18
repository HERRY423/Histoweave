# Validation report — `spagcn`

**Protocol:** `histoweave.method_validation.sota_batch.v1`
**Category:** `domain_detection`
**Decision:** **VALIDATED**
**Datasets (n=5):** 151507, 151669, 151670, 151673, 151674

## Gates

| Gate | Pass |
|------|:----:|
| `multi_slice_ari_ge_0.12` | ✅ |
| `multi_seed_runs_ge_9` | ✅ |
| `official_backend_not_substituted` | ✅ |
| `limitations_documented` | ✅ |

## Metrics

### sota_csv

```json
{
  "mean_ari": 0.3171446309976544,
  "std_ari": 0.09873356261670699,
  "per_dataset": {
    "151507": 0.3834553663668036,
    "151669": 0.19925228348214896,
    "151670": 0.22162956713600845,
    "151673": 0.3852591432425521,
    "151674": 0.3961267947607585
  },
  "n_datasets": 5,
  "n_runs": 15,
  "source": "5x15_spatial_aware\\sota_benchmark_long.csv",
  "protocol": "histoweave.sota_dlpfc.v1",
  "backend": "official SpaGCN"
}
```

### live_smoke

```json
{
  "available": true,
  "rows": [
    {
      "dataset": "dlpfc_151507",
      "success": true,
      "ari": 0.34366515783302065,
      "n_obs": 500
    },
    {
      "dataset": "dlpfc_151669",
      "success": true,
      "ari": 0.08184013689276551,
      "n_obs": 500
    },
    {
      "dataset": "dlpfc_151673",
      "success": true,
      "ari": 0.26875756575544674,
      "n_obs": 500
    }
  ],
  "n_success": 3,
  "n_total": 3,
  "mean_ari": 0.23142095349374428
}
```


## Sources

- `5x15_spatial_aware/sota_benchmark_long.csv`
- `research/method_validation/run_sota_batch_multidataset.py`

## Notes

- SOTA DLPFC grid mean ARI=0.317 across 5 slices / 15 runs.
- Live SpaGCN smoke success=3/3 mean_ari=0.23142095349374428

## Limitations (independent review)

- Primary multi-slice ARI from official SpaGCN SOTA grid (5 DLPFC × 3 seeds).
- Live smoke uses reduced epochs/HVG subsample for runtime.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
