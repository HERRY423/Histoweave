# Validation report — `agglomerative`

**Protocol:** `histoweave.method_validation.multidataset.v1`
**Category:** `domain_detection`
**Decision:** **VALIDATED**
**Datasets (n=8):** 151507, 151669, 151670, 151673, 151674, clean_easy, noisy_hard, sparse_scattered

## Gates

| Gate | Pass |
|------|:----:|
| `synthetic_ari_ge_0.40` | ✅ |
| `dlpfc_ari_ge_0.12` | ✅ |
| `multi_dataset_coverage` | ✅ |
| `limitations_documented` | ✅ |

## Metrics

### figure3_synthetic

```json
{
  "mean_ari": 0.8768795791978808,
  "std_ari": 0.04077251778228479,
  "per_dataset": {
    "clean_easy": 0.8511725584403583,
    "noisy_hard": 0.8450345535571109,
    "sparse_scattered": 0.9344316255961732
  },
  "n_datasets": 3,
  "source": "figure3_results/benchmark_long.csv",
  "protocol": "histoweave.figure3.synthetic.v1"
}
```

### dlpfc_5x10

```json
{
  "mean_ari": 0.20106127706443508,
  "std_ari": 0.05030732868949965,
  "per_dataset": {
    "151507": 0.2106957133522052,
    "151669": 0.1454074251652835,
    "151670": 0.280014239444171,
    "151673": 0.1479333288104907,
    "151674": 0.2212556785500248
  },
  "n_datasets": 5,
  "n_runs": 15,
  "source": "5x10_dlpfc_benchmark/benchmark_long.csv",
  "protocol": "histoweave.landscape.dlpfc_real.v1"
}
```

### dlpfc_5x15_spatial_aware

```json
{
  "mean_ari_all_sw": 0.16158105853438973,
  "mean_ari_best_sw": 0.24619047261732385,
  "std_ari_best_sw": 0.040439742937948005,
  "per_dataset_best_sw": {
    "151507": {
      "ari": 0.2157613517899146,
      "spatial_weight": 0.8
    },
    "151669": {
      "ari": 0.1895454791432171,
      "spatial_weight": 0.8
    },
    "151670": {
      "ari": 0.280014239444171,
      "spatial_weight": 0.3
    },
    "151673": {
      "ari": 0.24560875008751207,
      "spatial_weight": 0.8
    },
    "151674": {
      "ari": 0.3000225426218044,
      "spatial_weight": 0.8
    }
  },
  "n_datasets": 5,
  "n_runs": 45,
  "source": "5x15_spatial_aware/benchmark_long.csv",
  "protocol": "histoweave.dlpfc_spatial_aware.v1"
}
```


## Sources

- `figure3_results/benchmark_long.csv`
- `5x10_dlpfc_benchmark/benchmark_long.csv`
- `5x15_spatial_aware/benchmark_long.csv`

## Notes

- _None._

## Limitations (independent review)

- Oracle or estimate *k* policies affect ARI; report the protocol used in each table.
- DLPFC ARI vs manual layers is a domain-recovery proxy, not biological ground truth of cell state.
- Spatial weight is a major lever (5×15 study); expression-only configs under-estimate spatial methods.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
