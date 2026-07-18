# Validation report — `birch`

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
  "mean_ari": 0.17446196627314486,
  "std_ari": 0.045038975117845374,
  "per_dataset": {
    "151507": 0.1320529975079289,
    "151669": 0.1296685490683335,
    "151670": 0.1876685778839524,
    "151673": 0.17007462879487054,
    "151674": 0.2528450781106391
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
  "mean_ari_all_sw": 0.14670742993014885,
  "mean_ari_best_sw": 0.22759032241112148,
  "std_ari_best_sw": 0.035468770219201425,
  "per_dataset_best_sw": {
    "151507": {
      "ari": 0.2513527623493769,
      "spatial_weight": 0.8
    },
    "151669": {
      "ari": 0.1870453690031455,
      "spatial_weight": 0.8
    },
    "151670": {
      "ari": 0.1876685778839524,
      "spatial_weight": 0.3
    },
    "151673": {
      "ari": 0.23496401745875692,
      "spatial_weight": 0.8
    },
    "151674": {
      "ari": 0.2769208853603757,
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
