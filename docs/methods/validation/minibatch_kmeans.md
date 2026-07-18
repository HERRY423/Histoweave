# Validation report — `minibatch_kmeans`

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
  "mean_ari": 0.5708504385114233,
  "std_ari": 0.30514610034357326,
  "per_dataset": {
    "clean_easy": 0.9914326197318956,
    "noisy_hard": 0.4442503597376289,
    "sparse_scattered": 0.2768683360647453
  },
  "n_datasets": 3,
  "source": "figure3_results/benchmark_long.csv",
  "protocol": "histoweave.figure3.synthetic.v1"
}
```

### dlpfc_5x10

```json
{
  "mean_ari": 0.2229264432193757,
  "std_ari": 0.04691286852782013,
  "per_dataset": {
    "151507": 0.23722780389968542,
    "151669": 0.17216394104595115,
    "151670": 0.21123493042993471,
    "151673": 0.22937839342418823,
    "151674": 0.2646271472971191
  },
  "n_datasets": 5,
  "n_runs": 15,
  "source": "5x10_dlpfc_benchmark/benchmark_long.csv",
  "protocol": "histoweave.landscape.dlpfc_real.v1"
}
```


## Sources

- `figure3_results/benchmark_long.csv`
- `5x10_dlpfc_benchmark/benchmark_long.csv`

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
