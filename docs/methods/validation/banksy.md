# Validation report — `banksy`

**Protocol:** `histoweave.method_validation.multidataset.v1`
**Category:** `domain_detection`
**Decision:** **VALIDATED**
**Datasets (n=5):** 151507, 151669, 151670, 151673, 151674

## Gates

| Gate | Pass |
|------|:----:|
| `multi_slice_ari_ge_0.12` | ✅ |
| `r_container_contract_tests` | ✅ |
| `proxy_implementation_disclosed` | ✅ |
| `limitations_documented` | ✅ |

## Metrics

### sota_banksy_py_proxy

```json
{
  "mean_ari": 0.222860698214684,
  "std_ari": 0.04658856193904745,
  "per_dataset": {
    "151507": 0.2189200084121582,
    "151669": 0.15634155018555188,
    "151670": 0.2945022373356372,
    "151673": 0.19808490819807867,
    "151674": 0.24645478694199408
  },
  "n_datasets": 5,
  "n_runs": 15,
  "source": "5x15_spatial_aware/sota_benchmark_long.csv",
  "protocol": "histoweave.sota_domains.v1",
  "implementation_note": "Official multi-dataset ARI uses native banksy_py scaffold; R Bioconductor::Banksy wrap (name=banksy) is contract-validated separately."
}
```


## Sources

- `5x15_spatial_aware/sota_benchmark_long.csv`
- `tests/test_banksy_spatialde.py`

## Notes

- Multi-dataset ARI measured on native banksy_py (same algorithmic family); R Bioconductor::Banksy is the production wrap with container contract tests.

## Limitations (independent review)

- Numeric multi-slice ARI is from banksy_py, not a full R Banksy grid (container cost).
- Users requiring exact Bioconductor numerics should pin the R image and re-run the SOTA protocol.
- Lambda / algorithm hyperparameters remain tissue-dependent.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
