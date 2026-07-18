# Validation report — `spatialde`

**Protocol:** `histoweave.method_validation.sota_batch.v1`
**Category:** `svg`
**Decision:** **contract_validated** (mock multi-dataset SVG I/O gates — not scientific concordance)
**Datasets (n=6):** synth_domain_a, synth_domain_b, synth_domain_c, dlpfc_151507, dlpfc_151669, dlpfc_151673

## Gates

| Gate | Pass |
|------|:----:|
| `multi_dataset_ge_3` | ✅ |
| `contract_success_rate_1.0` | ✅ |
| `limitations_documented` | ✅ |
| `exports_svg_ranking` | ✅ |

## Metrics

### multidataset

```json
{
  "category": "svg",
  "backend": "mock_SpatialDE_NaiveDE",
  "rows": [
    {
      "dataset": "synth_domain_a",
      "success": true,
      "n_top": 20,
      "n_significant": 9,
      "n_tested": 60,
      "n_obs": 180
    },
    {
      "dataset": "synth_domain_b",
      "success": true,
      "n_top": 20,
      "n_significant": 9,
      "n_tested": 60,
      "n_obs": 180
    },
    {
      "dataset": "synth_domain_c",
      "success": true,
      "n_top": 20,
      "n_significant": 9,
      "n_tested": 60,
      "n_obs": 180
    },
    {
      "dataset": "dlpfc_151507",
      "success": true,
      "n_top": 20,
      "n_significant": 76,
      "n_tested": 600,
      "n_obs": 500
    },
    {
      "dataset": "dlpfc_151669",
      "success": true,
      "n_top": 20,
      "n_significant": 76,
      "n_tested": 600,
      "n_obs": 500
    },
    {
      "dataset": "dlpfc_151673",
      "success": true,
      "n_top": 20,
      "n_significant": 76,
      "n_tested": 600,
      "n_obs": 500
    }
  ],
  "n_success": 6,
  "n_total": 6,
  "datasets": [
    "synth_domain_a",
    "synth_domain_b",
    "synth_domain_c",
    "dlpfc_151507",
    "dlpfc_151669",
    "dlpfc_151673"
  ],
  "mean_n_significant": 42.5,
  "sources": [
    "research/method_validation/run_sota_batch_multidataset.py",
    "tests/test_banksy_spatialde.py",
    "tests/test_core_real_method_contracts.py"
  ],
  "limitations": [
    "Mock SpatialDE/NaiveDE for multi-dataset I/O and ranking contract.",
    "Install histoweave-spatial[spatialde] for real GP p-values."
  ]
}
```


## Sources

- `research/method_validation/run_sota_batch_multidataset.py`
- `tests/test_banksy_spatialde.py`
- `tests/test_core_real_method_contracts.py`

## Notes

- Structural multi-dataset contract 6/6 (mock_SpatialDE_NaiveDE).

## Limitations (independent review)

- Mock SpatialDE/NaiveDE for multi-dataset I/O and ranking contract.
- Install histoweave-spatial[spatialde] for real GP p-values.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
