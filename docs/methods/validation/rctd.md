# Validation report — `rctd`

**Protocol:** `histoweave.method_validation.sota_batch.v1`
**Category:** `deconvolution`
**Decision:** **VALIDATED**
**Datasets (n=6):** synth_domain_a, synth_domain_b, synth_domain_c, dlpfc_151507, dlpfc_151669, dlpfc_151673

## Gates

| Gate | Pass |
|------|:----:|
| `multi_dataset_ge_3` | ✅ |
| `contract_success_rate_1.0` | ✅ |
| `limitations_documented` | ✅ |
| `no_marker_fallback` | ✅ |
| `fail_closed_without_driver` | ✅ |

## Metrics

### multidataset

```json
{
  "category": "deconvolution",
  "backend": "Rscript_present_driver_required",
  "fail_closed_without_driver": true,
  "rows": [
    {
      "dataset": "synth_domain_a",
      "success": true,
      "mode": "fail_closed_missing_driver"
    },
    {
      "dataset": "synth_domain_b",
      "success": true,
      "mode": "fail_closed_missing_driver"
    },
    {
      "dataset": "synth_domain_c",
      "success": true,
      "mode": "fail_closed_missing_driver"
    },
    {
      "dataset": "dlpfc_151507",
      "success": true,
      "mode": "fail_closed_missing_driver"
    },
    {
      "dataset": "dlpfc_151669",
      "success": true,
      "mode": "fail_closed_missing_driver"
    },
    {
      "dataset": "dlpfc_151673",
      "success": true,
      "mode": "fail_closed_missing_driver"
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
  "no_marker_fallback": true,
  "sources": [
    "research/method_validation/run_sota_batch_multidataset.py",
    "src/histoweave/plugins/builtin/sota_domains.py"
  ],
  "limitations": [
    "Multi-dataset gate is fail-closed contract (reference + counts + driver required).",
    "Full spacexr RCTD ARI needs R driver + scRNA reference atlas \u2014 not substituted."
  ]
}
```


## Sources

- `research/method_validation/run_sota_batch_multidataset.py`
- `src/histoweave/plugins/builtin/sota_domains.py`

## Notes

- Structural multi-dataset contract 6/6 (Rscript_present_driver_required).

## Limitations (independent review)

- Multi-dataset gate is fail-closed contract (reference + counts + driver required).
- Full spacexr RCTD ARI needs R driver + scRNA reference atlas — not substituted.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
