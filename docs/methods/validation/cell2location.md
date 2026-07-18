# Validation report — `cell2location`

**Protocol:** `histoweave.method_validation.multidataset.v1`
**Category:** `deconvolution`
**Decision:** **contract_validated** (structural / mock multi-dataset gates — not scientific ARI)
**Datasets (n=6):** synth_mixture_a, synth_mixture_b, synth_mixture_c, dlpfc_151507, dlpfc_151669, dlpfc_151673

## Gates

| Gate | Pass |
|------|:----:|
| `multi_dataset_ge_3` | ✅ |
| `contract_success_rate_1.0` | ✅ |
| `mean_shared_genes_ge_5` | ✅ |
| `no_marker_fallback` | ✅ |
| `limitations_documented` | ✅ |

## Metrics

### multidataset

```json
{
  "protocol": "histoweave.method_validation.cell2location_structural.v1",
  "backend": "mock_cell2location.models.Cell2location",
  "datasets": [
    "synth_mixture_a",
    "synth_mixture_b",
    "synth_mixture_c",
    "dlpfc_151507",
    "dlpfc_151669",
    "dlpfc_151673"
  ],
  "rows": [
    {
      "dataset": "synth_mixture_a",
      "n_obs": 200,
      "n_vars": 80,
      "shared_genes": 26,
      "n_types": 5,
      "success": true,
      "abundance_key": "cell_abundance",
      "proportion_key": "proportions",
      "proportion_row_sum_mean": 1.0,
      "train_called": true,
      "export_called": true
    },
    {
      "dataset": "synth_mixture_b",
      "n_obs": 300,
      "n_vars": 100,
      "shared_genes": 26,
      "n_types": 5,
      "success": true,
      "abundance_key": "cell_abundance",
      "proportion_key": "proportions",
      "proportion_row_sum_mean": 1.0,
      "train_called": true,
      "export_called": true
    },
    {
      "dataset": "synth_mixture_c",
      "n_obs": 250,
      "n_vars": 90,
      "shared_genes": 26,
      "n_types": 5,
      "success": true,
      "abundance_key": "cell_abundance",
      "proportion_key": "proportions",
      "proportion_row_sum_mean": 1.0,
      "train_called": true,
      "export_called": true
    },
    {
      "dataset": "dlpfc_151507",
      "n_obs": 800,
      "n_vars": 33538,
      "shared_genes": 26,
      "n_types": 5,
      "success": true,
      "abundance_key": "cell_abundance",
      "proportion_key": "proportions",
      "proportion_row_sum_mean": 1.0,
      "train_called": true,
      "export_called": true
    },
    {
      "dataset": "dlpfc_151669",
      "n_obs": 800,
      "n_vars": 33538,
      "shared_genes": 26,
      "n_types": 5,
      "success": true,
      "abundance_key": "cell_abundance",
      "proportion_key": "proportions",
      "proportion_row_sum_mean": 1.0,
      "train_called": true,
      "export_called": true
    },
    {
      "dataset": "dlpfc_151673",
      "n_obs": 800,
      "n_vars": 33538,
      "shared_genes": 26,
      "n_types": 5,
      "success": true,
      "abundance_key": "cell_abundance",
      "proportion_key": "proportions",
      "proportion_row_sum_mean": 1.0,
      "train_called": true,
      "export_called": true
    }
  ],
  "n_success": 6,
  "n_total": 6,
  "mean_shared_genes": 26.0,
  "no_marker_fallback": true,
  "sources": [
    "research/method_validation/run_cell2location_multidataset.py",
    "tests/test_dlpfc_cell2location.py"
  ],
  "limitations": [
    "Mock backend exercises adapter I/O, not full Pyro posterior quality.",
    "Synthetic mixtures plant markers; DLPFC rows use marker-derived signatures, not full scRNA atlases.",
    "Do not interpret mock abundances as biological cell-type maps."
  ]
}
```


## Sources

- `research/method_validation/run_cell2location_multidataset.py`
- `tests/test_dlpfc_cell2location.py`

## Notes

- _None._

## Limitations (independent review)

- Mock backend exercises adapter I/O, not full Pyro posterior quality.
- Synthetic mixtures plant markers; DLPFC rows use marker-derived signatures, not full scRNA atlases.
- Do not interpret mock abundances as biological cell-type maps.

## Claim bounds

1. Validation promotes **wrapper maturity**, not universal SOTA.
2. Metrics are protocol-bound (oracle *k*, spatial weight, mock backend as disclosed).
3. Re-run compile after any benchmark CSV regeneration.
