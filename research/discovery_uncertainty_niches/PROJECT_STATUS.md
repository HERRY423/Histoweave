# Discovery project status (frozen)

```json
{
  "protocol": "histoweave.discovery_cohort.v1",
  "n_slices": 12,
  "n_slices_ok": 12,
  "n_L3_components": 15,
  "n_L3_direction_ok": 14,
  "n_L3_shift_rest_ok": 3,
  "n_L3_hard_pass": 0,
  "n_L6_components": 9,
  "n_L6_hard_pass": 2,
  "claim": "L3 cryptic niches show replicable RNA direction (mid-layer program up, myelin down vs rest) across the DLPFC cohort; same-layer hard gates and named cell states require IF (see IF_PROTOCOL.md).",
  "next": [
    "Run IF on high-confidence L3 ROIs (direction_ok + shift_rest_ok)",
    "Do not claim new cell state without same-layer protein pass"
  ],
  "donor_bootstrap_l3": {
    "n_components_direction_ok": 14,
    "n_donors": 3,
    "l3_delta_rest_point": 0.2878,
    "l3_delta_rest_ci95": [0.2221, 0.3442],
    "myelin_delta_rest_point": -0.3544,
    "myelin_delta_rest_ci95": [-0.3778, -0.3237],
    "direction_rate": 1.0,
    "ci_excludes_zero_both_directions": true
  },
  "cli": [
    "histoweave discovery bootstrap-ci",
    "histoweave discovery cohort",
    "histoweave stats-review --landscape …",
    "histoweave sota"
  ]
}
```

See [COHORT_META_REPORT.md](COHORT_META_REPORT.md) and [DONOR_BOOTSTRAP_L3.md](DONOR_BOOTSTRAP_L3.md).
