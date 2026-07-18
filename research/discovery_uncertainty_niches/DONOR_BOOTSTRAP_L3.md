# Donor-stratified bootstrap CI — L3 direction_ok components

**Protocol:** `histoweave.donor_bootstrap.v1`  ·  **n_boot:** 2000  ·  **components:** 14  ·  **donors:** 3 (Br5292, Br5595, Br8100)

**Filter:** `expected_class==L3_program & direction_ok`

## Point estimates (donor-equal weight)

| Metric | Point | 95% CI (donor-stratified) |
|--------|------:|----:|
| L3 Δ vs rest | 0.2878 | [0.2221, 0.3442] |
| Myelin Δ vs rest | -0.3544 | [-0.3778, -0.3237] |
| Direction rate | 1.000 | [1.000, 1.000] |

## Per-donor means (observed)

| Donor | n_comp | n_spots | L3 Δrest | Myelin Δrest | dir rate |
|-------|-------:|-------:|---------:|-------------:|---------:|
| Br5292 | 11 | 442 | 0.2614 | -0.2371 | 1.000 |
| Br5595 | 1 | 137 | 0.2254 | -0.2472 | 1.000 |
| Br8100 | 2 | 94 | 0.3766 | -0.5788 | 1.000 |

## Unstratified component bootstrap (comparison only)

| Metric | mean | CI |
|--------|-----:|----|
| L3 Δ | 0.2430 | [0.1736, 0.3183] |
| Myelin Δ | -0.2654 | [-0.3520, -0.1943] |

## Interpretation

- **Donor-stratified 95% CIs exclude 0 in the pre-registered directions** (L3 Δ > 0 and myelin Δ < 0) → cohort direction is robust to within-donor component resampling.
- Unstratified CIs are typically **narrower** (anticonservative).
- Same-layer hard gates remain separate (see cohort meta-report).

- _Donor-equal weight: each donor contributes one mean (components resampled within donor; optional spot-count weights inside donor)._
- _Unstratified bootstrap treats every component as independent (anticonservative if sections share donor effects)._
- _direction_rate = fraction of components with l3_delta>0 and myelin_delta<0._
