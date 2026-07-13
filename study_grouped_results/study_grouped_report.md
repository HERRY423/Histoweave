# Study-Grouped Recommendation Validation

- **Queries**: 3
- **Training datasets**: 6
- **Top-1 accuracy**: 33.33%
- **Top-3 accuracy**: 66.67%
- **Mean selection regret**: 0.0520
- **k-NN beats global-best**: 100%
- **k-NN beats random**: 100%

| Held-out | Oracle | Recommended | Score | Top1 | Regret |
|----------|--------|-------------|-------|------|--------|
| dlpfc_151507 | kmeans | kmeans | 0.4351 | Y | 0.0000 |
| dlpfc_151508 | kmeans | gaussian_mixture | 0.4521 | · | 0.0921 |
| dlpfc_151509 | gaussian_mixture | kmeans | 0.3215 | · | 0.0639 |

## Caveats
- Only 3 real-data queries — insufficient for precision.
- DLPFC slices share donor, platform, and tissue — not cross-study.
- Ground truth is GMM-derived, not manual layer annotation.
- Requires {>=5} heterogeneous training datasets for k-NN discrimination.