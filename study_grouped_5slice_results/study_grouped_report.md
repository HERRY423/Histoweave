# Study-Grouped Recommendation Validation

- **Queries**: 5
- **Training datasets**: 8
- **Top-1 accuracy**: 40.00%
- **Top-3 accuracy**: 100.00%
- **Mean selection regret**: 0.0451
- **k-NN beats global-best**: 80%
- **k-NN beats random**: 80%

| Held-out | Oracle | Recommended | Score | Top1 | Regret |
|----------|--------|-------------|-------|------|--------|
| dlpfc_151507 | kmeans | kmeans | 0.4351 | Y | 0.0000 |
| dlpfc_151508 | kmeans | kmeans | 0.5442 | Y | 0.0000 |
| dlpfc_151509 | gaussian_mixture | kmeans | 0.3215 | · | 0.0639 |
| dlpfc_151510 | bisecting_kmeans | kmeans | 0.3892 | · | 0.0273 |
| dlpfc_151673 | gaussian_mixture | kmeans | 0.3157 | · | 0.1342 |

## Caveats
- Multi-study validation supports generalisation claims.