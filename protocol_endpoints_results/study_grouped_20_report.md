# Study-Grouped Recommendation Validation

- **Protocol**: `study_grouped_holdout`
- **Queries**: 20
- **Training pool**: 20
- **Meets ≥20 target**: yes
- **Top-1 accuracy**: 35.00%
- **Top-3 accuracy**: 60.00%
- **Mean selection regret**: 0.0468
- **Mean global-best regret**: 0.0290
- **Beats global-best (mean regret)**: False
- **k-NN beats global rate**: 65%
- **k-NN beats random rate**: 55%

| Held-out | Platform | Oracle | Selected | Conf | Top1 | Regret | Δ vs global |
|----------|----------|--------|----------|------|------|--------|-------------|
| 151507 | visium | bisecting_kmeans | minibatch_kmeans | 0.167 | · | 0.0132 | -0.0370 |
| 151508 | visium | gaussian_mixture | gaussian_mixture | 1.000 | Y | 0.0000 | -0.0375 |
| 151509 | visium | gaussian_mixture | gaussian_mixture | 1.000 | Y | 0.0000 | -0.0646 |
| 151510 | visium | agglomerative | gaussian_mixture | 1.000 | · | 0.0693 | +0.0468 |
| 151669 | visium | bisecting_kmeans | kmeans | 1.000 | · | 0.0174 | +0.0023 |
| 151670 | visium | spectral | kmeans | 1.000 | · | 0.1250 | +0.1250 |
| 151671 | visium | spectral | gaussian_mixture | 1.000 | · | 0.1812 | +0.1812 |
| 151672 | visium | kmeans | gaussian_mixture | 1.000 | · | 0.0509 | +0.0404 |
| 151673 | visium | kmeans | spectral | 1.000 | · | 0.0078 | +0.0000 |
| 151674 | visium | kmeans | spectral | 1.000 | · | 0.1219 | +0.0000 |
| 151675 | visium | gaussian_mixture | spectral | 1.000 | · | 0.1162 | +0.0000 |
| 151676 | visium | minibatch_kmeans | spectral | 1.000 | · | 0.0870 | +0.0000 |
| allen_merfish_brain_section | merfish | spectral | spectral | 1.000 | Y | 0.0000 | +0.0000 |
| merfish | merfish | spectral | kmeans | 1.000 | · | 0.0987 | +0.0987 |
| slideseqv2 | slideseq | agglomerative | gaussian_mixture | 1.000 | · | 0.0157 | +0.0017 |
| visium_hd_crc | visium | spectral | spectral | 1.000 | Y | 0.0000 | +0.0000 |
| visium_mouse_brain | visium | spectral | spectral | 1.000 | Y | 0.0000 | +0.0000 |
| xenium | xenium | agglomerative | spectral | 1.000 | · | 0.0319 | +0.0000 |
| xenium_lung_cancer | xenium | spectral | spectral | 1.000 | Y | 0.0000 | +0.0000 |
| xenium_ovarian_cancer | xenium | spectral | spectral | 1.000 | Y | 0.0000 | +0.0000 |

## Caveats
- Mean selection regret does not beat the global-best comparator (fallback / global_default remains justified).