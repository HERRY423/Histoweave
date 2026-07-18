# spectral

**Category:** domain detection · **Maturity:** validated · **Implementation:** external (sklearn)

Spectral clustering on a spatial k-NN affinity of a PCA (+ optional spatial-weight)
embedding.

## When to use

- Layered / manifold-shaped domains with a defensible domain count.
- You want a strong **sklearn baseline** that still respects geometry via affinity.
- Spatial-weight sweeps (`spectral@sw0.3`, `spectral@sw0.8`).

## When not to use

- Unknown *k* and highly irregular density → try density methods (`dbscan` / `optics`).
- You need published spatial GNN SOTA → SpaGCN / GraphST / STAGATE.

## Failure modes

- `n_neighbors` larger than n_obs − 1 (auto-clamped, but graphs become near-complete).
- Giving oracle *k* inflates ARI vs real analyses — report k-misspecification.

## Evidence

DLPFC 5-slice spatial-aware landscape (often top configuration at high spatial weight).
