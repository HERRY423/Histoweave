# gaussian_mixture

**Category:** domain detection · **Maturity:** validated · **Implementation:** external (sklearn)

Gaussian mixture model on a PCA + spatial-neighbourhood embedding.

## When to use

- Soft / overlapping compartments.
- Contiguous domains when spatial weight is turned on (`@sw0.8` often wins on DLPFC).
- Need per-spot responsibilities (probabilistic labels).

## When not to use

- Strongly non-elliptical domains without a spatial term.
- Huge n without mini-batch alternatives.

## Failure modes

- Expression-only GMM under-performs spatial-native methods on layered cortex.
- Sensitive to spatial_weight (largest lever in the variance-decomposition study).

## Evidence

DLPFC multi-slice landscape; frequent winner among high-spatial-weight configs.
