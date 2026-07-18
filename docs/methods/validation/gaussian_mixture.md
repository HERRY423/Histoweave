# Validation report — `gaussian_mixture`

**Decision:** scientific **validated**  
**Category:** domain_detection  
**Protocol:** `histoweave.landscape.dlpfc_real.v1` + `dlpfc_spatial_aware.v1`

## Datasets

DLPFC Visium five-slice difficulty gradient (Maynard et al. 2021).

## Metric

ARI vs manual layers; multi-seed means across spatial_weight configurations.

## Outcome

sklearn GaussianMixture with spatial-context blending is repeatedly among the best
configurations on the DLPFC spatial-aware landscape.

## Limitations

- Probabilistic labels can be unstable at very small `n` or weak signal.
- Still a general-purpose mixture model, not a tissue-specific spatial prior.

## Evidence pointer

`VALIDATION_EVIDENCE["gaussian_mixture"]` in `release_manifest.py`.
