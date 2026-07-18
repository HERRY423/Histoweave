# Validation report — `spectral`

**Decision:** scientific **validated**  
**Category:** domain_detection  
**Protocol:** `histoweave.landscape.dlpfc_real.v1` + `dlpfc_spatial_aware.v1`

## Datasets

DLPFC Visium five-slice difficulty gradient (Maynard et al. 2021).

## Metric

ARI vs manual layers across seeds and spatial_weight policies (`@sw0.0/0.3/0.8`).

## Outcome

sklearn SpectralClustering on the spatial-neighbourhood PCA embedding is a consistent
strong baseline on DLPFC and frequently ranks near the top of spatial-aware grids.

## Limitations

- Requires a declared or estimated `n_domains`.
- Spectral methods can be sensitive to graph construction and feature scaling.

## Evidence pointer

`VALIDATION_EVIDENCE["spectral"]` in `release_manifest.py`; landscape CSVs under
`5x10_dlpfc_benchmark/` and `5x15_spatial_aware/`.
