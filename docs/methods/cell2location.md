# cell2location

**Category:** deconvolution · **Maturity:** validated · **Implementation:** external

Bayesian spatial cell-type abundance estimation with a scRNA-derived reference signature.

## When to use

- Spot-based assays (Visium / Slide-seq) with a matched single-cell reference.
- You need absolute abundance estimates, not just hard domains.

## When not to use

- No reference signature in `uns[reference_key]` — hard error (no marker-score fallback).
- Imaging single-cell platforms where segmentation + annotation is more appropriate.

## Failure modes

- Normalized values instead of raw counts → validation error.
- Too few shared genes with the reference.
- Long GPU/CPU training (`max_epochs` defaults are production-scale).

## Evidence

Multi-dataset structural validation (`histoweave.method_validation.cell2location_structural.v1`)
on 3 synthetic marker mixtures + DLPFC 151507/669/673 (subsampled): contract success 6/6,
shared-gene coverage, abundance/proportion simplex; no marker-score fallback.
Formal report: [validation/cell2location.md](validation/cell2location.md).

Real Pyro posterior training requires `histoweave-spatial[cell2location]` (out of CI scope).
