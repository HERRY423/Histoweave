# banksy_py

**Category:** domain detection · **Maturity:** validated · **Implementation:** native

Neighbourhood-augmented domain detection following the BANKSY idea (own expression +
neighbourhood mean + azimuthal gradient features → PCA → k-means).

## When to use

- Contiguous tissue domains (layers, niches, compartments).
- You want a **spatially native** method without Docker/R.
- You will not tune spatial weight carefully — BANKSY is relatively flat to λ.

## When not to use

- Pure cell-type identity recovery (prefer annotation methods / expression-first clustering).
- You need the **official** Bioconductor BANKSY paper claim → use `banksy` (R container).
- Very large imaging assays without subsampling (pair with tiled analysis).

## Failure modes

- Missing `obsm['spatial']` → hard error.
- `n_domains` omitted and `uns['n_domains']` absent → hard error.
- Extreme λ → over-smoothing of thin layers.

## Evidence

Multi-slice DLPFC ARI landscape + variance-decomposition factorial (see
`VALIDATION_EVIDENCE` in `release_manifest.py`).
