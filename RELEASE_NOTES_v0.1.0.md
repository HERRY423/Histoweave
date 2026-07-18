# HistoWeave v0.1.0 — submission freeze

**Date:** 2026-07-18  
**Package:** `histoweave-spatial==0.1.0`  
**Tag:** `v0.1.0`

## Validation ledger (canonical)

| Kind | Maturity | n | Methods |
|------|----------|--:|---------|
| Scientific | `validated` | **10** | agglomerative, banksy, banksy_py, birch, gaussian_mixture, graphst, minibatch_kmeans, spagcn, spectral, stagate |
| Contract | `contract_validated` | **3** | cell2location, rctd, spatialde |
| **Total multi-dataset packages** | — | **13** | `10 + 3` |

Source of truth: `src/histoweave/plugins/builtin/release_manifest.py`.

## Highlights

- Maturity ladder: experimental → beta → production → **contract_validated** → **validated**
- Task contracts, non-oracle K defaults, honest recommender baselines
- Digital-twin validation, spatial AutoML, failure fingerprints, active calibration
- PyPI: `histoweave-spatial` via Trusted Publishing on GitHub Release
- Immutable archive: Zenodo DOI minted from this GitHub Release (see CITATION.cff)

## Install

```bash
pip install histoweave-spatial==0.1.0
histoweave version
histoweave run --demo --out report.html
```

## Citation

Cite this software version and the original method papers it orchestrates.
After Zenodo archives the release, record the version DOI in `CITATION.cff`.
