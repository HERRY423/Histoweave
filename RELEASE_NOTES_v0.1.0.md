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

## Post-release ops (maintainers)

### PyPI (`histoweave-spatial==0.1.0`)

GitHub Actions **Publish to PyPI** builds the wheel from tag `v0.1.0` successfully.
Trusted Publishing failed once with `invalid-publisher` (no matching publisher for
`repo:HERRY423/Histoweave:environment:pypi`).

Configure once on [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/):

| Field | Value |
|-------|--------|
| Owner | `HERRY423` |
| Repository | `Histoweave` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Then re-run the failed workflow:

```bash
gh run rerun 29641064542 --failed
```

Or upload the already-built artifacts from that run with a PyPI API token:

```bash
gh run download 29641064542 -n python-package-distributions -D dist-pypi
python -m twine upload dist-pypi/*
```

### Zenodo DOI (immutable archive)

1. Enable GitHub → Zenodo integration for `HERRY423/Histoweave`.
2. Toggle the repository ON in Zenodo; the next (or this) GitHub Release is archived.
3. Copy the version DOI into `CITATION.cff` (`doi: 10.5281/zenodo.…`) and optionally
   re-tag a patch if the journal requires the DOI in the frozen tree.

`.zenodo.json` ships with this release for metadata.
