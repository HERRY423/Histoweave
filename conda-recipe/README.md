# HistoWeave conda-forge feedstock

This directory contains the conda recipe for publishing `histoweave-spatial`
on [conda-forge](https://conda-forge.org/).

## Keeping the recipe in sync

The recipe pins a `version` that **must** match `__version__` in
`src/histoweave/__init__.py`. When you cut a release:

1. Bump `{% set version = "..." %}` in `meta.yaml` to the new version.
2. Publish to PyPI first (the `Publish to PyPI` workflow does this on GitHub Release).
3. Fetch the sha256 of the **PyPI** sdist and paste it over `PLACEHOLDER`:
   ```bash
   VERSION=0.1.0b1
   curl -sL "https://pypi.io/packages/source/h/histoweave-spatial/histoweave_spatial-${VERSION}.tar.gz" \
     | sha256sum
   ```
   (On the feedstock itself the conda-forge autotick bot does this for you on each
   subsequent release; you only paste it by hand for the initial staged-recipes PR.)

The core recipe intentionally lists only the light runtime deps (numpy, pandas,
jinja2). Method-specific stacks (harmonypy, spatialdata, scvi-tools, cellpose, nnSVG's
R side) are optional pip extras and are **not** conda `run` deps — they stay opt-in.

## One-time feedstock setup

1. Fork [staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `conda-recipe/meta.yaml` into `staged-recipes/recipes/histoweave-spatial/meta.yaml`
3. Fill the real sha256 (see above) into `source.sha256`
4. Open a PR — the conda-forge bot will review, build, and auto-create
   `conda-forge/histoweave-spatial-feedstock`

## Local test build

```bash
conda install conda-build
conda build conda-recipe/
```

## Post-publication install

```bash
conda install -c conda-forge histoweave-spatial
```
