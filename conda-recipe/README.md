# HistoWeave conda-forge feedstock

This directory contains the conda recipe for publishing `histoweave-spatial`
on [conda-forge](https://conda-forge.org/).

## One-time feedstock setup

1. Fork [staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `conda-recipe/meta.yaml` into `staged-recipes/recipes/histoweave-spatial/meta.yaml`
3. Open a PR — the conda-forge bot will review, build, and auto-create
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
