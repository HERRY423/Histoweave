# Tutorial 1 — End-to-end analysis of a real Visium slide (DLPFC)

This tutorial runs a complete HistoWeave workflow on a **real 10x Visium** dataset:
one dorsolateral prefrontal cortex (DLPFC) slice from Maynard et al. (2021), which
ships hand-annotated cortical-layer ground truth via *spatialLIBD*. You will:

1. pull the slide from the versioned dataset registry,
2. run QC and normalization,
3. build a PCA embedding and integrate it with **Harmony**,
4. annotate cell types with **scANVI**,
5. rank spatially variable genes with **nnSVG**,
6. detect spatial domains and score them against the ground-truth layers,
7. write a self-contained interactive report.

!!! note "What you need"
    ```bash
    pip install "histoweave-spatial[io,spatial,scanpy,scanvi,harmony]"
    ```
    Steps that wrap R/Bioconductor methods (**nnSVG**) additionally need the
    `histoweave-r` container image (`ghcr.io/herry423/histoweave-r`); the
    tutorial shows how to run those steps and how to skip them if the image is not
    available. The first run downloads ~50 MB and caches it under
    `~/.cache/histoweave/datasets`.

## 1. Load the slide

```python
import logging

import histoweave as ts
from histoweave.datasets import get_dataset, list_datasets

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# The registry lists every benchmark-ready public dataset.
for d in list_datasets():
    if d["assay"] == "visium":
        _LOGGER.info("%s - %s", d["name"], d["description"])

entry = get_dataset("dlpfc_151507")
data = entry.load()          # -> SpatialTable, cached & checksum-verified
_LOGGER.info("%s", data)                  # SpatialTable(n_obs=..., n_vars=..., obsm=[spatial], ...)
```

The loaded `SpatialTable` carries the spot × gene count matrix in `X`, the array
coordinates in `obsm["spatial"]`, and (when present) the *spatialLIBD* layer labels in
`obs`. Everything downstream is written against `SpatialTable`, so the exact same code
runs on synthetic data, other Visium slides, or Xenium/CosMx.

## 2. QC and normalization

```python
from histoweave.plugins import create_method

data = create_method("qc", "basic_qc").run(data)
data = create_method("qc", "mitochondrial_qc").run(data)

# Log-normalize; the raw counts are moved to layers["counts"] so count-based
# methods (scANVI, nnSVG) can still recover them.
data = create_method("normalization", "log1p_cp10k").run(data)
```

Every `run(...)` returns a **new** `SpatialTable` with a provenance entry appended, so
the full processing chain stays auditable:

```python
for step in data.provenance:
    _LOGGER.info("%s -> %s %s", step["step"], step["method"], step["method_version"])
```

## 3. Batch integration with Harmony

A single slide has no batch structure, but the moment you concatenate slides (see
[Tutorial 3](03_batch_effect_correction.md)) you need integration. Harmony corrects a
low-dimensional embedding rather than the expression matrix:

```python
# Harmony builds a PCA embedding on the fly if obsm["X_pca"] is absent.
data = create_method(
    "integration", "harmony",
    batch_key="sample_id",      # any obs column; a single value is a no-op pass-through
    n_pcs=30,
    theta=2.0,
).run(data)

# Corrected embedding lands in obsm["X_pca_harmony"], ready for neighbours/clustering.
_LOGGER.info("%s", data.obsm["X_pca_harmony"].shape)
```

## 4. Cell-type annotation with scANVI

scANVI is semi-supervised: seed a subset of spots with labels (here, the ground-truth
layer for a fraction of spots) and it propagates calibrated labels to the rest.

```python
import numpy as np

# Build a partial-label column: keep 30% of known layers, blank the rest.
rng = np.random.default_rng(0)
if "spatialLIBD_layer" in data.obs:
    seed = data.obs["spatialLIBD_layer"].astype("object").to_numpy().copy()
    mask = rng.random(data.n_obs) > 0.30
    seed[mask] = "Unknown"
    data.obs["cell_type_seed"] = seed

    data = create_method(
        "annotation", "scanvi",
        labels_key="cell_type_seed",
        unlabeled_category="Unknown",
        layer="counts",          # scANVI needs raw counts
        scvi_epochs=50, scanvi_epochs=25,   # small for a tutorial; raise for real runs
    ).run(data)
    _LOGGER.info("%s", data.obs[["cell_type", "scanvi_confidence"]].head())
```

## 5. Spatially variable genes with nnSVG

nnSVG fits a nearest-neighbour Gaussian process per gene and returns a calibrated,
multiple-testing-corrected ranking. It runs inside the `histoweave-r` container:

```python
try:
    data = create_method(
        "svg", "nnsvg",
        n_top=50, n_neighbors=10, assay_name="logcounts",
    ).run(data)
    top = data.var.sort_values("nnsvg_rank").head(10)
    _LOGGER.info("%s", top[["nnsvg_rank", "nnsvg_LR_stat", "nnsvg_padj"]])
except (RuntimeError, FileNotFoundError) as exc:
    # No R container available — fall back to the pure-Python Moran's I SVG method.
    _LOGGER.warning("nnSVG unavailable, using Moran's I: %s", exc)
    data = create_method("svg", "morans_i", n_top=50).run(data)
```

Running the R step from the command line instead:

```bash
histoweave step --in dlpfc.ttab --category svg --method nnsvg \
    --param n_top=50 --param n_neighbors=10
```

## 6. Spatial domains + evaluation

```python
from histoweave._math import adjusted_rand_index

data = create_method("domain_detection", "banksy", n_domains=7).run(data)

if "spatialLIBD_layer" in data.obs:
    ari = adjusted_rand_index(
        data.obs["spatialLIBD_layer"].to_numpy(),
        data.obs["domain"].to_numpy(),
    )
    _LOGGER.info("Domain ARI vs spatialLIBD layers: %.3f", ari)
```

## 7. Interactive report

```python
out = ts.build_report(data, "dlpfc_report.html")
_LOGGER.info("Open %s", out)
```

The HTML report is self-contained (CDN-loaded Vitessce viewer, no Python server) and
embeds the spatial scatter, the domain map, the SVG table, and the full provenance
chain.

## Runnable script

A non-interactive version that degrades gracefully when optional
dependencies/containers are missing is provided at
[`examples/tutorial_real_visium.py`](https://github.com/HERRY423/Histoweave/blob/main/examples/tutorial_real_visium.py):

```bash
python examples/tutorial_real_visium.py
```
