# Troubleshooting

Concrete symptoms → causes → fixes. Ordered roughly by how often they bite.

---

## Installation

### `ModuleNotFoundError` for scanpy / scvi / sklearn

The core install is intentionally minimal. Method backends live in extras:

```bash
pip install "histoweave-spatial[scanpy]"   # sklearn domain detection
pip install "histoweave-spatial[scanvi]"   # scANVI
pip install "histoweave-spatial[all]"      # everything
```

### `histoweave: command not found`

The console script is not on `PATH` (common in fresh virtualenvs). Run it as a module:

```bash
python -m histoweave --help
```

---

## R-backed methods (BANKSY, nnSVG)

### `RuntimeError: R container not available` / Docker errors

R-backed plugins run inside the `histoweave-r` container. If Docker is missing or the image
is not pulled:

* **Preferred:** use the native Python fallback — e.g. `banksy_py` instead of `banksy`.
  It reproduces BANKSY feature construction without R/Docker.
* Or install Docker and pre-pull the image so the first run is not blocked on a download.

```bash
# check what the wrapper sees
docker info            # must succeed
python -c "import histoweave, logging; logging.basicConfig(level=logging.INFO); logging.info('ok')"
```

!!! note
    `banksy_py` is a native re-implementation, not the Bioconductor package. Use it for
    Docker-free runs and prototyping; use the R `banksy` wrapper for canonical results.

---

## scANVI / deep-learning methods

### Training is very slow

Without a GPU, scVI/scANVI train on CPU. Expect tens of seconds to minutes for a few
thousand spots. To keep it tractable:

* Lower `max_epochs` (50 for scVI / 25 for scANVI is enough on small slices).
* Subset to HVGs (2000 is plenty) before training.
* Set `accelerator="cpu"` explicitly to avoid CUDA probing overhead.

### `KeyError: 'counts'` when building scANVI

scVI needs **raw counts**, not normalized values. Stash them before normalizing:

```python
adata.layers["counts"] = adata.X.copy()   # BEFORE normalize_total/log1p
```

and pass that layer to the model setup.

### Results change every run

Set seeds. Deep models are stochastic:

```python
import scvi; scvi.settings.seed = 0
```

---

## Data ingestion

### Checksum / hash mismatch on download

The dataset registry verifies integrity. A mismatch means a truncated or partially updated
download. Fixes:

* Delete the cached file and re-download (partial downloads are the usual cause).
* Verify network stability; retry a large H5 over a fresh connection.
* If a public source legitimately changed, update the expected hash in the dataset entry.

### `in_tissue` filtering drops all spots / coordinate mismatch

Barcodes in the positions file must match the count matrix. Symptoms: zero spots after
join, or coordinates that look scrambled. Check:

* Positions columns are `barcode, in_tissue, array_row, array_col, pxl_row, pxl_col`.
* You joined on **barcode**, not row order.
* Only `in_tissue == 1` spots are kept.

### Imaging data (Xenium/MERFISH) has empty or tiny cells

Segmentation artefacts. Filter on counts/area before analysis (`min_counts`, `min_genes`),
and prefer variance-stabilizing normalization (`arcsinh`/`sqrt`) for small panels. See the
[Xenium/MERFISH tutorial](tutorials/04_xenium_merfish.md).

---

## Reports & Vitessce

### Vitessce viewer does not load

The report degrades gracefully to static SVG plots, so your results are never lost. The
viewer needs to fetch Vitessce + React from CDN:

* **Offline / air-gapped:** expected — use the static plots, or host the JS bundles locally
  and point the template at them.
* **Blocked scripts / strict CSP:** allow `cdn.jsdelivr.net` and `esm.sh`.
* **Old browser:** Vitessce needs a modern evergreen browser (ES modules + import maps).
* Open the browser dev console; a red network/import error confirms the CDN cause.

### Cluster selection does not update the heatmap

The linked-views behaviour requires the enhanced template and a config carrying per-cluster
markers. Check:

* The report was generated after the linked-views enhancement (config contains
  `cluster_top_markers`).
* The scatterplot, obsSets, and heatmap share the same coordination scope.
* No JS error in the console when you click a cluster.

### Heatmap is empty

No marker genes were embedded for the selected cluster (e.g. a cluster with too few cells).
Try another cluster, or regenerate with a larger `markers_per_cluster`.

---

## Benchmarking & recommendation

### `recommend` returns odd rankings

The recommender is only as good as its knowledge base. If rankings look off:

* Confirm the knowledge base was built on datasets comparable to yours (assay, tissue).
* Rebuild with more reference datasets — sparse KBs generalize poorly.
* Remember rankings are *similarity-weighted*: very unusual input data has few close
  neighbours and wider uncertainty.

### ARI is much lower than expected

A low ARI is not automatically evidence that the spatial term is too weak. First confirm
that the reference labels represent the intended output.

* For expert spatial-domain labels, compare `spatial_weight=0.3` and `0.8` against an
  expression-only control.
* For cell-type or transcriptomic proxy labels, begin with `spatial_weight=0.0`; heavy
  smoothing can reduce ARI substantially.
* Confirm `n_domains`, normalization, retained observations, and label granularity are
  matched across candidates.
* If only proxy labels exist, report the score as proxy-label recovery rather than
  spatial-domain accuracy.

See the [method-selection guide](method-selection.md) for the benchmark evidence and the
full validation checklist.

---

## Still stuck?

* Re-run with `--verbose` (CLI) to see the provenance/step log.
* Check the [FAQ](faq.md) and [method-selection guide](method-selection.md).
* File an issue with the provenance log and your HistoWeave version (`histoweave --version`).
