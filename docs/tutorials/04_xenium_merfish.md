# Tutorial 4 — Imaging-based data: Xenium & MERFISH

The first three tutorials used **sequencing-based** spatial data (Visium spots). This one
covers **imaging-based, single-cell-resolution** platforms — 10x **Xenium** and Vizgen
**MERFISH** — which differ enough that the defaults change. You will:

1. ingest a Xenium output folder and a MERFISH output folder,
2. apply QC and normalization tuned for small, low-count panels,
3. detect spatial domains at single-cell resolution,
4. write an interactive report.

!!! note "What you need"
    ```bash
    pip install "histoweave-spatial[io,spatial,scanpy]"
    ```
    The `spatial` extra pulls in `spatialdata-io`, used as the backend for the MERFISH /
    CosMx / MERSCOPE readers. Xenium can be read either through `spatialdata-io`
    (`engine="spatialdata"`) or with the built-in native reader (`engine="native"`), which
    needs only `cell_feature_matrix.h5`, `cells.parquet`/`cells.csv.gz`, and
    `experiment.xenium`.

## Why imaging-based data is different

| Property | Visium (sequencing) | Xenium / MERFISH (imaging) |
|---|---|---|
| Resolution | 55 µm spots (multi-cell) | single cell |
| Panel | whole transcriptome | targeted (~100–500 genes) |
| Counts / unit | high | **low** |
| Cell boundaries | none | from segmentation (can be noisy) |

Consequences for analysis:

* **Normalization:** library-size log (`log1p_cp10k`) can over-inflate noise on tiny panels.
  Prefer variance-stabilizing transforms — `arcsinh` or `sqrt`.
* **QC:** you must filter segmentation artefacts (empty / oversized cells) that don't exist
  in spot data.
* **Domain detection:** density is irregular (real cells, not a grid), so density-aware
  methods (`dbscan`/`optics`) and BANKSY behave well; keep the spatial term on.

## 1. Ingest

=== "Python — Xenium"

    ```python
    import histoweave as ts
    from histoweave.io import read

    # Native reader (no spatialdata-io required)
    data = read("xenium", "/path/to/xenium_output_bundle", engine="native")
    logging.info(data)   # SpatialTable(n_obs=<cells>, n_vars=<panel genes>, obsm=[spatial])
    ```

=== "Python — MERFISH"

    ```python
    import histoweave as ts
    from histoweave.io import read

    # MERFISH uses the spatialdata-io backend
    data = read("merfish", "/path/to/merfish_output", engine="spatialdata")
    logging.info(data)
    ```

=== "CLI"

    ```bash
    histoweave ingest --assay xenium --in /path/to/xenium_output --out sample.ttab
    histoweave ingest --assay merfish --in /path/to/merfish_output --out sample.ttab
    ```

## 2. QC — remove segmentation artefacts

Imaging data needs cell-level QC that spot data does not:

```python
import numpy as np

X = data.X
counts = np.asarray(X.sum(axis=1)).ravel()
n_genes = np.asarray((X > 0).sum(axis=1)).ravel()

# Typical thresholds for a ~300-gene panel; tune to your data's distribution.
keep = (counts >= 10) & (n_genes >= 5)
logging.info(f"keeping {keep.sum()} / {keep.size} cells")
data = data[keep]        # drop empty / poorly segmented cells
```

!!! tip
    Also inspect cell **area** if the reader exposes it (`data.obs`) and drop implausibly
    large cells — they are usually merged-segmentation artefacts.

## 3. Normalize — variance-stabilizing for small panels

```python
import numpy as np

X = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
lib = X.sum(axis=1, keepdims=True)
lib[lib == 0] = 1.0

# arcsinh on library-normalized counts — robust for low-count imaging panels
data_norm = np.arcsinh(X / lib * 1e4)
```

See the [method-selection guide](../method-selection.md#normalization) for when to prefer
`arcsinh` vs `sqrt` vs `log1p_cp10k`.

## 4. Detect spatial domains

Because cells sit on an irregular lattice, keep the **spatial term** on — it was the single
biggest accuracy driver in the HistoWeave variance experiment.

=== "BANKSY (robust default)"

    ```python
    from histoweave import get_method

    banksy = get_method("banksy_py")
    labels = banksy.run(data_norm, coords=data.obsm["spatial"],
                        lambda_param=0.8, k_geom=15, n_pcs=20)
    ```

=== "Density-aware (irregular cell density)"

    ```python
    from histoweave import get_method

    dbscan = get_method("dbscan")
    labels = dbscan.run(data_norm, coords=data.obsm["spatial"], spatial_weight=0.5)
    ```

!!! tip "Let the recommender choose"
    ```bash
    histoweave recommend --in sample.ttab --knowledge-base kb/landscape.json --json
    ```
    The engine reads target-free features from your imaging data and ranks methods by
    performance on the most similar reference datasets — no labels required.

## 5. Report

```python
from histoweave import build_report

data.obs["domain"] = labels
build_report(data, out_path="xenium_report.html")
```

Open `xenium_report.html`: the interactive Vitessce panel shows cells in space, and
selecting a cluster highlights it everywhere and updates the marker-gene heatmap (see the
[linked-views FAQ](../faq.md#reports-vitessce)). If the
viewer can't load, the report falls back to static SVG plots.

## Recap

* Imaging data is single-cell, low-count, small-panel — **normalize with `arcsinh`/`sqrt`**
  and **QC out segmentation artefacts**.
* Keep the **spatial term on** for domain detection; BANKSY and density-aware methods suit
  irregular cell layouts.
* The same report + recommender workflow applies as for Visium.

## See also
- [Method selection](../method-selection.md) · [FAQ](../faq.md) · [Troubleshooting](../troubleshooting.md)
- [Tutorial 1 — Visium DLPFC](01_real_visium_dlpfc.md)
