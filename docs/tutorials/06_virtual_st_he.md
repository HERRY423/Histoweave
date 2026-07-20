# Tutorial 6 — Virtual ST from real H&E (Visium mouse brain)

Predict spatial gene expression from a registered **H&E** image on a public
10x Visium adult mouse brain slide, then score predictions against **measured**
expression.

You will:

1. load a real paired H&E + ST table,
2. run three virtual_st methods (morphology / sCellST-style / STORM-style),
3. read mean gene Pearson correlations,
4. (optional) prepare a registry bundle for offline reuse.

!!! note "What you need"
    ```bash
    pip install "histoweave-spatial[io,spatial,scanpy]"
    # optional: squidpy for first-time download of the Visium H&E slide
    pip install squidpy
    ```
    The first squidpy download is ~330 MB and is cached under the working
    directory / squidpy cache. Prefer a local path if you already have
    `visium_hne_adata.h5ad`.

## 1. Load real H&E + expression

```python
import logging

from histoweave.datasets import load_visium_hne_paired
from histoweave.plugins.builtin import register_all

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)
register_all()

# Uses (in order): --source path, repo data/anndata/visium_hne_adata.h5ad,
# datasets_cache, or squidpy.datasets.visium_hne_adata().
data = load_visium_hne_paired(prefer="lowres", n_hvg=1000)
_LOGGER.info("%s", data)
_LOGGER.info("images: %s", {k: v.shape for k, v in data.images.items()})
_LOGGER.info("spatial: %s", None if data.spatial is None else data.spatial.shape)
```

The table carries:

* `X` / `layers['counts']` — measured expression (virtual_st ground truth),
* `obsm['spatial']` — spot coordinates aligned to the image,
* `images['image']` — canonical H&E array (lowres by default).

## 2. Run virtual ST predictors

```python
from histoweave.plugins import MethodCategory, create_method

methods = [
    "virtual_st_morphology",
    "virtual_st_scellst",
    "virtual_st_storm",
]
results = {}
for name in methods:
    out = create_method(
        MethodCategory.VIRTUAL_ST,
        name,
        mode="paired",
        image_key="image",
        n_genes=64,
        seed=0,
    ).run(data)
    meta = out.uns["virtual_st"][name]
    results[name] = meta
    _LOGGER.info(
        "%s  pearson=%.3f  spearman=%.3f  genes=%s",
        name,
        meta["mean_gene_pearson"],
        meta["mean_gene_spearman"],
        meta["n_genes_predicted"],
    )
```

Predicted expression lands in `layers['virtual_st']`. Morphology embeddings
land in `obsm['X_virtual_st']` for optional downstream domain methods.

## 3. Harness leaderboard

```python
from histoweave.benchmark import get_task, run_benchmark

task = get_task("virtual_st", dataset=data)
board = run_benchmark(
    task,
    methods=methods,
    method_params={
        name: {"mode": "paired", "n_genes": 64, "seed": 0} for name in methods
    },
)
for row in board.leaderboard:
    _LOGGER.info("rank %s  %s  score=%s", row["rank"], row["method"], row["score"])
```

Scores are **mean per-gene Pearson** correlation against measured expression —
not ARI.

## 4. Your own Visium `outs/` folder

```python
from histoweave.io import read
from histoweave.datasets import prepare_virtual_st_table

data = read("visium", "/path/to/spaceranger/outs")
data = prepare_virtual_st_table(data)  # attaches tissue_*_image.png when present

result = create_method(
    MethodCategory.VIRTUAL_ST,
    "virtual_st_storm",
    mode="paired",
    image_key="image",
).run(data)
```

## 5. Offline registry bundle (optional)

```bash
python scripts/prepare_visium_hne_virtual_st.py \
  --source data/anndata/visium_hne_adata.h5ad \
  --n-hvg 2000
```

Then:

```python
from histoweave.datasets import get_dataset, prepare_virtual_st_table

entry = get_dataset("visium_mouse_brain_hne")
data = prepare_virtual_st_table(entry.load())
_LOGGER.info("%s", entry.task_contract())
```

## Claim boundary

* This tutorial evaluates **expression agreement** under a paired H&E–ST contract.
* It does **not** establish clinical-grade virtual ST or a universal best predictor.
* Domain-partition rankings from protein/chromatin modalities are never mixed into
  this task (see [multimodal & virtual ST](../multimodal-virtual-st.md)).

## Next steps

* [Multimodal tasks & virtual ST](../multimodal-virtual-st.md)
* [Decision protocol](../decision-protocol.md)
* [Tutorial 1 — real Visium DLPFC](01_real_visium_dlpfc.md)
