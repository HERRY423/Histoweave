# Tutorial 3 — Batch-effect correction: ComBat vs Harmony

Multi-slide spatial experiments carry technical batch effects (different runs, slides,
donors) that must be removed before joint clustering or domain detection. HistoWeave
ships two complementary integration methods:

| Method | Corrects | Backend | When to use |
| --- | --- | --- | --- |
| `combat` | the expression matrix `X` | pure NumPy (Johnson et al. 2007) | fast first pass; keeps a gene-level corrected matrix |
| `harmony` | a low-dimensional embedding (`obsm`) | `harmonypy` (Korsunsky et al. 2019) | integration for neighbours/clustering; preserves biology across strong batch effects |

This tutorial builds a two-slide dataset with a deliberate batch offset, applies both
methods, and quantifies how well each mixes the batches while preserving cell-type
structure.

!!! note "What you need"
    ```bash
    pip install "histoweave-spatial[scanpy,harmony]"
    ```

## 1. Build a two-batch dataset

We concatenate two synthetic slides that share the same three cell types but sit at
different expression baselines — a controllable stand-in for a real slide-to-slide
batch effect.

```python
import logging

import numpy as np
import pandas as pd
import histoweave as ts
from histoweave.data import SpatialTable
from histoweave.plugins import create_method

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

rng = np.random.default_rng(0)
n_genes = 60
centers = [rng.normal(0, 5, size=n_genes) for _ in range(3)]   # 3 shared cell types

def make_slide(offset, spots_per_type=60):
    blocks = [rng.normal(c, 1.0, size=(spots_per_type, n_genes)) for c in centers]
    X = np.vstack(blocks) + offset                              # per-slide baseline shift
    ct = np.repeat([f"type{i}" for i in range(3)], spots_per_type)
    return X, ct

X_a, ct_a = make_slide(offset=0.0)
X_b, ct_b = make_slide(offset=6.0)                             # slide B is shifted

X = np.vstack([X_a, X_b])
n = X.shape[0]
obs = pd.DataFrame({
    "batch": ["slideA"] * len(ct_a) + ["slideB"] * len(ct_b),
    "cell_type": np.concatenate([ct_a, ct_b]),
}, index=[f"spot{i}" for i in range(n)])
var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])

data = SpatialTable(X=X, obs=obs, var=var)
data.obsm["spatial"] = rng.random((n, 2)) * 100
_LOGGER.info("%s", data)
```

## 2. A metric for "how bad is the batch effect?"

We use two quantities on an embedding:

* **batch mixing** — distance between the two batch centroids (smaller = better mixed),
* **biology preserved** — distance between cell-type centroids (should stay large).

```python
from sklearn.decomposition import PCA

def batch_dist(emb, obs):
    a = obs["batch"].to_numpy() == "slideA"
    return float(np.linalg.norm(emb[a].mean(0) - emb[~a].mean(0)))

def celltype_spread(emb, obs):
    cents = [emb[(obs["cell_type"] == t).to_numpy()].mean(0)
             for t in obs["cell_type"].unique()]
    cents = np.stack(cents)
    return float(np.mean([np.linalg.norm(a - b)
                          for i, a in enumerate(cents) for b in cents[i + 1:]]))

raw_emb = PCA(n_components=15, random_state=0).fit_transform(np.asarray(data.X))
_LOGGER.info(
    "RAW    batch_dist=%.2f  celltype_spread=%.2f",
    batch_dist(raw_emb, data.obs),
    celltype_spread(raw_emb, data.obs),
)
```

## 3. Harmony (embedding-space correction)

```python
h = create_method("integration", "harmony",
                  batch_key="batch", n_pcs=15, theta=2.0,
                  max_iter_harmony=20, seed=0).run(data)

harmony_emb = h.obsm["X_pca_harmony"]
_LOGGER.info(
    "HARMONY batch_dist=%.2f  celltype_spread=%.2f",
    batch_dist(harmony_emb, h.obs),
    celltype_spread(harmony_emb, h.obs),
)
_LOGGER.info("provenance: %s %s", h.provenance[-1]["method"], h.uns["integration"])
```

Harmony pulls the batch centroids together (batch_dist drops) while keeping the
cell-type centroids apart (celltype_spread stays high) — batches mixed, biology kept.

## 4. ComBat (expression-space correction)

ComBat corrects `X` itself, which is what you want if a downstream method needs a
batch-corrected *gene* matrix rather than an embedding:

```python
c = create_method("integration", "combat", batch_key="batch").run(data)

combat_emb = PCA(n_components=15, random_state=0).fit_transform(np.asarray(c.X))
_LOGGER.info(
    "COMBAT  batch_dist=%.2f  celltype_spread=%.2f",
    batch_dist(combat_emb, c.obs),
    celltype_spread(combat_emb, c.obs),
)

# The pre-correction matrix is preserved so the step is reversible.
assert "pre_combat" in c.layers
```

## 5. Choosing between them

* Use **Harmony** when the next step consumes an embedding (neighbour graph →
  Leiden/BANKSY domains, UMAP). It handles strong, non-linear batch effects best and
  never touches your counts.
* Use **ComBat** when a downstream method needs a corrected expression matrix, or when
  you want a fast, interpretable, dependency-light first pass.
* You can chain them: ComBat on `X`, then Harmony on a PCA of the corrected matrix.

Feed the integrated result straight into domain detection:

```python
# BANKSY/Leiden pick up obsm["X_pca_harmony"] as the embedding to cluster.
domains = create_method("domain_detection", "banksy", n_domains=3).run(h)
```

## Runnable script

[`examples/tutorial_batch_correction.py`](https://github.com/HERRY423/Histoweave/blob/main/examples/tutorial_batch_correction.py)
runs the full comparison and prints the metric table:

```bash
python examples/tutorial_batch_correction.py
```
