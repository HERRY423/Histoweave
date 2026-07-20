# Category: virtual_st

H&E (or other registered histology) → predicted spatial transcriptomics.

## Methods

| Name | Maturity | Summary |
|------|----------|---------|
| `virtual_st_morphology` | beta | Patch-statistic morphology → expression baseline |
| `virtual_st_scellst` | beta | sCellST-inspired weakly supervised morphology encoder |
| `virtual_st_storm` | beta | STORM-inspired hierarchical morphology + spatial fusion |

## Contract

* **Analysis task:** `AnalysisTask.VIRTUAL_ST`
* **Ground truth:** `GroundTruthKind.MEASURED_EXPRESSION` (or `none` for inference-only)
* **Primary metric:** `mean_gene_pearson`
* **Required inputs:** `images[image_key]`, `obsm['spatial']`; paired mode also needs measured expression in `X` or `expression_layer`
* **Primary outputs:** `layers['virtual_st']`, `obsm['X_virtual_st']`, `uns['virtual_st']`

## Cross-modal rule

Virtual ST evidence is **never** mixed with `spatial_domain` /
`spatial_protein_domain` / `spatial_chromatin_domain` rankings. See
[multimodal & virtual ST](../../multimodal-virtual-st.md).

## Usage

```python
from histoweave.datasets import load_visium_hne_paired
from histoweave.plugins import MethodCategory, create_method
from histoweave.plugins.builtin import register_all

register_all()
data = load_visium_hne_paired(prefer="lowres", n_hvg=1000)
result = create_method(
    MethodCategory.VIRTUAL_ST,
    "virtual_st_storm",
    mode="paired",
    image_key="image",
).run(data)
```
