# Method guide (when to use / when not)

HistoWeave documents **every registered method**. Use this page to start,
the [complete catalog](catalog.md) for the full inventory, and category
pages for decision notes within a task.

**Inventory:** 85 registered · 30 field-facing (production/beta/validated in analysis categories) · 5 deep hand-written guides.

## Deep guides (start here)

| Method | Maturity | Start here when… |
|--------|----------|------------------|
| [BANKSY (native)](banksy_py.md) | validated | Robust spatial-domain default without R |
| [Spectral clustering](spectral.md) | validated | Contiguous domains, known/defensible *k* |
| [Gaussian mixture](gaussian_mixture.md) | validated | Soft domains / elliptical compartments |
| [SpaGCN](spagcn.md) | beta | Visium-scale graph-conv SOTA comparison |
| [cell2location](cell2location.md) | beta | Spot deconvolution with a scRNA reference |

## Complete coverage

- **[Full catalog](catalog.md)** — every registered method in one table
- **Category guides:**

    - [annotation](categories/annotation.md) (4)
    - [ccc](categories/ccc.md) (1)
    - [deconvolution](categories/deconvolution.md) (4)
    - [domain_detection](categories/domain_detection.md) (20)
    - [ingestion](categories/ingestion.md) (7)
    - [integration](categories/integration.md) (12)
    - [neighborhood](categories/neighborhood.md) (4)
    - [normalization](categories/normalization.md) (12)
    - [qc](categories/qc.md) (7)
    - [segmentation](categories/segmentation.md) (1)
    - [svg](categories/svg.md) (10)
    - [virtual_st](categories/virtual_st.md) (3)

!!! tip "Selection under uncertainty"
    Prefer a short multi-method ensemble + boundary-uncertainty map when two
    configurations are within ~0.03 ARI, or when the recommender does **not**
    beat the global-best baseline.  Use non-oracle *K* (`k_policy='estimate'`)
    for realistic domain benchmarks — see [Statistical review](../statistical-review.md).

Related: [Method selection guide](../method-selection.md) ·
[Method lifecycle](../method-lifecycle.md) ·
[Research incubator](../research-methods.md) ·
[Contributing](https://github.com/HERRY423/Histoweave/blob/main/CONTRIBUTING.md)
